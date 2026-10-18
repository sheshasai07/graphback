"""
GraphRAG - Fast Offline Entity & Relationship Extractor
Extracts named entities, domain terms, key-value data, and semantic relations
purely using local rule-based heuristics & regex with zero LLM API costs and sub-millisecond latency.
Specially tuned for tabular data (Excel, CSV, database dumps) and technical documentation.
"""

import re
from typing import List, Dict, Any, Tuple, Set
from loguru import logger

STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once",
    "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
    "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd",
    "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where",
    "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
    "yourselves", "true", "false", "yes", "no", "none", "null", "item", "row", "sheet", "columns",
    "excel", "spreadsheet", "file", "document", "nan", "n/a", "id", "ii", "iv", "vi", "ok", "vs", "code"
}

RELATION_TRIGGERS = [
    (r"\b(connects\s+to|connects\s+with|interfaces\s+with)\b", "CONNECTS_TO"),
    (r"\b(maintained\s+by|managed\s+by|owned\s+by|authored\s+by)\b", "MAINTAINED_BY"),
    (r"\b(maintains|manages|owns|authors|leads)\b", "MANAGES"),
    (r"\b(depends\s+on|relies\s+on|requires|needs)\b", "DEPENDS_ON"),
    (r"\b(routes\s+to|forwards\s+to|redirects\s+to)\b", "ROUTES_TO"),
    (r"\b(authenticates|authorizes|validates|verifies)\b", "AUTHENTICATES"),
    (r"\b(stores|persists|caches|saves)\b", "STORES"),
    (r"\b(triggers|invokes|calls|emits|dispatches)\b", "CALLS"),
    (r"\b(part\s+of|belongs\s+to|member\s+of|contained\s+in)\b", "MEMBER_OF"),
    (r"\b(implements|uses|utilizes|leverages)\b", "USES"),
    (r"\b(produces|generates|returns|creates)\b", "PRODUCES"),
]

TYPE_PATTERNS = [
    (r"(?i)\b.*(Service|Gateway|Server|Worker|Daemon|API|Cluster|Queue|Broker|Database|DB|Vault|Store)\b", "Service"),
    (r"(?i)\b.*(Engineer|Architect|Manager|Developer|Lead|Director|Officer|Admin|Technician|Scientist|Analyst)\b", "Role"),
    (r"(?i)\b.*(Framework|Library|Protocol|Language|Runtime|SDK|Algorithm|Model|Format|Encoding)\b", "Technology"),
    (r"(?i)\b.*(Company|Corp|Inc|Department|Team|Group|Org|Organization|Division)\b", "Department"),
    (r"(?i)\b.*(Feature|Module|Component|Policy|Rule|Flow|Pipeline|Process|Method)\b", "Feature"),
    (r"(?i)\b.*(Laptop|MacBook|Dell|ThinkPad|Desktop|Device|Server|Hardware|Phone|iPhone|Android)\b", "Device"),
    (r"(?i)\b.*(Active|Suspended|Pending|Terminated|Disabled|Enabled|Locked)\b", "Status"),
]

# Patterns for values that should NEVER become Entity nodes (dates, numbers, currency, booleans)
IS_DATE_PATTERN = re.compile(r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}(?:\s+\d{2}:\d{2}(?::\d{2})?)?$|^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$")
IS_NUMERIC_PATTERN = re.compile(r"^[\$€£]?\s*[\d,]+(?:\.\d+)?\s*[%]?$")
IS_FILE_HEADER_PATTERN = re.compile(r"(?i)^(excel spreadsheet|sheet:|columns:|row \d+:|--- sheet:)")


class OfflineExtractor:
    """
    Zero-cost, offline NLP entity & relation extractor.
    Operates via compiled regular expressions, linguistic heuristics, and tabular parsers.
    """

    def __init__(self, max_entities_per_chunk: int = 25, max_relations_per_chunk: int = 35):
        self.max_entities = max_entities_per_chunk
        self.max_relations = max_relations_per_chunk

    @staticmethod
    def _is_non_entity_value(val: str) -> bool:
        """Check if a string is a date, numeric metric, or flag that should not be an entity."""
        v = val.strip()
        if not v or v.lower() in STOP_WORDS or len(v) <= 2:
            return True
        if IS_DATE_PATTERN.match(v):
            return True
        if IS_NUMERIC_PATTERN.match(v):
            return True
        if v.lower() in {"active", "suspended", "on leave", "pending", "terminated", "disabled", "enabled", "locked"}:
            return False
        if v.lower() in {"yes", "no", "true", "false", "y", "n", "nan", "none", "null", "n/a", "unnamed"}:
            return True
        if "unnamed:" in v.lower() or "license cost" in v.lower() or "support tickets" in v.lower():
            return True
        if any(v.lower().endswith(ext) for ext in [".xlsx", ".xls", ".csv", ".pdf", ".docx", ".txt"]):
            return True
        return False

    def _infer_type(self, key_or_name: str) -> str:
        k = key_or_name.strip()
        for pattern, ent_type in TYPE_PATTERNS:
            if re.search(pattern, k):
                return ent_type
        if re.search(r"(?i)(status|state)", k):
            return "Status"
        if re.search(r"(?i)(role|title|position|job)", k):
            return "Role"
        if re.search(r"(?i)(department|division|team|org)", k):
            return "Department"
        if re.search(r"(?i)(device|laptop|hardware|machine)", k):
            return "Device"
        if re.search(r"(?i)(access|tier|level|permission)", k):
            return "AccessLevel"
        if re.search(r"(?i)(name|person|user|employee|client|customer)", k):
            return "Person"
        return "Concept"

    def extract_entities_and_relations(
        self,
        text: str,
        chunk_id: str = "",
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if not text or not text.strip():
            return [], []

        entities_dict: Dict[str, Dict[str, Any]] = {}
        relations: List[Dict[str, Any]] = []

        # ─── 1. Tabular Structured Extraction (Excel, CSV, Database Rows) ────
        tabular_rows = []
        if " | " in text and (":" in text or "Row " in text):
            raw_splits = re.split(r"(?m)^(?=Row\s+\d+:)|(?<=\s)(?=Row\s+\d+:)", text)
            for r in raw_splits:
                r_clean = r.strip()
                if " | " in r_clean and ":" in r_clean:
                    tabular_rows.append(r_clean)

        if tabular_rows:
            for row_line in tabular_rows:
                # Strip leading "Row 1:" or "Row 26:" if present
                clean_row = re.sub(r"^Row\s+\d+:\s*", "", row_line.strip())
                parts = [p.strip() for p in clean_row.split("|") if ":" in p]
                if not parts:
                    continue

                row_dict: Dict[str, str] = {}
                for p in parts:
                    colon_idx = p.find(":")
                    if colon_idx != -1:
                        k = p[:colon_idx].strip()
                        v = p[colon_idx + 1:].strip()
                        if k and v:
                            row_dict[k] = v

                # Determine primary subject of the row
                subject_name = None
                subject_type = "Person"

                # Priority 1: Full Name / Name / Employee Name
                for name_key in ["Full Name", "Employee Name", "Name", "User Name", "Account Name", "Username", "Person"]:
                    if name_key in row_dict and row_dict[name_key].strip():
                        subject_name = row_dict[name_key].strip()
                        subject_type = "Person"
                        break

                # Priority 2: Employee ID / Account ID / User ID
                if not subject_name:
                    for id_key in ["Employee ID", "Account ID", "User ID", "Customer ID", "ID"]:
                        if id_key in row_dict and row_dict[id_key].strip():
                            subject_name = row_dict[id_key].strip()
                            subject_type = "Account"
                            break

                # Priority 3: First valid key
                if not subject_name and row_dict:
                    for k, v in row_dict.items():
                        if not self._is_non_entity_value(v):
                            subject_name = v
                            subject_type = self._infer_type(k)
                            break

                if not subject_name:
                    continue

                # Build rich subject description with all row attributes
                desc_parts = []
                for k, v in row_dict.items():
                    desc_parts.append(f"{k}: {v}")
                subject_desc = f"{subject_name} ({', '.join(desc_parts)})"

                # Register subject entity
                entities_dict[subject_name.lower()] = {
                    "name": subject_name,
                    "type": subject_type,
                    "description": subject_desc,
                }

                # Link subject to categorical row entities
                for k, v in row_dict.items():
                    if v == subject_name or self._is_non_entity_value(v):
                        continue

                    # Filter out purely non-entity fields (dates, amounts, numbers, emails)
                    if IS_DATE_PATTERN.match(v) or IS_NUMERIC_PATTERN.match(v) or "@" in v:
                        continue

                    k_lower = k.lower()
                    if re.search(r"\b(cost|tickets?|dates?|resets?|created|updated|price|amounts?|counts?|ytd|spent)\b", k_lower):
                        continue
                    if any(term in k_lower for term in ["department", "division", "team"]):
                        ent_type = "Department"
                        rel_type = "MEMBER_OF"
                    elif any(term in k_lower for term in ["role", "title", "job", "position"]):
                        ent_type = "Role"
                        rel_type = "HAS_ROLE"
                    elif any(term in k_lower for term in ["device", "laptop", "hardware", "machine"]):
                        ent_type = "Device"
                        rel_type = "ASSIGNED_DEVICE"
                    elif any(term in k_lower for term in ["status", "state"]):
                        ent_type = "Status"
                        rel_type = "ACCOUNT_STATUS"
                    elif any(term in k_lower for term in ["access", "tier", "level", "permission"]):
                        ent_type = "AccessLevel"
                        rel_type = "HAS_ACCESS_LEVEL"
                    elif any(term in k_lower for term in ["id", "code"]):
                        ent_type = "Identifier"
                        rel_type = "HAS_ID"
                    else:
                        ent_type = self._infer_type(k)
                        rel_type = re.sub(r"[^A-Za-z0-9_]", "_", k.strip()).upper()

                    # Avoid single-letter, stop-word, or generic header entities
                    if (
                        len(v) >= 2
                        and v.lower() not in STOP_WORDS
                        and v.lower() not in {"department", "headcount", "role", "job role", "status", "device", "summary", "total"}
                    ):
                        entities_dict[v.lower()] = {
                            "name": v,
                            "type": ent_type,
                            "description": f"{ent_type}: {v} (associated with {subject_name})",
                        }
                        relations.append({
                            "source": subject_name,
                            "target": v,
                            "relation": rel_type,
                            "description": f"{subject_name} has {k}: {v}",
                        })

        # ─── 2. Freeform Text Extraction (Paragraphs, RFCs, SOPs) ────────────
        else:
            # 2a. CamelCase identifiers (e.g., AuthService, APIGateway)
            camel_cases = re.findall(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+\b", text)
            for cc in camel_cases:
                if cc.lower() not in STOP_WORDS and len(cc) >= 3 and not IS_FILE_HEADER_PATTERN.match(cc):
                    entities_dict[cc.lower()] = {
                        "name": cc,
                        "type": self._infer_type(cc),
                        "description": f"Component or service: {cc}",
                    }

            # 2b. Technical acronyms (JWT, HTTP, AWS, MFA, RBAC, OAuth2)
            NOISE_ACRONYMS = {"ID", "II", "IV", "VI", "OK", "VS", "NO", "RE", "DO", "SO", "MY", "UP", "WE", "HE", "ME", "US", "AM", "AN", "AS", "AT", "BE", "BY", "IF", "IN", "IS", "IT", "OF", "ON", "OR", "TO"}
            acronyms = re.findall(r"\b[A-Z]{2,6}(?:[0-9]{1,2})?\b", text)
            for acr in acronyms:
                if acr.lower() not in STOP_WORDS and len(acr) >= 2 and acr.upper() not in NOISE_ACRONYMS:
                    entities_dict[acr.lower()] = {
                        "name": acr,
                        "type": "Technology",
                        "description": f"Technical standard or protocol ({acr})",
                    }

            # 2c. Proper phrases (e.g. Elena Rostova, Stripe Payments)
            proper_phrases = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b", text)
            for phrase in proper_phrases:
                clean_phrase = phrase.strip()
                if self._is_non_entity_value(clean_phrase) or IS_FILE_HEADER_PATTERN.match(clean_phrase):
                    continue
                words = [w.lower() for w in clean_phrase.split()]
                if any(w in STOP_WORDS for w in words):
                    continue
                if clean_phrase.lower() not in entities_dict:
                    entities_dict[clean_phrase.lower()] = {
                        "name": clean_phrase,
                        "type": self._infer_type(clean_phrase),
                        "description": f"Named concept: {clean_phrase}",
                    }

            # 2d. Co-occurrence relations across sentences
            sentences = re.split(r"(?<=[.!?\n])\s+", text)
            for sent in sentences:
                if len(sent.strip()) < 10:
                    continue
                in_sentence = []
                for k, ent_data in entities_dict.items():
                    if re.search(rf"\b{re.escape(ent_data['name'])}\b", sent, re.IGNORECASE):
                        in_sentence.append(ent_data['name'])

                if len(in_sentence) >= 2:
                    matched_rel = None
                    for trigger_pattern, rel_name in RELATION_TRIGGERS:
                        if re.search(trigger_pattern, sent, re.IGNORECASE):
                            matched_rel = rel_name
                            break

                    for i in range(len(in_sentence) - 1):
                        src = in_sentence[i]
                        tgt = in_sentence[i + 1]
                        if src != tgt:
                            relations.append({
                                "source": src,
                                "target": tgt,
                                "relation": matched_rel or "RELATED_TO",
                                "description": sent.strip()[:180],
                            })

        # Final cleanup: strictly eliminate non-entity nodes
        clean_entities = []
        for ent in entities_dict.values():
            n = ent["name"].strip()
            if self._is_non_entity_value(n):
                continue
            if IS_FILE_HEADER_PATTERN.match(n):
                continue
            clean_entities.append(ent)

        if tabular_rows:
            # Structured tables: retain all valid row entities and relationships without artificial truncation
            return clean_entities, relations

        final_entities = clean_entities[:self.max_entities]
        final_relations = relations[:self.max_relations]

        return final_entities, final_relations


offline_extractor = OfflineExtractor()
