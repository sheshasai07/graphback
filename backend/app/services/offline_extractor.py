"""
GraphRAG - Fast Offline Entity & Relationship Extractor
Extracts named entities, domain terms, key-value data, and semantic relations
purely using local rule-based heuristics & regex with zero LLM API costs and sub-millisecond latency.
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
    "yourselves", "this", "that", "these", "those", "true", "false", "none", "null", "item", "row"
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
    (r"\b(part\s+of|belongs\s+to|member\s+of|contained\s+in)\b", "PART_OF"),
    (r"\b(implements|uses|utilizes|leverages)\b", "USES"),
    (r"\b(produces|generates|returns|creates)\b", "PRODUCES"),
]

TYPE_PATTERNS = [
    (r"(?i)\b.*(Service|Gateway|Server|Worker|Daemon|API|Cluster|Queue|Broker|Database|DB|Vault|Store)\b", "Service"),
    (r"(?i)\b.*(Engineer|Architect|Manager|Developer|Lead|Director|Officer|Admin|User|Client|Customer)\b", "Person"),
    (r"(?i)\b.*(Framework|Library|Protocol|Language|Runtime|SDK|Algorithm|Model|Format|Encoding)\b", "Technology"),
    (r"(?i)\b.*(Company|Corp|Inc|Department|Team|Group|Org|Organization|Institute|Agency)\b", "Organization"),
    (r"(?i)\b.*(Feature|Module|Component|Policy|Rule|Flow|Pipeline|Process|Method)\b", "Feature"),
    (r"(?i)\b.*(Latency|Throughput|Error|Rate|Cost|Count|Size|Memory|CPU|Duration|Score)\b", "Metric"),
]


class OfflineExtractor:
    """
    Zero-cost, offline NLP entity & relation extractor.
    Operates via compiled regular expressions, linguistic heuristics, and tabular parsers.
    """

    def __init__(self, max_entities_per_chunk: int = 25, max_relations_per_chunk: int = 35):
        self.max_entities = max_entities_per_chunk
        self.max_relations = max_relations_per_chunk

    def _infer_type(self, name: str) -> str:
        for pattern, ent_type in TYPE_PATTERNS:
            if re.search(pattern, name):
                return ent_type
        if re.match(r"^[A-Z][a-z]+(\s+[A-Z][a-z]+)+$", name):
            # Two capitalized words typically person or organization
            return "Person"
        if re.match(r"^[A-Z0-9_\-\.]{2,8}$", name):
            return "Technology"
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

        # ─── 1. Key-Value & Tabular Extraction (CSVs, Excel rows, Configs) ───
        # Matches patterns like: Header: Value | Header2: Value2
        kv_pairs = re.findall(r"([A-Za-z0-9_\-\s]{2,30}):\s*([^|\n,;]+)", text)
        row_subject = None
        for key, val in kv_pairs:
            k = key.strip()
            v = val.strip()
            if not v or v.lower() in STOP_WORDS or len(v) > 60:
                continue
            
            # Treat important keys as entities
            if k.lower() not in {"row", "sheet", "item", "index", "id"}:
                ent_name = v
                ent_type = self._infer_type(k)
                if ent_name.lower() not in STOP_WORDS and len(ent_name) >= 2:
                    entities_dict[ent_name.lower()] = {
                        "name": ent_name,
                        "type": ent_type,
                        "description": f"Extracted as '{k}' in data table",
                    }
                    if not row_subject:
                        row_subject = ent_name
                    elif row_subject != ent_name:
                        relations.append({
                            "source": row_subject,
                            "target": ent_name,
                            "relation": k.upper().replace(" ", "_"),
                            "description": f"{row_subject} has {k}: {ent_name}",
                        })

        # ─── 2. CamelCase & System Identifiers (AuthService, APIGateway) ────
        camel_cases = re.findall(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+\b", text)
        for cc in camel_cases:
            if cc.lower() not in STOP_WORDS and len(cc) >= 3:
                entities_dict[cc.lower()] = {
                    "name": cc,
                    "type": self._infer_type(cc),
                    "description": f"Technical component or service identified in text",
                }

        # ─── 3. Acronyms & Short Codes (JWT, HTTP, AWS, SES, RBAC, OAuth2) ───
        acronyms = re.findall(r"\b[A-Z]{2,6}(?:[0-9]{1,2})?\b", text)
        for acr in acronyms:
            if acr.lower() not in STOP_WORDS and len(acr) >= 2:
                entities_dict[acr.lower()] = {
                    "name": acr,
                    "type": "Technology",
                    "description": f"Standard protocol or technological specification ({acr})",
                }

        # ─── 4. Capitalized Multi-Word Phrases (Elena Rostova, Stripe Payments) ──
        proper_phrases = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b", text)
        for phrase in proper_phrases:
            clean_phrase = phrase.strip()
            words = [w.lower() for w in clean_phrase.split()]
            if any(w in STOP_WORDS for w in words):
                continue
            if len(clean_phrase) >= 4:
                entities_dict[clean_phrase.lower()] = {
                    "name": clean_phrase,
                    "type": self._infer_type(clean_phrase),
                    "description": f"Named entity extracted from context",
                }

        # ─── 5. Sentence-Level Relational Heuristics ──────────────────────────
        sentences = re.split(r"(?<=[.!?\n])\s+", text)
        found_names = list(entities_dict.keys())

        for sent in sentences:
            if len(sent.strip()) < 10:
                continue
            # Find entities co-occurring in this sentence
            in_sentence = []
            for k, ent_data in entities_dict.items():
                if re.search(rf"\b{re.escape(ent_data['name'])}\b", sent, re.IGNORECASE):
                    in_sentence.append(ent_data['name'])

            if len(in_sentence) >= 2:
                # Check for explicit verbal triggers
                matched_rel = None
                for trigger_pattern, rel_name in RELATION_TRIGGERS:
                    if re.search(trigger_pattern, sent, re.IGNORECASE):
                        matched_rel = rel_name
                        break
                
                # Pair entities up
                for i in range(len(in_sentence) - 1):
                    src = in_sentence[i]
                    tgt = in_sentence[i + 1]
                    if src != tgt:
                        relations.append({
                            "source": src,
                            "target": tgt,
                            "relation": matched_rel or "ASSOCIATED_WITH",
                            "description": sent.strip()[:180],
                        })

        # Cap entities and relations per chunk to avoid bloated graphs
        final_entities = list(entities_dict.values())[:self.max_entities]
        final_relations = relations[:self.max_relations]

        return final_entities, final_relations


offline_extractor = OfflineExtractor()
