"""
GraphRAG Backend - LLM Service (OpenRouter)
Handles entity extraction, relation extraction, and answer synthesis
using OpenRouter's OpenAI-compatible API, with built-in heuristic fallbacks
when API keys are missing or offline.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from loguru import logger

from app.config import settings


class LLMService:
    """
    OpenRouter-backed LLM service.
    Uses OpenAI-compatible API with models such as Nemotron.
    Includes smart local fallbacks for entity extraction and synthesis
    when no API key is provided.
    """

    def __init__(self):
        self._client: Optional[AsyncOpenAI] = None

    def _has_api_key(self) -> bool:
        return bool(
            settings.openrouter_api_key
            and settings.openrouter_api_key.strip()
            and "your_openrouter_api_key" not in settings.openrouter_api_key
        )

    def _get_client(self) -> Optional[AsyncOpenAI]:
        if not self._has_api_key():
            return None
        clean_key = settings.openrouter_api_key.strip("\"' ")
        if self._client is None or getattr(self, "_cached_key", None) != clean_key:
            self._cached_key = clean_key
            self._client = AsyncOpenAI(
                api_key=clean_key,
                base_url=settings.openrouter_base_url,
                default_headers={
                    "HTTP-Referer": "http://localhost:5173",
                    "X-Title": "GraphRAG Knowledge Assistant",
                },
            )
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def _chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        response_format: Optional[Dict] = None,
        model: Optional[str] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """Core chat completion call with retry logic."""
        client = self._get_client()
        if client is None:
            raise RuntimeError("OpenRouter API key is not configured.")

        active_model = model or settings.openrouter_model
        kwargs = {
            "model": active_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if top_p is not None:
            kwargs["top_p"] = top_p
        if response_format:
            kwargs["response_format"] = response_format

        resp = await client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    # -------------------------------------------------------------------------
    # Rule-Based Heuristic Fallback Extractor
    # -------------------------------------------------------------------------

    def _heuristic_extract(self, text: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Fast heuristic entity and relation extractor used when LLM is unavailable.
        Extracts proper nouns, technological terms, acronyms, and builds co-occurrence edges.
        """
        stop_words = {
            "The", "This", "That", "These", "Those", "There", "Here", "When", "Where",
            "What", "Which", "Who", "Whom", "How", "Why", "In", "On", "At", "By",
            "For", "With", "About", "Against", "Between", "Into", "Through", "During",
            "Before", "After", "Above", "Below", "To", "From", "Up", "Down", "If", "Then"
        }

        # Match capitalized phrases (1 to 3 words) or uppercase acronyms
        pattern = r"\b[A-Z][a-zA-Z0-9_\-\.]+(?:\s+[A-Z][a-zA-Z0-9_\-\.]+){0,2}\b"
        raw_matches = re.findall(pattern, text)

        entities_dict: Dict[str, Dict] = {}
        ordered_entity_names: List[str] = []

        for m in raw_matches:
            name = m.strip()
            if len(name) < 2 or name in stop_words:
                continue

            # Classify entity roughly
            etype = "Concept"
            if re.match(r"^[A-Z0-9]{2,6}$", name):
                etype = "Technology"
            elif any(s in name.lower() for s in ["inc", "corp", "ltd", "google", "meta", "microsoft", "falkordb", "redis"]):
                etype = "Organization"
            elif any(s in name.lower() for s in ["graph", "rag", "database", "vector", "api", "system"]):
                etype = "Technology"

            if name not in entities_dict:
                entities_dict[name] = {
                    "name": name,
                    "type": etype,
                    "description": f"Key concept identified in document chunk",
                }
                ordered_entity_names.append(name)

        # Build relationships between co-occurring entities in same sentence
        relations = []
        sentences = re.split(r"[.!?\n]+", text)
        for sent in sentences:
            found_in_sent = [e for e in ordered_entity_names if e in sent]
            for i in range(len(found_in_sent) - 1):
                s = found_in_sent[i]
                t = found_in_sent[i + 1]
                if s != t:
                    relations.append({
                        "source": s,
                        "relation": "RELATED_TO",
                        "target": t,
                        "weight": 0.8,
                    })

        entities = list(entities_dict.values())[:settings.max_entities_per_chunk]
        relations = relations[:settings.max_relations_per_chunk]
        return entities, relations

    # -------------------------------------------------------------------------
    # Entity & Relation Extraction
    # -------------------------------------------------------------------------

    async def extract_entities_and_relations(
        self, text: str, chunk_id: str, model: Optional[str] = None
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Extract entities and relationships from a text chunk using the LLM.
        Falls back to rule-based heuristic extraction if LLM is unavailable.
        """
        if not self._has_api_key():
            logger.info(f"Using rule-based heuristic entity extraction for chunk {chunk_id}")
            return self._heuristic_extract(text)

        system_prompt = """You are an expert knowledge graph construction AI.
Your task is to extract entities and relationships from text to build a knowledge graph.

ENTITY TYPES: Person, Organization, Location, Concept, Technology, Event, Product, Date, Metric, Other
RELATION TYPES: IS_A, HAS, WORKS_FOR, LOCATED_IN, PART_OF, RELATED_TO, MENTIONS, CAUSED_BY, LEADS_TO, USES, CREATED_BY, OCCURRED_AT, MEASURED_BY, DEPENDS_ON

RULES:
- Extract specific, meaningful entities only (not generic terms like "the system")
- Entity names should be concise (1-5 words)
- Relationships should be directional (source → relation → target)
- Only extract relationships where BOTH entities appear in the text
- Return ONLY valid JSON, no explanation

OUTPUT FORMAT:
{
  "entities": [
    {"name": "Entity Name", "type": "EntityType", "description": "Brief 1-sentence description"}
  ],
  "relations": [
    {"source": "Entity A", "relation": "RELATION_TYPE", "target": "Entity B", "weight": 0.9}
  ]
}"""

        user_prompt = f"""Extract entities and relationships from this text:

TEXT:
{text}

Return JSON only."""

        try:
            raw = await self._chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=2048,
                model=model,
            )
            # Robust JSON extraction: find outermost JSON object
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
            else:
                raw_clean = re.sub(r"```json\s*|\s*```", "", raw).strip()
                data = json.loads(raw_clean)

            entities = data.get("entities", [])[:settings.max_entities_per_chunk]
            relations = data.get("relations", [])[:settings.max_relations_per_chunk]

            logger.debug(
                f"Extracted {len(entities)} entities and {len(relations)} relations from chunk {chunk_id}"
            )
            return entities, relations

        except Exception as e:
            logger.warning(f"LLM extraction error for chunk {chunk_id}: {e}. Falling back to heuristic extractor.")
            return self._heuristic_extract(text)

    # -------------------------------------------------------------------------
    # Answer Synthesis (GraphRAG)
    # -------------------------------------------------------------------------

    async def synthesize_answer(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]],
        graph_context: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate a final answer using retrieved chunks and graph context.
        This is the core RAG synthesis step.
        """
        # Format chunk context
        chunk_text = "\n\n".join(
            f"[Source: {c.get('document_name', 'Unknown')} | Score: {c.get('score', 0):.3f}]\n{c.get('text', '')}"
            for c in context_chunks[:8]
        )

        # Format graph context
        graph_text = ""
        if graph_context:
            facts = []
            for item in graph_context[:30]:
                if "entity" in item:
                    facts.append(f"• Entity: {item['entity']} ({item.get('type', '')}) — {item.get('description', '')}")
                elif "source" in item and "target" in item:
                    facts.append(f"• {item['source']} —[{item.get('relation', 'RELATED_TO')}]→ {item['target']}")
            graph_text = "\n".join(facts)

        if not self._has_api_key():
            # Heuristic synthesis fallback
            response_lines = [
                f"### GraphRAG Analysis for: *{query}*",
                "",
                "#### Key Knowledge Graph Relationships & Entities:",
            ]
            if graph_context:
                for item in graph_context[:10]:
                    if "source" in item and "target" in item:
                        response_lines.append(f"- **{item['source']}** `{item.get('relation', 'RELATED_TO')}` **{item['target']}**")
                    elif "entity" in item:
                        response_lines.append(f"- **{item['entity']}** ({item.get('type', 'Entity')})")
            else:
                response_lines.append("*No direct knowledge graph edges found for this query.*")

            response_lines.append("\n#### Evidence from Ingested Documents:")
            if context_chunks:
                for i, c in enumerate(context_chunks[:3], 1):
                    doc = c.get("document_name", "Document")
                    text_snippet = c.get("text", "")[:350].strip()
                    response_lines.append(f"{i}. **[{doc}]**: \"{text_snippet}...\"")
            else:
                response_lines.append("*No matching document chunks found.*")

            response_lines.append(
                "\n> *Note: Set `OPENROUTER_API_KEY` in `.env` to enable full generative synthesis with Nemotron.*"
            )
            return "\n".join(response_lines)

        system_prompt = """You are an expert AI knowledge assistant using GraphRAG (Graph-enhanced Retrieval-Augmented Generation).

You have access to:
1. TEXT CHUNKS - Direct excerpts from source documents
2. GRAPH CONTEXT - Structured knowledge graph facts about entities and their relationships

Your task: Provide a comprehensive, accurate, and well-structured answer to the user's question.

INSTRUCTIONS:
- Use BOTH text chunks and graph context to formulate your answer
- Cite specific sources when possible
- If information is insufficient, say so clearly
- Structure your response with clear sections when appropriate
- Be factual and precise"""

        user_prompt = f"""USER QUESTION: {query}

--- TEXT CONTEXT (from vector search) ---
{chunk_text if chunk_text else "No relevant text chunks found."}

--- KNOWLEDGE GRAPH CONTEXT ---
{graph_text if graph_text else "No graph context available."}

Please provide a comprehensive answer:"""

        try:
            answer = await self._chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature if temperature is not None else 0.2,
                max_tokens=max_tokens or 3000,
                model=model,
                top_p=top_p,
            )
            return answer
        except Exception as e:
            logger.error(f"Answer synthesis failed: {e}")
            return f"Error communicating with LLM ({e}). Check your OPENROUTER_API_KEY configuration."

    # -------------------------------------------------------------------------
    # Query Decomposition (Advanced)
    # -------------------------------------------------------------------------

    async def decompose_query(self, query: str) -> List[str]:
        """Decompose a complex query into simpler sub-queries for better retrieval."""
        if not self._has_api_key():
            return [query]

        system_prompt = """You are a query decomposition expert.
Break complex questions into 2-4 simpler, atomic sub-questions that together cover the original question.
Return ONLY a JSON array of strings. Example: ["sub-question 1", "sub-question 2"]"""

        try:
            raw = await self._chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Decompose this query: {query}"},
                ],
                temperature=0.0,
                max_tokens=512,
            )
            raw = re.sub(r"```json\s*|\s*```", "", raw).strip()
            sub_queries = json.loads(raw)
            if isinstance(sub_queries, list) and all(isinstance(q, str) for q in sub_queries):
                return sub_queries[:4]
        except Exception as e:
            logger.debug(f"Query decomposition failed: {e}, using original query")

        return [query]


# Singleton
llm_service = LLMService()
