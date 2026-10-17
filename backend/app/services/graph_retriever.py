"""
GraphRAG Backend - Graph RAG Retriever
Implements the hybrid retrieval strategy:
  1. Vector search on Chunk embeddings (semantic similarity)
  2. Entity vector search (find relevant entities)
  3. Graph traversal from matched entities (1-2 hops)
  4. Combine and rank results for LLM synthesis

This is the core differentiator of GraphRAG over vanilla RAG.
"""

from typing import Any, Dict, List, Optional, Tuple
from loguru import logger

from app.db.falkordb_client import db_client
from app.services.embedding_service import embedding_service
from app.services.llm_service import llm_service
from app.config import settings


class GraphRAGRetriever:
    """
    Hybrid GraphRAG retrieval engine.
    Combines vector similarity search with knowledge graph traversal.
    """

    def __init__(
        self,
        top_k_chunks: int = 8,
        top_k_entities: int = 10,
        graph_hops: int = 2,
        vector_weight: float = 0.6,
        graph_weight: float = 0.4,
    ):
        self.top_k_chunks = top_k_chunks
        self.top_k_entities = top_k_entities
        self.graph_hops = graph_hops
        self.vector_weight = vector_weight
        self.graph_weight = graph_weight

    # -------------------------------------------------------------------------
    # Main Retrieval Entry Point
    # -------------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        use_query_decomposition: bool = True,
        graph_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Full GraphRAG retrieval pipeline with dynamic graph workspace and embedding model.
        Returns structured context for LLM synthesis.
        """
        if graph_name and graph_name.strip():
            db_client.select_graph(graph_name.strip())

        logger.info(f"GraphRAG retrieval for: '{query}' (graph='{db_client._in_memory_engine.current_workspace if db_client._use_in_memory else settings.falkordb_graph_name}')")

        # 1) Optionally decompose complex queries
        queries = [query]
        if use_query_decomposition:
            try:
                queries = await llm_service.decompose_query(query)
                if len(queries) > 1:
                    logger.info(f"Query decomposed into {len(queries)} sub-queries")
            except Exception as e:
                logger.debug(f"Query decomposition skipped: {e}")
                queries = [query]

        # 2) Embed all queries using Voyage AI query input_type
        query_embeddings = embedding_service.embed(queries, input_type="query")

        # 3) Vector search for relevant chunks
        all_chunks = []
        for q_text, q_emb in zip(queries, query_embeddings):
            chunks = await self._vector_search_chunks(q_emb)
            all_chunks.extend(chunks)

        # Deduplicate chunks by ID, keep highest score
        chunk_map: Dict[str, Dict] = {}
        for chunk in all_chunks:
            cid = chunk["id"]
            if cid not in chunk_map or chunk["score"] > chunk_map[cid]["score"]:
                chunk_map[cid] = chunk
        top_chunks = sorted(chunk_map.values(), key=lambda x: x["score"], reverse=True)[:self.top_k_chunks]

        # 4) Entity vector search
        all_entities = []
        for q_emb in query_embeddings:
            entities = await self._vector_search_entities(q_emb)
            all_entities.extend(entities)

        # Deduplicate entities
        entity_map: Dict[str, Dict] = {}
        for ent in all_entities:
            eid = ent["id"]
            if eid not in entity_map or ent["score"] > entity_map[eid]["score"]:
                entity_map[eid] = ent
        top_entities = sorted(entity_map.values(), key=lambda x: x["score"], reverse=True)[:self.top_k_entities]

        # 5) Graph traversal from top entities
        graph_context = await self._graph_traversal(
            entity_ids=[e["id"] for e in top_entities],
            hops=self.graph_hops,
        )

        # 6) Entity mentions from top chunks
        chunk_entities = await self._get_chunk_entities(
            chunk_ids=[c["id"] for c in top_chunks]
        )

        # 7) Merge graph context
        all_graph_context = graph_context + chunk_entities

        logger.info(
            f"Retrieved: {len(top_chunks)} chunks, "
            f"{len(top_entities)} entities, "
            f"{len(all_graph_context)} graph facts"
        )

        return {
            "query": query,
            "sub_queries": queries,
            "chunks": top_chunks,
            "entities": top_entities,
            "graph_context": all_graph_context,
        }

    # -------------------------------------------------------------------------
    # Vector Search
    # -------------------------------------------------------------------------

    async def _vector_search_chunks(
        self, query_embedding: List[float], top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks using FalkorDB vector index.
        Uses KNN vector search with cosine similarity.
        """
        top_k = top_k or self.top_k_chunks

        # FalkorDB vector search syntax
        cypher = """
        CALL db.idx.vector.queryNodes('Chunk', 'embedding', $k, vecf32($embedding))
        YIELD node, score
        RETURN 
            node.id AS id,
            node.text AS text,
            node.document_id AS document_id,
            node.document_name AS document_name,
            node.chunk_index AS chunk_index,
            score
        ORDER BY score DESC
        LIMIT $k
        """
        try:
            result = await db_client.aquery(cypher, {
                "embedding": query_embedding,
                "k": top_k,
            })
            chunks = []
            for row in result.result_set:
                chunks.append({
                    "id": row[0],
                    "text": row[1],
                    "document_id": row[2],
                    "document_name": row[3],
                    "chunk_index": row[4],
                    "score": float(row[5]),
                })
            return chunks
        except Exception as e:
            logger.warning(f"Chunk vector search failed: {e}")
            # Fallback: linear scan (slower but works without vector index)
            return await self._fallback_chunk_search(query_embedding, top_k)

    async def _fallback_chunk_search(
        self, query_embedding: List[float], top_k: int
    ) -> List[Dict[str, Any]]:
        """Fallback linear scan for chunk similarity (no vector index)."""
        try:
            cypher = """
            MATCH (c:Chunk)
            WHERE c.embedding IS NOT NULL
            RETURN c.id, c.text, c.document_id, c.document_name, c.chunk_index, c.embedding
            LIMIT 500
            """
            result = await db_client.aquery(cypher)
            chunks = []
            for row in result.result_set:
                if row[5] is None:
                    continue
                score = embedding_service.cosine_similarity(query_embedding, list(row[5]))
                chunks.append({
                    "id": row[0],
                    "text": row[1],
                    "document_id": row[2],
                    "document_name": row[3],
                    "chunk_index": row[4],
                    "score": score,
                })
            return sorted(chunks, key=lambda x: x["score"], reverse=True)[:top_k]
        except Exception as e:
            logger.error(f"Fallback chunk search failed: {e}")
            return []

    async def _vector_search_entities(
        self, query_embedding: List[float], top_k: int = None
    ) -> List[Dict[str, Any]]:
        """Search for relevant entities using vector similarity."""
        top_k = top_k or self.top_k_entities

        cypher = """
        CALL db.idx.vector.queryNodes('Entity', 'embedding', $k, vecf32($embedding))
        YIELD node, score
        RETURN
            node.id AS id,
            node.name AS name,
            node.type AS type,
            node.description AS description,
            node.mention_count AS mention_count,
            score
        ORDER BY score DESC
        LIMIT $k
        """
        try:
            result = await db_client.aquery(cypher, {
                "embedding": query_embedding,
                "k": top_k,
            })
            entities = []
            for row in result.result_set:
                entities.append({
                    "id": row[0],
                    "name": row[1],
                    "type": row[2],
                    "description": row[3],
                    "mention_count": row[4],
                    "score": float(row[5]),
                    "entity": f"{row[1]} ({row[2]})",
                })
            return entities
        except Exception as e:
            logger.warning(f"Entity vector search failed: {e}")
            return await self._fallback_entity_search(query_embedding, top_k)

    async def _fallback_entity_search(
        self, query_embedding: List[float], top_k: int
    ) -> List[Dict[str, Any]]:
        """Fallback linear scan for entity similarity."""
        try:
            cypher = """
            MATCH (e:Entity)
            WHERE e.embedding IS NOT NULL
            RETURN e.id, e.name, e.type, e.description, e.mention_count, e.embedding
            LIMIT 1000
            """
            result = await db_client.aquery(cypher)
            entities = []
            for row in result.result_set:
                if row[5] is None:
                    continue
                score = embedding_service.cosine_similarity(query_embedding, list(row[5]))
                entities.append({
                    "id": row[0],
                    "name": row[1],
                    "type": row[2],
                    "description": row[3],
                    "mention_count": row[4],
                    "score": score,
                    "entity": f"{row[1]} ({row[2]})",
                })
            return sorted(entities, key=lambda x: x["score"], reverse=True)[:top_k]
        except Exception as e:
            logger.error(f"Fallback entity search failed: {e}")
            return []

    # -------------------------------------------------------------------------
    # Graph Traversal
    # -------------------------------------------------------------------------

    async def _graph_traversal(
        self, entity_ids: List[str], hops: int = 2
    ) -> List[Dict[str, Any]]:
        """
        Traverse the knowledge graph from seed entities.
        Returns relationships and connected entities up to N hops.
        """
        if not entity_ids:
            return []

        # FalkorDB variable-length path traversal
        cypher = f"""
        MATCH (start:Entity)
        WHERE start.id IN $entity_ids
        MATCH path = (start)-[r*1..{hops}]-(connected:Entity)
        WITH 
            start.name AS source,
            type(last(relationships(path))) AS relation,
            connected.name AS target,
            connected.type AS target_type,
            connected.description AS target_desc,
            length(path) AS hop_distance
        RETURN DISTINCT source, relation, target, target_type, target_desc, hop_distance
        ORDER BY hop_distance ASC
        LIMIT 50
        """
        try:
            result = await db_client.aquery(cypher, {"entity_ids": entity_ids})
            facts = []
            for row in result.result_set:
                facts.append({
                    "source": row[0],
                    "relation": row[1],
                    "target": row[2],
                    "type": row[3],
                    "description": row[4],
                    "hop_distance": row[5],
                })
            return facts
        except Exception as e:
            logger.warning(f"Graph traversal failed: {e}")
            return []

    async def _get_chunk_entities(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        """Get entities mentioned in the retrieved chunks."""
        if not chunk_ids:
            return []

        cypher = """
        MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
        WHERE c.id IN $chunk_ids
        RETURN DISTINCT
            e.name AS entity,
            e.type AS type,
            e.description AS description,
            e.mention_count AS mention_count
        ORDER BY e.mention_count DESC
        LIMIT 30
        """
        try:
            result = await db_client.aquery(cypher, {"chunk_ids": chunk_ids})
            entities = []
            for row in result.result_set:
                entities.append({
                    "entity": row[0],
                    "type": row[1],
                    "description": row[2],
                    "mention_count": row[3],
                })
            return entities
        except Exception as e:
            logger.warning(f"Chunk entity fetch failed: {e}")
            return []

    # -------------------------------------------------------------------------
    # Graph Statistics
    # -------------------------------------------------------------------------

    async def get_graph_stats(self) -> Dict[str, Any]:
        """Return statistics about the knowledge graph."""
        stats = {}

        queries = {
            "documents": "MATCH (d:Document) RETURN count(d) AS count",
            "chunks": "MATCH (c:Chunk) RETURN count(c) AS count",
            "entities": "MATCH (e:Entity) RETURN count(e) AS count",
            "relations": "MATCH ()-[r]->() RETURN count(r) AS count",
        }

        for key, cypher in queries.items():
            try:
                result = await db_client.aquery(cypher)
                stats[key] = result.result_set[0][0] if result.result_set else 0
            except Exception as e:
                logger.warning(f"Stats query failed for {key}: {e}")
                stats[key] = 0

        return stats

    async def get_entity_neighborhood(
        self, entity_name: str, hops: int = 1
    ) -> Dict[str, Any]:
        """
        Get the graph neighborhood around a specific entity.
        Used for graph visualization.
        """
        cypher = f"""
        MATCH (e:Entity {{name: $name}})
        OPTIONAL MATCH path = (e)-[r*1..{hops}]-(neighbor:Entity)
        WITH e,
            collect(DISTINCT {{
                source: startNode(last(relationships(path))).name,
                source_type: startNode(last(relationships(path))).type,
                relation: type(last(relationships(path))),
                target: endNode(last(relationships(path))).name,
                target_type: endNode(last(relationships(path))).type
            }}) AS edges
        RETURN e.name, e.type, e.description, e.mention_count, edges
        """
        try:
            result = await db_client.aquery(cypher, {"name": entity_name})
            if not result.result_set:
                return {}
            row = result.result_set[0]
            return {
                "name": row[0],
                "type": row[1],
                "description": row[2],
                "mention_count": row[3],
                "edges": row[4] or [],
            }
        except Exception as e:
            logger.error(f"Entity neighborhood query failed: {e}")
            return {}


# Singleton
graph_retriever = GraphRAGRetriever()
