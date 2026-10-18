"""
GraphRAG Backend - Graph Builder
Writes extracted entities, relations, and chunks into FalkorDB.
Core of the enterprise GraphRAG knowledge graph construction pipeline.

Graph Schema:
  (:Document {id, name, type, metadata, created_at})
  (:Chunk {id, text, document_id, document_name, chunk_index, token_count, embedding})
  (:Entity {id, name, type, description, embedding})
  
  (:Document)-[:HAS_CHUNK]->(:Chunk)
  (:Chunk)-[:MENTIONS]->(:Entity)
  (:Entity)-[:{RELATION_TYPE}]->(:Entity)
"""

import hashlib
import json
import re
import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone
from loguru import logger

from app.db.falkordb_client import db_client
from app.services.embedding_service import embedding_service
from app.services.chunker import TextChunk


class GraphBuilder:
    """
    Builds and maintains the knowledge graph in FalkorDB.
    Handles deduplication of entities and atomic graph writes.
    """

    # -------------------------------------------------------------------------
    # Document Node
    # -------------------------------------------------------------------------

    async def create_document_node(
        self,
        doc_id: str,
        name: str,
        doc_type: str,
        metadata: Dict[str, Any],
    ) -> str:
        """Create or update a Document node in FalkorDB."""
        meta_str = json.dumps(metadata, ensure_ascii=False)
        created_at = datetime.now(timezone.utc).isoformat()

        cypher = """
        MERGE (d:Document {id: $id})
        SET d.name = $name,
            d.type = $type,
            d.metadata = $metadata,
            d.created_at = $created_at,
            d.chunk_count = 0
        RETURN d.id AS id
        """
        await db_client.aquery(cypher, {
            "id": doc_id,
            "name": name,
            "type": doc_type,
            "metadata": meta_str,
            "created_at": created_at,
        })
        logger.info(f"Document node created: {name} ({doc_id})")
        return doc_id

    async def update_document_chunk_count(self, doc_id: str, count: int):
        """Update chunk count after ingestion completes."""
        cypher = "MATCH (d:Document {id: $id}) SET d.chunk_count = $count"
        await db_client.aquery(cypher, {"id": doc_id, "count": count})

    # -------------------------------------------------------------------------
    # Chunk Nodes
    # -------------------------------------------------------------------------

    async def create_chunk_node(self, chunk: TextChunk, embedding: List[float]) -> str:
        """Create a Chunk node with embedding and link it to its Document."""
        meta_str = json.dumps(chunk.metadata, ensure_ascii=False)

        cypher = """
        MERGE (c:Chunk {id: $id})
        SET c.text = $text,
            c.document_id = $document_id,
            c.document_name = $document_name,
            c.chunk_index = $chunk_index,
            c.token_count = $token_count,
            c.embedding = vecf32($embedding),
            c.metadata = $metadata
        WITH c
        MATCH (d:Document {id: $document_id})
        MERGE (d)-[:HAS_CHUNK]->(c)
        RETURN c.id AS id
        """
        await db_client.aquery(cypher, {
            "id": chunk.id,
            "text": chunk.text,
            "document_id": chunk.document_id,
            "document_name": chunk.document_name,
            "chunk_index": chunk.chunk_index,
            "token_count": chunk.token_count,
            "embedding": embedding,
            "metadata": meta_str,
        })
        return chunk.id

    # -------------------------------------------------------------------------
    # Entity Nodes
    # -------------------------------------------------------------------------

    @staticmethod
    def _entity_id(name: str, entity_type: str) -> str:
        """Deterministic entity ID based on name+type for deduplication."""
        key = f"{entity_type.upper()}::{name.lower().strip()}"
        return hashlib.md5(key.encode()).hexdigest()

    async def upsert_entity(
        self,
        name: str,
        entity_type: str,
        description: str,
        embedding: List[float],
    ) -> str:
        """
        Create or merge an Entity node. Deduplication is by (name, type).
        Uses MERGE to avoid duplicates.
        """
        entity_id = self._entity_id(name, entity_type)

        cypher = """
        MERGE (e:Entity {id: $id})
        ON CREATE SET
            e.name = $name,
            e.type = $type,
            e.description = $description,
            e.embedding = vecf32($embedding),
            e.mention_count = 1
        ON MATCH SET
            e.mention_count = e.mention_count + 1,
            e.description = CASE WHEN $description <> '' THEN $description ELSE e.description END
        RETURN e.id AS id
        """
        await db_client.aquery(cypher, {
            "id": entity_id,
            "name": name,
            "type": entity_type,
            "description": description,
            "embedding": embedding,
        })
        return entity_id

    async def link_chunk_to_entity(self, chunk_id: str, entity_id: str):
        """Create MENTIONS relationship between a Chunk and an Entity."""
        cypher = """
        MATCH (c:Chunk {id: $chunk_id})
        MATCH (e:Entity {id: $entity_id})
        MERGE (c)-[:MENTIONS]->(e)
        """
        await db_client.aquery(cypher, {
            "chunk_id": chunk_id,
            "entity_id": entity_id,
        })

    # -------------------------------------------------------------------------
    # Relation Edges
    # -------------------------------------------------------------------------

    async def create_relation(
        self,
        source_name: str,
        source_type: str,
        relation_type: str,
        target_name: str,
        target_type: str,
        weight: float = 1.0,
        chunk_id: str = None,
    ):
        """
        Create a relationship edge between two entities.
        Uses dynamic relation types (FalkorDB Cypher supports dynamic labels
        only via APOC-style, so we use a fixed RELATES_TO with type property,
        plus a few well-known typed relationships).
        """
        source_id = self._entity_id(source_name, source_type)
        target_id = self._entity_id(target_name, target_type)

        # Sanitize relation type into a valid Cypher relationship identifier
        clean_rel = re.sub(r"[^A-Za-z0-9_]", "_", relation_type.strip()).upper()
        # Ensure it starts with a letter and is between 2 and 40 characters
        if not clean_rel or not clean_rel[0].isalpha():
            clean_rel = "RELATED_TO"
        rel_type = clean_rel[:40]

        cypher = f"""
        MATCH (s:Entity {{id: $source_id}})
        MATCH (t:Entity {{id: $target_id}})
        MERGE (s)-[r:{rel_type}]->(t)
        ON CREATE SET r.weight = $weight, r.count = 1, r.chunk_id = $chunk_id
        ON MATCH SET r.count = r.count + 1, r.weight = (r.weight + $weight) / 2.0
        """
        try:
            await db_client.aquery(cypher, {
                "source_id": source_id,
                "target_id": target_id,
                "weight": weight,
                "chunk_id": chunk_id or "",
            })
        except Exception as e:
            logger.warning(f"Relation creation failed ({source_name} -[{rel_type}]-> {target_name}): {e}")

    # -------------------------------------------------------------------------
    # Batch Processing
    # -------------------------------------------------------------------------

    async def process_extracted_entities(
        self,
        chunk_id: str,
        entities: List[Dict],
        relations: List[Dict],
        entity_embeddings: Optional[Dict[str, List[float]]] = None,
    ):
        """
        Batch-process extracted entities and relations for a chunk.
        Uses precomputed batch embeddings if provided to minimize API requests.
        """
        entity_map: Dict[str, str] = {}  # name -> entity_id
        entity_embeddings = entity_embeddings or {}

        # 1) Upsert entities
        for ent in entities:
            name = ent.get("name", "").strip()
            ent_type = ent.get("type", "Other").strip()
            desc = ent.get("description", "").strip()

            if not name:
                continue

            # Use pre-computed embedding from batch, or fallback to single
            embedding = entity_embeddings.get(name.lower())
            if embedding is None:
                entity_text = f"{name}: {desc}" if desc else name
                embedding = embedding_service.embed_single(entity_text)

            entity_id = await self.upsert_entity(name, ent_type, desc, embedding)
            entity_map[name.lower()] = entity_id

            # Link chunk to entity
            await self.link_chunk_to_entity(chunk_id, entity_id)

        # 2) Create relations
        for rel in relations:
            source = rel.get("source", "").strip()
            target = rel.get("target", "").strip()
            rel_type = rel.get("relation", "RELATED_TO").strip()
            weight = float(rel.get("weight", 1.0))

            if not source or not target:
                continue

            # Find types from extracted entities
            source_type = next(
                (e["type"] for e in entities if e.get("name", "").lower() == source.lower()),
                "Other"
            )
            target_type = next(
                (e["type"] for e in entities if e.get("name", "").lower() == target.lower()),
                "Other"
            )

            await self.create_relation(
                source, source_type, rel_type, target, target_type,
                weight=weight, chunk_id=chunk_id
            )

    # -------------------------------------------------------------------------
    # Deletion
    # -------------------------------------------------------------------------

    async def delete_document(self, doc_id: str):
        """
        Delete a document and all its chunks from the graph.
        Note: Entities are NOT deleted (they may be shared across documents).
        """
        # Remove MENTIONS edges from chunks of this document
        await db_client.aquery(
            "MATCH (c:Chunk {document_id: $doc_id})-[r:MENTIONS]->() DELETE r",
            {"doc_id": doc_id}
        )
        # Remove HAS_CHUNK edges
        await db_client.aquery(
            "MATCH (d:Document {id: $doc_id})-[r:HAS_CHUNK]->() DELETE r",
            {"doc_id": doc_id}
        )
        # Delete chunks
        await db_client.aquery(
            "MATCH (c:Chunk {document_id: $doc_id}) DELETE c",
            {"doc_id": doc_id}
        )
        # Delete document
        await db_client.aquery(
            "MATCH (d:Document {id: $doc_id}) DELETE d",
            {"doc_id": doc_id}
        )
        logger.info(f"Deleted document {doc_id} and all its chunks from graph.")


# Singleton
graph_builder = GraphBuilder()
