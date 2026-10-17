"""
GraphRAG Backend - Graph Router
Provides graph exploration, statistics, and document management endpoints.
Used for the knowledge graph visualization panel in the UI.
"""

import json
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from loguru import logger

from app.db.falkordb_client import db_client
from app.models.schemas import (
    DocumentInfo,
    EntityNeighborhoodRequest,
    EntityNeighborhoodResponse,
    GraphStatsResponse,
)
from app.services.graph_retriever import graph_retriever
from app.services.graph_builder import graph_builder

router = APIRouter(prefix="/api/v1/graph", tags=["Graph"])


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/stats", response_model=GraphStatsResponse)
async def get_graph_stats():
    """Return statistics about the knowledge graph."""
    stats = await graph_retriever.get_graph_stats()
    return GraphStatsResponse(**stats)


# ─────────────────────────────────────────────────────────────────────────────
# Documents
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/documents", response_model=List[DocumentInfo])
async def list_documents(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
):
    """List all ingested documents with metadata."""
    cypher = """
    MATCH (d:Document)
    RETURN d.id, d.name, d.type, d.chunk_count, d.created_at, d.metadata
    ORDER BY d.created_at DESC
    SKIP $offset
    LIMIT $limit
    """
    try:
        result = await db_client.aquery(cypher, {"limit": limit, "offset": offset})
        documents = []
        for row in result.result_set:
            meta = {}
            try:
                meta = json.loads(row[5]) if row[5] else {}
            except Exception:
                pass
            documents.append(DocumentInfo(
                id=row[0] or "",
                name=row[1] or "",
                type=row[2] or "unknown",
                chunk_count=row[3] or 0,
                created_at=row[4] or "",
                metadata=meta,
            ))
        return documents
    except Exception as e:
        logger.error(f"List documents failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document and all its associated chunks from the graph."""
    try:
        # Verify document exists
        check = await db_client.aquery(
            "MATCH (d:Document {id: $id}) RETURN d.id",
            {"id": doc_id}
        )
        if not check.result_set:
            raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")

        await graph_builder.delete_document(doc_id)
        return {"message": f"Document {doc_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document deletion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Entities
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/entities")
async def list_entities(
    entity_type: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """List entities in the knowledge graph, optionally filtered by type."""
    if entity_type:
        cypher = """
        MATCH (e:Entity {type: $type})
        RETURN e.id, e.name, e.type, e.description, e.mention_count
        ORDER BY e.mention_count DESC
        SKIP $offset LIMIT $limit
        """
        params = {"type": entity_type, "limit": limit, "offset": offset}
    else:
        cypher = """
        MATCH (e:Entity)
        RETURN e.id, e.name, e.type, e.description, e.mention_count
        ORDER BY e.mention_count DESC
        SKIP $offset LIMIT $limit
        """
        params = {"limit": limit, "offset": offset}

    try:
        result = await db_client.aquery(cypher, params)
        entities = []
        for row in result.result_set:
            entities.append({
                "id": row[0],
                "name": row[1],
                "type": row[2],
                "description": row[3],
                "mention_count": row[4],
            })
        return {"entities": entities, "count": len(entities)}
    except Exception as e:
        logger.error(f"List entities failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/entities/types")
async def get_entity_types():
    """Get all unique entity types and their counts."""
    cypher = """
    MATCH (e:Entity)
    RETURN e.type AS type, count(e) AS count
    ORDER BY count DESC
    """
    try:
        result = await db_client.aquery(cypher)
        return {
            "types": [
                {"type": row[0], "count": row[1]}
                for row in result.result_set
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/entities/neighborhood", response_model=EntityNeighborhoodResponse)
async def get_entity_neighborhood(request: EntityNeighborhoodRequest):
    """
    Get the graph neighborhood around a specific entity.
    Returns the entity and all connected nodes up to N hops away.
    Used for graph visualization.
    """
    try:
        result = await graph_retriever.get_entity_neighborhood(
            entity_name=request.entity_name,
            hops=request.hops,
        )
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"Entity not found: {request.entity_name}"
            )
        return EntityNeighborhoodResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Entity neighborhood failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Graph Visualization Data
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/visualization")
async def get_graph_visualization(
    limit_nodes: int = Query(default=100, ge=10, le=500),
):
    """
    Return graph data in a format suitable for D3.js / Vis.js visualization.
    Returns nodes and edges as arrays.
    """
    # Get top entities (by mention count)
    entity_cypher = """
    MATCH (e:Entity)
    RETURN e.id, e.name, e.type, e.mention_count
    ORDER BY e.mention_count DESC
    LIMIT $limit
    """

    edge_cypher = """
    MATCH (s:Entity)-[r]->(t:Entity)
    WHERE s.id IN $entity_ids AND t.id IN $entity_ids
    RETURN s.name, type(r), t.name, r.weight, r.count
    LIMIT 300
    """

    try:
        ent_result = await db_client.aquery(entity_cypher, {"limit": limit_nodes})
        entities = []
        entity_ids = []
        for row in ent_result.result_set:
            entities.append({
                "id": row[0],
                "label": row[1],
                "type": row[2],
                "mention_count": row[3] or 1,
                "group": row[2],  # Group by entity type for coloring
            })
            entity_ids.append(row[0])

        edges = []
        if entity_ids:
            edge_result = await db_client.aquery(edge_cypher, {"entity_ids": entity_ids})
            for row in edge_result.result_set:
                edges.append({
                    "source": row[0],
                    "relation": row[1],
                    "target": row[2],
                    "weight": float(row[3]) if row[3] else 1.0,
                    "count": row[4] or 1,
                })

        return {
            "nodes": entities,
            "edges": edges,
            "node_count": len(entities),
            "edge_count": len(edges),
        }
    except Exception as e:
        logger.error(f"Graph visualization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Chunks
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/documents/{doc_id}/chunks")
async def get_document_chunks(doc_id: str, limit: int = Query(default=20, ge=1, le=100)):
    """Get chunks for a specific document."""
    cypher = """
    MATCH (d:Document {id: $doc_id})-[:HAS_CHUNK]->(c:Chunk)
    RETURN c.id, c.text, c.chunk_index, c.token_count
    ORDER BY c.chunk_index ASC
    LIMIT $limit
    """
    try:
        result = await db_client.aquery(cypher, {"doc_id": doc_id, "limit": limit})
        chunks = [
            {
                "id": row[0],
                "text": row[1],
                "chunk_index": row[2],
                "token_count": row[3],
            }
            for row in result.result_set
        ]
        return {"doc_id": doc_id, "chunks": chunks, "count": len(chunks)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
