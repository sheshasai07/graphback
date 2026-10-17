"""
GraphRAG Backend - Pydantic Models
Request/Response schemas for all API endpoints.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


# ─────────────────────────────────────────────────────────────────────────────
# Ingestion
# ─────────────────────────────────────────────────────────────────────────────

class IngestionStatusEnum(str, Enum):
    PENDING = "pending"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    EXTRACTING = "extracting"
    WRITING = "writing"
    COMPLETED = "completed"
    FAILED = "failed"


class IngestionJobResponse(BaseModel):
    doc_id: str
    doc_name: str
    status: IngestionStatusEnum
    total_chunks: int
    processed_chunks: int
    progress_pct: float
    total_entities: int
    total_relations: int
    error: Optional[str] = None
    started_at: str
    completed_at: Optional[str] = None


class RawTextIngestRequest(BaseModel):
    title: str = Field(default="Untitled", description="Document title")
    content: str = Field(..., min_length=1, description="Raw text content to ingest")
    async_processing: bool = Field(default=False, description="Whether to process in background")
    chunk_size: Optional[int] = Field(default=None, description="Max tokens per chunk (default from settings)")
    chunk_overlap: Optional[int] = Field(default=None, description="Overlap tokens between chunks")
    chunking_strategy: Optional[str] = Field(default="semantic", description="'semantic', 'recursive', or 'fixed'")
    enable_llm_extraction: bool = Field(default=False, description="Enable LLM entity extraction (default is offline)")
    llm_model: Optional[str] = Field(default=None, description="Optional custom OpenRouter model for extraction")
    graph_name: Optional[str] = Field(default=None, description="Optional target graph/workspace")


# ─────────────────────────────────────────────────────────────────────────────
# Query / RAG
# ─────────────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000, description="The user query")
    use_query_decomposition: bool = Field(
        default=True,
        description="Decompose complex queries into sub-queries for better retrieval"
    )
    top_k_chunks: int = Field(default=8, ge=1, le=50, description="Top document chunks to retrieve")
    top_k_entities: int = Field(default=10, ge=1, le=50, description="Top graph entities to retrieve")
    graph_hops: int = Field(default=2, ge=1, le=5, description="Graph traversal depth")
    temperature: float = Field(default=0.2, ge=0.0, le=2.0, description="LLM sampling temperature")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="LLM nucleus sampling top_p")
    max_tokens: int = Field(default=1024, ge=64, le=8192, description="Max generation tokens")
    llm_model: Optional[str] = Field(default=None, description="OpenRouter model (e.g. nvidia/nemotron-3-super-120b-a12b:free)")
    embedding_model: Optional[str] = Field(default=None, description="Voyage AI model (e.g. voyage-4-lite)")
    graph_name: Optional[str] = Field(default=None, description="Target FalkorDB graph workspace")
    stream: bool = Field(default=False, description="Stream the response (SSE)")


class ChunkResult(BaseModel):
    id: str
    text: str
    document_id: str
    document_name: str
    chunk_index: int
    score: float


class EntityResult(BaseModel):
    id: str
    name: str
    type: str
    description: Optional[str] = ""
    mention_count: Optional[int] = 0
    score: float


class GraphFact(BaseModel):
    source: Optional[str] = None
    relation: Optional[str] = None
    target: Optional[str] = None
    type: Optional[str] = None
    description: Optional[str] = None
    entity: Optional[str] = None
    mention_count: Optional[int] = None


class QueryResponse(BaseModel):
    query: str
    answer: str
    sub_queries: List[str] = Field(default_factory=list)
    chunks: List[ChunkResult] = Field(default_factory=list)
    entities: List[EntityResult] = Field(default_factory=list)
    graph_context: List[Dict[str, Any]] = Field(default_factory=list)
    processing_time_ms: float
    sources: List[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Graph / Knowledge Base
# ─────────────────────────────────────────────────────────────────────────────

class GraphStatsResponse(BaseModel):
    documents: int
    chunks: int
    entities: int
    relations: int


class DocumentInfo(BaseModel):
    id: str
    name: str
    type: str
    chunk_count: int
    created_at: str
    metadata: Optional[Dict[str, Any]] = None


class EntityNeighborhoodRequest(BaseModel):
    entity_name: str
    hops: int = Field(default=1, ge=1, le=3)


class EntityNeighborhoodResponse(BaseModel):
    name: str
    type: str
    description: Optional[str] = ""
    mention_count: int = 0
    edges: List[Dict[str, Any]] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# General
# ─────────────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    falkordb_connected: bool
    database_type: str = Field(default="in_memory", description="'falkordb_cloud' or 'in_memory'")
    database_status: str = Field(default="", description="Detailed human-readable connection status")
    cloud_endpoint: Optional[str] = Field(default="", description="Configured FalkorDB Cloud host or endpoint")
    graph_name: str
    embedding_model: str
    llm_model: str


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
