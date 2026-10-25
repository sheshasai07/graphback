"""
GraphRAG Backend - Health Router
System health checks and configuration info.
"""

from fastapi import APIRouter
from loguru import logger

from app.db.falkordb_client import db_client
from app.models.schemas import HealthResponse
from app.config import settings

router = APIRouter(prefix="/api/v1", tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    System health check.
    Returns connectivity status and service configuration.
    """
    endpoint = settings.falkordb_url or f"{settings.falkordb_host}:{settings.falkordb_port}"
    if db_client.is_connected:
        db_type = "falkordb_cloud"
        db_status = f"Connected to FalkorDB Cloud ({endpoint})"
    else:
        db_type = "in_memory"
        db_status = (
            "Using In-Memory Graph Engine fallback. "
            "To connect to FalkorDB Cloud, verify your FALKORDB_URL / FALKORDB_PASSWORD in backend/.env"
        )

    return HealthResponse(
        status="healthy",
        falkordb_connected=db_client.is_connected,
        database_type=db_type,
        database_status=db_status,
        cloud_endpoint=endpoint,
        graph_name=settings.falkordb_graph_name,
        embedding_model=settings.embedding_model_name,
        llm_model=settings.openrouter_model,
    )


@router.get("/config")
async def get_config():
    """Return non-sensitive configuration for the frontend."""
    return {
        "embedding_model": settings.embedding_model_name,
        "embedding_dimension": settings.embedding_dimension,
        "llm_model": settings.openrouter_model,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "graph_name": settings.falkordb_graph_name,
        "falkordb_host": settings.falkordb_host,
        "falkordb_port": settings.falkordb_port,
        "falkordb_connected": db_client.is_connected,
        "database_type": "falkordb_cloud" if db_client.is_connected else "in_memory",
    }


@router.get("/config/options")
async def get_rag_options():
    """
    Return all user-selectable options for the UI:
    - Embedding models
    - LLM models
    - Chunking strategies
    - Extraction modes
    - Default parameter ranges
    """
    return {
        "chunking_strategies": [
            {"id": "semantic", "name": "Semantic (Paragraph & Sentence Aware)", "description": "Respects natural paragraph and sentence boundaries with sliding overlap"},
            {"id": "recursive", "name": "Recursive (Markdown & Hierarchical)", "description": "Splits by headings (#, ##), then sections, then paragraphs"},
            {"id": "fixed", "name": "Fixed Window (Strict Sizing)", "description": "Splits by exact word/token window with constant overlap"},
        ],
        "extraction_modes": [
            {"id": "offline", "name": "Fast Offline NLP (Default - 0 LLM Cost)", "description": "Extracts entities and relationships locally using fast linguistic rules with zero token cost", "is_default": True},
            {"id": "llm", "name": "LLM Extraction (For Tough Documents)", "description": "Passes chunks through OpenRouter LLM for deep reasoning and complex relationships", "is_default": False},
        ],
        "embedding_models": [
            {"id": "voyage-4-lite", "name": "Voyage 4 Lite (1024-dim, Fast & Accurate)", "provider": "Voyage AI", "dimension": 1024},
            {"id": "voyage-3", "name": "Voyage 3 (1024-dim, General Purpose)", "provider": "Voyage AI", "dimension": 1024},
            {"id": "voyage-3-lite", "name": "Voyage 3 Lite (512-dim, Lightweight)", "provider": "Voyage AI", "dimension": 512},
            {"id": "voyage-large-2-instruct", "name": "Voyage Large 2 Instruct (1536-dim)", "provider": "Voyage AI", "dimension": 1536},
        ],
        "llm_models": [
            {"id": "nvidia/nemotron-3-ultra-550b-a55b:free", "name": "NVIDIA: Nemotron 3 Ultra (free)", "provider": "OpenRouter"},
            {"id": "meta-llama/llama-3.3-70b-instruct:free", "name": "Meta Llama 3.3 70B Instruct (Free)", "provider": "OpenRouter"},
            {"id": "google/gemini-2.0-flash-exp:free", "name": "Google Gemini 2.0 Flash (Free)", "provider": "OpenRouter"},
            {"id": "deepseek/deepseek-r1:free", "name": "DeepSeek R1 (Free)", "provider": "OpenRouter"},
            {"id": "qwen/qwen-2.5-72b-instruct:free", "name": "Qwen 2.5 72B Instruct (Free)", "provider": "OpenRouter"},
        ],
        "default_parameters": {
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "chunking_strategy": "semantic",
            "enable_llm_extraction": False,
            "top_k_chunks": 8,
            "top_k_entities": 10,
            "graph_hops": 2,
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": 1024,
            "graph_name": settings.falkordb_graph_name,
            "active_database": "FalkorDB Cloud" if db_client.is_connected else "In-Memory Fallback",
        }
    }
