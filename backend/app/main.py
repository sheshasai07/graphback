"""
GraphRAG Backend - FastAPI Application Entry Point
Configures the app, middleware, lifespan, and routes.
"""

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.config import settings
from app.db.falkordb_client import db_client
from app.routers import health, ingest, query, graph, agent


# ─────────────────────────────────────────────────────────────────────────────
# Lifespan: startup & shutdown
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    - Startup: Connect to FalkorDB, initialize schema, pre-load embedding model
    - Shutdown: Close DB connection
    """
    logger.info("🚀 GraphGenie Backend starting up...")

    # 1) Connect to FalkorDB & Initialize Schema
    try:
        db_client.connect()
        db_client.initialize_schema()
    except Exception as e:
        logger.error(f"FalkorDB startup failed: {e}")
        logger.warning("Running without FalkorDB — queries will use In-Memory fallback")

    # 2) Pre-warm embedding model (lazy-load in background)
    async def warm_embeddings():
        try:
            from app.services.embedding_service import embedding_service
            await asyncio.get_event_loop().run_in_executor(
                None, embedding_service.embed_single, "warm up"
            )
            logger.info("Embedding model pre-warmed ✓")
        except Exception as e:
            logger.warning(f"Embedding pre-warm failed: {e}")

    asyncio.create_task(warm_embeddings())

    logger.info("✅ GraphRAG Backend ready")
    logger.info(f"   FalkorDB: {settings.falkordb_host}:{settings.falkordb_port}")
    logger.info(f"   Graph: {settings.falkordb_graph_name}")
    logger.info(f"   LLM: {settings.openrouter_model}")
    logger.info(f"   Embeddings: {settings.embedding_model_name}")
    logger.info(f"   API: http://{settings.api_host}:{settings.api_port}")

    yield

    # Shutdown
    logger.info("Shutting down GraphRAG Backend...")
    db_client.disconnect()


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="GraphRAG Knowledge Assistant API",
    description="""
Enterprise-grade Graph RAG (Retrieval-Augmented Generation) backend powered by FalkorDB.

## Features
- 📄 **Multi-format Document Ingestion**: PDF, DOCX, TXT, Markdown, CSV, JSON, HTML
- 🧠 **Knowledge Graph Construction**: LLM-extracted entities and relations stored in FalkorDB
- 🔍 **Hybrid Retrieval**: Vector search + multi-hop graph traversal
- 💬 **GraphRAG Q&A**: LLM synthesis with structured graph context
- 📊 **Graph Visualization**: D3.js-ready node/edge data

## Tech Stack
- **Graph DB**: FalkorDB (Redis-based, OpenCypher)
- **LLM**: Nemotron via OpenRouter
- **Embeddings**: Sentence-Transformers (local)
- **Framework**: FastAPI + Python asyncio
""",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ─────────────────────────────────────────────────────────────────────────────
# Middleware
# ─────────────────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Routers
# ─────────────────────────────────────────────────────────────────────────────

app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(graph.router)
app.include_router(agent.router)


# ─────────────────────────────────────────────────────────────────────────────
# Root
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Root"])
async def root():
    return {
        "service": "GraphGenie: Context-Aware Multi-Agent System API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
        "agent": "/api/v1/agent",
        "powered_by": "FalkorDB + Multi-Agent GraphRAG + OpenRouter Nemotron",
    }
