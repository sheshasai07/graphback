"""
GraphRAG Backend - Ingestion Router
Handles file uploads and triggers the ingestion pipeline.
Supports streaming progress via Server-Sent Events (SSE).
"""

import asyncio
import os
import uuid
from typing import Optional

import aiofiles
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from loguru import logger

from app.models.schemas import IngestionJobResponse, RawTextIngestRequest
from app.services.ingestion_pipeline import ingestion_pipeline
from app.services.storage_service import storage_service
from app.config import settings

router = APIRouter(prefix="/api/v1/ingest", tags=["Ingestion"])

ALLOWED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".txt", ".md", ".markdown",
    ".csv", ".tsv", ".json", ".html", ".htm",
    ".xlsx", ".xls", ".xlsm",
}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB


@router.post("/upload", response_model=IngestionJobResponse, status_code=202)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    async_processing: bool = Form(default=True),
    chunk_size: Optional[int] = Form(default=None),
    chunk_overlap: Optional[int] = Form(default=None),
    chunking_strategy: str = Form(default="semantic"),
    enable_llm_extraction: bool = Form(default=False),
    llm_model: Optional[str] = Form(default=None),
    graph_name: Optional[str] = Form(default=None),
):
    """
    Upload a document for ingestion into the knowledge graph with user-defined RAG settings.
    
    Supported formats: PDF, DOCX, TXT, Markdown, CSV, TSV, XLSX, XLS, XLSM, JSON, HTML
    
    - `chunking_strategy`: 'semantic' (default), 'recursive', or 'fixed'
    - `enable_llm_extraction`: False (default: zero-cost fast offline NLP), True (OpenRouter LLM)
    - `llm_model`: Optional custom model for extraction
    - `graph_name`: Target FalkorDB graph workspace
    """
    # Validate extension
    filename = file.filename or "unknown"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Save uploaded file via StorageService (local + Zoho Catalyst Cloud File Store)
    doc_id = str(uuid.uuid4())

    try:
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"File too large. Max size: 100 MB")

        storage_info = await storage_service.save_file(
            file_content=content,
            filename=filename,
            doc_id=doc_id,
        )
        file_path = storage_info["local_path"]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File save error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    if async_processing:
        # Start ingestion in background
        background_tasks.add_task(
            ingestion_pipeline.ingest_document,
            file_path=file_path,
            filename=filename,
            doc_id=doc_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunking_strategy=chunking_strategy,
            enable_llm_extraction=enable_llm_extraction,
            llm_model=llm_model,
            graph_name=graph_name,
            storage_info=storage_info,
        )
        # Return initial pending status
        return {
            "doc_id": doc_id,
            "doc_name": filename,
            "status": "pending",
            "total_chunks": 0,
            "processed_chunks": 0,
            "progress_pct": 0.0,
            "total_entities": 0,
            "total_relations": 0,
            "error": None,
            "started_at": "",
            "completed_at": None,
        }
    else:
        # Synchronous ingestion
        progress = await ingestion_pipeline.ingest_document(
            file_path=file_path,
            filename=filename,
            doc_id=doc_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            chunking_strategy=chunking_strategy,
            enable_llm_extraction=enable_llm_extraction,
            llm_model=llm_model,
            graph_name=graph_name,
            storage_info=storage_info,
        )
        return progress.to_dict()


@router.post("/raw", response_model=IngestionJobResponse)
async def ingest_raw_text(
    payload: RawTextIngestRequest,
    background_tasks: BackgroundTasks,
):
    """Directly ingest a raw text block into the knowledge graph with custom RAG settings."""
    doc_id = str(uuid.uuid4())
    filename = f"{payload.title or 'raw_document'}.txt"
    os.makedirs(settings.upload_dir, exist_ok=True)
    file_path = os.path.join(settings.upload_dir, f"{doc_id}.txt")

    async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
        await f.write(payload.content)

    if payload.async_processing:
        background_tasks.add_task(
            ingestion_pipeline.ingest_document,
            file_path=file_path,
            filename=filename,
            doc_id=doc_id,
            chunk_size=payload.chunk_size,
            chunk_overlap=payload.chunk_overlap,
            chunking_strategy=payload.chunking_strategy or "semantic",
            enable_llm_extraction=payload.enable_llm_extraction,
            llm_model=payload.llm_model,
            graph_name=payload.graph_name,
        )
        return {
            "doc_id": doc_id,
            "doc_name": filename,
            "status": "pending",
            "total_chunks": 0,
            "processed_chunks": 0,
            "progress_pct": 0.0,
            "total_entities": 0,
            "total_relations": 0,
            "error": None,
            "started_at": "",
            "completed_at": None,
        }
    else:
        progress = await ingestion_pipeline.ingest_document(
            file_path=file_path,
            filename=filename,
            doc_id=doc_id,
            chunk_size=payload.chunk_size,
            chunk_overlap=payload.chunk_overlap,
            chunking_strategy=payload.chunking_strategy or "semantic",
            enable_llm_extraction=payload.enable_llm_extraction,
            llm_model=payload.llm_model,
            graph_name=payload.graph_name,
        )
        return progress.to_dict()


@router.get("/status/{doc_id}", response_model=IngestionJobResponse)
async def get_ingestion_status(doc_id: str):
    """Get the current status of a document ingestion job."""
    status = ingestion_pipeline.get_job_status(doc_id)
    if not status:
        raise HTTPException(status_code=404, detail=f"Job not found: {doc_id}")
    return status


@router.get("/status/{doc_id}/stream")
async def stream_ingestion_status(doc_id: str):
    """
    Stream real-time ingestion progress via Server-Sent Events (SSE).
    Connect to this endpoint to receive live progress updates.
    """
    import json

    async def event_generator():
        while True:
            status = ingestion_pipeline.get_job_status(doc_id)
            if not status:
                yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
                break

            yield f"data: {json.dumps(status)}\n\n"

            # Stop streaming when terminal state reached
            if status["status"] in ("completed", "failed"):
                break

            await asyncio.sleep(1.0)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/jobs", response_model=list)
async def list_ingestion_jobs():
    """List all ingestion jobs and their current status."""
    return ingestion_pipeline.get_all_jobs()


@router.get("/storage/status")
async def get_storage_status():
    """
    Check whether Zoho Catalyst File Store is configured and test cloud connectivity.
    Returns connection status and folder details if configured, or unconfigured status.
    """
    return await storage_service.test_connection()

