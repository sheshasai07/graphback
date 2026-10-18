"""
GraphRAG Backend - Ingestion Pipeline
Orchestrates the full document ingestion workflow:
  1. Parse document → 2. Chunk text → 3. Embed chunks → 
  4. Extract entities & relations (LLM) → 5. Write to FalkorDB graph

Enterprise-grade: async, streaming progress, error recovery.
"""

import asyncio
import os
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional
from loguru import logger

from app.services.document_parser import document_parser
from app.services.chunker import text_chunker, TextChunk
from app.services.embedding_service import embedding_service
from app.services.llm_service import llm_service
from app.services.offline_extractor import offline_extractor
from app.services.graph_builder import graph_builder
from app.db.falkordb_client import db_client
from app.config import settings


class IngestionStatus(str, Enum):
    PENDING = "pending"
    PARSING = "parsing"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    EXTRACTING = "extracting"
    WRITING = "writing"
    COMPLETED = "completed"
    FAILED = "failed"


class IngestionProgress:
    """Tracks progress of a single document ingestion job."""

    def __init__(self, doc_id: str, doc_name: str, total_chunks: int = 0):
        self.doc_id = doc_id
        self.doc_name = doc_name
        self.status = IngestionStatus.PENDING
        self.total_chunks = total_chunks
        self.processed_chunks = 0
        self.total_entities = 0
        self.total_relations = 0
        self.error: Optional[str] = None
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.completed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "doc_name": self.doc_name,
            "status": self.status.value,
            "total_chunks": self.total_chunks,
            "processed_chunks": self.processed_chunks,
            "progress_pct": round(
                (self.processed_chunks / max(self.total_chunks, 1)) * 100, 1
            ),
            "total_entities": self.total_entities,
            "total_relations": self.total_relations,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


class IngestionPipeline:
    """
    End-to-end document ingestion pipeline for GraphRAG.
    Produces a populated FalkorDB knowledge graph.
    """

    _jobs: Dict[str, IngestionProgress] = {}

    async def ingest_document(
        self,
        file_path: str,
        filename: str,
        doc_id: Optional[str] = None,
        progress_callback: Optional[Callable[[IngestionProgress], None]] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        chunking_strategy: str = "semantic",
        enable_llm_extraction: bool = False,
        llm_model: Optional[str] = None,
        graph_name: Optional[str] = None,
        storage_info: Optional[Dict[str, Any]] = None,
    ) -> IngestionProgress:
        """
        Run the full ingestion pipeline for a document file with user-configured parameters.
        """
        doc_id = doc_id or str(uuid.uuid4())
        progress = IngestionProgress(doc_id, filename)
        self._jobs[doc_id] = progress

        if graph_name and graph_name.strip():
            db_client.select_graph(graph_name.strip())

        def _notify():
            if progress_callback:
                try:
                    asyncio.create_task(progress_callback(progress))
                except Exception:
                    pass

        try:
            # ─── Step 1: Parse ────────────────────────────────────────────────
            progress.status = IngestionStatus.PARSING
            _notify()
            logger.info(f"[{doc_id}] Parsing: {filename}")

            text, doc_metadata = document_parser.parse(file_path, filename)
            if not text.strip():
                raise ValueError(f"No extractable text found in {filename}")

            doc_metadata["file_size"] = os.path.getsize(file_path)
            doc_metadata["original_filename"] = filename
            if storage_info:
                doc_metadata.update({
                    "storage_type": storage_info.get("storage_type", "local"),
                    "catalyst_file_id": storage_info.get("catalyst_file_id"),
                    "catalyst_folder_id": storage_info.get("catalyst_folder_id"),
                    "catalyst_download_url": storage_info.get("catalyst_download_url"),
                })

            # ─── Step 2: Create Document Node ─────────────────────────────────
            await graph_builder.create_document_node(
                doc_id=doc_id,
                name=filename,
                doc_type=doc_metadata.get("type", "unknown"),
                metadata=doc_metadata,
            )

            # ─── Step 3: Chunk ────────────────────────────────────────────────
            progress.status = IngestionStatus.CHUNKING
            _notify()
            logger.info(f"[{doc_id}] Chunking text ({len(text):,} chars) with strategy '{chunking_strategy}'")

            chunks: List[TextChunk] = text_chunker.chunk_document(
                text=text,
                document_id=doc_id,
                document_name=filename,
                extra_metadata={"file_type": doc_metadata.get("type", "unknown")},
                strategy=chunking_strategy,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )

            if not chunks:
                raise ValueError(f"No chunks generated for {filename}")

            progress.total_chunks = len(chunks)
            logger.info(f"[{doc_id}] Generated {len(chunks)} chunks")

            # ─── Step 4: Embed + Extract + Write ──────────────────────────────
            await self._process_chunks(
                chunks=chunks,
                progress=progress,
                progress_callback=progress_callback,
                enable_llm_extraction=enable_llm_extraction,
                llm_model=llm_model,
            )

            # ─── Step 5: Update Document chunk count ──────────────────────────
            await graph_builder.update_document_chunk_count(doc_id, len(chunks))

            # ─── Done ─────────────────────────────────────────────────────────
            progress.status = IngestionStatus.COMPLETED
            progress.completed_at = datetime.now(timezone.utc).isoformat()
            _notify()

            logger.success(
                f"[{doc_id}] Ingestion complete: {len(chunks)} chunks, "
                f"{progress.total_entities} entities, {progress.total_relations} relations "
                f"(LLM extraction: {enable_llm_extraction})"
            )

        except Exception as e:
            progress.status = IngestionStatus.FAILED
            progress.error = str(e)
            progress.completed_at = datetime.now(timezone.utc).isoformat()
            logger.error(f"[{doc_id}] Ingestion failed: {e}")

        return progress

    async def _process_chunks(
        self,
        chunks: List[TextChunk],
        progress: IngestionProgress,
        progress_callback: Optional[Callable],
        enable_llm_extraction: bool = False,
        llm_model: Optional[str] = None,
    ):
        """
        Process chunks with maximum cost efficiency:
        1. Batch embed ALL chunks in 1 single Voyage AI API call.
        2. Write chunk nodes to graph.
        3. Extract entities & relations:
           - Default: Fast offline zero-cost NLP extraction
           - If enable_llm_extraction=True: OpenRouter LLM extraction
        4. Batch embed ALL unique entities across document in 1 single Voyage AI API call.
        5. Write entities & relations to graph.
        """
        # ─── 1) Batch embed all chunks at once (1 Voyage API call) ───
        progress.status = IngestionStatus.EMBEDDING
        chunk_texts = [c.text for c in chunks]
        logger.info(f"[{progress.doc_id}] Batch embedding {len(chunk_texts)} chunks via Voyage AI...")
        chunk_embeddings = await embedding_service.aembed(chunk_texts, input_type="document")

        # ─── 2) Write chunk nodes ───
        progress.status = IngestionStatus.WRITING
        for chunk, emb in zip(chunks, chunk_embeddings):
            await graph_builder.create_chunk_node(chunk, emb)

        # ─── 3) Extract entities & relations ───
        progress.status = IngestionStatus.EXTRACTING
        extracted_data: List[tuple] = []  # (chunk, entities, relations)

        if not enable_llm_extraction:
            # High-speed offline extraction (zero token cost)
            logger.info(f"[{progress.doc_id}] Running offline rule-based entity extraction (0 LLM cost)...")
            for chunk in chunks:
                ents, rels = offline_extractor.extract_entities_and_relations(
                    text=chunk.text,
                    chunk_id=chunk.id,
                )
                extracted_data.append((chunk, ents, rels))
                progress.processed_chunks += 1
                if progress_callback:
                    try:
                        asyncio.create_task(progress_callback(progress))
                    except Exception:
                        pass
        else:
            # LLM-powered extraction: process ALL document chunks in 1 single batched LLM request
            # Prevents OpenRouter rate limits (RPM / 429 errors) and provides faster, cost-controlled extraction
            logger.info(f"[{progress.doc_id}] Running batched LLM entity extraction for {len(chunks)} chunks in 1 call ({llm_model or 'default'})...")
            batch_results = await llm_service.extract_entities_and_relations_batch(
                chunks=chunks,
                model=llm_model,
            )
            for chunk in chunks:
                ents, rels = batch_results.get(chunk.id, ([], []))
                extracted_data.append((chunk, ents, rels))
                progress.processed_chunks += 1
                if progress_callback:
                    try:
                        asyncio.create_task(progress_callback(progress))
                    except Exception:
                        pass

        # ─── 4) Batch embed ALL unique entities at once (1 Voyage API call) ───
        unique_entities: Dict[str, Dict] = {}
        for _, ents, _ in extracted_data:
            for ent in ents:
                name = ent.get("name", "").strip()
                if name and name.lower() not in unique_entities:
                    unique_entities[name.lower()] = ent

        entity_emb_map: Dict[str, List[float]] = {}
        if unique_entities:
            entity_items = list(unique_entities.values())
            entity_texts = [
                f"{e['name']}: {e.get('description', '')}" if e.get("description") else e["name"]
                for e in entity_items
            ]
            logger.info(f"[{progress.doc_id}] Batch embedding {len(entity_texts)} unique entities via Voyage AI...")
            entity_embeddings = await embedding_service.aembed(entity_texts, input_type="document")
            entity_emb_map = {
                e["name"].lower(): emb for e, emb in zip(entity_items, entity_embeddings)
            }

        # ─── 5) Write entities & relations to graph ───
        progress.status = IngestionStatus.WRITING
        for chunk, ents, rels in extracted_data:
            if ents or rels:
                await graph_builder.process_extracted_entities(
                    chunk_id=chunk.id,
                    entities=ents,
                    relations=rels,
                    entity_embeddings=entity_emb_map,
                )
                progress.total_entities += len(ents)
                progress.total_relations += len(rels)

    def get_job_status(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status of an ingestion job."""
        progress = self._jobs.get(doc_id)
        return progress.to_dict() if progress else None

    def get_all_jobs(self) -> List[Dict[str, Any]]:
        """Get status of all ingestion jobs."""
        return [p.to_dict() for p in self._jobs.values()]


# Singleton
ingestion_pipeline = IngestionPipeline()
