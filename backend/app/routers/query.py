"""
GraphRAG Backend - Query Router
Handles GraphRAG queries with hybrid vector + graph retrieval.
Supports both standard JSON responses and streaming (SSE).
"""

import json
import time
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from app.models.schemas import QueryRequest, QueryResponse
from app.services.graph_retriever import graph_retriever, GraphRAGRetriever
from app.services.llm_service import llm_service

router = APIRouter(prefix="/api/v1/query", tags=["Query"])


@router.post("", response_model=QueryResponse)
async def query_knowledge_graph(request: QueryRequest):
    """
    Query the knowledge graph using GraphRAG.
    
    Performs hybrid retrieval:
    1. Semantic vector search on document chunks
    2. Entity vector search 
    3. Knowledge graph traversal (multi-hop)
    4. LLM synthesis with combined context
    
    Returns the answer with supporting evidence from the graph.
    """
    start_time = time.monotonic()

    try:
        # Initialize retriever with request parameters
        retriever = GraphRAGRetriever(
            top_k_chunks=request.top_k_chunks,
            top_k_entities=request.top_k_entities,
            graph_hops=request.graph_hops,
        )

        # Retrieve context with user-selected graph & embedding model
        context = await retriever.retrieve(
            query=request.query,
            use_query_decomposition=request.use_query_decomposition,
            graph_name=request.graph_name,
            embedding_model=request.embedding_model,
        )

        # Synthesize answer with user-selected LLM model and hyperparameters
        answer = await llm_service.synthesize_answer(
            query=request.query,
            context_chunks=context["chunks"],
            graph_context=context["graph_context"],
            model=request.llm_model,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
        )

        elapsed_ms = (time.monotonic() - start_time) * 1000

        # Deduplicate sources
        sources = list({
            c["document_name"]
            for c in context["chunks"]
            if c.get("document_name")
        })

        return QueryResponse(
            query=request.query,
            answer=answer,
            sub_queries=context.get("sub_queries", [request.query]),
            chunks=context["chunks"],
            entities=context["entities"],
            graph_context=context["graph_context"],
            processing_time_ms=round(elapsed_ms, 2),
            sources=sources,
        )

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def stream_query(request: QueryRequest):
    """
    Stream GraphRAG query response via Server-Sent Events.
    Useful for long-form answers that take time to generate.
    """
    async def event_generator():
        start_time = time.monotonic()
        try:
            # Step 1: Notify retrieval start
            yield f"data: {json.dumps({'type': 'status', 'message': 'Retrieving context...'})}\n\n"

            retriever = GraphRAGRetriever(
                top_k_chunks=request.top_k_chunks,
                top_k_entities=request.top_k_entities,
                graph_hops=request.graph_hops,
            )
            context = await retriever.retrieve(
                query=request.query,
                use_query_decomposition=request.use_query_decomposition,
                graph_name=request.graph_name,
                embedding_model=request.embedding_model,
            )

            # Step 2: Emit context metadata
            yield f"data: {json.dumps({'type': 'context', 'chunks': len(context['chunks']), 'entities': len(context['entities'])})}\n\n"

            # Step 3: Generate answer
            yield f"data: {json.dumps({'type': 'status', 'message': 'Generating answer...'})}\n\n"

            answer = await llm_service.synthesize_answer(
                query=request.query,
                context_chunks=context["chunks"],
                graph_context=context["graph_context"],
                model=request.llm_model,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
            )

            elapsed_ms = (time.monotonic() - start_time) * 1000
            sources = list({c["document_name"] for c in context["chunks"] if c.get("document_name")})

            # Step 4: Final answer
            payload = {
                "type": "answer",
                "query": request.query,
                "answer": answer,
                "sub_queries": context.get("sub_queries", []),
                "chunks": context["chunks"][:5],  # Top 5 for brevity
                "entities": context["entities"][:10],
                "sources": sources,
                "processing_time_ms": round(elapsed_ms, 2),
            }
            yield f"data: {json.dumps(payload)}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Stream query error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
