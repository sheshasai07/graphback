# GraphRAG Backend — Architecture & Developer Guide

## Overview

This is an **enterprise-grade GraphRAG (Graph Retrieval-Augmented Generation)** backend built using **FalkorDB Cloud** as the managed knowledge graph store, **Voyage AI (`voyage-4-lite`, 1024 dimensions)** for high-precision asymmetric vector embeddings, and **OpenRouter** as the LLM engine.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Backend                          │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │ /ingest  │  │ /query   │  │ /graph   │  │  /health     │   │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────────────┘   │
│       │              │             │                             │
│  ┌────▼─────────────────────────────────────────────────────┐   │
│  │                    Service Layer                          │   │
│  │                                                           │   │
│  │  IngestionPipeline  │  GraphRAGRetriever  │  LLMService  │   │
│  │  DocumentParser     │  GraphBuilder       │  Voyage AI   │   │
│  │  TextChunker        │                    │  Embeddings  │   │
│  └────┬──────────────────────────┬───────────────┬──────────┘   │
│       │                          │               │              │
│  ┌────▼──────────┐   ┌───────────▼──────┐  ┌────▼──────────┐  │
│  │ FalkorDB Cloud │   │    Voyage AI     │  │  OpenRouter   │  │
│  │  (Graph+Vec)  │   │ (voyage-4-lite)  │  │  (Nemotron)   │  │
│  └───────────────┘   └──────────────────┘  └───────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Ingestion Pipeline

```
File Upload
    │
    ▼
DocumentParser ── PDF/DOCX/TXT/MD/CSV/JSON/HTML ──► Plain Text
    │
    ▼
TextChunker ── Semantic paragraphs + overlap ──► [Chunk, Chunk, ...]
    │
    ▼ (Batch Voyage Embedding: 1 API Call for all chunks)
All Chunk Embeddings (1024-dim, input_type="document")
    │
    ▼
LLM Entity & Relationship Extraction
    │
    ▼ (Batch Voyage Embedding: 1 API Call for all unique entities)
All Entity Embeddings (1024-dim, input_type="document")
    │
    ▼
GraphBuilder:
    ├── FalkorDB: CREATE (:Chunk {embedding: vecf32(...)})
    ├── MERGE (:Entity {id: md5(name+type), embedding: vecf32(...)})
    ├── (Chunk)-[:MENTIONS]->(:Entity)
    └── (Entity)-[:RELATION_TYPE]->(:Entity)
```

---

## Retrieval Pipeline (GraphRAG)

```
User Query
    │
    ▼
LLMService.decompose_query() ── optional multi-query
    │
    ▼
EmbeddingService.embed(query)
    │
    ├──► Vector KNN Search on :Chunk.embedding
    │         └── FalkorDB: db.idx.vector.queryNodes(...)
    │
    ├──► Vector KNN Search on :Entity.embedding
    │         └── Top K entities
    │
    ├──► Graph Traversal
    │         └── MATCH (e:Entity)-[r*1..2]->(neighbor)
    │
    └──► Chunk Entity Extraction
              └── MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)

    ▼
LLMService.synthesize_answer(query, chunks, graph_context)
    │
    ▼
QueryResponse { answer, sources, entities, graph_context }
```

---

## Graph Schema (FalkorDB / OpenCypher)

### Nodes

| Label | Key Properties |
|-------|---------------|
| `:Document` | `id`, `name`, `type`, `chunk_count`, `created_at` |
| `:Chunk` | `id`, `text`, `document_id`, `chunk_index`, `embedding: vecf32(384)` |
| `:Entity` | `id`, `name`, `type`, `description`, `embedding: vecf32(384)`, `mention_count` |

### Relationships

| Pattern | Properties |
|---------|-----------|
| `(:Document)-[:HAS_CHUNK]->(:Chunk)` | — |
| `(:Chunk)-[:MENTIONS]->(:Entity)` | — |
| `(:Entity)-[:IS_A or HAS or WORKS_FOR or ...]->(:Entity)` | `weight`, `count` |

### Indexes

- `CREATE INDEX FOR (d:Document) ON (d.id)`
- `CREATE INDEX FOR (e:Entity) ON (e.name, e.type)`
- `CREATE VECTOR INDEX FOR (c:Chunk) ON (c.embedding)` cosine similarity
- `CREATE VECTOR INDEX FOR (e:Entity) ON (e.embedding)` cosine similarity

---

## API Endpoints

### Ingestion
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/ingest/upload` | Upload document for ingestion |
| `GET` | `/api/v1/ingest/status/{doc_id}` | Get ingestion job status |
| `GET` | `/api/v1/ingest/status/{doc_id}/stream` | SSE streaming progress |
| `GET` | `/api/v1/ingest/jobs` | List all jobs |

### Query
| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/query` | GraphRAG query (JSON response) |
| `POST` | `/api/v1/query/stream` | GraphRAG query (SSE streaming) |

### Graph
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/graph/stats` | Graph statistics |
| `GET` | `/api/v1/graph/documents` | List documents |
| `DELETE` | `/api/v1/graph/documents/{id}` | Delete document |
| `GET` | `/api/v1/graph/entities` | List entities |
| `GET` | `/api/v1/graph/entities/types` | Entity type distribution |
| `POST` | `/api/v1/graph/entities/neighborhood` | Entity subgraph |
| `GET` | `/api/v1/graph/visualization` | D3.js graph data |

### System
| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/health` | Health check |
| `GET` | `/api/v1/config` | Backend configuration |

---

## Getting Started

### 1. Start FalkorDB
```bash
cd backend
docker compose up -d
```
FalkorDB Browser UI: http://localhost:3000

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env — add your OPENROUTER_API_KEY
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Backend
```bash
python run.py
```

API Docs: http://localhost:8000/docs

---

## Key Design Decisions

### Why FalkorDB?
- Built on Redis for ultra-low latency
- Native vector index (`vecf32`) for similarity search
- OpenCypher standard query language
- Combines vector + graph in one store — no extra infrastructure

### Why Sentence-Transformers locally?
- Zero API cost for embeddings
- `all-MiniLM-L6-v2` = fast, 384-dim, great quality/speed tradeoff
- No rate limits or latency from external API calls

### Entity Deduplication
- Entity IDs are `md5(type + name.lower())` — deterministic
- `MERGE` in Cypher ensures no duplicate nodes
- `mention_count` tracks entity co-occurrence frequency

### Hybrid Retrieval
- **Vector search**: Finds semantically similar chunks
- **Entity vector search**: Finds relevant concept nodes
- **Graph traversal**: Expands context via 1-2 hop relationships
- Combined context gives LLM both exact passages AND relational knowledge
