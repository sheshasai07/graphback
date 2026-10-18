# GraphGenie: Context-Aware Multi-Agent GraphRAG & Living Company Brain

Built on **FalkorDB Cloud**, **FastAPI**, **Voyage AI (`voyage-4-lite`)**, **OpenRouter (Nvidia Nemotron)**, and **Zoho Catalyst Cloud File Store**.

## Hackathon Tracks Covered
- **Track 01: Agents That Act on Connected Data** — Hybrid Vector Search (`vecf32`) + Multi-Hop Traversal + `shortestPath()` Algorithm + Explainable OpenCypher Reasoning Trail.
- **Track 02: Agent Memory & Coordination** — Investigator & Resolver Agents coordinating via `:HandoffTask` Blackboard nodes + Persistent `:MemoryEpisode` nodes + Multi-Tenant `select_graph()` isolation.
- **Track 03: Company Brain** — Multi-Format Ingestion (PDF, DOCX, Markdown, CSV, Excel) + Live Semantic Ownership Mutation via natural language chat.

---

## Graph Data Model (FalkorDB Property Graph Schema)

### Node Labels & Properties
| Label | Key Properties | Description |
| :--- | :--- | :--- |
| `:Document` | `id`, `name`, `type`, `chunk_count`, `created_at`, `metadata` | Ingested enterprise RFCs, runbooks, PDFs, CSVs, and Excel sheets |
| `:Chunk` | `id`, `text`, `document_id`, `document_name`, `chunk_index`, `embedding: vecf32(1024)` | Semantic text segments indexed with FalkorDB cosine vector index |
| `:Entity` | `id` (MD5), `name`, `type`, `description`, `mention_count`, `embedding: vecf32(1024)` | Extracted domain nodes (`Service`, `Feature`, `Person`, `Role`, `Technology`, `Status`) |
| `:HandoffTask` | `id`, `status` (`OPEN`/`COMPLETED`), `symptom`, `target_service`, `assigned_to`, `resolution_fix` | Operational blackboard task nodes shared between Investigator and Resolver agents |
| `:MemoryEpisode` | `id`, `session_id`, `timestamp`, `symptom`, `root_cause`, `fix`, `service_name` | Long-term episodic memory records preventing cross-session RAG amnesia |

### Relationships
- `(:Document)-[:HAS_CHUNK]->(:Chunk)`
- `(:Chunk)-[:MENTIONS]->(:Entity)`
- `(:Entity)-[:DEPENDS_ON | :CALLS | :MAINTAINED_BY | :MANAGES | :MEMBER_OF | :RELATED_TO]->(:Entity)`
- `(:HandoffTask)-[:TARGETS]->(:Entity)`
- `(:MemoryEpisode)-[:RESOLVED_FOR]->(:Entity)`

---

## Core OpenCypher Queries & Graph Algorithms

### 1. Native Vector Index Creation & K-NN Search (`vecf32`)
```cypher
CREATE VECTOR INDEX FOR (c:Chunk) ON (c.embedding)
OPTIONS {dimension: 1024, similarityFunction: 'cosine'};

CALL db.idx.vector.queryNodes('Chunk', 'embedding', $k, vecf32($embedding))
YIELD node, score
RETURN node.id, node.text, node.document_id, node.document_name, score
ORDER BY score DESC;
```

### 2. Multi-Hop Neighborhood Traversal & `shortestPath()` Algorithm
```cypher
MATCH path = (src)-[r*1..2]-(tgt)
WHERE src.name IN $seed_names AND src <> tgt AND NOT src:Chunk AND NOT tgt:Chunk
UNWIND relationships(path) AS rel
WITH startNode(rel) AS u, rel, endNode(rel) AS v
RETURN DISTINCT labels(u)[0], u.id, u.name, type(rel), labels(v)[0], v.id, v.name;

MATCH (start {name: $start_name}), (target)
WHERE start <> target AND target.name IN $candidates
WITH coalesce(shortestPath((start)-[*1..4]->(target)), shortestPath((target)-[*1..4]->(start))) AS p
WHERE p IS NOT NULL
RETURN [n IN nodes(p) | n.name] AS path_nodes, length(p) AS hops
ORDER BY hops ASC LIMIT 1;
```

### 3. Multi-Agent Blackboard Handoff & Episodic Memory Persistence
```cypher
// Investigator Agent dispatches OPEN task
CREATE (t:HandoffTask {id: $id, status: 'OPEN', symptom: $symptom, target_service: $srv, assigned_to: $assigned})
WITH t
OPTIONAL MATCH (s) WHERE (s:Service OR s:Entity) AND toLower(s.name) = toLower($srv)
FOREACH (_ IN CASE WHEN s IS NOT NULL THEN [1] ELSE [] END | MERGE (t)-[:TARGETS]->(s));

// Resolver Agent completes task & logs permanent MemoryEpisode
MATCH (t:HandoffTask {id: $task_id})
SET t.status = 'COMPLETED', t.resolved_at = $resolved_at, t.resolution_fix = $fix;

CREATE (e:MemoryEpisode {id: $id, session_id: $session_id, timestamp: $timestamp, symptom: $symptom, root_cause: $root_cause, fix: $fix, service_name: $srv})
WITH e
OPTIONAL MATCH (s) WHERE (s:Service OR s:Entity) AND toLower(s.name) = toLower($srv)
FOREACH (_ IN CASE WHEN s IS NOT NULL THEN [1] ELSE [] END | MERGE (e)-[:RESOLVED_FOR]->(s));
```

### 4. Live Semantic Ownership Mutation
```cypher
MERGE (new_m:Entity {name: $new_maintainer})
ON CREATE SET new_m.type = 'Person', new_m.id = $new_id
WITH new_m
MATCH (s:Entity) WHERE toLower(s.name) = toLower($service_name)
OPTIONAL MATCH (s)-[old_rel:MAINTAINED_BY]->(old_m)
DELETE old_rel
CREATE (s)-[:MAINTAINED_BY]->(new_m)
RETURN s.name, old_m.name, new_m.name;
```

---

## Local Setup & Installation Instructions

### 1. Clone the Repository
```bash
git clone [https://github.com/sheshasai07/graphback.git](https://github.com/sheshasai07/graphback.git)
cd graphback/backend
```

### 2. Configure Environment Variables (`.env`)
Create a `.env` file inside the `backend/` directory:
```env
# FalkorDB Cloud Configuration
FALKORDB_URL=falkors://default:YOUR_PASSWORD@your-falkordb-host:port
FALKORDB_HOST=your-falkordb-host
FALKORDB_PORT=6379
FALKORDB_USERNAME=default
FALKORDB_PASSWORD=your-password
FALKORDB_SSL=true
FALKORDB_GRAPH_NAME=graphrag

# OpenRouter LLM Configuration
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=nvidia/nemotron-3-super-120b-a12b:free

# Voyage AI Embeddings Configuration
VOYAGE_API_KEY=your_voyage_api_key
EMBEDDING_MODEL_NAME=voyage-4-lite
EMBEDDING_DIMENSION=1024

# Zoho Catalyst File Store (Cloud Storage)
CATALYST_PROJECT_ID=
CATALYST_FOLDER_ID=
CATALYST_ENVIRONMENT=Development
CATALYST_API_DOMAIN=api.catalyst.zoho.in
CATALYST_ACCOUNTS_URL=[https://accounts.zoho.in](https://accounts.zoho.in)

# OAuth 2.0 Credentials (from Zoho API Console)
CATALYST_CLIENT_ID=
CATALYST_CLIENT_SECRET=
CATALYST_REFRESH_TOKEN=
```

### 3. Install Dependencies & Start Server
```bash
pip install -r requirements.txt
python check_connection.py
python run.py
```

- **API Documentation (Swagger UI):** `http://localhost:8000/docs`
- **Judge Validation Suite:** `python test_all_judge_criteria.py`
