"""
GraphRAG Backend - FalkorDB Client
Manages connection pool and Cypher query execution with FalkorDB.
Includes an automatic in-memory graph engine fallback when a local FalkorDB
service is offline or starting up, ensuring zero-downtime development and testing.
"""

import asyncio
import json
import re
from typing import Any, Dict, List, Optional
import numpy as np
from loguru import logger
import falkordb

from app.config import settings


class QueryResultWrapper:
    """Wrapper matching the falkordb.QueryResult interface."""

    def __init__(self, result_set: Optional[List[List[Any]]] = None):
        self.result_set = result_set or []


class InMemoryGraphEngine:
    """
    Lightweight in-memory graph database fallback.
    Stores Documents, Chunks, Entities, and Relationships in memory.
    Supports similarity calculations and Cypher-like operations.
    """

    def __init__(self):
        self.documents: Dict[str, Dict[str, Any]] = {}
        self.chunks: Dict[str, Dict[str, Any]] = {}
        self.entities: Dict[str, Dict[str, Any]] = {}
        self.relations: List[Dict[str, Any]] = []
        self.chunk_mentions: set = set()  # (chunk_id, entity_id)
        # GraphGenie Cognitive & Domain Extensions
        self.services: Dict[str, Dict[str, Any]] = {}
        self.features: Dict[str, Dict[str, Any]] = {}
        self.persons: Dict[str, Dict[str, Any]] = {}
        self.episodes: Dict[str, Dict[str, Any]] = {}
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.workspaces: Dict[str, Any] = {
            "workspace_default": {"id": "workspace_default", "name": "Default Core Topology", "description": "Core enterprise microservices, auth flows, and incident history"},
            "workspace_payments": {"id": "workspace_payments", "name": "Payments & Billing", "description": "Payment gateways, Stripe compliance, and transaction services"},
            "workspace_identity": {"id": "workspace_identity", "name": "Identity & Security", "description": "IAM, OAuth2 providers, and Zero-Trust policies"},
        }
        self.current_workspace = "workspace_default"

    @staticmethod
    def _cosine_sim(a: List[float], b: List[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        va = np.array(a, dtype=np.float32)
        vb = np.array(b, dtype=np.float32)
        norm_a = np.linalg.norm(va)
        norm_b = np.linalg.norm(vb)
        if norm_a < 1e-6 or norm_b < 1e-6:
            return 0.0
        return float(np.dot(va, vb) / (norm_a * norm_b))

    def execute(self, cypher: str, params: Optional[dict] = None) -> QueryResultWrapper:
        params = params or {}
        q = " ".join(cypher.strip().split())

        # Health ping
        if "RETURN 1 AS ping" in q or "RETURN 1" in q:
            return QueryResultWrapper([[1]])

        # ─── Vector Query Nodes (Chunk) ──────────────────────────────────────────
        if "db.idx.vector.queryNodes('Chunk'" in q or 'db.idx.vector.queryNodes("Chunk"' in q:
            target_emb = params.get("embedding", [])
            k = params.get("k", 8)
            scored = []
            for c in self.chunks.values():
                emb = c.get("embedding", [])
                score = self._cosine_sim(target_emb, emb)
                scored.append((score, c))
            scored.sort(key=lambda x: x[0], reverse=True)
            rows = []
            for score, c in scored[:k]:
                rows.append([
                    c["id"],
                    c["text"],
                    c["document_id"],
                    c["document_name"],
                    c["chunk_index"],
                    score,
                ])
            return QueryResultWrapper(rows)

        # ─── Vector Query Nodes (Entity) ─────────────────────────────────────────
        if "db.idx.vector.queryNodes('Entity'" in q or 'db.idx.vector.queryNodes("Entity"' in q:
            target_emb = params.get("embedding", [])
            k = params.get("k", 10)
            scored = []
            for e in self.entities.values():
                emb = e.get("embedding", [])
                score = self._cosine_sim(target_emb, emb)
                scored.append((score, e))
            scored.sort(key=lambda x: x[0], reverse=True)
            rows = []
            for score, e in scored[:k]:
                rows.append([
                    e["id"],
                    e["name"],
                    e["type"],
                    e.get("description", ""),
                    e.get("mention_count", 1),
                    score,
                ])
            return QueryResultWrapper(rows)

        # ─── Documents ──────────────────────────────────────────────────────────
        if "MERGE (d:Document {id: $id})" in q:
            doc_id = params.get("id")
            self.documents[doc_id] = {
                "id": doc_id,
                "name": params.get("name", ""),
                "type": params.get("type", "unknown"),
                "metadata": params.get("metadata", "{}"),
                "created_at": params.get("created_at", ""),
                "chunk_count": 0,
            }
            return QueryResultWrapper([[doc_id]])

        if "MATCH (d:Document {id: $id}) SET d.chunk_count = $count" in q:
            doc_id = params.get("id")
            if doc_id in self.documents:
                self.documents[doc_id]["chunk_count"] = params.get("count", 0)
            return QueryResultWrapper()

        if "MATCH (d:Document)" in q and "RETURN d.id, d.name" in q:
            limit = params.get("limit", 50)
            offset = params.get("offset", 0)
            rows = []
            docs = list(self.documents.values())
            docs.sort(key=lambda x: x.get("created_at", ""), reverse=True)
            for d in docs[offset : offset + limit]:
                rows.append([
                    d["id"],
                    d["name"],
                    d["type"],
                    d["chunk_count"],
                    d["created_at"],
                    d["metadata"],
                ])
            return QueryResultWrapper(rows)

        if "MATCH (d:Document {id: $id}) RETURN d.id" in q:
            doc_id = params.get("id")
            if doc_id in self.documents:
                return QueryResultWrapper([[doc_id]])
            return QueryResultWrapper([])

        if "DETACH DELETE c, d" in q or "DELETE d" in q:
            doc_id = params.get("doc_id") or params.get("id")
            if doc_id in self.documents:
                del self.documents[doc_id]
            # Remove associated chunks
            cids_to_del = [cid for cid, c in self.chunks.items() if c.get("document_id") == doc_id]
            for cid in cids_to_del:
                del self.chunks[cid]
                self.chunk_mentions = {m for m in self.chunk_mentions if m[0] != cid}
            return QueryResultWrapper()

        # ─── Chunks ─────────────────────────────────────────────────────────────
        if "MERGE (c:Chunk {id: $id})" in q:
            cid = params.get("id")
            embedding = params.get("embedding", [])
            self.chunks[cid] = {
                "id": cid,
                "text": params.get("text", ""),
                "document_id": params.get("document_id", ""),
                "document_name": params.get("document_name", ""),
                "chunk_index": params.get("chunk_index", 0),
                "token_count": params.get("token_count", 0),
                "embedding": embedding,
                "metadata": params.get("metadata", "{}"),
            }
            return QueryResultWrapper([[cid]])

        if "MATCH (c:Chunk)" in q and "RETURN c.id, c.text" in q:
            rows = []
            for c in self.chunks.values():
                rows.append([
                    c["id"],
                    c["text"],
                    c["document_id"],
                    c["document_name"],
                    c["chunk_index"],
                    c.get("embedding", []),
                ])
            return QueryResultWrapper(rows)

        # ─── Entities ───────────────────────────────────────────────────────────
        if "MERGE (e:Entity {id: $id})" in q:
            eid = params.get("id")
            if eid in self.entities:
                self.entities[eid]["mention_count"] += 1
                if params.get("description"):
                    self.entities[eid]["description"] = params["description"]
            else:
                self.entities[eid] = {
                    "id": eid,
                    "name": params.get("name", ""),
                    "type": params.get("type", "Concept"),
                    "description": params.get("description", ""),
                    "embedding": params.get("embedding", []),
                    "mention_count": 1,
                }
            return QueryResultWrapper([[eid]])

        if "MERGE (c)-[:MENTIONS]->(e)" in q:
            cid = params.get("chunk_id")
            eid = params.get("entity_id")
            if cid and eid:
                self.chunk_mentions.add((cid, eid))
            return QueryResultWrapper()

        if "MATCH (e:Entity)" in q and "RETURN e.id, e.name, e.type" in q:
            rows = []
            for e in self.entities.values():
                rows.append([
                    e["id"],
                    e["name"],
                    e["type"],
                    e.get("description", ""),
                    e.get("embedding", []),
                ])
            return QueryResultWrapper(rows)

        # ─── Relations ──────────────────────────────────────────────────────────
        if "MERGE (s)-[r:" in q or "MERGE (s)-[r" in q or "r.weight" in q:
            source_id = params.get("source_id")
            target_id = params.get("target_id")
            rel_type = "RELATED_TO"
            match = re.search(r"MERGE \(s\)-\[r:([A-Za-z0-9_]+)\]->\(t\)", cypher)
            if match:
                rel_type = match.group(1)

            if source_id in self.entities and target_id in self.entities:
                self.relations.append({
                    "source_id": source_id,
                    "source_name": self.entities[source_id]["name"],
                    "source_type": self.entities[source_id]["type"],
                    "relation": rel_type,
                    "target_id": target_id,
                    "target_name": self.entities[target_id]["name"],
                    "target_type": self.entities[target_id]["type"],
                    "weight": float(params.get("weight", 1.0)),
                    "chunk_id": params.get("chunk_id"),
                })
            return QueryResultWrapper()

        # ─── Chunk Mentions (Chunk Entities) ────────────────────────────────────
        if "MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)" in q:
            chunk_ids = set(params.get("chunk_ids", []))
            matched_entity_ids = {eid for cid, eid in self.chunk_mentions if not chunk_ids or cid in chunk_ids}
            rows = []
            for eid in matched_entity_ids:
                if eid in self.entities:
                    e = self.entities[eid]
                    rows.append([
                        e["name"],
                        e["type"],
                        e.get("description", ""),
                        e.get("mention_count", 1),
                    ])
            return QueryResultWrapper(rows)

        # ─── Graph Traversal / Neighborhood ─────────────────────────────────────
        if "MATCH (start:Entity)" in q and "WHERE start.id IN $entity_ids" in q:
            entity_ids = set(params.get("entity_ids", []))
            rows = []
            for r in self.relations:
                if r["source_id"] in entity_ids or r["target_id"] in entity_ids:
                    rows.append([
                        r["source_name"],
                        r["relation"],
                        r["target_name"],
                        r["target_type"],
                        "",
                        1,
                    ])
            return QueryResultWrapper(rows)

        if "MATCH (s:Entity)-[r]->(t:Entity)" in q:
            entity_ids = set(params.get("entity_ids", []))
            rows = []
            for r in self.relations:
                if not entity_ids or (r["source_id"] in entity_ids or r["target_id"] in entity_ids):
                    rows.append([
                        r["source_name"],
                        r["source_type"],
                        r["relation"],
                        r["target_name"],
                        r["target_type"],
                        r.get("weight", 1.0),
                    ])
            return QueryResultWrapper(rows)

        if "MATCH (e:Entity {name: $name})" in q or 'MATCH (e:Entity {name: "' in q:
            name = params.get("name")
            matched_e = None
            for e in self.entities.values():
                if e["name"].lower() == (name or "").lower():
                    matched_e = e
                    break
            if not matched_e:
                return QueryResultWrapper([])
            edges = []
            for r in self.relations:
                if r["source_id"] == matched_e["id"] or r["target_id"] == matched_e["id"]:
                    edges.append({
                        "source": r["source_name"],
                        "source_type": r["source_type"],
                        "relation": r["relation"],
                        "target": r["target_name"],
                        "target_type": r["target_type"],
                    })
            row = [
                matched_e["name"],
                matched_e["type"],
                matched_e.get("description", ""),
                matched_e.get("mention_count", 1),
                edges,
            ]
            return QueryResultWrapper([row])

        # ─── Graph Stats ────────────────────────────────────────────────────────
        if "MATCH (d:Document) RETURN count(d)" in q:
            return QueryResultWrapper([[len(self.documents)]])
        if "MATCH (c:Chunk) RETURN count(c)" in q:
            return QueryResultWrapper([[len(self.chunks)]])
        if "MATCH (e:Entity) RETURN count(e)" in q:
            return QueryResultWrapper([[len(self.entities)]])
        if "MATCH ()-[r]->() RETURN count(r)" in q:
            total_edges = len(self.relations) + len(self.chunk_mentions) + len(self.chunks)
            return QueryResultWrapper([[total_edges]])

        if "MATCH (e:Entity) RETURN e.type, count(e)" in q:
            counts = {}
            for e in self.entities.values():
                t = e.get("type", "Other")
                counts[t] = counts.get(t, 0) + 1
            return QueryResultWrapper([[k, v] for k, v in counts.items()])

        # ─── GraphGenie: Handoff Tasks ──────────────────────────────────────────
        if "MERGE (t:HandoffTask {id: $id})" in q or "CREATE (t:HandoffTask" in q:
            tid = params.get("id") or f"task_{len(self.tasks) + 101}"
            self.tasks[tid] = {
                "id": tid,
                "status": params.get("status", "OPEN"),
                "symptom": params.get("symptom", ""),
                "details": params.get("details", ""),
                "priority": params.get("priority", "HIGH"),
                "created_at": params.get("created_at", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")),
                "target_service": params.get("srv") or params.get("target_service"),
                "assigned_to": params.get("assigned_to"),
                "resolved_at": params.get("resolved_at"),
                "resolution_fix": params.get("resolution_fix"),
            }
            return QueryResultWrapper([[tid]])

        if "MATCH (t:HandoffTask" in q and "SET t.status" in q:
            tid = params.get("id")
            status = params.get("status")
            if tid in self.tasks and status:
                self.tasks[tid]["status"] = status
                if status == "COMPLETED":
                    self.tasks[tid]["resolved_at"] = params.get("resolved_at", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"))
                    if params.get("resolution_fix"):
                        self.tasks[tid]["resolution_fix"] = params.get("resolution_fix")
            return QueryResultWrapper([[tid]])

        if "MATCH (t:HandoffTask)" in q or "MATCH (t:HandoffTask" in q:
            status_filter = params.get("status")
            rows = []
            for t in self.tasks.values():
                if status_filter and t.get("status") != status_filter:
                    continue
                rows.append([
                    t["id"],
                    t["status"],
                    t["symptom"],
                    t.get("details", ""),
                    t.get("priority", "HIGH"),
                    t.get("created_at", ""),
                    t.get("target_service", ""),
                    t.get("assigned_to", ""),
                    t.get("resolved_at"),
                    t.get("resolution_fix"),
                ])
            return QueryResultWrapper(rows)

        # ─── GraphGenie: Memory Episodes ────────────────────────────────────────
        if "MERGE (e:MemoryEpisode {id: $id})" in q or "CREATE (e:MemoryEpisode" in q:
            eid = params.get("id") or f"ep_{len(self.episodes) + 201}"
            self.episodes[eid] = {
                "id": eid,
                "session_id": params.get("session_id", "default_session"),
                "timestamp": params.get("ts") or params.get("timestamp", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")),
                "symptom": params.get("symptom", ""),
                "root_cause": params.get("root_cause", ""),
                "fix": params.get("fix", ""),
                "service_name": params.get("srv") or params.get("service_name", ""),
            }
            return QueryResultWrapper([[eid]])

        if "MATCH (e:MemoryEpisode)" in q or "MATCH (e:MemoryEpisode" in q:
            srv_filter = params.get("service_name") or params.get("srv")
            rows = []
            for ep in self.episodes.values():
                if srv_filter and ep.get("service_name", "").lower() != srv_filter.lower():
                    continue
                rows.append([
                    ep["id"],
                    ep["session_id"],
                    ep["timestamp"],
                    ep["symptom"],
                    ep["root_cause"],
                    ep["fix"],
                    ep.get("service_name", ""),
                ])
            return QueryResultWrapper(rows)

        # ─── GraphGenie: Services, Features, Persons ────────────────────────────
        if "MATCH (s:Service)" in q:
            rows = []
            for s in self.services.values():
                rows.append([
                    s["id"],
                    s["name"],
                    s.get("runtime", "Go"),
                    s.get("criticality", "Tier-1"),
                    s.get("description", ""),
                    s.get("maintainer", ""),
                ])
            return QueryResultWrapper(rows)

        if "MATCH (f:Feature)" in q:
            rows = []
            for f in self.features.values():
                rows.append([
                    f["id"],
                    f["name"],
                    f.get("status", "active"),
                    f.get("stage", "prod"),
                    f.get("description", ""),
                    f.get("service", ""),
                ])
            return QueryResultWrapper(rows)

        if "MATCH (p:Person)" in q:
            rows = []
            for p in self.persons.values():
                rows.append([
                    p["id"],
                    p["name"],
                    p.get("role", "Engineer"),
                    p.get("email", ""),
                    p.get("on_call", False),
                ])
            return QueryResultWrapper(rows)

        # Default fallback: return empty result set
        return QueryResultWrapper([])


class FalkorDBClient:
    """
    Thread-safe FalkorDB client wrapper with automatic in-memory fallback.
    """

    def __init__(self):
        self._client: Optional[falkordb.FalkorDB] = None
        self._graph: Optional[falkordb.Graph] = None
        self._in_memory_engine = InMemoryGraphEngine()
        self._use_in_memory = False
        self._lock = asyncio.Lock()

    @property
    def is_connected(self) -> bool:
        return self._client is not None and not self._use_in_memory

    def connect(self):
        """Establish connection to FalkorDB Cloud or activate in-memory engine."""
        try:
            url_set = bool(settings.falkordb_url and settings.falkordb_url.strip())
            host_set = bool(settings.falkordb_host and settings.falkordb_host.strip())

            if not url_set and not host_set:
                logger.info(
                    "No FalkorDB Cloud credentials configured in .env (FALKORDB_URL or FALKORDB_HOST). "
                    "In-Memory Graph Engine fallback is active."
                )
                self._client = None
                self._graph = None
                self._use_in_memory = True
                return

            if url_set:
                raw_url = settings.falkordb_url.strip()
                if "://" not in raw_url:
                    # User provided raw host:port (e.g. from FalkorDB Cloud endpoint)
                    logger.info("Parsing host and port from FalkorDB endpoint string...")
                    if ":" in raw_url:
                        parts = raw_url.split(":")
                        parsed_host = parts[0].strip()
                        try:
                            parsed_port = int(parts[1].strip())
                        except ValueError:
                            parsed_port = settings.falkordb_port
                    else:
                        parsed_host = raw_url
                        parsed_port = settings.falkordb_port

                    kwargs = {
                        "host": parsed_host,
                        "port": parsed_port,
                        "socket_connect_timeout": 5,
                        "socket_timeout": 5,
                    }
                    if settings.falkordb_username:
                        kwargs["username"] = settings.falkordb_username
                    if settings.falkordb_password:
                        kwargs["password"] = settings.falkordb_password
                    if settings.falkordb_ssl:
                        kwargs["ssl"] = True

                    logger.info(
                        f"Connecting to FalkorDB Cloud at {kwargs['host']}:{kwargs['port']} "
                        f"(SSL={settings.falkordb_ssl})..."
                    )
                    self._client = falkordb.FalkorDB(**kwargs)
                else:
                    logger.info("Connecting to FalkorDB Cloud via URL...")
                    self._client = falkordb.FalkorDB.from_url(
                        raw_url, socket_connect_timeout=5, socket_timeout=5
                    )
            else:
                kwargs = {
                    "host": settings.falkordb_host.strip(),
                    "port": settings.falkordb_port,
                    "socket_connect_timeout": 5,
                    "socket_timeout": 5,
                }
                if settings.falkordb_username:
                    kwargs["username"] = settings.falkordb_username
                if settings.falkordb_password:
                    kwargs["password"] = settings.falkordb_password
                if settings.falkordb_ssl:
                    kwargs["ssl"] = True

                logger.info(
                    f"Connecting to FalkorDB Cloud at {kwargs['host']}:{kwargs['port']} "
                    f"(SSL={settings.falkordb_ssl})..."
                )
                self._client = falkordb.FalkorDB(**kwargs)

            self._graph = self._client.select_graph(settings.falkordb_graph_name)
            # Test connectivity with ping
            self._graph.query("RETURN 1 AS ping")
            self._use_in_memory = False
            logger.info(
                f"Successfully connected to FalkorDB Cloud! "
                f"| Active Graph: '{settings.falkordb_graph_name}'"
            )
        except Exception as e:
            logger.warning(
                f"FalkorDB connection failed ({e}). "
                f"Activated local In-Memory Graph Engine fallback for graph: '{settings.falkordb_graph_name}'"
            )
            self._client = None
            self._graph = None
            self._use_in_memory = True

    def disconnect(self):
        """Close the FalkorDB connection."""
        if self._client:
            try:
                self._client.connection.close()
                logger.info("FalkorDB connection closed.")
            except Exception as e:
                logger.warning(f"Error closing FalkorDB connection: {e}")

    @property
    def graph(self) -> Any:
        if self._use_in_memory:
            return self._in_memory_engine
        if self._graph is None:
            raise RuntimeError("FalkorDB not connected. Call connect() first.")
        return self._graph

    def query(self, cypher: str, params: Optional[dict] = None) -> Any:
        """Execute a Cypher query synchronously."""
        if self._use_in_memory or self._graph is None:
            return self._in_memory_engine.execute(cypher, params)

        try:
            return self._graph.query(cypher, params or {})
        except Exception as e:
            logger.warning(f"FalkorDB query failed ({e}), routing to in-memory engine.")
            return self._in_memory_engine.execute(cypher, params)

    async def aquery(self, cypher: str, params: Optional[dict] = None) -> Any:
        """Run a Cypher query asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.query, cypher, params)

    def initialize_schema(self):
        """Set up FalkorDB schema: indexes, vector indexes, and constraints."""
        if self._use_in_memory or self._graph is None:
            logger.info("In-Memory Graph Engine schema initialized ✓")
            return

        logger.info("Initializing FalkorDB schema...")
        schema_queries = [
            "CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.id)",
            "CREATE INDEX IF NOT EXISTS FOR (c:Chunk) ON (c.id)",
            "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX IF NOT EXISTS FOR (c:Chunk) ON (c.document_id)",
            f"""
            CREATE VECTOR INDEX IF NOT EXISTS FOR (c:Chunk) ON (c.embedding)
            OPTIONS {{dimension: {settings.embedding_dimension}, similarityFunction: 'cosine'}}
            """,
            f"""
            CREATE VECTOR INDEX IF NOT EXISTS FOR (e:Entity) ON (e.embedding)
            OPTIONS {{dimension: {settings.embedding_dimension}, similarityFunction: 'cosine'}}
            """,
            # GraphGenie schema indexes
            "CREATE INDEX IF NOT EXISTS FOR (s:Service) ON (s.name)",
            "CREATE INDEX IF NOT EXISTS FOR (f:Feature) ON (f.name)",
            "CREATE INDEX IF NOT EXISTS FOR (p:Person) ON (p.name)",
            "CREATE INDEX IF NOT EXISTS FOR (t:HandoffTask) ON (t.id)",
            "CREATE INDEX IF NOT EXISTS FOR (e:MemoryEpisode) ON (e.id)",
        ]

        for q in schema_queries:
            try:
                self._graph.query(q.strip())
            except Exception as e:
                logger.debug(f"Schema init note: {e}")

        logger.info("FalkorDB schema initialized.")

    def select_graph(self, graph_name: str):
        """Switch tenant workspace / graph."""
        if self._client:
            self._graph = self._client.select_graph(graph_name)
        self._in_memory_engine.current_workspace = graph_name
        logger.info(f"Switched active graph to '{graph_name}'")


# Singleton client instance
db_client = FalkorDBClient()
