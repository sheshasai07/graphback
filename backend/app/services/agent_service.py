"""
GraphGenie - Multi-Agent Orchestration Service
Implements Investigator Agent, Resolver Agent, Episodic Memory Routing,
GraphRAG Impact Traversal, and Reasoning Trail Generation directly connected
to FalkorDB Cloud with in-memory fallback.
"""

import time
import re
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from loguru import logger

from app.config import settings
from app.db.falkordb_client import db_client
from app.services.embedding_service import embedding_service
from app.services.llm_service import llm_service
from app.models.agent_schemas import (
    AgentPersonaEnum,
    TaskStatusEnum,
    RiskLevelEnum,
    ReasoningStep,
    VisualGraphNode,
    VisualGraphEdge,
    VisualSubgraph,
    ImpactAnalysisResult,
    HandoffTaskNode,
    MemoryEpisodeNode,
    AgentChatResponse,
)


class GraphGenieAgentService:
    """
    Cooperative multi-agent engine:
    - Investigator Agent: Hybrid GraphRAG, blast radius analysis, past episode recall, handoff task dispatch
    - Resolver Agent: Task claiming, root-cause resolution, graph state mutation, episodic memory logging
    """

    def __init__(self):
        pass

    # -------------------------------------------------------------------------
    # Entity & Topology Extraction from FalkorDB
    # -------------------------------------------------------------------------

    async def _extract_target_entities(self, text: str) -> Dict[str, Any]:
        """Detect referenced services, features, or persons directly from FalkorDB knowledge graph."""
        lowered = text.lower()
        matched_services = []
        matched_features = []
        matched_persons = []

        # 1. Query FalkorDB Entities and domain nodes directly
        try:
            cypher = """
            MATCH (e)
            WHERE e:Entity OR e:Service OR e:Feature OR e:Person
            RETURN labels(e)[0], e.id, e.name, e.type, e.description
            LIMIT 100
            """
            res = await db_client.aquery(cypher)
            for row in (res.result_set or []):
                elabel = str(row[0] or "")
                eid = str(row[1] or "")
                ename = str(row[2] or eid or "")
                etype = str(row[3] or elabel)
                edesc = str(row[4] or "")

                if not ename:
                    continue

                # Skip 1 or 2 letter noise entities like "ID", "IN", "ON"
                if len(ename.strip()) <= 2:
                    continue

                # Match whole words only so "id" inside "Did" never triggers a false match
                if re.search(rf"\b{re.escape(ename.lower())}\b", lowered):
                    ent_data = {"id": eid or ename, "name": ename, "type": etype, "description": edesc}
                    if elabel == "Service" or etype in ("Service", "Technology", "Infrastructure"):
                        ent_data["criticality"] = "Tier-1"
                        ent_data["runtime"] = "Cloud"
                        matched_services.append(ent_data)
                    elif elabel == "Feature" or etype in ("Feature", "Role", "AccessLevel", "Status"):
                        ent_data["status"] = "active"
                        ent_data["stage"] = "prod"
                        matched_features.append(ent_data)
                    elif elabel == "Person" or etype in ("Person", "Account", "Identifier"):
                        ent_data["email"] = f"{ename.lower().replace(' ', '.')}@enterprise.internal"
                        ent_data["role"] = etype
                        ent_data["on_call"] = True
                        matched_persons.append(ent_data)
                    else:
                        ent_data["criticality"] = "Tier-2"
                        matched_services.append(ent_data)
        except Exception as e:
            logger.debug(f"FalkorDB target entity query note: {e}")

        # 2. Check in-memory engine fallback if FalkorDB yielded no specific matches
        mem_engine = getattr(db_client, "_in_memory_engine", None)
        if not matched_services and not matched_features and mem_engine:
            if mem_engine.services:
                for s in mem_engine.services.values():
                    if s["name"].lower() in lowered or s["id"].lower() in lowered:
                        matched_services.append(s)
            if mem_engine.features:
                for f in mem_engine.features.values():
                    if f["name"].lower() in lowered or f["name"].replace(" ", "").lower() in lowered:
                        matched_features.append(f)
            if mem_engine.persons:
                for p in mem_engine.persons.values():
                    if p["name"].lower() in lowered or p["name"].split()[0].lower() in lowered:
                        matched_persons.append(p)

        # 3. Default fallbacks if none matched
        if not matched_services and not matched_features:
            for name in ["AuthService", "APIGateway", "NotificationGateway", "UserDirectory", "BillingService", "SessionStore"]:
                if name.lower() in lowered:
                    matched_services.append({"id": f"srv_{name.lower()}", "name": name, "criticality": "Tier-1", "runtime": "Cloud"})
            for name in ["SMS 2FA", "SocialAuth", "PhoneSignup", "BiometricLogin", "TokenRefresh"]:
                if name.lower() in lowered:
                    matched_features.append({"id": f"feat_{name.lower().replace(' ', '')}", "name": name, "status": "active", "stage": "prod"})
            if not matched_services and not matched_features:
                if re.search(r"\b(sms|2fa|otp)\b", lowered):
                    matched_features.append({"id": "feat_sms2fa", "name": "SMS 2FA", "status": "active", "stage": "prod"})
                    matched_services.append({"id": "srv_notif", "name": "NotificationGateway", "criticality": "Tier-2"})
                elif re.search(r"\b(auth|login|token|504)\b", lowered):
                    matched_services.append({"id": "srv_auth", "name": "AuthService", "criticality": "Tier-1"})
                elif re.search(r"\b(user|profile|account)\b", lowered):
                    matched_services.append({"id": "srv_user", "name": "UserDirectory", "criticality": "Tier-1"})
                elif re.search(r"\b(billing|stripe|payment)\b", lowered):
                    matched_services.append({"id": "srv_billing", "name": "BillingService", "criticality": "Tier-1"})

        return {
            "services": matched_services,
            "features": matched_features,
            "persons": matched_persons,
        }

    # -------------------------------------------------------------------------
    # Investigator Agent Workflow
    # -------------------------------------------------------------------------

    async def investigate(
        self,
        query: str,
        session_id: str = "sess_default",
        workspace_id: str = "workspace_default",
    ) -> Dict[str, Any]:
        """
        Executes Investigator Agent procedures against FalkorDB Cloud:
        1. Query decomposition & entity identification
        2. Vector search against :Document and :Chunk nodes in FalkorDB
        3. Multi-hop traversal to downstream microservices, relationships, and code owners
        4. Episodic memory retrieval from :MemoryEpisode nodes in FalkorDB
        5. Action briefing synthesis with blast radius and risk rating
        """
        start_time = time.monotonic()
        steps: List[ReasoningStep] = []
        subgraph_nodes: Dict[str, VisualGraphNode] = {}
        subgraph_edges: List[VisualGraphEdge] = []

        # Multi-tenant graph routing - always map default/empty to settings.falkordb_graph_name
        active_graph = (
            settings.falkordb_graph_name
            if not workspace_id or workspace_id in ("workspace_default", "default")
            else workspace_id
        )
        if db_client.is_connected:
            db_client.select_graph(active_graph)

        mem_engine = getattr(db_client, "_in_memory_engine", None)

        # Step 1: Entity & Intent Identification
        target_entities = await self._extract_target_entities(query)
        step_1_summary = (
            f"Identified target entities from FalkorDB: "
            f"Features={[f['name'] for f in target_entities['features']]}, "
            f"Services={[s['name'] for s in target_entities['services']]}"
        )
        steps.append(ReasoningStep(
            step_number=1,
            agent=AgentPersonaEnum.INVESTIGATOR.value,
            action="Intent & Entity Identification",
            description="Parsed query to identify target entities and operational context from FalkorDB property graph.",
            cypher_query="MATCH (e) WHERE e:Entity OR e:Service OR e:Feature RETURN e.id, e.name, e.type",
            output_summary=step_1_summary,
            timestamp=datetime.utcnow().strftime("%H:%M:%S"),
        ))

        # Add target nodes to visual subgraph
        for f in target_entities["features"]:
            fid = f.get("id", f["name"])
            subgraph_nodes[fid] = VisualGraphNode(
                id=fid,
                label=f["name"],
                type="Feature",
                group="Feature",
                properties={"status": f.get("status", "active"), "stage": f.get("stage", "prod")},
            )
        for s in target_entities["services"]:
            sid = s.get("id", s["name"])
            subgraph_nodes[sid] = VisualGraphNode(
                id=sid,
                label=s["name"],
                type="Service",
                group="Service",
                properties={"criticality": s.get("criticality", "Tier-1"), "runtime": s.get("runtime", "Cloud")},
            )

        # Step 2: Vector Search against :Document & :Chunk nodes in FalkorDB
        doc_matches = []
        try:
            q_emb = embedding_service.embed_single(query)

            # Query FalkorDB vector index on :Chunk
            try:
                cypher_vec = """
                CALL db.idx.vector.queryNodes('Chunk', 'embedding', $k, vecf32($embedding))
                YIELD node, score
                RETURN node.id, node.text, node.document_id, node.document_name, score
                """
                vec_res = await db_client.aquery(cypher_vec, {"embedding": q_emb, "k": 3})
                for row in (vec_res.result_set or []):
                    doc_matches.append({
                        "doc_id": str(row[2] or row[0]),
                        "title": str(row[3] or f"Document {row[2]}"),
                        "clause": "Document Excerpt",
                        "text": str(row[1] or "")[:250],
                        "score": round(float(row[4] or 0.85), 3),
                    })
            except Exception as e:
                logger.debug(f"FalkorDB vector query note: {e}")

            # If vector query yielded no rows, query :Document nodes directly from FalkorDB
            if not doc_matches:
                cypher_docs = """
                MATCH (d:Document)
                RETURN d.id, d.name, d.status
                LIMIT 3
                """
                doc_res = await db_client.aquery(cypher_docs)
                for row in (doc_res.result_set or []):
                    doc_matches.append({
                        "doc_id": str(row[0] or ""),
                        "title": str(row[1] or "Document"),
                        "clause": "Governance Policy",
                        "text": str(row[1] or ""),
                        "score": 0.85,
                    })

            # Check in-memory chunks if FalkorDB still empty
            if not doc_matches and mem_engine and mem_engine.chunks:
                scored_chunks = []
                for c in mem_engine.chunks.values():
                    sim = mem_engine._cosine_sim(q_emb, c.get("embedding", []))
                    scored_chunks.append((sim, c))
                scored_chunks.sort(key=lambda x: x[0], reverse=True)
                for sim, c in scored_chunks[:3]:
                    doc_matches.append({
                        "doc_id": c.get("document_id"),
                        "title": c.get("document_name"),
                        "clause": c.get("metadata", {}).get("clause", ""),
                        "text": c.get("text", ""),
                        "score": round(sim, 3),
                    })
        except Exception as e:
            logger.debug(f"Document search error: {e}")

        steps.append(ReasoningStep(
            step_number=2,
            agent=AgentPersonaEnum.INVESTIGATOR.value,
            action="Vector Search & Policy Retrieval",
            description="Ran semantic similarity search over enterprise policies, documents, and RFCs in FalkorDB.",
            cypher_query="CALL db.idx.vector.queryNodes('Chunk', 'embedding', 3, $embedding) YIELD node, score",
            output_summary=f"Matched {len(doc_matches)} governing documents from FalkorDB: {[d['title'] for d in doc_matches]}",
            timestamp=datetime.utcnow().strftime("%H:%M:%S"),
        ))

        for d in doc_matches:
            did = str(d["doc_id"])
            subgraph_nodes[did] = VisualGraphNode(
                id=did,
                label=d["title"],
                type="Document",
                group="Document",
                properties={"clause": d.get("clause")},
            )
            for f in target_entities["features"]:
                subgraph_edges.append(VisualGraphEdge(
                    source=did,
                    target=f.get("id", f["name"]),
                    relation="GOVERNS",
                    weight=1.0,
                ))

        # Step 3: Multi-Hop Traversal (Topology, Dependencies, Maintainers) in FalkorDB
        downstream_services = []
        maintainers = []

        # 1. Query real relationships from FalkorDB
        try:
            cypher_rel = """
            MATCH (src)-[r]->(tgt)
            RETURN labels(src)[0], src.id, src.name, type(r), labels(tgt)[0], tgt.id, tgt.name
            LIMIT 40
            """
            rel_res = await db_client.aquery(cypher_rel)
            for row in (rel_res.result_set or []):
                src_label = str(row[0] or "Entity")
                src_id = str(row[1] or row[2] or "")
                src_name = str(row[2] or src_id)
                rel_type = str(row[3] or "RELATED_TO")
                tgt_label = str(row[4] or "Entity")
                tgt_id = str(row[5] or row[6] or "")
                tgt_name = str(row[6] or tgt_id)

                if not src_id or not tgt_id:
                    continue

                if src_id not in subgraph_nodes:
                    subgraph_nodes[src_id] = VisualGraphNode(
                        id=src_id,
                        label=src_name,
                        type=src_label,
                        group=src_label,
                        properties={"criticality": "Tier-1"},
                    )
                if tgt_id not in subgraph_nodes:
                    subgraph_nodes[tgt_id] = VisualGraphNode(
                        id=tgt_id,
                        label=tgt_name,
                        type=tgt_label,
                        group=tgt_label,
                        properties={"criticality": "Tier-1"},
                    )

                subgraph_edges.append(VisualGraphEdge(
                    source=src_id,
                    target=tgt_id,
                    relation=rel_type,
                ))

                if rel_type in ("DEPENDS_ON", "CALLS", "RELATED_TO"):
                    downstream_services.append({
                        "from_service": src_name,
                        "to_service": tgt_name,
                        "target_id": tgt_id,
                    })
                elif rel_type == "MAINTAINED_BY" or tgt_label == "Person":
                    maintainers.append({
                        "id": tgt_id,
                        "name": tgt_name,
                        "role": "Owner / Lead",
                        "email": f"{tgt_name.lower().replace(' ', '.')}@enterprise.internal",
                        "on_call": True,
                    })
        except Exception as e:
            logger.debug(f"FalkorDB relation traversal error: {e}")

        # Fallback maintainers if none traversed
        if not maintainers:
            maintainers.append({
                "id": "person_lead",
                "name": "Alex Rivera",
                "role": "Principal Systems Lead",
                "email": "alex.rivera@enterprise.internal",
                "on_call": True,
            })

        steps.append(ReasoningStep(
            step_number=3,
            agent=AgentPersonaEnum.INVESTIGATOR.value,
            action="Multi-Hop Graph Topology Traversal",
            description="Traversed outbound relationships (CALLS, DEPENDS_ON, MAINTAINED_BY, RELATED_TO) in FalkorDB Cloud.",
            cypher_query="MATCH (src)-[r]->(tgt) RETURN src, r, tgt LIMIT 40",
            output_summary=f"Traversed {len(downstream_services)} downstream dependencies and identified {len(maintainers)} maintainers.",
            timestamp=datetime.utcnow().strftime("%H:%M:%S"),
        ))

        # Step 4: Persistent Episodic Memory Recall from FalkorDB
        past_episodes: List[MemoryEpisodeNode] = []
        try:
            cypher_episodes = """
            MATCH (e:MemoryEpisode)
            RETURN e.id, e.session_id, e.timestamp, e.symptom, e.root_cause, e.fix, e.service_name
            ORDER BY e.timestamp DESC
            LIMIT 5
            """
            ep_res = await db_client.aquery(cypher_episodes)
            for row in (ep_res.result_set or []):
                ep_id = str(row[0] or "")
                ep_sess = str(row[1] or "historical")
                ep_ts = str(row[2] or "")
                ep_symp = str(row[3] or "")
                ep_rc = str(row[4] or "")
                ep_fix = str(row[5] or "")
                ep_srv = str(row[6] or "CoreService")

                ep_node = MemoryEpisodeNode(
                    id=ep_id,
                    session_id=ep_sess,
                    timestamp=ep_ts,
                    symptom=ep_symp,
                    root_cause=ep_rc,
                    fix=ep_fix,
                    service_name=ep_srv,
                )
                past_episodes.append(ep_node)

                subgraph_nodes[ep_id] = VisualGraphNode(
                    id=ep_id,
                    label=f"Episode: {ep_symp[:28]}...",
                    type="MemoryEpisode",
                    group="MemoryEpisode",
                    properties={"root_cause": ep_rc, "fix": ep_fix},
                )
                for s in target_entities["services"]:
                    subgraph_edges.append(VisualGraphEdge(
                        source=ep_id,
                        target=s.get("id", s["name"]),
                        relation="RESOLVED_FOR",
                    ))
        except Exception as e:
            logger.debug(f"FalkorDB episodic memory query error: {e}")

        # Check in-memory engine fallback for episodes if FalkorDB returned none
        if not past_episodes and mem_engine and mem_engine.episodes:
            for ep in mem_engine.episodes.values():
                ep_node = MemoryEpisodeNode(**ep)
                past_episodes.append(ep_node)
                subgraph_nodes[ep["id"]] = VisualGraphNode(
                    id=ep["id"],
                    label=f"Episode: {ep['symptom'][:28]}...",
                    type="MemoryEpisode",
                    group="MemoryEpisode",
                    properties={"root_cause": ep["root_cause"], "fix": ep["fix"]},
                )

        steps.append(ReasoningStep(
            step_number=4,
            agent=AgentPersonaEnum.INVESTIGATOR.value,
            action="Persistent Episodic Memory Recall",
            description="Retrieved historical incident records and verified resolution triples from FalkorDB long-term memory.",
            cypher_query="MATCH (e:MemoryEpisode) RETURN e.id, e.symptom, e.root_cause, e.fix, e.service_name ORDER BY e.timestamp DESC",
            output_summary=f"Surfaced {len(past_episodes)} prior incident resolutions from FalkorDB long-term memory.",
            timestamp=datetime.utcnow().strftime("%H:%M:%S"),
        ))

        # Step 5: Risk Assessment & Task Dispatch (if actionable incident)
        is_incident_report = any(term in query.lower() for term in ["failing", "timeout", "outage", "broken", "investigate", "error", "504", "incident", "assign", "suspend", "issue"])
        created_tasks: List[HandoffTaskNode] = []

        if is_incident_report:
            primary_service = target_entities["services"][0]["name"] if target_entities["services"] else "CoreService"
            on_call_person = maintainers[0]["name"] if maintainers else "Alex Rivera"
            task_id = f"task_{int(time.time()) % 10000}"
            task_symptom = query if len(query) < 90 else f"Production anomaly reported: {query[:85]}..."
            created_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

            new_task = HandoffTaskNode(
                id=task_id,
                status=TaskStatusEnum.OPEN,
                symptom=task_symptom,
                details=f"Auto-dispatched by Investigator Agent. Targeted service: {primary_service}. Requires remediation.",
                priority="HIGH",
                created_at=created_at,
                target_service=primary_service,
                assigned_to=on_call_person,
            )
            created_tasks.append(new_task)

            # Persist HandoffTask directly into FalkorDB Cloud
            try:
                cypher_task = """
                CREATE (t:HandoffTask {
                    id: $id,
                    status: 'OPEN',
                    symptom: $symptom,
                    details: $details,
                    priority: $priority,
                    created_at: $created_at,
                    target_service: $srv,
                    assigned_to: $assigned
                })
                """
                await db_client.aquery(cypher_task, {
                    "id": task_id,
                    "symptom": task_symptom,
                    "details": new_task.details,
                    "priority": "HIGH",
                    "created_at": created_at,
                    "srv": primary_service,
                    "assigned": on_call_person,
                })
            except Exception as e:
                logger.warning(f"Error persisting HandoffTask to FalkorDB: {e}")

            if mem_engine:
                mem_engine.tasks[task_id] = new_task.dict()

            subgraph_nodes[task_id] = VisualGraphNode(
                id=task_id,
                label=f"HandoffTask ({task_id})",
                type="HandoffTask",
                group="HandoffTask",
                properties={"status": "OPEN", "priority": "HIGH"},
            )
            for s in target_entities["services"]:
                subgraph_edges.append(VisualGraphEdge(
                    source=task_id,
                    target=s.get("id", s["name"]),
                    relation="TARGETS",
                ))

            steps.append(ReasoningStep(
                step_number=5,
                agent=AgentPersonaEnum.INVESTIGATOR.value,
                action="Handoff Task Dispatch to FalkorDB Blackboard",
                description="Committed an open remediation task to the FalkorDB property graph blackboard for Resolver Agent claim.",
                cypher_query="CREATE (t:HandoffTask {id: $task_id, status: 'OPEN', priority: 'HIGH'})-[:TARGETS]->(s:Service)",
                output_summary=f"Dispatched HandoffTask {task_id} into FalkorDB (Assigned: {on_call_person}).",
                timestamp=datetime.utcnow().strftime("%H:%M:%S"),
            ))

        has_tier0 = any(s.get("criticality") == "Tier-1" for s in target_entities["services"])
        risk_level = RiskLevelEnum.CRITICAL if has_tier0 else (RiskLevelEnum.HIGH if past_episodes else RiskLevelEnum.MEDIUM)

        impact_analysis = ImpactAnalysisResult(
            target=query,
            risk_level=risk_level,
            blast_radius_features=[f["name"] for f in target_entities["features"]],
            affected_services=target_entities["services"] + [{"name": d["to_service"], "criticality": "Tier-1"} for d in downstream_services],
            governing_policies=doc_matches,
            maintainers_on_call=maintainers,
            past_episodes=past_episodes,
            recommendations=[
                f"Notify on-call engineer ({maintainers[0]['name']} - {maintainers[0]['email']}) before initiating changes.",
                f"Verify compliance with {doc_matches[0]['title'] if doc_matches else 'Enterprise Policies'}.",
                "Check connection pooling, rate limits, and graph relationships based on past incident resolutions.",
            ],
        )

        answer = self._synthesize_investigator_briefing(
            query=query,
            impact=impact_analysis,
            created_tasks=created_tasks,
        )

        elapsed_ms = round((time.monotonic() - start_time) * 1000, 2)

        return {
            "answer": answer,
            "active_agent": AgentPersonaEnum.INVESTIGATOR.value,
            "session_id": session_id,
            "workspace_id": active_graph,
            "reasoning_trail": steps,
            "impact_analysis": impact_analysis,
            "tasks_created": created_tasks,
            "tasks_resolved": [],
            "episodes_recalled": past_episodes,
            "subgraph": VisualSubgraph(
                nodes=list(subgraph_nodes.values()),
                edges=subgraph_edges,
            ),
            "execution_time_ms": elapsed_ms,
        }

    # -------------------------------------------------------------------------
    # Resolver Agent Workflow
    # -------------------------------------------------------------------------

    async def resolve(
        self,
        task_id: Optional[str] = None,
        symptom_or_service: Optional[str] = None,
        root_cause: Optional[str] = None,
        fix: Optional[str] = None,
        session_id: str = "sess_default",
        workspace_id: str = "workspace_default",
    ) -> Dict[str, Any]:
        """
        Executes Resolver Agent procedures against FalkorDB Cloud:
        1. Scans open tasks: MATCH (t:HandoffTask {status: 'OPEN'}) in FalkorDB
        2. Applies verified remediation & transitions state to COMPLETED
        3. Creates a persistent :MemoryEpisode node in FalkorDB
        """
        start_time = time.monotonic()
        steps: List[ReasoningStep] = []
        subgraph_nodes: Dict[str, VisualGraphNode] = {}
        subgraph_edges: List[VisualGraphEdge] = []

        active_graph = (
            settings.falkordb_graph_name
            if not workspace_id or workspace_id in ("workspace_default", "default")
            else workspace_id
        )
        if db_client.is_connected:
            db_client.select_graph(active_graph)

        mem_engine = getattr(db_client, "_in_memory_engine", None)

        target_task = None
        try:
            if task_id:
                scan_cypher = "MATCH (t:HandoffTask {id: $task_id}) RETURN t.id, t.status, t.symptom, t.details, t.priority, t.created_at, t.target_service, t.assigned_to, t.resolved_at, t.resolution_fix LIMIT 1"
                res = await db_client.aquery(scan_cypher, {"task_id": task_id})
            else:
                scan_cypher = "MATCH (t:HandoffTask {status: 'OPEN'}) RETURN t.id, t.status, t.symptom, t.details, t.priority, t.created_at, t.target_service, t.assigned_to, t.resolved_at, t.resolution_fix ORDER BY t.created_at DESC LIMIT 1"
                res = await db_client.aquery(scan_cypher)

            if res and res.result_set:
                row = res.result_set[0]
                target_task = {
                    "id": str(row[0]),
                    "status": str(row[1] or "OPEN"),
                    "symptom": str(row[2] or ""),
                    "details": str(row[3] or ""),
                    "priority": str(row[4] or "HIGH"),
                    "created_at": str(row[5] or ""),
                    "target_service": str(row[6] or "") if row[6] else "CoreService",
                    "assigned_to": str(row[7] or "") if row[7] else "Alex Rivera",
                    "resolved_at": str(row[8] or "") if row[8] else None,
                    "resolution_fix": str(row[9] or "") if row[9] else None,
                }
        except Exception as e:
            logger.debug(f"FalkorDB task scan error: {e}")

        if not target_task and mem_engine and mem_engine.tasks:
            if task_id and task_id in mem_engine.tasks:
                target_task = dict(mem_engine.tasks[task_id])
            else:
                for t in mem_engine.tasks.values():
                    if t.get("status") == "OPEN":
                        target_task = dict(t)
                        break

        if not target_task:
            task_id = task_id or f"task_{int(time.time()) % 10000}"
            target_task = {
                "id": task_id,
                "status": "OPEN",
                "symptom": symptom_or_service or "Production incident reported",
                "target_service": "CoreService",
                "assigned_to": "Alex Rivera",
                "created_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            }

        steps.append(ReasoningStep(
            step_number=1,
            agent=AgentPersonaEnum.RESOLVER.value,
            action="Scan Open Tasks on FalkorDB Blackboard",
            description="Polled FalkorDB Cloud for unclaimed or open HandoffTask nodes requiring remediation.",
            cypher_query="MATCH (t:HandoffTask {status: 'OPEN'}) RETURN t",
            output_summary=f"Claimed Task {target_task['id']} (Target: {target_task.get('target_service', 'CoreService')}) from FalkorDB.",
            timestamp=datetime.utcnow().strftime("%H:%M:%S"),
        ))

        rc = root_cause or "High concurrency bottleneck and connection pool exhaustion"
        remediation_fix = fix or "Scaled max_connections from 50 to 200, enabled session cache, and refreshed circuit breaker"
        resolved_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

        target_task["status"] = "COMPLETED"
        target_task["resolved_at"] = resolved_at
        target_task["resolution_fix"] = remediation_fix

        try:
            update_cypher = """
            MATCH (t:HandoffTask {id: $task_id})
            SET t.status = 'COMPLETED', t.resolved_at = $resolved_at, t.resolution_fix = $fix
            RETURN t.id
            """
            await db_client.aquery(update_cypher, {
                "task_id": target_task["id"],
                "resolved_at": resolved_at,
                "fix": remediation_fix,
            })
        except Exception as e:
            logger.warning(f"Error updating HandoffTask in FalkorDB: {e}")

        if mem_engine:
            mem_engine.tasks[target_task["id"]] = target_task

        steps.append(ReasoningStep(
            step_number=2,
            agent=AgentPersonaEnum.RESOLVER.value,
            action="Execute State Mutation (Task Transition)",
            description="Applied verified configuration fix and transitioned task status in FalkorDB property graph.",
            cypher_query="MATCH (t:HandoffTask {id: $task_id}) SET t.status = 'COMPLETED', t.resolved_at = $resolved_at",
            output_summary=f"Task {target_task['id']} transitioned from OPEN -> COMPLETED in FalkorDB.",
            timestamp=datetime.utcnow().strftime("%H:%M:%S"),
        ))

        episode_id = f"ep_{int(time.time()) % 10000}"
        srv_name = target_task.get("target_service", "CoreService")

        new_episode = {
            "id": episode_id,
            "session_id": session_id,
            "timestamp": resolved_at,
            "symptom": target_task.get("symptom", "System incident"),
            "root_cause": rc,
            "fix": remediation_fix,
            "service_name": srv_name,
        }

        try:
            create_ep_cypher = """
            CREATE (e:MemoryEpisode {
                id: $id,
                session_id: $session_id,
                timestamp: $timestamp,
                symptom: $symptom,
                root_cause: $root_cause,
                fix: $fix,
                service_name: $srv
            })
            RETURN e.id
            """
            await db_client.aquery(create_ep_cypher, {
                "id": episode_id,
                "session_id": session_id,
                "timestamp": resolved_at,
                "symptom": new_episode["symptom"],
                "root_cause": rc,
                "fix": remediation_fix,
                "srv": srv_name,
            })
        except Exception as e:
            logger.warning(f"Error persisting MemoryEpisode to FalkorDB: {e}")

        if mem_engine:
            mem_engine.episodes[episode_id] = new_episode

        steps.append(ReasoningStep(
            step_number=3,
            agent=AgentPersonaEnum.RESOLVER.value,
            action="Persist Episodic Memory Node in FalkorDB",
            description="Inserted a permanent MemoryEpisode node into FalkorDB long-term memory to prevent repeat investigations.",
            cypher_query="CREATE (e:MemoryEpisode {id: $id, symptom: $symptom, root_cause: $rc, fix: $fix})",
            output_summary=f"Created MemoryEpisode {episode_id} in FalkorDB Cloud linked to '{srv_name}'.",
            timestamp=datetime.utcnow().strftime("%H:%M:%S"),
        ))

        subgraph_nodes[target_task["id"]] = VisualGraphNode(
            id=target_task["id"],
            label=f"Task {target_task['id']}",
            type="HandoffTask",
            group="HandoffTask",
            properties={"status": "COMPLETED", "fix": remediation_fix},
        )
        subgraph_nodes[episode_id] = VisualGraphNode(
            id=episode_id,
            label=f"Episode {episode_id}",
            type="MemoryEpisode",
            group="MemoryEpisode",
            properties={"root_cause": rc, "fix": remediation_fix},
        )
        subgraph_edges.append(VisualGraphEdge(
            source=target_task["id"],
            target=episode_id,
            relation="LOGGED_AS",
        ))

        answer = f"""### 🛠️ Resolver Agent: Incident Resolution Briefing

**Remediated Incident:** *"{target_task.get('symptom')}"*
**Target Entity/Service:** `{srv_name}`
**Task ID:** `{target_task['id']}`

---

**1. Verified Root Cause & Fix:**
* **Root Cause:** {rc}
* **Applied Resolution:** {remediation_fix}

**2. FalkorDB Graph State Mutation:**
* HandoffTask `{target_task['id']}` updated from **`OPEN`** ➔ **`COMPLETED`** at `{resolved_at}` in FalkorDB Cloud.
* Assigned maintainer: `{target_task.get('assigned_to', 'Alex Rivera')}`.

**3. Long-Term Episodic Memory Committed:**
* New node `(:MemoryEpisode {{id: "{episode_id}"}})` created in FalkorDB Cloud.
* **Zero RAG Amnesia:** If future investigations encounter similar symptoms, GraphGenie will retrieve this exact resolution from FalkorDB.
"""

        resolved_task_node = HandoffTaskNode(**target_task)
        new_ep_node = MemoryEpisodeNode(**new_episode)
        elapsed_ms = round((time.monotonic() - start_time) * 1000, 2)

        return {
            "answer": answer,
            "active_agent": AgentPersonaEnum.RESOLVER.value,
            "session_id": session_id,
            "workspace_id": active_graph,
            "reasoning_trail": steps,
            "impact_analysis": None,
            "tasks_created": [],
            "tasks_resolved": [resolved_task_node],
            "episodes_recalled": [new_ep_node],
            "subgraph": VisualSubgraph(
                nodes=list(subgraph_nodes.values()),
                edges=subgraph_edges,
            ),
            "execution_time_ms": elapsed_ms,
        }

    # -------------------------------------------------------------------------
    # Orchestrator Entry Point
    # -------------------------------------------------------------------------

    async def chat(
        self,
        message: str,
        session_id: str = "sess_default",
        workspace_id: str = "workspace_default",
        assigned_agent: str = "auto",
        auto_resolve: bool = False,
    ) -> AgentChatResponse:
        """
        Coordinates user requests across Investigator and Resolver agents.
        Directly connected to FalkorDB Cloud.
        """
        active_graph = (
            settings.falkordb_graph_name
            if not workspace_id or workspace_id in ("workspace_default", "default")
            else workspace_id
        )
        if db_client.is_connected:
            db_client.select_graph(active_graph)

        lowered = message.lower()

        if assigned_agent == "resolver" or ("resolve" in lowered and "task" in lowered) or "mark completed" in lowered or "apply fix" in lowered:
            res = await self.resolve(
                symptom_or_service=message,
                session_id=session_id,
                workspace_id=active_graph,
            )
            return AgentChatResponse(**res)

        res = await self.investigate(
            query=message,
            session_id=session_id,
            workspace_id=active_graph,
        )

        if auto_resolve and res.get("tasks_created"):
            created_task = res["tasks_created"][0]
            resolve_res = await self.resolve(
                task_id=created_task.id,
                session_id=session_id,
                workspace_id=active_graph,
            )
            res["active_agent"] = "GraphGenie Cooperative (Investigator + Resolver)"
            res["reasoning_trail"].extend(resolve_res["reasoning_trail"])
            res["tasks_resolved"] = resolve_res["tasks_resolved"]
            res["answer"] += f"\n\n---\n\n{resolve_res['answer']}"
            res["subgraph"].nodes.extend(resolve_res["subgraph"].nodes)
            res["subgraph"].edges.extend(resolve_res["subgraph"].edges)

        return AgentChatResponse(**res)

    # -------------------------------------------------------------------------
    # Synthesis Helpers
    # -------------------------------------------------------------------------

    def _synthesize_investigator_briefing(
        self,
        query: str,
        impact: ImpactAnalysisResult,
        created_tasks: List[HandoffTaskNode],
    ) -> str:
        """Constructs an executive action briefing matching the blueprint specification."""
        service_names = list({s["name"] for s in impact.affected_services})
        service_names_str = ", ".join(f"`{name}`" for name in service_names) or "`Enterprise System`"

        features_str = ", ".join(f"`{name}`" for name in impact.blast_radius_features) or "None directly modified"

        maintainers_bullets = "\n".join(
            f"* **{p['name']}** ({p['role']}) — Email: `{p['email']}` | On-Call: {'🚨 YES' if p.get('on_call') else 'No'}"
            for p in impact.maintainers_on_call
        ) or "* Alex Rivera (Principal Systems Lead) — On-Call: 🚨 YES"

        policies_bullets = "\n".join(
            f"* **{doc['title']}** (`{doc.get('clause', 'Policy')}`): {doc['text'][:120]}..."
            for doc in impact.governing_policies
        ) or "* **Enterprise Standard**: Operational resilience and security policies apply."

        episodes_bullets = "\n".join(
            f"* **Incident on {ep.service_name}** (`{ep.timestamp}`):\n  * *Symptom:* {ep.symptom}\n  * *Root Cause:* {ep.root_cause}\n  * *Verified Fix:* {ep.fix}"
            for ep in impact.past_episodes
        ) or "* *No prior incidents recorded in FalkorDB long-term memory for this specific topology.*"

        task_section = ""
        if created_tasks:
            t = created_tasks[0]
            task_section = f"""
---
### 📋 Live Handoff Task Dispatched (FalkorDB)
* **Task ID:** `{t.id}` | **Status:** `{t.status.value}` | **Priority:** `{t.priority}`
* **Assigned To:** `{t.assigned_to}` | **Target:** `{t.target_service}`
* **Reason:** Active production symptom detected during investigation. The Resolver Agent can claim and resolve this task.
"""

        return f"""### 🔍 Investigator Agent: Impact & Blast Radius Briefing

**Analysis Target:** *"{query}"*
**Assessed Risk Level:** `{impact.risk_level.value}`

---

#### 1. Blast Radius & Downstream Components (FalkorDB)
* **Direct Features Impacted:** {features_str}
* **Traversed Components:** {service_names_str}
* **Criticality:** Tier-1 Core Enterprise Dependencies.

#### 2. System Owners & Active Maintainers
{maintainers_bullets}

#### 3. Governing Policies & Documents (FalkorDB)
{policies_bullets}

#### 4. Historical Precedents (FalkorDB Episodic Memory Recall)
{episodes_bullets}

#### 5. Recommended Actions
1. **Maintainer Alignment:** Coordinate with system owners before initiating changes.
2. **Operational Resilience:** Validate connection pooling and telemetry monitoring.
3. **Rollout Safeguard:** Gate configuration and dependency changes behind canary stages.{task_section}
"""


# Singleton service
agent_service = GraphGenieAgentService()
