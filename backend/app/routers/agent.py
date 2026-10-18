"""
GraphGenie - Multi-Agent Router
Exposes REST endpoints for Investigator & Resolver agents,
Handoff Tasks, Episodic Memory, and Multi-Tenant Workspaces.
"""

from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from loguru import logger

from app.models.agent_schemas import (
    AgentChatRequest,
    AgentChatResponse,
    CreateTaskRequest,
    ResolveTaskRequest,
    HandoffTaskNode,
    MemoryEpisodeNode,
    WorkspaceInfo,
)
from app.services.agent_service import agent_service
from app.services.seed_data import seed_graphgenie_database
from app.db.falkordb_client import db_client
from app.config import settings

router = APIRouter(prefix="/api/v1/agent", tags=["Multi-Agent System"])


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Agent Chat & Orchestration
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/chat", response_model=AgentChatResponse)
async def agent_chat(request: AgentChatRequest):
    """
    Execute GraphGenie multi-agent workflow:
    - Automatically routes between Investigator Agent and Resolver Agent
    - Traverses GraphRAG policies and microservice topologies
    - Recalls persistent episodic memory records
    - Dispatches or claims live HandoffTask nodes
    - Returns step-by-step reasoning trail and visual subgraph
    """
    try:
        response = await agent_service.chat(
            message=request.message,
            session_id=request.session_id or "sess_default",
            workspace_id=request.workspace_id or "workspace_default",
            assigned_agent=request.assigned_agent or "auto",
            auto_resolve=request.auto_resolve or False,
        )
        return response
    except Exception as e:
        logger.error(f"Agent chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/investigate")
async def agent_investigate(request: AgentChatRequest):
    """
    Run dedicated Investigator Agent flow:
    - Blast radius analysis on features and downstream services
    - Maintainer and on-call detection
    - Memory recall of past outages
    """
    try:
        return await agent_service.investigate(
            query=request.message,
            session_id=request.session_id or "sess_default",
            workspace_id=request.workspace_id or "workspace_default",
        )
    except Exception as e:
        logger.error(f"Agent investigation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/resolve")
async def agent_resolve(
    task_id: Optional[str] = None,
    symptom: Optional[str] = None,
    root_cause: Optional[str] = None,
    fix: Optional[str] = None,
    session_id: str = "sess_default",
    workspace_id: str = "workspace_default",
):
    """
    Run dedicated Resolver Agent flow:
    - Claims an open HandoffTask
    - Applies state mutation (SET status = 'COMPLETED')
    - Records persistent MemoryEpisode node in FalkorDB
    """
    try:
        return await agent_service.resolve(
            task_id=task_id,
            symptom_or_service=symptom,
            root_cause=root_cause,
            fix=fix,
            session_id=session_id,
            workspace_id=workspace_id,
        )
    except Exception as e:
        logger.error(f"Agent resolution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Handoff Task Management (Graph Blackboard)
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/tasks", response_model=List[HandoffTaskNode])
async def list_handoff_tasks(status: Optional[str] = Query(None, description="Filter by OPEN, CLAIMED, COMPLETED")):
    """
    List all HandoffTask nodes stored on the FalkorDB operational blackboard.
    """
    tasks = []
    try:
        cypher = """
        MATCH (t:HandoffTask)
        RETURN t.id, t.status, t.symptom, t.details, t.priority, t.created_at, t.target_service, t.assigned_to, t.resolved_at, t.resolution_fix
        ORDER BY t.created_at DESC
        """
        result = await db_client.aquery(cypher)
        for row in result.result_set:
            t_status = str(row[1] or "OPEN")
            if status and t_status != status:
                continue
            tasks.append(HandoffTaskNode(
                id=str(row[0] or ""),
                status=t_status,
                symptom=str(row[2] or ""),
                details=str(row[3] or ""),
                priority=str(row[4] or "HIGH"),
                created_at=str(row[5] or ""),
                target_service=str(row[6] or "") if row[6] else None,
                assigned_to=str(row[7] or "") if row[7] else None,
                resolved_at=str(row[8] or "") if row[8] else None,
                resolution_fix=str(row[9] or "") if row[9] else None,
            ))
    except Exception as e:
        logger.warning(f"FalkorDB tasks query error: {e}")

    # Fallback to in-memory engine if FalkorDB returns nothing and in-memory has tasks
    if not tasks:
        mem_engine = getattr(db_client, "_in_memory_engine", None)
        if mem_engine and mem_engine.tasks:
            for t in mem_engine.tasks.values():
                if status and t.get("status") != status:
                    continue
                tasks.append(HandoffTaskNode(**t))

    tasks.sort(key=lambda x: x.created_at or "", reverse=True)
    return tasks


@router.post("/tasks", response_model=HandoffTaskNode)
async def create_handoff_task(request: CreateTaskRequest):
    """
    Insert a new HandoffTask node into FalkorDB targeted at a microservice.
    """
    import time
    task_id = f"task_{int(time.time()) % 10000}"
    created_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    assigned = request.assigned_to or "Sam Chen"
    target_srv = request.target_service or ""

    new_task = HandoffTaskNode(
        id=task_id,
        status="OPEN",
        symptom=request.symptom,
        details=request.details or "",
        priority=request.priority or "HIGH",
        created_at=created_at,
        target_service=target_srv or None,
        assigned_to=assigned,
    )

    try:
        cypher = """
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
        RETURN t.id
        """
        await db_client.aquery(cypher, {
            "id": task_id,
            "symptom": request.symptom,
            "details": request.details or "",
            "priority": request.priority or "HIGH",
            "created_at": created_at,
            "srv": target_srv,
            "assigned": assigned,
        })
        if target_srv:
            await db_client.aquery("""
            MATCH (t:HandoffTask {id: $id}), (s:Entity)
            WHERE toLower(s.name) = toLower($srv)
            MERGE (t)-[:TARGETS]->(s)
            """, {"id": task_id, "srv": target_srv})
    except Exception as e:
        logger.warning(f"Error persisting HandoffTask to FalkorDB: {e}")

    # Mirror to in-memory engine as well
    mem_engine = getattr(db_client, "_in_memory_engine", None)
    if mem_engine:
        mem_engine.tasks[task_id] = new_task.dict()

    return new_task


@router.post("/tasks/{task_id}/resolve")
async def resolve_task_by_id(task_id: str, request: ResolveTaskRequest):
    """
    Resolve a specific HandoffTask by ID and commit a new MemoryEpisode to the graph.
    """
    return await agent_service.resolve(
        task_id=task_id,
        root_cause=request.root_cause,
        fix=request.fix,
        session_id=request.session_id or "sess_default",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Episodic Memory Timeline
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/episodes", response_model=List[MemoryEpisodeNode])
async def list_memory_episodes(service: Optional[str] = Query(None, description="Filter by service name")):
    """
    List all persistent MemoryEpisode records from FalkorDB.
    """
    episodes = []
    try:
        cypher = """
        MATCH (e:MemoryEpisode)
        RETURN e.id, e.session_id, e.timestamp, e.symptom, e.root_cause, e.fix, e.service_name
        ORDER BY e.timestamp DESC
        """
        result = await db_client.aquery(cypher)
        for row in result.result_set:
            srv = str(row[6] or "")
            if service and srv.lower() != service.lower():
                continue
            episodes.append(MemoryEpisodeNode(
                id=str(row[0] or ""),
                session_id=str(row[1] or "historical"),
                timestamp=str(row[2] or ""),
                symptom=str(row[3] or ""),
                root_cause=str(row[4] or ""),
                fix=str(row[5] or ""),
                service_name=srv or "AuthService",
            ))
    except Exception as e:
        logger.warning(f"FalkorDB episodes query error: {e}")

    # Fallback to in-memory if FalkorDB returns nothing and in-memory has episodes
    if not episodes:
        mem_engine = getattr(db_client, "_in_memory_engine", None)
        if mem_engine and mem_engine.episodes:
            for ep in mem_engine.episodes.values():
                if service and ep.get("service_name", "").lower() != service.lower():
                    continue
                episodes.append(MemoryEpisodeNode(**ep))

    episodes.sort(key=lambda x: x.timestamp or "", reverse=True)
    return episodes


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Tenant Workspaces
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/workspaces", response_model=List[WorkspaceInfo])
async def list_workspaces():
    """
    List available multi-tenant graph workspaces and show live FalkorDB stats.
    """
    node_count = 0
    edge_count = 0
    try:
        n_res = await db_client.aquery("MATCH (n) RETURN count(n)")
        if n_res and n_res.result_set:
            node_count = int(n_res.result_set[0][0])
        e_res = await db_client.aquery("MATCH ()-[r]->() RETURN count(r)")
        if e_res and e_res.result_set:
            edge_count = int(e_res.result_set[0][0])
    except Exception as e:
        logger.debug(f"Count query error: {e}")

    return [
        WorkspaceInfo(
            id="workspace_default",
            name=f"FalkorDB Active Graph ({settings.falkordb_graph_name})",
            description=f"Primary enterprise knowledge graph on FalkorDB Cloud ({settings.falkordb_graph_name})",
            is_active=True,
            node_count=node_count,
            edge_count=edge_count,
        ),
        WorkspaceInfo(
            id="workspace_payments",
            name="Payments & Billing",
            description="Payment gateways, Stripe compliance, and transaction services",
            is_active=False,
            node_count=0,
            edge_count=0,
        ),
        WorkspaceInfo(
            id="workspace_identity",
            name="Identity & Security",
            description="IAM, OAuth2 providers, and Zero-Trust policies",
            is_active=False,
            node_count=0,
            edge_count=0,
        ),
    ]


@router.post("/workspaces/{workspace_id}/select")
async def select_workspace(workspace_id: str):
    """
    Switch active tenant workspace / graph partition.
    """
    active_graph = settings.falkordb_graph_name if workspace_id in ("workspace_default", "default", "", None) else workspace_id
    db_client.select_graph(active_graph)
    return {"message": f"Active workspace switched to '{workspace_id}' (Graph: '{active_graph}')", "workspace_id": workspace_id}


# ─────────────────────────────────────────────────────────────────────────────
# Seed / Reset Graph
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/seed")
async def seed_graph():
    """
    Seed/Reset the enterprise domain topology, documents, policies, and memory episodes.
    """
    seed_graphgenie_database(force=True)
    return {"message": "GraphGenie enterprise graph seeded successfully."}
