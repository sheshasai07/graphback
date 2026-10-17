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
    mem_engine = getattr(db_client, "_in_memory_engine", None)
    tasks = []
    if mem_engine and mem_engine.tasks:
        for t in mem_engine.tasks.values():
            if status and t.get("status") != status:
                continue
            tasks.append(HandoffTaskNode(**t))
    tasks.sort(key=lambda x: x.created_at, reverse=True)
    return tasks


@router.post("/tasks", response_model=HandoffTaskNode)
async def create_handoff_task(request: CreateTaskRequest):
    """
    Insert a new HandoffTask node into the graph targeted at a microservice.
    """
    import time
    task_id = f"task_{int(time.time()) % 10000}"
    new_task = HandoffTaskNode(
        id=task_id,
        status="OPEN",
        symptom=request.symptom,
        details=request.details or "",
        priority=request.priority or "HIGH",
        created_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        target_service=request.target_service,
        assigned_to=request.assigned_to or "Sam Chen",
    )
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
    List all persistent MemoryEpisode records (incident timeline & root causes).
    """
    mem_engine = getattr(db_client, "_in_memory_engine", None)
    episodes = []
    if mem_engine and mem_engine.episodes:
        for ep in mem_engine.episodes.values():
            if service and ep.get("service_name", "").lower() != service.lower():
                continue
            episodes.append(MemoryEpisodeNode(**ep))
    episodes.sort(key=lambda x: x.timestamp, reverse=True)
    return episodes


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Tenant Workspaces
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/workspaces", response_model=List[WorkspaceInfo])
async def list_workspaces():
    """
    List available multi-tenant graph workspaces.
    """
    mem_engine = getattr(db_client, "_in_memory_engine", None)
    current = getattr(mem_engine, "current_workspace", "workspace_default")
    workspaces = []
    if mem_engine and mem_engine.workspaces:
        for wid, w in mem_engine.workspaces.items():
            workspaces.append(WorkspaceInfo(
                id=wid,
                name=w["name"],
                description=w["description"],
                is_active=(wid == current),
                node_count=len(mem_engine.services) + len(mem_engine.features) + len(mem_engine.persons) + len(mem_engine.documents),
                edge_count=len(mem_engine.relations),
            ))
    return workspaces


@router.post("/workspaces/{workspace_id}/select")
async def select_workspace(workspace_id: str):
    """
    Switch active tenant workspace / graph partition.
    """
    db_client.select_graph(workspace_id)
    return {"message": f"Active workspace switched to '{workspace_id}'", "workspace_id": workspace_id}


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
