"""
GraphGenie - Multi-Agent & Memory Schemas
Pydantic models for Investigator Agent, Resolver Agent, Handoff Tasks,
Episodic Memory, and Graph Reasoning Trail.
"""

from typing import Any, Dict, List, Optional
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class TaskStatusEnum(str, Enum):
    OPEN = "OPEN"
    CLAIMED = "CLAIMED"
    COMPLETED = "COMPLETED"


class AgentPersonaEnum(str, Enum):
    INVESTIGATOR = "Investigator Agent"
    RESOLVER = "Resolver Agent"
    ORCHESTRATOR = "GraphGenie Orchestrator"


class RiskLevelEnum(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# ─────────────────────────────────────────────────────────────────────────────
# Graph Domain Entities
# ─────────────────────────────────────────────────────────────────────────────

class ServiceNode(BaseModel):
    id: str
    name: str
    runtime: str = "Go"
    criticality: str = "Tier-1"
    description: Optional[str] = ""
    maintainer: Optional[str] = None


class FeatureNode(BaseModel):
    id: str
    name: str
    status: str = "active"
    stage: str = "prod"
    description: Optional[str] = ""


class PersonNode(BaseModel):
    id: str
    name: str
    role: str
    email: str
    on_call: bool = False


class MemoryEpisodeNode(BaseModel):
    id: str
    session_id: str = "default_session"
    timestamp: str
    symptom: str
    root_cause: str
    fix: str
    service_name: str
    service_id: Optional[str] = None


class HandoffTaskNode(BaseModel):
    id: str
    status: TaskStatusEnum = TaskStatusEnum.OPEN
    symptom: str
    details: Optional[str] = ""
    priority: str = "HIGH"
    created_at: str
    target_service: Optional[str] = None
    assigned_to: Optional[str] = None
    resolved_at: Optional[str] = None
    resolution_fix: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Reasoning Trail & Visual Subgraph
# ─────────────────────────────────────────────────────────────────────────────

class ReasoningStep(BaseModel):
    step_number: int
    agent: str
    action: str
    description: str
    cypher_query: Optional[str] = None
    output_summary: Optional[str] = None
    timestamp: str


class VisualGraphNode(BaseModel):
    id: str
    label: str
    type: str
    group: Optional[str] = None
    properties: Optional[Dict[str, Any]] = None


class VisualGraphEdge(BaseModel):
    source: str
    target: str
    relation: str
    weight: Optional[float] = 1.0


class VisualSubgraph(BaseModel):
    nodes: List[VisualGraphNode] = Field(default_factory=list)
    edges: List[VisualGraphEdge] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Impact Analysis
# ─────────────────────────────────────────────────────────────────────────────

class ImpactAnalysisResult(BaseModel):
    target: str
    risk_level: RiskLevelEnum = RiskLevelEnum.MEDIUM
    blast_radius_features: List[str] = Field(default_factory=list)
    affected_services: List[Dict[str, Any]] = Field(default_factory=list)
    governing_policies: List[Dict[str, Any]] = Field(default_factory=list)
    maintainers_on_call: List[Dict[str, Any]] = Field(default_factory=list)
    past_episodes: List[MemoryEpisodeNode] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# API Requests & Responses
# ─────────────────────────────────────────────────────────────────────────────

class AgentChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User instruction or query for GraphGenie multi-agent system")
    session_id: Optional[str] = "sess_default"
    workspace_id: Optional[str] = "workspace_default"
    assigned_agent: Optional[str] = "auto"  # 'auto', 'investigator', 'resolver'
    auto_resolve: Optional[bool] = False    # Whether resolver agent immediately resolves newly created tasks


class AgentChatResponse(BaseModel):
    answer: str
    active_agent: str
    session_id: str
    workspace_id: str
    reasoning_trail: List[ReasoningStep] = Field(default_factory=list)
    impact_analysis: Optional[ImpactAnalysisResult] = None
    tasks_created: List[HandoffTaskNode] = Field(default_factory=list)
    tasks_resolved: List[HandoffTaskNode] = Field(default_factory=list)
    episodes_recalled: List[MemoryEpisodeNode] = Field(default_factory=list)
    subgraph: VisualSubgraph = Field(default_factory=VisualSubgraph)
    execution_time_ms: float


class CreateTaskRequest(BaseModel):
    symptom: str
    target_service: str
    details: Optional[str] = ""
    priority: Optional[str] = "HIGH"
    assigned_to: Optional[str] = None


class ResolveTaskRequest(BaseModel):
    root_cause: str
    fix: str
    session_id: Optional[str] = "sess_default"


class WorkspaceInfo(BaseModel):
    id: str
    name: str
    description: str
    is_active: bool = False
    node_count: int = 0
    edge_count: int = 0
