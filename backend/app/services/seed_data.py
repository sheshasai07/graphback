"""
GraphGenie - Enterprise Knowledge Graph & Memory Seeder
Seeds standard enterprise topology (Features, Services, People, Policies,
Memory Episodes, and Handoff Tasks) into FalkorDB or InMemoryGraphEngine.
"""

from typing import List, Dict, Any
from datetime import datetime
from loguru import logger

from app.db.falkordb_client import db_client
from app.services.embedding_service import embedding_service


# ─────────────────────────────────────────────────────────────────────────────
# Seed Definitions
# ─────────────────────────────────────────────────────────────────────────────

SERVICES_DATA = [
    {
        "id": "srv_auth",
        "name": "AuthService",
        "runtime": "Go / Kubernetes",
        "criticality": "Tier-1",
        "description": "Core authentication, JWT issuance, session validation, and OAuth2 provider.",
        "maintainer": "Alex Rivera",
    },
    {
        "id": "srv_gateway",
        "name": "APIGateway",
        "runtime": "Rust / Envoy",
        "criticality": "Tier-0",
        "description": "Edge routing, global rate limiting, JWT validation, and SSL termination.",
        "maintainer": "Sam Chen",
    },
    {
        "id": "srv_notif",
        "name": "NotificationGateway",
        "runtime": "Node.js / Lambda",
        "criticality": "Tier-2",
        "description": "Outbound SMS/Email delivery via Twilio shortcodes and AWS SES.",
        "maintainer": "Alex Rivera",
    },
    {
        "id": "srv_user",
        "name": "UserDirectory",
        "runtime": "Java / Spring Boot",
        "criticality": "Tier-1",
        "description": "User profile store, RBAC permissions, and credential hash vault.",
        "maintainer": "Elena Rostova",
    },
    {
        "id": "srv_billing",
        "name": "BillingService",
        "runtime": "Python / FastAPI",
        "criticality": "Tier-1",
        "description": "Subscription lifecycle, payment processing, invoice generation, and Stripe gateway.",
        "maintainer": "Maya Patel",
    },
    {
        "id": "srv_session",
        "name": "SessionStore",
        "runtime": "Redis Cluster",
        "criticality": "Tier-1",
        "description": "Sub-millisecond token blacklist, active session cache, and rate-limit counters.",
        "maintainer": "Sam Chen",
    },
]

FEATURES_DATA = [
    {
        "id": "feat_sms2fa",
        "name": "SMS 2FA",
        "status": "active",
        "stage": "prod",
        "description": "Two-factor verification via SMS one-time passcode (OTP).",
        "service": "NotificationGateway",
    },
    {
        "id": "feat_social",
        "name": "SocialAuth",
        "status": "active",
        "stage": "prod",
        "description": "Federated social sign-in with Google, Apple, and GitHub OAuth2.",
        "service": "AuthService",
    },
    {
        "id": "feat_signup",
        "name": "PhoneSignup",
        "status": "beta",
        "stage": "staging",
        "description": "Passwordless account creation directly using mobile phone verification.",
        "service": "NotificationGateway",
    },
    {
        "id": "feat_bio",
        "name": "BiometricLogin",
        "status": "planned",
        "stage": "design",
        "description": "FIDO2 / WebAuthn passkey authentication eliminating SMS fallback.",
        "service": "AuthService",
    },
    {
        "id": "feat_refresh",
        "name": "TokenRefresh",
        "status": "active",
        "stage": "prod",
        "description": "OIDC JWT refresh token rotation with single-use replay detection.",
        "service": "AuthService",
    },
]

PERSONS_DATA = [
    {
        "id": "usr_alex",
        "name": "Alex Rivera",
        "role": "Principal Tech Lead",
        "email": "alex.rivera@corp.internal",
        "on_call": True,
    },
    {
        "id": "usr_elena",
        "name": "Elena Rostova",
        "role": "Compliance & Security Officer",
        "email": "elena.rostova@corp.internal",
        "on_call": False,
    },
    {
        "id": "usr_sam",
        "name": "Sam Chen",
        "role": "Staff DevOps / SRE",
        "email": "sam.chen@corp.internal",
        "on_call": True,
    },
    {
        "id": "usr_maya",
        "name": "Maya Patel",
        "role": "Lead Security Architect",
        "email": "maya.patel@corp.internal",
        "on_call": False,
    },
]

# Predefined test documents removed - documents are only created via user uploads
DOCUMENTS_DATA = []


EPISODES_DATA = [
    {
        "id": "ep_101",
        "session_id": "sess_incident_2026_09",
        "timestamp": "2026-09-14 14:22:00",
        "symptom": "504 Gateway Timeout on /auth/v1/token under peak load",
        "root_cause": "Connection pool exhaustion under 4000 req/s concurrency",
        "fix": "Scaled max_connections from 50 to 200 and enabled Redis session cache",
        "service_name": "AuthService",
    },
    {
        "id": "ep_102",
        "session_id": "sess_incident_2026_09",
        "timestamp": "2026-09-28 09:45:00",
        "symptom": "SMS OTP delivery latency exceeding 180s during flash sale",
        "root_cause": "Downstream Twilio carrier throttling on single toll-free sender ID",
        "fix": "Implemented round-robin shortcode pool with SQS exponential backoff queue",
        "service_name": "NotificationGateway",
    },
    {
        "id": "ep_103",
        "session_id": "sess_incident_2026_10",
        "timestamp": "2026-10-05 18:10:00",
        "symptom": "UserDirectory read-replica lag causing stale credential checks",
        "root_cause": "Replication thread stalled during cross-region DB failover drill",
        "fix": "Increased replication buffer size and tuned failover heartbeat threshold",
        "service_name": "UserDirectory",
    },
]

INITIAL_TASKS_DATA = [
    {
        "id": "task_901",
        "status": "OPEN",
        "symptom": "504 Gateway Timeout on /auth/v1/token",
        "details": "Spike in 504 timeouts reported by APIGateway for /auth/v1/token during EU peak traffic window. Suspected connection pool exhaustion or redis timeout.",
        "priority": "HIGH",
        "created_at": "2026-10-17 10:15:00",
        "target_service": "AuthService",
        "assigned_to": "Sam Chen",
    }
]


# ─────────────────────────────────────────────────────────────────────────────
# Seeder Implementation
# ─────────────────────────────────────────────────────────────────────────────

def seed_graphgenie_database(force: bool = False):
    """
    Seeds the enterprise domain topology, documents with embeddings,
    past episodic memory, and active handoff tasks into FalkorDB/InMemoryGraphEngine.
    """
    logger.info("⚡ Seeding GraphGenie enterprise topology & memory graph...")

    # Embed document texts (only if documents are defined)
    doc_texts = [d["text"] for d in DOCUMENTS_DATA]
    embeddings = []
    if doc_texts:
        try:
            embeddings = embedding_service.embed(doc_texts)
        except Exception as e:
            logger.warning(f"Could not compute local embeddings during seed: {e}")
            embeddings = [[0.0] * 384 for _ in doc_texts]

    # In-Memory engine direct injection (for instant zero-latency local fallback)
    mem_engine = getattr(db_client, "_in_memory_engine", None)
    if mem_engine is not None:
        # Populate in-memory collections
        for s in SERVICES_DATA:
            mem_engine.services[s["id"]] = s
            # also index as entity for hybrid compatibility
            mem_engine.entities[s["id"]] = {
                "id": s["id"],
                "name": s["name"],
                "type": "Service",
                "description": s["description"],
                "mention_count": 10,
            }

        for f in FEATURES_DATA:
            mem_engine.features[f["id"]] = f
            mem_engine.entities[f["id"]] = {
                "id": f["id"],
                "name": f["name"],
                "type": "Feature",
                "description": f["description"],
                "mention_count": 8,
            }

        for p in PERSONS_DATA:
            mem_engine.persons[p["id"]] = p
            mem_engine.entities[p["id"]] = {
                "id": p["id"],
                "name": p["name"],
                "type": "Person",
                "description": f"{p['role']} ({p['email']})",
                "mention_count": 5,
            }

        for i, d in enumerate(DOCUMENTS_DATA):
            mem_engine.documents[d["id"]] = {
                "id": d["id"],
                "name": d["title"],
                "type": d["doc_type"],
                "metadata": {"clause": d["clause"]},
                "chunk_count": 1,
                "created_at": "2026-10-17 08:00:00",
            }
            mem_engine.chunks[f"{d['id']}_c0"] = {
                "id": f"{d['id']}_c0",
                "text": d["text"],
                "document_id": d["id"],
                "document_name": d["title"],
                "chunk_index": 0,
                "token_count": len(d["text"].split()),
                "embedding": embeddings[i],
                "metadata": {"clause": d["clause"]},
            }

        for ep in EPISODES_DATA:
            mem_engine.episodes[ep["id"]] = ep

        for t in INITIAL_TASKS_DATA:
            mem_engine.tasks[t["id"]] = t

        # Relations
        # Service dependencies: APIGateway -> AuthService -> UserDirectory & NotificationGateway & SessionStore
        # BillingService -> AuthService & UserDirectory
        mem_engine.relations.extend([
            {"source_id": "srv_gateway", "source_name": "APIGateway", "source_type": "Service", "relation": "DEPENDS_ON", "target_id": "srv_auth", "target_name": "AuthService", "target_type": "Service", "weight": 1.0},
            {"source_id": "srv_auth", "source_name": "AuthService", "source_type": "Service", "relation": "DEPENDS_ON", "target_id": "srv_user", "target_name": "UserDirectory", "target_type": "Service", "weight": 1.0},
            {"source_id": "srv_auth", "source_name": "AuthService", "source_type": "Service", "relation": "DEPENDS_ON", "target_id": "srv_session", "target_name": "SessionStore", "target_type": "Service", "weight": 1.0},
            {"source_id": "srv_auth", "source_name": "AuthService", "source_type": "Service", "relation": "DEPENDS_ON", "target_id": "srv_notif", "target_name": "NotificationGateway", "target_type": "Service", "weight": 1.0},
            {"source_id": "srv_billing", "source_name": "BillingService", "source_type": "Service", "relation": "DEPENDS_ON", "target_id": "srv_user", "target_name": "UserDirectory", "target_type": "Service", "weight": 1.0},
            # Features CALLS Services
            {"source_id": "feat_sms2fa", "source_name": "SMS 2FA", "source_type": "Feature", "relation": "CALLS", "target_id": "srv_notif", "target_name": "NotificationGateway", "target_type": "Service", "weight": 1.0},
            {"source_id": "feat_sms2fa", "source_name": "SMS 2FA", "source_type": "Feature", "relation": "CALLS", "target_id": "srv_auth", "target_name": "AuthService", "target_type": "Service", "weight": 1.0},
            {"source_id": "feat_social", "source_name": "SocialAuth", "source_type": "Feature", "relation": "CALLS", "target_id": "srv_auth", "target_name": "AuthService", "target_type": "Service", "weight": 1.0},
            {"source_id": "feat_signup", "source_name": "PhoneSignup", "source_type": "Feature", "relation": "CALLS", "target_id": "srv_notif", "target_name": "NotificationGateway", "target_type": "Service", "weight": 1.0},
            {"source_id": "feat_bio", "source_name": "BiometricLogin", "source_type": "Feature", "relation": "CALLS", "target_id": "srv_auth", "target_name": "AuthService", "target_type": "Service", "weight": 1.0},
            {"source_id": "feat_refresh", "source_name": "TokenRefresh", "source_type": "Feature", "relation": "CALLS", "target_id": "srv_session", "target_name": "SessionStore", "target_type": "Service", "weight": 1.0},
            # People maintain services
            {"source_id": "srv_auth", "source_name": "AuthService", "source_type": "Service", "relation": "MAINTAINED_BY", "target_id": "usr_alex", "target_name": "Alex Rivera", "target_type": "Person", "weight": 1.0},
            {"source_id": "srv_gateway", "source_name": "APIGateway", "source_type": "Service", "relation": "MAINTAINED_BY", "target_id": "usr_sam", "target_name": "Sam Chen", "target_type": "Person", "weight": 1.0},
            {"source_id": "srv_notif", "source_name": "NotificationGateway", "source_type": "Service", "relation": "MAINTAINED_BY", "target_id": "usr_alex", "target_name": "Alex Rivera", "target_type": "Person", "weight": 1.0},
            {"source_id": "srv_user", "source_name": "UserDirectory", "source_type": "Service", "relation": "MAINTAINED_BY", "target_id": "usr_elena", "target_name": "Elena Rostova", "target_type": "Person", "weight": 1.0},
            {"source_id": "srv_billing", "source_name": "BillingService", "source_type": "Service", "relation": "MAINTAINED_BY", "target_id": "usr_maya", "target_name": "Maya Patel", "target_type": "Person", "weight": 1.0},
        ])

    # If real FalkorDB is connected, execute OpenCypher Cypher queries to seed DB
    if db_client.is_connected:
        try:
            # 1. Services
            for s in SERVICES_DATA:
                db_client.query("""
                MERGE (s:Service {id: $id})
                SET s.name = $name, s.runtime = $runtime, s.criticality = $criticality, s.description = $desc
                """, {"id": s["id"], "name": s["name"], "runtime": s["runtime"], "criticality": s["criticality"], "desc": s["description"]})

            # 2. Features
            for f in FEATURES_DATA:
                db_client.query("""
                MERGE (f:Feature {id: $id})
                SET f.name = $name, f.status = $status, f.stage = $stage, f.description = $desc
                """, {"id": f["id"], "name": f["name"], "status": f["status"], "stage": f["stage"], "desc": f["description"]})

            # 3. Persons
            for p in PERSONS_DATA:
                db_client.query("""
                MERGE (p:Person {id: $id})
                SET p.name = $name, p.role = $role, p.email = $email, p.on_call = $on_call
                """, {"id": p["id"], "name": p["name"], "role": p["role"], "email": p["email"], "on_call": p["on_call"]})

            # 4. Episodes
            for ep in EPISODES_DATA:
                db_client.query("""
                MERGE (e:MemoryEpisode {id: $id})
                SET e.session_id = $session_id, e.timestamp = $ts, e.symptom = $symptom, e.root_cause = $root_cause, e.fix = $fix
                WITH e
                MATCH (s:Service {name: $srv})
                MERGE (e)-[:RESOLVED_FOR]->(s)
                """, {"id": ep["id"], "session_id": ep["session_id"], "ts": ep["timestamp"], "symptom": ep["symptom"], "root_cause": ep["root_cause"], "fix": ep["fix"], "srv": ep["service_name"]})

            # 5. Handoff Tasks
            for t in INITIAL_TASKS_DATA:
                db_client.query("""
                MERGE (t:HandoffTask {id: $id})
                SET t.status = $status, t.symptom = $symptom, t.details = $details, t.priority = $priority, t.created_at = $created_at
                WITH t
                MATCH (s:Service {name: $srv})
                MERGE (t)-[:TARGETS]->(s)
                """, {"id": t["id"], "status": t["status"], "symptom": t["symptom"], "details": t["details"], "priority": t["priority"], "created_at": t["created_at"], "srv": t["target_service"]})

            # 6. Service dependencies
            deps = [
                ("APIGateway", "AuthService"),
                ("AuthService", "UserDirectory"),
                ("AuthService", "SessionStore"),
                ("AuthService", "NotificationGateway"),
                ("BillingService", "UserDirectory"),
            ]
            for src, tgt in deps:
                db_client.query("""
                MATCH (s:Service {name: $src}), (t:Service {name: $tgt})
                MERGE (s)-[:DEPENDS_ON]->(t)
                """, {"src": src, "tgt": tgt})

            # 7. Feature calls Service
            calls = [
                ("SMS 2FA", "NotificationGateway"),
                ("SMS 2FA", "AuthService"),
                ("SocialAuth", "AuthService"),
                ("PhoneSignup", "NotificationGateway"),
                ("BiometricLogin", "AuthService"),
                ("TokenRefresh", "SessionStore"),
            ]
            for feat, srv in calls:
                db_client.query("""
                MATCH (f:Feature {name: $feat}), (s:Service {name: $srv})
                MERGE (f)-[:CALLS]->(s)
                """, {"feat": feat, "srv": srv})

            # 8. Service maintained by Person
            maintains = [
                ("AuthService", "Alex Rivera"),
                ("APIGateway", "Sam Chen"),
                ("NotificationGateway", "Alex Rivera"),
                ("UserDirectory", "Elena Rostova"),
                ("BillingService", "Maya Patel"),
                ("SessionStore", "Sam Chen"),
            ]
            for srv, person in maintains:
                db_client.query("""
                MATCH (s:Service {name: $srv}), (p:Person {name: $person})
                MERGE (s)-[:MAINTAINED_BY {status: 'active'}]->(p)
                """, {"srv": srv, "person": person})

            logger.info("FalkorDB graph successfully seeded with GraphGenie enterprise schema.")
        except Exception as e:
            logger.warning(f"Error seeding FalkorDB live instance: {e}")

    logger.info("✅ GraphGenie enterprise topology & memory seed complete.")
