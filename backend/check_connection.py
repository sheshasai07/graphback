"""
FalkorDB Cloud Connection & Diagnostic Verifier
Run this script to test whether your backend is connected to FalkorDB Cloud or the In-Memory engine.

Usage:
    python check_connection.py
"""

import sys
from loguru import logger
from app.config import settings
from app.db.falkordb_client import db_client

def verify():
    print("=" * 65)
    print("   FalkorDB Cloud Connection Verification")
    print("=" * 65)
    
    print(f"\n[1] Environment Configuration (.env):")
    print(f"    - FALKORDB_URL      : {settings.falkordb_url or '(not set)'}")
    print(f"    - FALKORDB_HOST     : {settings.falkordb_host or '(not set)'}")
    print(f"    - FALKORDB_PORT     : {settings.falkordb_port}")
    print(f"    - FALKORDB_USERNAME : {settings.falkordb_username}")
    print(f"    - FALKORDB_PASSWORD : {'********' if settings.falkordb_password else '(empty / not set)'}")
    print(f"    - FALKORDB_SSL      : {settings.falkordb_ssl}")
    print(f"    - Active Graph Name : {settings.falkordb_graph_name}")

    print(f"\n[2] Connecting...")
    db_client.connect()

    print(f"\n[3] Connection Status:")
    if db_client.is_connected:
        print("    STATUS: CONNECTED TO FALKORDB CLOUD! [OK]")
        print(f"    Engine: Managed FalkorDB Cloud Instance")
        
        # Test query
        try:
            res = db_client.query("RETURN 1 AS ping")
            print("    Ping Test: SUCCESS [OK]")
            
            # Query stats
            doc_res = db_client.query("MATCH (d:Document) RETURN count(d)")
            chunk_res = db_client.query("MATCH (c:Chunk) RETURN count(c)")
            entity_res = db_client.query("MATCH (e:Entity) RETURN count(e)")
            
            docs = doc_res.result_set[0][0] if doc_res.result_set else 0
            chunks = chunk_res.result_set[0][0] if chunk_res.result_set else 0
            entities = entity_res.result_set[0][0] if entity_res.result_set else 0
            
            print(f"\n[4] Cloud Graph Database Contents:")
            print(f"    - Documents : {docs}")
            print(f"    - Chunks    : {chunks}")
            print(f"    - Entities  : {entities}")
            print("\nAll graph data is storing directly in your FalkorDB Cloud instance!")
        except Exception as e:
            print(f"    Query failed: {e}")
    else:
        print("    STATUS: RUNNING ON IN-MEMORY GRAPH ENGINE FALLBACK [WARNING]")
        print("    Reason: The backend could not establish an authenticated connection to FalkorDB Cloud.")
        print("\n    To connect to FalkorDB Cloud, ensure backend/.env contains:")
        print("    FALKORDB_URL=falkors://default:YOUR_PASSWORD@your-host.cloud:port")
        print("    OR:")
        print("    FALKORDB_HOST=your-host.cloud")
        print("    FALKORDB_PORT=your-port")
        print("    FALKORDB_PASSWORD=your-password")
        print("    FALKORDB_SSL=true")
        print("\n    (Note: The UI still displays graphs because the In-Memory Fallback keeps the app alive)")
    print("=" * 65)

if __name__ == "__main__":
    verify()
