"""
GraphRAG Backend - Development Server Entry Point
Run with: python run.py
Or with uvicorn: uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

import uvicorn
from app.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level="info",
        access_log=True,
    )
