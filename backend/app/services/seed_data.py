"""
GraphGenie - Knowledge Graph & Memory Seeder
All predefined mock test data (microservices, features, episodes, mock tasks, and dummy documents)
have been removed so that only user-uploaded documents and user-created data exist in the graph.
"""

from typing import List, Dict, Any
from loguru import logger

from app.db.falkordb_client import db_client

# Predefined collections are empty by default - no hidden test files or dummy services
SERVICES_DATA: List[Dict[str, Any]] = []
FEATURES_DATA: List[Dict[str, Any]] = []
PERSONS_DATA: List[Dict[str, Any]] = []
DOCUMENTS_DATA: List[Dict[str, Any]] = []
EPISODES_DATA: List[Dict[str, Any]] = []
INITIAL_TASKS_DATA: List[Dict[str, Any]] = []


def seed_graphgenie_database(force: bool = False):
    """
    Seeder entry point. Retained for backwards-compatibility.
    No-op: All pre-defined test files and dummy entities are excluded.
    """
    logger.debug("seed_graphgenie_database called: Predefined seed data is disabled.")
