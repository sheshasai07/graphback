"""
GraphRAG Backend - Configuration
Enterprise-grade settings management using Pydantic Settings
"""

from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import json


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- FalkorDB Cloud ---
    falkordb_url: str = Field(default="", description="FalkorDB Cloud connection URL (falkors://...)")
    falkordb_host: str = Field(default="localhost", description="FalkorDB host")
    falkordb_port: int = Field(default=6379, description="FalkorDB port")
    falkordb_username: str = Field(default="default", description="FalkorDB username")
    falkordb_password: str = Field(default="", description="FalkorDB password")
    falkordb_ssl: bool = Field(default=False, description="Enable SSL/TLS for FalkorDB Cloud")
    falkordb_graph_name: str = Field(default="graphrag", description="Graph name in FalkorDB")

    # --- OpenRouter LLM ---
    openrouter_api_key: str = Field(default="", description="OpenRouter API key")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter base URL"
    )
    openrouter_model: str = Field(
        default="nvidia/nemotron-3.5-super-vl:free",
        description="OpenRouter model for generation and entity extraction"
    )

    # --- Voyage AI Embeddings ---
    voyage_api_key: str = Field(default="", description="Voyage AI API Key")
    embedding_model_name: str = Field(
        default="voyage-4-lite",
        description="Voyage AI embedding model"
    )
    embedding_dimension: int = Field(default=1024, description="Embedding vector dimension")

    # --- Ingestion ---
    chunk_size: int = Field(default=512, description="Max tokens per chunk")
    chunk_overlap: int = Field(default=64, description="Overlap tokens between chunks")
    max_entities_per_chunk: int = Field(default=20)
    max_relations_per_chunk: int = Field(default=30)

    # --- Server ---
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    debug: bool = Field(default=True)
    cors_origins: List[str] = Field(
        default=["http://localhost:5173", "http://localhost:3000"]
    )

    # --- Paths ---
    upload_dir: str = Field(default="./uploads")


settings = Settings()
