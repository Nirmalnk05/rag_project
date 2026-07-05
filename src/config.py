"""Centralized configuration loaded from environment."""
from __future__ import annotations

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Anthropic
    anthropic_api_key: str

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "maintenance_docs"

    # Models
    embed_model: str = "BAAI/bge-small-en-v1.5"
    rerank_model: str = "BAAI/bge-reranker-base"
    answer_model: str = "claude-sonnet-4-20250514"
    context_aug_model: str = "claude-haiku-4-5-20251001"
    grader_model: str = "claude-haiku-4-5-20251001"

    # Retrieval
    top_k_dense: int = 20
    top_k_sparse: int = 20
    top_k_rerank: int = 5
    max_retries: int = 2

    # Tracing (optional)
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    @property
    def tracing_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
