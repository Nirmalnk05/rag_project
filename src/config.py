"""Centralized configuration loaded from environment."""
from __future__ import annotations

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

VALID_PROVIDERS = {"anthropic", "ollama", "openai_compatible"}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Provider selection
    llm_provider: str = "anthropic"  # anthropic | ollama | openai_compatible

    # Anthropic (required only when llm_provider == anthropic)
    anthropic_api_key: str = ""

    # Ollama
    ollama_base_url: str = "http://localhost:11434"

    # OpenAI-compatible (vLLM, LM Studio, OpenAI, ...)
    openai_base_url: str = "http://localhost:8080/v1"
    openai_api_key: str = ""

    # LLM call behavior
    llm_timeout_s: float = 60.0
    llm_max_attempts: int = 3
    fallback_model: str = ""  # empty = disabled

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "maintenance_docs"

    # Models (interpreted by the active provider)
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

    # Security
    api_keys: str = ""       # comma-separated; empty = auth disabled
    rate_limit_rpm: int = 30

    # Guardrails
    redact_pii_in_answers: bool = False

    # Figure captioning (vision)
    enable_figure_caption: bool = True
    vision_model: str = ""  # e.g. a Claude vision model, or "llama3.2-vision"

    # Tracing (optional)
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    @property
    def tracing_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)

    @property
    def api_key_set(self) -> set[str]:
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}

    def validate_provider(self) -> None:
        """Fail fast at startup with a clear message."""
        if self.llm_provider not in VALID_PROVIDERS:
            raise ValueError(
                f"LLM_PROVIDER='{self.llm_provider}' is not one of {sorted(VALID_PROVIDERS)}"
            )
        if self.llm_provider == "anthropic" and not self.anthropic_api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic. "
                "Set it in .env, or switch to a local provider: LLM_PROVIDER=ollama"
            )


@lru_cache
def get_settings() -> Settings:
    return Settings()
