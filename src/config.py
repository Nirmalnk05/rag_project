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
    rate_limit_backend: str = "memory"   # memory | redis
    redis_url: str = ""                  # required when rate_limit_backend=redis

    # Guardrails
    redact_pii_in_answers: bool = False

    # Agent (tool-calling)
    agent_mode: str = "agent"              # agent | pipeline
    agent_max_steps: int = 6
    agent_max_tool_results_chars: int = 4000

    # Conversation memory
    enable_memory: bool = True
    memory_backend: str = "memory"         # memory | redis
    memory_max_turns: int = 10             # user+assistant pairs kept per session
    memory_ttl_s: int = 3600

    # Figure captioning (vision)
    enable_figure_caption: bool = True
    vision_model: str = ""  # e.g. a Claude vision model, or "llama3.2-vision"

    # Document parsing
    parse_backend: str = "hybrid"          # local | cloud | hybrid
    cloud_parse_provider: str = "llamaparse"  # llamaparse | azure
    llamaparse_api_key: str = ""
    azure_di_endpoint: str = ""
    azure_di_key: str = ""
    cloud_parse_min_chars: int = 200       # local result below this ⇒ low-confidence
    ocr_engine: str = "easyocr"            # easyocr | tesseract
    ocr_languages: str = "en"

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

    @property
    def cloud_parse_configured(self) -> bool:
        if self.cloud_parse_provider == "llamaparse":
            return bool(self.llamaparse_api_key)
        if self.cloud_parse_provider == "azure":
            return bool(self.azure_di_endpoint and self.azure_di_key)
        return False

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
        if self.parse_backend == "cloud" and not self.cloud_parse_configured:
            raise ValueError(
                f"PARSE_BACKEND=cloud but no {self.cloud_parse_provider} credentials "
                "are set. Configure the key, or use PARSE_BACKEND=local|hybrid."
            )
        if self.rate_limit_backend == "redis" and not self.redis_url:
            raise ValueError(
                "RATE_LIMIT_BACKEND=redis requires REDIS_URL. "
                "Set it, or use RATE_LIMIT_BACKEND=memory (single-instance)."
            )


@lru_cache
def get_settings() -> Settings:
    return Settings()
