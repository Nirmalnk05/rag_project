"""Centralized configuration loaded from environment."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

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

    # Chunking (sized in tokens: bge-small takes 512 and truncates the rest in
    # silence, and measured tokens-per-word spans 1.23 prose -> 5.21 CJK, so no
    # word target fits all content)
    chunk_max_tokens: int = 512            # the embedding model's hard window
    chunk_overlap_tokens: int = 64
    chunk_context_reserve_tokens: int = 128  # room for the contextual blurb

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

    # Access control (per-user retrieval ACLs)
    enable_acl: bool = False               # off = every chunk visible (dev default)
    # X-User-Id / X-Roles / X-Sites are caller-supplied: with ENABLE_ACL on and
    # nothing in front of the app, `X-Roles: admin` is a full ACL bypass. Set
    # this only when a gateway strips and reconstructs those headers from a
    # verified identity (OIDC/LDAP). validate_access() refuses to start without it.
    trust_identity_headers: bool = False

    # Document freshness / versioning
    retrieve_superseded: bool = False      # False = superseded revisions are hidden

    # Index reload: how often a serving process checks index_meta.json for a
    # newer committed generation and rebuilds chunks.json + BM25 off the
    # request path. 0 disables it (the snapshot then lives until a restart).
    index_reload_poll_s: int = 60

    # Answer safety + faithfulness
    enable_safety_escalation: bool = True  # prepend safety notices for high-risk topics
    enable_faithfulness_check: bool = False  # extra LLM-judge call per answer

    # Semantic answer cache
    enable_semantic_cache: bool = False
    semantic_cache_threshold: float = 0.95
    semantic_cache_max: int = 1000

    # Guardrails backend
    guardrails_backend: str = "heuristic"  # heuristic | external

    # Feedback + online eval
    feedback_path: str = "data/feedback.jsonl"
    online_eval_sample_rate: float = 0.0   # fraction of answers logged for review

    # Prompt registry
    prompt_version: str = "v1"

    # CMMS / system-of-record connector (agent tool)
    enable_cmms: bool = False
    cmms_connector: str = "mock"           # mock | (extend with real connectors)

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

    def validate_access(self, acl_path: Path | None = None) -> None:
        """Fail closed on the two ways ACLs silently become no-ops.

        A deliberately public deployment may run with ACLs off; one that ships
        an ACL sidecar may not, because `_access_for` then returns None and
        `acl_visible` short-circuits to visible for every chunk.
        """
        path = Path("data/acl.json") if acl_path is None else Path(acl_path)
        if path.exists() and not self.enable_acl:
            raise ValueError(
                f"{path} defines document ACLs but ENABLE_ACL is false, so every "
                "chunk is served to every caller. Set ENABLE_ACL=true, or remove "
                "the sidecar if this deployment is intentionally public."
            )
        if self.enable_acl and not self.trust_identity_headers:
            raise ValueError(
                "ENABLE_ACL=true requires TRUST_IDENTITY_HEADERS=true, which "
                "asserts that a gateway authenticates the user and rebuilds "
                "X-User-Id / X-Roles / X-Sites from verified claims. Without one, "
                "any caller can send 'X-Roles: admin' and read every document."
            )


@lru_cache
def get_settings() -> Settings:
    return Settings()
