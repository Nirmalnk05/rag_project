"""Provider selection from settings."""
from __future__ import annotations

from src.config import get_settings
from src.providers.base import LLMProvider

_provider: LLMProvider | None = None


def get_provider() -> LLMProvider:
    global _provider
    if _provider is None:
        s = get_settings()
        if s.llm_provider == "anthropic":
            from src.providers.anthropic_provider import AnthropicProvider
            _provider = AnthropicProvider(s)
        elif s.llm_provider == "ollama":
            from src.providers.ollama_provider import OllamaProvider
            _provider = OllamaProvider(s)
        elif s.llm_provider == "openai_compatible":
            from src.providers.openai_compat_provider import OpenAICompatProvider
            _provider = OpenAICompatProvider(s)
        else:
            raise ValueError(f"unknown LLM_PROVIDER: {s.llm_provider}")
    return _provider


def reset_provider_cache() -> None:
    global _provider
    _provider = None
