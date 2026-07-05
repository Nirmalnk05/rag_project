"""Provider-neutral LLM helpers. Kept as a shim so existing imports work.

complete()/stream_complete() are async; complete_sync() is for scripts
(ingestion) that run outside an event loop.
"""
from __future__ import annotations

import asyncio
from typing import AsyncIterator

from src.config import get_settings
from src.providers.base import LLMResult
from src.providers.factory import get_provider
from src.providers.retry import call_with_retries


async def complete(
    prompt: str,
    *,
    model: str | None = None,
    system: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> LLMResult:
    s = get_settings()
    provider = get_provider()
    chosen = model or s.answer_model

    async def make_call(override: str | None) -> LLMResult:
        return await provider.complete(
            prompt, system=system, model=override or chosen,
            max_tokens=max_tokens, temperature=temperature,
        )

    return await call_with_retries(
        make_call, max_attempts=s.llm_max_attempts, fallback_model=s.fallback_model
    )


async def stream_complete(
    prompt: str,
    *,
    model: str | None = None,
    system: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
) -> AsyncIterator[str]:
    """Streaming has no mid-stream retry (tokens already sent); failures
    before/during the stream raise typed LLMError to the caller."""
    s = get_settings()
    provider = get_provider()
    async for token in provider.stream(
        prompt, system=system, model=model or s.answer_model,
        max_tokens=max_tokens, temperature=temperature,
    ):
        yield token


def complete_sync(prompt: str, **kwargs) -> LLMResult:
    return asyncio.run(complete(prompt, **kwargs))
