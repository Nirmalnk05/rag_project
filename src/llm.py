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


_sync_loop: asyncio.AbstractEventLoop | None = None


def _get_sync_loop() -> asyncio.AbstractEventLoop:
    """A persistent event loop reused across complete_sync() calls.

    asyncio.run() would create and close a new loop on every call. The LLM
    provider (get_provider()) is a process-wide singleton whose async client
    (e.g. httpx.AsyncClient / AsyncAnthropic) binds its connections to
    whichever loop is running when it's first used. Closing that loop out
    from under it (as asyncio.run() does) breaks the client on the next
    call — surfacing as "RuntimeError: Event loop is closed" during
    ingestion, which calls complete_sync() once per chunk. Reusing one loop
    for the lifetime of the process keeps the provider's client valid.
    """
    global _sync_loop
    if _sync_loop is None or _sync_loop.is_closed():
        _sync_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_sync_loop)
    return _sync_loop


def complete_sync(prompt: str, **kwargs) -> LLMResult:
    return _get_sync_loop().run_until_complete(complete(prompt, **kwargs))
