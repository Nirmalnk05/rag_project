"""Thin Anthropic client wrapper with caching-friendly helpers."""
from __future__ import annotations

from typing import AsyncIterator, Iterable

from anthropic import Anthropic, AsyncAnthropic

from src.config import get_settings


_settings = get_settings()
_sync_client: Anthropic | None = None
_async_client: AsyncAnthropic | None = None


def sync_client() -> Anthropic:
    global _sync_client
    if _sync_client is None:
        _sync_client = Anthropic(api_key=_settings.anthropic_api_key)
    return _sync_client


def async_client() -> AsyncAnthropic:
    global _async_client
    if _async_client is None:
        _async_client = AsyncAnthropic(api_key=_settings.anthropic_api_key)
    return _async_client


def complete(
    prompt: str,
    *,
    model: str | None = None,
    system: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> str:
    """Sync single-turn completion. Used by graph nodes (rewriter, grader, contextual aug)."""
    client = sync_client()
    kwargs = {
        "model": model or _settings.answer_model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system
    msg = client.messages.create(**kwargs)
    # join all text blocks (usually one)
    return "".join(b.text for b in msg.content if b.type == "text")


async def stream_complete(
    prompt: str,
    *,
    model: str | None = None,
    system: str | None = None,
    max_tokens: int = 2048,
    temperature: float = 0.0,
) -> AsyncIterator[str]:
    """Async streaming completion. Yields text deltas. Used by the API for SSE."""
    client = async_client()
    kwargs = {
        "model": model or _settings.answer_model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        kwargs["system"] = system
    async with client.messages.stream(**kwargs) as stream:
        async for text in stream.text_stream:
            yield text
