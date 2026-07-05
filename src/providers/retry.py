"""Timeout/backoff wrapper around provider calls, with optional model fallback."""
from __future__ import annotations

import asyncio
import random
from typing import Awaitable, Callable

from src.providers.base import LLMError, LLMResult, LLMRetryableError

MakeCall = Callable[[str | None], Awaitable[LLMResult]]


async def call_with_retries(
    make_call: MakeCall,
    *,
    max_attempts: int,
    fallback_model: str = "",
    base_delay: float = 0.5,
    max_delay: float = 8.0,
) -> LLMResult:
    """Call make_call(None) up to max_attempts times on retryable errors,
    with exponential backoff + jitter. If still failing and fallback_model
    is set, one final attempt with make_call(fallback_model)."""
    last: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return await make_call(None)
        except LLMRetryableError as e:
            last = e
            if attempt < max_attempts - 1:
                delay = min(max_delay, base_delay * 2**attempt) * (0.5 + random.random())
                await asyncio.sleep(delay)
        except LLMError:
            raise  # terminal — don't retry
    if fallback_model:
        try:
            return await make_call(fallback_model)
        except LLMRetryableError as e:
            last = e
    raise LLMError(f"LLM call failed after {max_attempts} attempt(s)") from last
