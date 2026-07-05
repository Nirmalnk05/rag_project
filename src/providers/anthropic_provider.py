"""Anthropic provider. System prompts get cache_control for prompt caching."""
from __future__ import annotations

import base64
from typing import Any, AsyncIterator

import anthropic
from anthropic import AsyncAnthropic

from src.config import Settings
from src.providers.base import LLMError, LLMProvider, LLMResult, LLMRetryableError

_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 529}


class AnthropicProvider(LLMProvider):
    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self._client = client or AsyncAnthropic(
            api_key=settings.anthropic_api_key, timeout=settings.llm_timeout_s
        )

    @staticmethod
    def _kwargs(prompt, system, model, max_tokens, temperature) -> dict:
        kw: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kw["system"] = [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ]
        return kw

    @staticmethod
    def _map_error(e: Exception) -> LLMError:
        if isinstance(e, (anthropic.APIConnectionError, anthropic.APITimeoutError)):
            return LLMRetryableError(str(e))
        if isinstance(e, anthropic.APIStatusError):
            status = getattr(e.response, "status_code", 0)
            cls = LLMRetryableError if status in _RETRYABLE_STATUS else LLMError
            return cls(f"anthropic status {status}: {e}")
        return LLMError(str(e))

    async def complete(self, prompt, *, system=None, model, max_tokens=1024, temperature=0.0) -> LLMResult:
        try:
            msg = await self._client.messages.create(
                **self._kwargs(prompt, system, model, max_tokens, temperature)
            )
        except Exception as e:  # noqa: BLE001 — mapped to typed errors
            raise self._map_error(e) from e
        text = "".join(b.text for b in msg.content if b.type == "text")
        return LLMResult(
            text=text,
            model=getattr(msg, "model", model),
            input_tokens=getattr(msg.usage, "input_tokens", 0),
            output_tokens=getattr(msg.usage, "output_tokens", 0),
        )

    async def stream(self, prompt, *, system=None, model, max_tokens=2048, temperature=0.0) -> AsyncIterator[str]:
        try:
            async with self._client.messages.stream(
                **self._kwargs(prompt, system, model, max_tokens, temperature)
            ) as s:
                async for text in s.text_stream:
                    yield text
        except LLMError:
            raise
        except Exception as e:  # noqa: BLE001
            raise self._map_error(e) from e

    async def caption(self, *, image, media_type, prompt, model, max_tokens=512) -> str:
        content = [
            {"type": "image", "source": {
                "type": "base64", "media_type": media_type,
                "data": base64.b64encode(image).decode(),
            }},
            {"type": "text", "text": prompt},
        ]
        try:
            msg = await self._client.messages.create(
                model=model, max_tokens=max_tokens,
                messages=[{"role": "user", "content": content}],
            )
        except Exception as e:  # noqa: BLE001
            raise self._map_error(e) from e
        return "".join(b.text for b in msg.content if b.type == "text")
