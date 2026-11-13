"""Anthropic provider. System prompts get cache_control for prompt caching."""
from __future__ import annotations

import base64
from typing import Any, AsyncIterator

import anthropic
from anthropic import AsyncAnthropic

from src.config import Settings
from src.providers.base import (
    ChatResult, LLMError, LLMProvider, LLMResult, LLMRetryableError, ToolCall, ToolSpec,
)

_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 529}


def _to_anthropic_messages(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Translate neutral messages → (system, anthropic messages). Tool results
    are grouped into a single user turn, as the Anthropic API requires."""
    system: str | None = None
    out: list[dict] = []
    pending_results: list[dict] = []

    def flush_results():
        if pending_results:
            out.append({"role": "user", "content": list(pending_results)})
            pending_results.clear()

    for m in messages:
        role = m["role"]
        if role == "system":
            system = m["content"]
        elif role == "tool":
            pending_results.append({
                "type": "tool_result", "tool_use_id": m["tool_call_id"],
                "content": m["content"],
            })
        else:
            flush_results()
            if role == "assistant" and m.get("tool_calls"):
                content = []
                if m.get("content"):
                    content.append({"type": "text", "text": m["content"]})
                for tc in m["tool_calls"]:
                    content.append({"type": "tool_use", "id": tc.id,
                                    "name": tc.name, "input": tc.arguments})
                out.append({"role": "assistant", "content": content})
            else:
                out.append({"role": role, "content": m["content"]})
    flush_results()
    return system, out


class AnthropicProvider(LLMProvider):
    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self._client = client or AsyncAnthropic(
            api_key=settings.anthropic_api_key, timeout=settings.llm_timeout_s
        )

    @staticmethod
    def _kwargs(prompt, system, model, max_tokens, temperature) -> dict:
        # `temperature` is accepted for interface parity but never sent: Claude
        # 5-family models reject sampling params with a 400, and the API is
        # removing them on every model.
        kw: dict = {
            "model": model,
            "max_tokens": max_tokens,
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

    async def chat(self, messages, *, tools=None, model, max_tokens=1024, temperature=0.0) -> ChatResult:
        system, native = _to_anthropic_messages(messages)
        kw: dict = {"model": model, "max_tokens": max_tokens, "messages": native}
        if system:
            kw["system"] = system
        if tools:
            kw["tools"] = [{"name": t.name, "description": t.description,
                            "input_schema": t.parameters} for t in tools]
        try:
            msg = await self._client.messages.create(**kw)
        except Exception as e:  # noqa: BLE001
            raise self._map_error(e) from e
        text = "".join(b.text for b in msg.content if b.type == "text")
        calls = [ToolCall(id=b.id, name=b.name, arguments=b.input)
                 for b in msg.content if b.type == "tool_use"]
        usage = getattr(msg, "usage", None)
        return ChatResult(
            text=text, tool_calls=calls,
            stop_reason=getattr(msg, "stop_reason", ""),
            model=getattr(msg, "model", model),
            input_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
            output_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
        )
