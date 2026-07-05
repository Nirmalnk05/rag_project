"""OpenAI-compatible /v1/chat/completions provider (vLLM, LM Studio, OpenAI...)."""
from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from src.config import Settings
from src.providers.base import LLMError, LLMProvider, LLMResult, LLMRetryableError

_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


def _map_status(status: int, body: str) -> LLMError:
    cls = LLMRetryableError if status in _RETRYABLE_STATUS else LLMError
    return cls(f"openai-compat status {status}: {body[:200]}")


class OpenAICompatProvider(LLMProvider):
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        headers = {}
        if settings.openai_api_key:
            headers["Authorization"] = f"Bearer {settings.openai_api_key}"
        self._client = client or httpx.AsyncClient(
            base_url=settings.openai_base_url, timeout=settings.llm_timeout_s, headers=headers
        )
        # when a test injects a client, still honor the key
        if client is not None and settings.openai_api_key:
            client.headers["Authorization"] = f"Bearer {settings.openai_api_key}"

    @staticmethod
    def _body(prompt, system, model, max_tokens, temperature, stream) -> dict:
        msgs = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        return {
            "model": model, "messages": msgs, "max_tokens": max_tokens,
            "temperature": temperature, "stream": stream,
        }

    async def complete(self, prompt, *, system=None, model, max_tokens=1024, temperature=0.0) -> LLMResult:
        try:
            resp = await self._client.post(
                "chat/completions",
                json=self._body(prompt, system, model, max_tokens, temperature, False),
            )
        except httpx.HTTPError as e:
            raise LLMRetryableError(f"openai-compat transport error: {e}") from e
        if resp.status_code != 200:
            raise _map_status(resp.status_code, resp.text)
        data = resp.json()
        usage = data.get("usage") or {}
        return LLMResult(
            text=(data.get("choices") or [{}])[0].get("message", {}).get("content", ""),
            model=data.get("model", model),
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )

    async def stream(self, prompt, *, system=None, model, max_tokens=2048, temperature=0.0) -> AsyncIterator[str]:
        try:
            async with self._client.stream(
                "POST", "chat/completions",
                json=self._body(prompt, system, model, max_tokens, temperature, True),
            ) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode(errors="replace")
                    raise _map_status(resp.status_code, body)
                async for line in resp.aiter_lines():
                    line = line.strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[len("data:"):].strip()
                    if payload == "[DONE]":
                        break
                    chunk = json.loads(payload)
                    frag = (chunk.get("choices") or [{}])[0].get("delta", {}).get("content")
                    if frag:
                        yield frag
        except httpx.HTTPError as e:
            raise LLMRetryableError(f"openai-compat transport error: {e}") from e
