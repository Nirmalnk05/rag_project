"""Ollama native /api/chat provider (no SDK; httpx only)."""
from __future__ import annotations

import base64
import json
from typing import AsyncIterator

import httpx

from src.config import Settings
from src.providers.base import LLMError, LLMProvider, LLMResult, LLMRetryableError

_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


def _messages(prompt: str, system: str | None) -> list[dict]:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    return msgs


def _map_status(status: int, body: str) -> LLMError:
    cls = LLMRetryableError if status in _RETRYABLE_STATUS else LLMError
    return cls(f"ollama status {status}: {body[:200]}")


class OllamaProvider(LLMProvider):
    def __init__(self, settings: Settings, client: httpx.AsyncClient | None = None) -> None:
        self._client = client or httpx.AsyncClient(
            base_url=settings.ollama_base_url, timeout=settings.llm_timeout_s
        )

    def _body(self, prompt, system, model, max_tokens, temperature, stream) -> dict:
        return {
            "model": model,
            "messages": _messages(prompt, system),
            "stream": stream,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }

    async def complete(self, prompt, *, system=None, model, max_tokens=1024, temperature=0.0) -> LLMResult:
        try:
            resp = await self._client.post(
                "/api/chat", json=self._body(prompt, system, model, max_tokens, temperature, False)
            )
        except httpx.HTTPError as e:
            raise LLMRetryableError(f"ollama transport error: {e}") from e
        if resp.status_code != 200:
            raise _map_status(resp.status_code, resp.text)
        data = resp.json()
        return LLMResult(
            text=data.get("message", {}).get("content", ""),
            model=model,
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
        )

    async def stream(self, prompt, *, system=None, model, max_tokens=2048, temperature=0.0) -> AsyncIterator[str]:
        try:
            async with self._client.stream(
                "POST", "/api/chat",
                json=self._body(prompt, system, model, max_tokens, temperature, True),
            ) as resp:
                if resp.status_code != 200:
                    body = (await resp.aread()).decode(errors="replace")
                    raise _map_status(resp.status_code, body)
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    chunk = json.loads(line)
                    frag = chunk.get("message", {}).get("content", "")
                    if frag:
                        yield frag
                    if chunk.get("done"):
                        break
        except httpx.HTTPError as e:
            raise LLMRetryableError(f"ollama transport error: {e}") from e

    async def caption(self, *, image, media_type, prompt, model, max_tokens=512) -> str:
        body = {
            "model": model,
            "messages": [{
                "role": "user", "content": prompt,
                "images": [base64.b64encode(image).decode()],
            }],
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": max_tokens},
        }
        try:
            resp = await self._client.post("/api/chat", json=body)
        except httpx.HTTPError as e:
            raise LLMRetryableError(f"ollama transport error: {e}") from e
        if resp.status_code != 200:
            raise _map_status(resp.status_code, resp.text)
        return resp.json().get("message", {}).get("content", "")
