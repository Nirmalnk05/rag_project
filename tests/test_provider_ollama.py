import json

import httpx
import pytest

from src.config import Settings
from src.providers.base import LLMRetryableError
from src.providers.ollama_provider import OllamaProvider


def _settings():
    return Settings(_env_file=None, llm_provider="ollama")


def _provider(handler):
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://localhost:11434"
    )
    return OllamaProvider(_settings(), client=client)


async def test_complete_parses_response():
    def handler(request):
        body = json.loads(request.content)
        assert request.url.path == "/api/chat"
        assert body["model"] == "llama3" and body["stream"] is False
        assert body["messages"][0] == {"role": "system", "content": "sys"}
        return httpx.Response(200, json={
            "message": {"role": "assistant", "content": "hi there"},
            "prompt_eval_count": 12, "eval_count": 4,
        })

    r = await _provider(handler).complete("q", system="sys", model="llama3")
    assert r.text == "hi there" and r.input_tokens == 12 and r.output_tokens == 4


async def test_5xx_is_retryable():
    def handler(request):
        return httpx.Response(500, text="boom")

    with pytest.raises(LLMRetryableError):
        await _provider(handler).complete("q", model="llama3")


async def test_stream_yields_fragments():
    lines = [
        json.dumps({"message": {"content": "Hel"}, "done": False}),
        json.dumps({"message": {"content": "lo"}, "done": False}),
        json.dumps({"message": {"content": ""}, "done": True}),
    ]

    def handler(request):
        return httpx.Response(200, text="\n".join(lines))

    out = [t async for t in _provider(handler).stream("q", model="llama3")]
    assert "".join(out) == "Hello"
