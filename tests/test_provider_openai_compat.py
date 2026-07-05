import json

import httpx
import pytest

from src.config import Settings
from src.providers.base import LLMError, LLMRetryableError
from src.providers.openai_compat_provider import OpenAICompatProvider


def _settings(key=""):
    return Settings(_env_file=None, llm_provider="openai_compatible",
                    openai_base_url="http://x/v1", openai_api_key=key)


def _provider(handler, key=""):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://x/v1")
    return OpenAICompatProvider(_settings(key), client=client)


async def test_complete_parses_choice_and_sends_auth():
    def handler(request):
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer sk-1"
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "yo"}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 2},
            "model": "qwen",
        })

    r = await _provider(handler, key="sk-1").complete("q", system="s", model="qwen")
    assert r.text == "yo" and r.input_tokens == 7 and r.output_tokens == 2


async def test_no_key_no_auth_header():
    def handler(request):
        assert "authorization" not in request.headers
        return httpx.Response(200, json={"choices": [{"message": {"content": "a"}}]})

    await _provider(handler).complete("q", model="qwen")


async def test_401_terminal_503_retryable():
    def h401(request):
        return httpx.Response(401, text="no")

    with pytest.raises(LLMError) as ei:
        await _provider(h401).complete("q", model="m")
    assert not isinstance(ei.value, LLMRetryableError)

    def h503(request):
        return httpx.Response(503, text="busy")

    with pytest.raises(LLMRetryableError):
        await _provider(h503).complete("q", model="m")


async def test_stream_parses_sse_until_done():
    events = [
        'data: {"choices":[{"delta":{"content":"He"}}]}',
        'data: {"choices":[{"delta":{"content":"y"}}]}',
        "data: [DONE]",
    ]

    def handler(request):
        assert json.loads(request.content)["stream"] is True
        return httpx.Response(200, text="\n".join(events))

    out = [t async for t in _provider(handler).stream("q", model="m")]
    assert "".join(out) == "Hey"
