import base64
import json
from types import SimpleNamespace

import httpx
import pytest

from src.config import Settings
from src.providers.anthropic_provider import AnthropicProvider
from src.providers.base import LLMError, LLMProvider
from src.providers.ollama_provider import OllamaProvider
from src.providers.openai_compat_provider import OpenAICompatProvider

IMG = b"\x89PNG\r\n\x1a\nfakebytes"
B64 = base64.b64encode(IMG).decode()


# ---------- base default ----------

class _Bare(LLMProvider):
    async def complete(self, prompt, *, system=None, model, max_tokens=1024, temperature=0.0):
        raise NotImplementedError

    async def stream(self, prompt, *, system=None, model, max_tokens=2048, temperature=0.0):
        yield ""


async def test_base_caption_unsupported():
    with pytest.raises(LLMError, match="not supported"):
        await _Bare().caption(image=IMG, media_type="image/png", prompt="x", model="m")


# ---------- anthropic ----------

class FakeMessages:
    def __init__(self):
        self.last_kwargs = None

    async def create(self, **kwargs):
        self.last_kwargs = kwargs
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="a pump")])


async def test_anthropic_caption_builds_image_block():
    fake = FakeMessages()
    p = AnthropicProvider(Settings(_env_file=None, anthropic_api_key="k"),
                          client=SimpleNamespace(messages=fake))
    out = await p.caption(image=IMG, media_type="image/png", prompt="describe", model="claude-x")
    assert out == "a pump"
    content = fake.last_kwargs["messages"][0]["content"]
    img_block = next(b for b in content if b["type"] == "image")
    assert img_block["source"]["data"] == B64
    assert img_block["source"]["media_type"] == "image/png"


# ---------- ollama ----------

async def test_ollama_caption_sends_images_field():
    def handler(request):
        body = json.loads(request.content)
        assert request.url.path == "/api/chat"
        assert body["messages"][0]["images"] == [B64]
        return httpx.Response(200, json={"message": {"content": "a valve"}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler),
                              base_url="http://localhost:11434")
    p = OllamaProvider(Settings(_env_file=None, llm_provider="ollama"), client=client)
    assert await p.caption(image=IMG, media_type="image/png", prompt="x", model="llava") == "a valve"


# ---------- openai-compatible ----------

async def test_openai_compat_caption_builds_image_url():
    def handler(request):
        body = json.loads(request.content)
        assert request.url.path == "/v1/chat/completions"
        parts = body["messages"][0]["content"]
        img = next(p for p in parts if p["type"] == "image_url")
        assert img["image_url"]["url"] == f"data:image/png;base64,{B64}"
        return httpx.Response(200, json={"choices": [{"message": {"content": "a gauge"}}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://x/v1")
    p = OpenAICompatProvider(
        Settings(_env_file=None, llm_provider="openai_compatible", openai_base_url="http://x/v1"),
        client=client)
    assert await p.caption(image=IMG, media_type="image/png", prompt="x", model="qwen-vl") == "a gauge"
