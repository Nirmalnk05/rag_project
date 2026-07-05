import pytest

import src.llm as llm
from src.providers.base import LLMProvider, LLMResult, LLMRetryableError
from src.providers.factory import get_provider, reset_provider_cache


class FakeProvider(LLMProvider):
    def __init__(self):
        self.calls = []

    async def complete(self, prompt, *, system=None, model, max_tokens=1024, temperature=0.0):
        self.calls.append(model)
        if model == "always-fails":
            raise LLMRetryableError("down")
        return LLMResult(text=f"echo:{prompt}", model=model, input_tokens=1, output_tokens=1)

    async def stream(self, prompt, *, system=None, model, max_tokens=2048, temperature=0.0):
        for t in ["a", "b"]:
            yield t


@pytest.fixture(autouse=True)
def fake_provider(monkeypatch):
    fp = FakeProvider()
    monkeypatch.setattr("src.llm.get_provider", lambda: fp)
    return fp


async def test_complete_returns_llmresult(fake_provider):
    r = await llm.complete("hi", model="m1")
    assert r.text == "echo:hi" and fake_provider.calls == ["m1"]


async def test_stream_complete_yields(fake_provider):
    out = [t async for t in llm.stream_complete("hi", model="m1")]
    assert out == ["a", "b"]


def test_complete_sync_wraps_async(fake_provider):
    r = llm.complete_sync("hi", model="m1")
    assert r.text == "echo:hi"


def test_factory_selects_by_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    from src.config import get_settings
    get_settings.cache_clear()
    reset_provider_cache()
    from src.providers.ollama_provider import OllamaProvider
    assert isinstance(get_provider(), OllamaProvider)
    get_settings.cache_clear()
    reset_provider_cache()
