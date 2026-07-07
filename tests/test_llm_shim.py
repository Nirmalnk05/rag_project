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


def test_complete_sync_multiple_calls_reuse_open_loop(fake_provider):
    """Regression test: complete_sync() must not create-and-close a new event
    loop on every call (that breaks a provider client cached across calls —
    see the docstring on _get_sync_loop). Simulates the ingestion pattern of
    calling complete_sync() once per chunk in a loop."""
    r1 = llm.complete_sync("first", model="m1")
    loop_after_first = llm._sync_loop
    assert loop_after_first is not None and not loop_after_first.is_closed()

    r2 = llm.complete_sync("second", model="m1")
    r3 = llm.complete_sync("third", model="m1")

    assert [r1.text, r2.text, r3.text] == ["echo:first", "echo:second", "echo:third"]
    # same loop instance reused across all three calls, never closed in between
    assert llm._sync_loop is loop_after_first
    assert not llm._sync_loop.is_closed()


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
