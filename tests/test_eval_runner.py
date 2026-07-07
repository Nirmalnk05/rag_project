import pytest

from src.config import Settings
from src.eval_runner import _build_ragas_llm_and_embeddings, _run_traced
from src.observability import Tracer, _current_trace


class FakeHFEmbeddings:
    """Stand-in for langchain_huggingface.HuggingFaceEmbeddings — the real
    class downloads/loads an actual model at construction time, which would
    require network access and be slow in unit tests."""

    def __init__(self, *, model_name):
        self.model_name = model_name


@pytest.fixture(autouse=True)
def fake_hf_embeddings(monkeypatch):
    import langchain_huggingface
    monkeypatch.setattr(langchain_huggingface, "HuggingFaceEmbeddings", FakeHFEmbeddings)


def test_anthropic_provider_maps_to_chat_anthropic():
    from langchain_anthropic import ChatAnthropic
    s = Settings(_env_file=None, llm_provider="anthropic", anthropic_api_key="k",
                answer_model="claude-x")
    llm, emb = _build_ragas_llm_and_embeddings(s)
    assert isinstance(llm.langchain_llm, ChatAnthropic)
    assert llm.langchain_llm.model == "claude-x"


def test_ollama_provider_maps_to_chat_ollama():
    from langchain_ollama import ChatOllama
    s = Settings(_env_file=None, llm_provider="ollama", answer_model="llama3.1",
                ollama_base_url="http://x:11434")
    llm, emb = _build_ragas_llm_and_embeddings(s)
    assert isinstance(llm.langchain_llm, ChatOllama)
    assert llm.langchain_llm.model == "llama3.1"
    assert llm.langchain_llm.base_url == "http://x:11434"


def test_openai_compatible_maps_to_chat_openai():
    from langchain_openai import ChatOpenAI
    s = Settings(_env_file=None, llm_provider="openai_compatible",
                openai_base_url="http://x/v1", openai_api_key="", answer_model="qwen")
    llm, emb = _build_ragas_llm_and_embeddings(s)
    assert isinstance(llm.langchain_llm, ChatOpenAI)


def test_openai_compatible_uses_dummy_key_when_none_configured():
    s = Settings(_env_file=None, llm_provider="openai_compatible", openai_api_key="")
    llm, emb = _build_ragas_llm_and_embeddings(s)
    # SecretStr on the langchain object; just confirm it isn't empty (would
    # otherwise fail client-side validation before any request is made).
    key = llm.langchain_llm.openai_api_key.get_secret_value()
    assert key  # non-empty


def test_embeddings_always_local_regardless_of_provider():
    s = Settings(_env_file=None, llm_provider="ollama", embed_model="BAAI/bge-small-en-v1.5")
    llm, emb = _build_ragas_llm_and_embeddings(s)
    assert isinstance(emb.embeddings, FakeHFEmbeddings)
    assert emb.embeddings.model_name == "BAAI/bge-small-en-v1.5"


def test_unknown_provider_raises():
    s = Settings(_env_file=None, llm_provider="bogus")
    with pytest.raises(ValueError, match="bogus"):
        _build_ragas_llm_and_embeddings(s)


# ---------- _run_traced: Langfuse wiring for eval runs ----------

class FakeLangfuseTrace:
    def __init__(self):
        self.ended_with: str | None = None

    def span(self, *, name, input=None):
        raise AssertionError("not used by this test")

    def generation(self, **kwargs):
        raise AssertionError("not used by this test")

    def update(self, *, output):
        self.ended_with = output


class FakeLangfuseClient:
    def __init__(self):
        self.trace_calls: list[dict] = []
        self.last_trace: FakeLangfuseTrace | None = None

    def trace(self, **kwargs):
        self.trace_calls.append(kwargs)
        self.last_trace = FakeLangfuseTrace()
        return self.last_trace


class FakeOrchestrator:
    def __init__(self, answer="the answer", raise_error=None):
        self._answer = answer
        self._raise = raise_error
        self.trace_active_during_run = "unset"

    async def run(self, question):
        # capture whether a trace is set as current while the orchestrator runs
        self.trace_active_during_run = _current_trace.get() is not None
        if self._raise:
            raise self._raise
        return {"answer": self._answer}


async def test_run_traced_creates_trace_and_ends_with_answer():
    client = FakeLangfuseClient()
    orch = FakeOrchestrator(answer="4F means low suction pressure.")

    import src.eval_runner as eval_runner
    from unittest.mock import patch
    with patch.object(eval_runner, "get_tracer", return_value=Tracer(client)):
        out = await _run_traced(orch, "what is 4F?", session_id="eval-abc123",
                                request_id="eval-abc123-q1")

    assert client.trace_calls == [{
        "name": "eval", "id": "eval-abc123-q1", "input": "what is 4F?",
        "session_id": "eval-abc123",
    }]
    assert client.last_trace.ended_with == "4F means low suction pressure."
    assert out == {"answer": "4F means low suction pressure."}
    assert orch.trace_active_during_run is True


async def test_run_traced_sets_and_resets_current_trace_around_run():
    client = FakeLangfuseClient()
    orch = FakeOrchestrator()

    assert _current_trace.get() is None
    import src.eval_runner as eval_runner
    from unittest.mock import patch
    with patch.object(eval_runner, "get_tracer", return_value=Tracer(client)):
        await _run_traced(orch, "q", session_id="s", request_id="r1")
    assert _current_trace.get() is None  # reset after the call


async def test_run_traced_resets_current_trace_even_on_error():
    client = FakeLangfuseClient()
    orch = FakeOrchestrator(raise_error=ValueError("boom"))

    import src.eval_runner as eval_runner
    from unittest.mock import patch
    with patch.object(eval_runner, "get_tracer", return_value=Tracer(client)):
        with pytest.raises(ValueError, match="boom"):
            await _run_traced(orch, "q", session_id="s", request_id="r1")
    assert _current_trace.get() is None  # reset despite the exception


async def test_run_traced_noop_without_tracing_configured():
    orch = FakeOrchestrator(answer="ok")

    import src.eval_runner as eval_runner
    from unittest.mock import patch
    with patch.object(eval_runner, "get_tracer", return_value=Tracer(None)):
        out = await _run_traced(orch, "q", session_id="s", request_id="r1")

    assert out == {"answer": "ok"}
