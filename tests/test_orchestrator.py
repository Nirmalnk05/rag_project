import pytest

from src.config import Settings
from src.orchestrator import Orchestrator
from src.providers.base import ChatResult, ToolCall
from src.retrieval import Retrieved


class FakeRetriever:
    def __init__(self, results=None):
        self._results = results or []
        self._chunks = []

    async def hybrid_retrieve(self, query, filters=None):
        return self._results


class ScriptedProvider:
    def __init__(self, script):
        self._script = list(script)
        self.calls = 0

    async def chat(self, messages, *, tools=None, model, max_tokens=1024, temperature=0.0):
        r = self._script[min(self.calls, len(self._script) - 1)]
        self.calls += 1
        return r


class FakeGraph:
    def __init__(self, state):
        self._state = state

    async def ainvoke(self, initial):
        return {**initial, **self._state}


async def test_agent_mode_run():
    retr = FakeRetriever([Retrieved("c1", "d", "P", "S", "4F low pressure", "", 0.9, "rerank")])
    provider = ScriptedProvider([
        ChatResult(tool_calls=[ToolCall("t1", "search", {"query": "4F"})]),
        ChatResult(text="Low pressure [#1]."),
    ])
    orch = Orchestrator(retr, settings=Settings(_env_file=None, agent_mode="agent"),
                        provider=provider)
    out = await orch.run("what is 4F?")
    assert out["answer"] == "Low pressure [#1]." and out["steps"][0]["tool"] == "search"
    assert out["citations"]["valid"] is True
    assert "rewritten" in out and "grade" in out  # common shape


async def test_pipeline_mode_run():
    state = {"answer": "piped answer", "grade": "sufficient", "rewritten": "rw",
             "retrieval": [{"text": "x"}], "citations": {"valid": True}}
    orch = Orchestrator(FakeRetriever(), settings=Settings(_env_file=None, agent_mode="pipeline"),
                        graph_full=FakeGraph(state))
    out = await orch.run("q")
    assert out["answer"] == "piped answer" and out["grade"] == "sufficient"
    assert out["steps"] == []


async def test_agent_mode_stream_yields_events():
    retr = FakeRetriever([Retrieved("c1", "d", "P", "S", "t", "", 0.9, "rerank")])
    provider = ScriptedProvider([
        ChatResult(tool_calls=[ToolCall("t1", "search", {"query": "x"})]),
        ChatResult(text="done [#1]"),
    ])
    orch = Orchestrator(retr, settings=Settings(_env_file=None, agent_mode="agent"),
                        provider=provider)
    types = [e["type"] async for e in orch.stream("q")]
    assert "tool_call" in types and types[-1] == "final"


async def test_pipeline_mode_stream_delegates(monkeypatch):
    async def fake_stream_answer(graph, query, filters=None):
        yield {"type": "rewrite", "rewritten": "rw"}
        yield {"type": "final", "retrieval": []}

    monkeypatch.setattr("src.orchestrator.stream_answer", fake_stream_answer)
    orch = Orchestrator(FakeRetriever(), settings=Settings(_env_file=None, agent_mode="pipeline"),
                        graph_pre=object())
    types = [e["type"] async for e in orch.stream("q")]
    assert types == ["rewrite", "final"]
