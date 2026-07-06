import pytest

from src.config import Settings
from src.orchestrator import Orchestrator
from src.providers.base import ChatResult, ToolCall
from src.retrieval import Retrieved


class FakeRetriever:
    def __init__(self, results=None):
        self._results = results or []
        self._chunks = []

    async def hybrid_retrieve(self, query, filters=None, access=None):
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


async def test_session_history_roundtrips_agent_mode():
    from src.memory import SessionStore
    retr = FakeRetriever([Retrieved("c1", "d", "P", "S", "4F low pressure", "", 0.9, "rerank")])
    # first turn asks a tool then answers; second turn answers directly
    provider = ScriptedProvider([
        ChatResult(tool_calls=[ToolCall("t1", "search", {"query": "4F"})]),
        ChatResult(text="Low pressure [#1]."),
        ChatResult(text="Check the strainer [#1]."),
    ])
    store = SessionStore(Settings(_env_file=None, enable_memory=True, memory_max_turns=5))
    orch = Orchestrator(retr, settings=Settings(_env_file=None, agent_mode="agent"),
                        provider=provider, store=store)
    await orch.run("what is 4F?", session_id="s1")
    # history now has the first exchange
    assert len(store.get_history("s1")) == 2
    await orch.run("how do I fix it?", session_id="s1")
    hist = store.get_history("s1")
    assert len(hist) == 4
    assert hist[0]["content"] == "what is 4F?"
    assert hist[1]["content"] == "Low pressure [#1]."


async def test_semantic_cache_serves_second_identical_query():
    from src.cache import SemanticCache

    class EmbedRetriever(FakeRetriever):
        def embed_query(self, query):
            return [1.0, 0.0]  # identical vector → cache hit on repeat

    retr = EmbedRetriever([Retrieved("c1", "d", "P", "S", "text", "", 0.9, "rerank")])
    provider = ScriptedProvider([ChatResult(text="the answer [#1]")])
    cache = SemanticCache(Settings(_env_file=None, enable_semantic_cache=True,
                                   semantic_cache_threshold=0.95))
    orch = Orchestrator(retr, settings=Settings(_env_file=None, agent_mode="agent",
                                                enable_semantic_cache=True),
                        provider=provider, cache=cache)
    out1 = await orch.run("what is 4F?")
    assert out1["cached"] is False and provider.calls == 1
    out2 = await orch.run("what is 4F?")   # served from cache
    assert out2["cached"] is True and out2["answer"] == "the answer [#1]"
    assert provider.calls == 1             # LLM not called again


async def test_cache_skipped_when_acl_on():
    from src.access import AccessContext
    from src.cache import SemanticCache

    class EmbedRetriever(FakeRetriever):
        def embed_query(self, query):
            return [1.0, 0.0]

    retr = EmbedRetriever([Retrieved("c1", "d", "P", "S", "t", "", 0.9, "rerank")])
    provider = ScriptedProvider([ChatResult(text="a1"), ChatResult(text="a2")])
    cache = SemanticCache(Settings(_env_file=None, enable_semantic_cache=True))
    orch = Orchestrator(retr, settings=Settings(_env_file=None, agent_mode="agent",
                                                enable_semantic_cache=True),
                        provider=provider, cache=cache)
    await orch.run("q", access=AccessContext(roles=frozenset({"tech"})))
    await orch.run("q", access=AccessContext(roles=frozenset({"tech"})))
    assert provider.calls == 2  # no cache reuse under ACL


async def test_no_session_id_does_not_store():
    from src.memory import SessionStore
    retr = FakeRetriever([Retrieved("c1", "d", "P", "S", "t", "", 0.9, "rerank")])
    provider = ScriptedProvider([ChatResult(text="answer")])
    store = SessionStore(Settings(_env_file=None, enable_memory=True))
    orch = Orchestrator(retr, settings=Settings(_env_file=None, agent_mode="agent"),
                        provider=provider, store=store)
    await orch.run("q")  # no session_id
    assert store.get_history("") == []


async def test_pipeline_mode_stream_delegates(monkeypatch):
    async def fake_stream_answer(graph, query, filters=None, access=None):
        yield {"type": "rewrite", "rewritten": "rw"}
        yield {"type": "final", "retrieval": []}

    monkeypatch.setattr("src.orchestrator.stream_answer", fake_stream_answer)
    orch = Orchestrator(FakeRetriever(), settings=Settings(_env_file=None, agent_mode="pipeline"),
                        graph_pre=object())
    types = [e["type"] async for e in orch.stream("q")]
    assert types == ["rewrite", "final"]
