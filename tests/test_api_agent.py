import json

import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.config import Settings
from src.orchestrator import Orchestrator
from src.providers.base import ChatResult, ToolCall
from src.retrieval import Retrieved
from src.security import _reset_buckets


class FakeRetriever:
    def __init__(self):
        self._results = [Retrieved("c1", "d", "Pump Manual", "Alarms",
                                   "Code 4F = low suction pressure", "", 0.9, "rerank")]
        self._chunks = []

    async def hybrid_retrieve(self, query, filters=None, access=None):
        return self._results


class ScriptedProvider:
    def __init__(self, script):
        self._script, self.calls = list(script), 0

    async def chat(self, messages, *, tools=None, model, max_tokens=1024, temperature=0.0):
        r = self._script[min(self.calls, len(self._script) - 1)]
        self.calls += 1
        return r


@pytest.fixture
def agent_client():
    _reset_buckets()
    from src import memory
    provider = ScriptedProvider([
        ChatResult(tool_calls=[ToolCall("t1", "search", {"query": "4F"})]),
        ChatResult(text="Code 4F means low suction pressure [#1]."),
        ChatResult(text="Replace the strainer [#1]."),
    ])
    settings = Settings(_env_file=None, agent_mode="agent", enable_memory=True)
    store = memory.SessionStore(settings)
    memory.set_store(store)  # so /session/reset hits the same store
    app.state.retriever = FakeRetriever()
    app.state.orchestrator = Orchestrator(
        app.state.retriever, settings=settings, provider=provider, store=store)
    client = TestClient(app, raise_server_exceptions=False)
    client._store = store  # expose for assertions
    yield client
    memory.set_store(None)


def test_agent_ask_returns_answer_steps_citations(agent_client):
    r = agent_client.post("/ask", json={"query": "what is fault 4F?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "Code 4F means low suction pressure [#1]."
    assert body["steps"] == [{"tool": "search", "args": {"query": "4F"}}]
    assert body["citations"]["valid"] is True
    assert len(body["retrieval"]) == 1
    assert body["request_id"]


def test_agent_stream_emits_tool_events(agent_client):
    r = agent_client.post("/ask/stream", json={"query": "what is fault 4F?"})
    assert r.status_code == 200
    events = [json.loads(l[6:]) for l in r.text.splitlines()
              if l.startswith("data: ") and l != "data: [DONE]"]
    types = [e["type"] for e in events]
    assert "tool_call" in types and "tool_result" in types
    assert "token" in types and types[-1] == "final"
    tool_evt = next(e for e in events if e["type"] == "tool_call")
    assert tool_evt["tool"] == "search"
    assert events[-1]["request_id"]


def test_multiturn_builds_history_and_echoes_session(agent_client):
    r1 = agent_client.post("/ask", json={"query": "what is fault 4F?", "session_id": "sess-1"})
    assert r1.status_code == 200 and r1.json()["session_id"] == "sess-1"
    assert len(agent_client._store.get_history("sess-1")) == 2

    r2 = agent_client.post("/ask", json={"query": "how do I fix it?", "session_id": "sess-1"})
    assert r2.status_code == 200
    hist = agent_client._store.get_history("sess-1")
    assert len(hist) == 4 and hist[0]["content"] == "what is fault 4F?"


def test_electrical_query_gets_safety_notice(agent_client):
    r = agent_client.post("/ask", json={"query": "how to fix the 440V earth fault?"})
    assert r.status_code == 200
    body = r.json()
    assert "Safety notice" in body["safety"]
    assert "faithfulness" in body


def test_routine_query_no_safety(agent_client):
    r = agent_client.post("/ask", json={"query": "how often to clean condenser fins?"})
    assert r.json()["safety"] == ""


def test_session_reset_clears_history(agent_client):
    agent_client.post("/ask", json={"query": "what is fault 4F?", "session_id": "sess-2"})
    assert agent_client._store.get_history("sess-2")
    rr = agent_client.post("/session/reset", json={"session_id": "sess-2"})
    assert rr.status_code == 200 and rr.json()["status"] == "reset"
    assert agent_client._store.get_history("sess-2") == []
