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

    async def hybrid_retrieve(self, query, filters=None):
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
    provider = ScriptedProvider([
        ChatResult(tool_calls=[ToolCall("t1", "search", {"query": "4F"})]),
        ChatResult(text="Code 4F means low suction pressure [#1]."),
    ])
    app.state.retriever = FakeRetriever()
    app.state.orchestrator = Orchestrator(
        app.state.retriever, settings=Settings(_env_file=None, agent_mode="agent"),
        provider=provider)
    return TestClient(app, raise_server_exceptions=False)


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
