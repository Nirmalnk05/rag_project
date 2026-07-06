"""Mode dispatch: route queries to the tool-calling agent or the CRAG pipeline.

Both the API and the eval runner call this so the two modes share one entry
point and a common result shape.
"""
from __future__ import annotations

from typing import AsyncIterator

from src.agent.loop import run_agent, stream_agent
from src.config import get_settings
from src.graph import build_state_graph, run_graph, stream_answer
from src.memory import get_store


def _with_history_preamble(query: str, history: list[dict]) -> str:
    """Pipeline mode has no native history; fold recent turns into the query."""
    if not history:
        return query
    convo = "\n".join(f"{t['role']}: {t['content']}" for t in history)
    return f"Previous conversation:\n{convo}\n\nCurrent question: {query}"


class Orchestrator:
    def __init__(self, retriever, *, settings=None, provider=None,
                 graph_full=None, graph_pre=None, store=None) -> None:
        self._r = retriever
        self._s = settings or get_settings()
        self._provider = provider
        self._graph_full = graph_full
        self._graph_pre = graph_pre
        self._store = store if store is not None else get_store(self._s)

    @property
    def mode(self) -> str:
        return self._s.agent_mode

    def _full(self):
        if self._graph_full is None:
            self._graph_full = build_state_graph(self._r)
        return self._graph_full

    def _pre(self):
        if self._graph_pre is None:
            self._graph_pre = build_state_graph(self._r, include_generate=False)
        return self._graph_pre

    async def run(self, query: str, filters: dict | None = None,
                  session_id: str = "") -> dict:
        history = self._store.get_history(session_id)
        if self.mode == "agent":
            out = await run_agent(query, self._r, filters=filters,
                                  settings=self._s, provider=self._provider,
                                  history=history)
            out.setdefault("rewritten", "")
            out.setdefault("grade", "")
        else:
            state = await run_graph(self._full(),
                                    _with_history_preamble(query, history), filters)
            out = {
                "answer": state.get("answer", ""),
                "retrieval": state.get("retrieval", []),
                "citations": state.get("citations", {}),
                "steps": [],
                "rewritten": state.get("rewritten", ""),
                "grade": state.get("grade", ""),
            }
        if session_id and out.get("answer"):
            self._store.append(session_id, query, out["answer"])
        return out

    async def stream(self, query: str, filters: dict | None = None,
                     session_id: str = "") -> AsyncIterator[dict]:
        history = self._store.get_history(session_id)
        answer_parts: list[str] = []
        if self.mode == "agent":
            gen = stream_agent(query, self._r, filters=filters, settings=self._s,
                               provider=self._provider, history=history)
        else:
            gen = stream_answer(self._pre(), _with_history_preamble(query, history), filters)
        async for evt in gen:
            if evt.get("type") == "token":
                answer_parts.append(evt.get("text", ""))
            yield evt
        if session_id and answer_parts:
            self._store.append(session_id, query, "".join(answer_parts).strip())
