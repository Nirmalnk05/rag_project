"""Mode dispatch: route queries to the tool-calling agent or the CRAG pipeline.

Both the API and the eval runner call this so the two modes share one entry
point and a common result shape.
"""
from __future__ import annotations

from typing import AsyncIterator

from src.agent.loop import run_agent, stream_agent
from src.config import get_settings
from src.graph import build_state_graph, run_graph, stream_answer


class Orchestrator:
    def __init__(self, retriever, *, settings=None, provider=None,
                 graph_full=None, graph_pre=None) -> None:
        self._r = retriever
        self._s = settings or get_settings()
        self._provider = provider
        self._graph_full = graph_full
        self._graph_pre = graph_pre

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

    async def run(self, query: str, filters: dict | None = None) -> dict:
        if self.mode == "agent":
            out = await run_agent(query, self._r, filters=filters,
                                  settings=self._s, provider=self._provider)
            out.setdefault("rewritten", "")
            out.setdefault("grade", "")
            return out
        state = await run_graph(self._full(), query, filters)
        return {
            "answer": state.get("answer", ""),
            "retrieval": state.get("retrieval", []),
            "citations": state.get("citations", {}),
            "steps": [],
            "rewritten": state.get("rewritten", ""),
            "grade": state.get("grade", ""),
        }

    async def stream(self, query: str, filters: dict | None = None) -> AsyncIterator[dict]:
        if self.mode == "agent":
            async for evt in stream_agent(query, self._r, filters=filters,
                                          settings=self._s, provider=self._provider):
                yield evt
        else:
            async for evt in stream_answer(self._pre(), query, filters):
                yield evt
