"""Mode dispatch: route queries to the tool-calling agent or the CRAG pipeline.

Both the API and the eval runner call this so the two modes share one entry
point and a common result shape.
"""
from __future__ import annotations

import asyncio
import logging
import random
from typing import AsyncIterator

_logger = logging.getLogger("rag.eval")

from src.agent.loop import run_agent, stream_agent
from src.cache import cache_scope, get_cache
from src.config import get_settings
from src.faithfulness import check_faithfulness
from src.graph import build_state_graph, run_graph, stream_answer
from src.memory import get_store
from src.safety import assess_risk, safety_notice


def _with_history_preamble(query: str, history: list[dict]) -> str:
    """Pipeline mode has no native history; fold recent turns into the query."""
    if not history:
        return query
    convo = "\n".join(f"{t['role']}: {t['content']}" for t in history)
    return f"Previous conversation:\n{convo}\n\nCurrent question: {query}"


class Orchestrator:
    def __init__(self, retriever, *, settings=None, provider=None,
                 graph_full=None, graph_pre=None, store=None, cache=None) -> None:
        self._r = retriever
        self._s = settings or get_settings()
        self._provider = provider
        self._graph_full = graph_full
        self._graph_pre = graph_pre
        self._store = store if store is not None else get_store(self._s)
        self._cache = cache if cache is not None else get_cache(self._s)

    def _scope(self, access) -> str:
        """Partition key: what this principal can see, at this index generation.

        Read per request rather than cached: an ingest claims its generation
        before it touches Qdrant, so every scope is invalidated the moment the
        documents behind the cached answers start changing.
        """
        from src.ingestion import index_generation
        return cache_scope(access, index_generation())

    async def _cache_lookup(self, query: str, access):
        if not self._cache.enabled:
            return None
        try:
            vec = await asyncio.to_thread(self._r.embed_query, query)
        except Exception:
            return None
        return self._cache.get(vec, scope=self._scope(access))

    async def _cache_store(self, query: str, entry: dict, access) -> None:
        if not self._cache.enabled:
            return
        try:
            vec = await asyncio.to_thread(self._r.embed_query, query)
            self._cache.put(vec, entry, scope=self._scope(access))
        except Exception:
            pass

    @property
    def mode(self) -> str:
        return self._s.agent_mode

    async def _assess(self, query: str, answer: str, retrieval: list[dict]) -> dict:
        safety = (safety_notice(assess_risk(f"{query}\n{answer}"))
                  if self._s.enable_safety_escalation else "")
        faith = await check_faithfulness(answer, retrieval, provider=self._provider)
        return {"safety": safety, "faithfulness": faith}

    def _maybe_sample(self, query: str, out: dict) -> None:
        """Log a sampled fraction of answers for online-eval review."""
        if self._s.online_eval_sample_rate <= 0:
            return
        if random.random() < self._s.online_eval_sample_rate:
            from src.observability import log_event
            log_event(_logger, "answer_sampled", query=query[:200],
                      n_sources=len(out.get("retrieval", [])),
                      citations_valid=out.get("citations", {}).get("valid"),
                      grounded=out.get("faithfulness", {}).get("grounded"))

    def _full(self):
        if self._graph_full is None:
            self._graph_full = build_state_graph(self._r)
        return self._graph_full

    def _pre(self):
        if self._graph_pre is None:
            self._graph_pre = build_state_graph(self._r, include_generate=False)
        return self._graph_pre

    async def run(self, query: str, filters: dict | None = None,
                  session_id: str = "", access=None) -> dict:
        history = self._store.get_history(session_id)
        cacheable = not history and not filters
        if cacheable:
            hit = await self._cache_lookup(query, access)
            if hit is not None:
                out = {**hit, "cached": True}
                if session_id and out.get("answer"):
                    self._store.append(session_id, query, out["answer"])
                return out
        if self.mode == "agent":
            out = await run_agent(query, self._r, filters=filters,
                                  settings=self._s, provider=self._provider,
                                  history=history, access=access)
            out.setdefault("rewritten", "")
            out.setdefault("grade", "")
        else:
            state = await run_graph(self._full(),
                                    _with_history_preamble(query, history), filters, access)
            out = {
                "answer": state.get("answer", ""),
                "retrieval": state.get("retrieval", []),
                "citations": state.get("citations", {}),
                "steps": [],
                "rewritten": state.get("rewritten", ""),
                "grade": state.get("grade", ""),
            }
        out.update(await self._assess(query, out.get("answer", ""), out.get("retrieval", [])))
        out["cached"] = False
        self._maybe_sample(query, out)
        if cacheable and out.get("answer"):
            await self._cache_store(query, {k: v for k, v in out.items() if k != "cached"}, access)
        if session_id and out.get("answer"):
            self._store.append(session_id, query, out["answer"])
        return out

    async def stream(self, query: str, filters: dict | None = None,
                     session_id: str = "", access=None) -> AsyncIterator[dict]:
        history = self._store.get_history(session_id)
        cacheable = not history and not filters
        if cacheable:
            hit = await self._cache_lookup(query, access)
            if hit is not None:
                for word in (hit.get("answer", "") + " ").split(" "):
                    if word:
                        yield {"type": "token", "text": word + " "}
                yield {"type": "final", "retrieval": hit.get("retrieval", []),
                       "citations": hit.get("citations", {}), "steps": hit.get("steps", []),
                       "safety": hit.get("safety", ""), "faithfulness": hit.get("faithfulness", {}),
                       "cached": True}
                if session_id and hit.get("answer"):
                    self._store.append(session_id, query, hit["answer"])
                return

        answer_parts: list[str] = []
        if self.mode == "agent":
            gen = stream_agent(query, self._r, filters=filters, settings=self._s,
                               provider=self._provider, history=history, access=access)
        else:
            gen = stream_answer(self._pre(), _with_history_preamble(query, history),
                                filters, access)
        async for evt in gen:
            if evt.get("type") == "token":
                answer_parts.append(evt.get("text", ""))
            if evt.get("type") == "final":
                answer = "".join(answer_parts).strip()
                evt.update(await self._assess(query, answer, evt.get("retrieval", [])))
                if cacheable and answer:
                    await self._cache_store(query, {
                        "answer": answer, "retrieval": evt.get("retrieval", []),
                        "citations": evt.get("citations", {}), "steps": evt.get("steps", []),
                        "safety": evt.get("safety", ""),
                        "faithfulness": evt.get("faithfulness", {})}, access)
            yield evt
        if session_id and answer_parts:
            self._store.append(session_id, query, "".join(answer_parts).strip())
