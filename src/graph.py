"""LangGraph state machine for agentic RAG (fully async).

Nodes: rewrite → retrieve (hybrid, guardrail-screened) → grade → generate
Grade routes back through decrement_retry → rewrite while retries remain.

One wiring function builds the graph; include_generate=False compiles the
same wiring minus the generate node — used by the streaming path, which
streams generation tokens itself. (This replaces interrupt_before, which
would require a checkpointer.)
"""
from __future__ import annotations

import json
import logging
from typing import AsyncIterator, Literal, TypedDict

from langgraph.graph import END, StateGraph

from src.citations import verify_citations
from src.config import get_settings
from src.guardrails import screen_chunks
from src.llm import complete, stream_complete
from src.observability import log_event, node_span
from src.retrieval import Retriever

_settings = get_settings()
_logger = logging.getLogger("rag.graph")


class GraphState(TypedDict, total=False):
    query: str
    rewritten: str
    retries: int
    filters: dict
    access: object
    retrieval: list[dict]
    grade: str
    grade_reason: str
    answer: str
    citations: dict


from src.prompts import get_prompt

REWRITE_SYSTEM = get_prompt("rewrite")
REWRITE_WITH_HINT_SYSTEM = get_prompt("rewrite_hint")
GRADER_SYSTEM = get_prompt("grader")
GENERATE_SYSTEM = get_prompt("generate")


# ---------- node functions ----------

def _make_rewrite_node():
    async def rewrite_node(state: GraphState) -> GraphState:
        with node_span("rewrite"):
            if state.get("grade") == "insufficient" and state.get("rewritten"):
                sys = REWRITE_WITH_HINT_SYSTEM.format(
                    prev=state["rewritten"],
                    reason=state.get("grade_reason", "context did not match query intent"),
                )
            else:
                sys = REWRITE_SYSTEM
            try:
                res = await complete(state["query"], system=sys,
                                     model=_settings.grader_model, max_tokens=200)
                new_q = res.text.strip() or state["query"]
            except Exception:  # degrade: retrieval on the original query
                _logger.warning("rewrite_failed_using_original", exc_info=True)
                new_q = state["query"]
            return {"rewritten": new_q}
    return rewrite_node


def _make_retrieve_node(retriever: Retriever):
    async def retrieve_node(state: GraphState) -> GraphState:
        with node_span("retrieve"):
            q = state.get("rewritten") or state["query"]
            results = await retriever.hybrid_retrieve(
                q, state.get("filters") or None, state.get("access"))
            dicts = [r.to_dict() for r in results]
            clean, flagged = screen_chunks(dicts)
            if flagged:
                log_event(_logger, "chunk_flagged",
                          count=len(flagged), ids=[c["chunk_id"] for c in flagged])
            return {"retrieval": clean}
    return retrieve_node


def _make_grade_node():
    async def grade_node(state: GraphState) -> GraphState:
        with node_span("grade"):
            ctx = "\n\n".join(
                f"[{i+1}] ({r['doc_title']} > {r['section']})\n{r['text']}"
                for i, r in enumerate(state.get("retrieval", []))
            )
            prompt = (
                f"Original question: {state['query']}\n"
                f"Rewritten search query: {state.get('rewritten', state['query'])}\n\n"
                f"Context:\n{ctx}\n\nIs this context sufficient to answer the original question?"
            )
            try:
                raw = (await complete(prompt, system=GRADER_SYSTEM,
                                      model=_settings.grader_model, max_tokens=200)).text.strip()
            except Exception:  # degrade: proceed low-confidence, no retry burn
                _logger.warning("grader_failed_assuming_insufficient", exc_info=True)
                return {"grade": "insufficient", "grade_reason": "grader unavailable", "retries": 0}
            try:
                parsed = json.loads(raw)
                verdict = parsed.get("verdict", "insufficient")
                reason = parsed.get("reason", "")
            except json.JSONDecodeError:
                verdict = ("sufficient"
                           if "sufficient" in raw.lower() and "insuff" not in raw.lower()
                           else "insufficient")
                reason = raw[:200]
            return {"grade": verdict, "grade_reason": reason}
    return grade_node


def _generation_prompt(state: GraphState) -> str:
    ctx = _format_context(state.get("retrieval", []))
    note = ""
    if state.get("grade") == "insufficient":
        note = ("\n\nNote: the grader flagged the context as possibly insufficient. "
                "Answer best-effort and explicitly call out gaps.")
    return f"Question: {state['query']}\n\nContext:\n{ctx}{note}\n\nAnswer:"


def _make_generate_node():
    async def generate_node(state: GraphState) -> GraphState:
        with node_span("generate"):
            res = await complete(_generation_prompt(state), system=GENERATE_SYSTEM,
                                 model=_settings.answer_model, max_tokens=1024)
            answer = res.text.strip()
            citations = verify_citations(answer, len(state.get("retrieval", [])))
            return {"answer": answer, "citations": citations}
    return generate_node


# ---------- routing ----------

def _route_after_grade(state: GraphState) -> Literal["rewrite", "generate"]:
    if state.get("grade") == "sufficient":
        return "generate"
    if state.get("retries", 0) > 0:
        return "rewrite"
    return "generate"


async def _decrement_retry_node(state: GraphState) -> GraphState:
    return {"retries": state.get("retries", 0) - 1}


def _format_context(retrieval: list[dict]) -> str:
    return "\n\n".join(
        f"[#{i+1}] (source: {r.get('doc_title','?')} > {r.get('section','?')})\n{r['text']}"
        for i, r in enumerate(retrieval)
    )


# ---------- build & run ----------

def build_state_graph(retriever: Retriever, *, include_generate: bool = True):
    g = StateGraph(GraphState)
    g.add_node("rewrite", _make_rewrite_node())
    g.add_node("retrieve", _make_retrieve_node(retriever))
    g.add_node("grader", _make_grade_node())
    g.add_node("decrement_retry", _decrement_retry_node)
    if include_generate:
        g.add_node("generate", _make_generate_node())

    g.set_entry_point("rewrite")
    g.add_edge("rewrite", "retrieve")
    g.add_edge("retrieve", "grader")
    g.add_conditional_edges(
        "grader", _route_after_grade,
        {"rewrite": "decrement_retry",
         "generate": "generate" if include_generate else END},
    )
    g.add_edge("decrement_retry", "rewrite")
    if include_generate:
        g.add_edge("generate", END)
    return g.compile()


async def run_graph(graph, query: str, filters: dict | None = None, access=None) -> GraphState:
    return await graph.ainvoke({
        "query": query, "retries": _settings.max_retries,
        "filters": filters or {}, "access": access,
    })


# ---------- sync + streaming helpers ----------

_graph_cache: dict[int, object] = {}


def run_sync(retriever: Retriever, query: str) -> GraphState:
    """One-shot synchronous run (eval scripts). Caches the compiled graph."""
    import asyncio
    g = _graph_cache.get(id(retriever))
    if g is None:
        g = _graph_cache[id(retriever)] = build_state_graph(retriever)
    return asyncio.run(run_graph(g, query))


async def stream_answer(pre_generate_graph, query: str, filters: dict | None = None,
                        access=None) -> AsyncIterator[dict]:
    """Run rewrite/retrieve/grade via the pre-generate graph, then STREAM the
    final answer. Yields typed event dicts; 'final' carries retrieval + citations."""
    state = await run_graph(pre_generate_graph, query, filters, access)

    yield {"type": "rewrite", "rewritten": state.get("rewritten", "")}
    yield {"type": "grade", "verdict": state.get("grade", ""),
           "reason": state.get("grade_reason", "")}

    chunks_text: list[str] = []
    async for token in stream_complete(_generation_prompt(state), system=GENERATE_SYSTEM,
                                       model=_settings.answer_model, max_tokens=1024):
        chunks_text.append(token)
        yield {"type": "token", "text": token}

    answer = "".join(chunks_text)
    citations = verify_citations(answer, len(state.get("retrieval", [])))
    yield {"type": "final", "retrieval": state.get("retrieval", []),
           "citations": citations}
