"""LangGraph state machine for agentic RAG.

Nodes:
  rewrite      → expand acronyms, decompose, generate alternative phrasing
  retrieve     → hybrid (BM25 + dense) → RRF → cross-encoder rerank
  grade        → is the retrieved context sufficient? if not, route back to rewrite
  generate     → grounded generation with strict citation prompt

Edges:
  start → rewrite → retrieve → grade
  grade → (sufficient) → generate → END
  grade → (insufficient AND retries left) → rewrite
  grade → (insufficient AND no retries) → generate (with explicit "low confidence" hint)
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator, Literal, TypedDict

from langgraph.graph import END, StateGraph

from src.config import get_settings
from src.llm import complete, stream_complete
from src.retrieval import Retrieved, Retriever


_settings = get_settings()


# ---------- state ----------

class GraphState(TypedDict, total=False):
    query: str              # original user query
    rewritten: str          # current rewritten query
    retries: int            # remaining retries
    retrieval: list[dict]   # serialized Retrieved items
    grade: str              # 'sufficient' | 'insufficient'
    grade_reason: str       # for prompting next rewrite
    answer: str             # final answer (filled in by generate)


# ---------- prompts ----------

REWRITE_SYSTEM = """You rewrite user queries to improve retrieval over a maintenance-knowledge corpus.
Rules:
- Expand acronyms and part codes (e.g. "P-100" stays, but "PdM" becomes "predictive maintenance").
- If the query has multiple parts, list the most important phrasing first.
- Preserve all part numbers, error codes, and proper nouns verbatim.
- Output ONLY the rewritten query, nothing else."""


REWRITE_WITH_HINT_SYSTEM = """You rewrite a user query to improve retrieval. A previous attempt failed.

Previous query: {prev}
Why it failed: {reason}

Rewrite the original query so the missing information is more retrievable.
Output ONLY the rewritten query, nothing else."""


GRADER_SYSTEM = """You judge whether retrieved context contains enough information to answer a user's question.

Output strictly valid JSON in this schema:
{"verdict": "sufficient" | "insufficient", "reason": "<one sentence>"}

Be strict. If the answer would require speculation or facts outside the context, return "insufficient"."""


GENERATE_SYSTEM = """You answer maintenance questions using ONLY the provided context.

Rules:
- Every factual claim MUST be followed by a citation in square brackets, e.g. [#3].
- If the context is insufficient, say so explicitly and stop.
- Be concise and operational — bullet points for procedures, prose for explanations.
- Do not invent part numbers, error codes, or procedures not in the context."""


# ---------- node functions ----------

def _make_rewrite_node(retriever: Retriever):
    def rewrite_node(state: GraphState) -> GraphState:
        if state.get("grade") == "insufficient" and state.get("rewritten"):
            sys = REWRITE_WITH_HINT_SYSTEM.format(
                prev=state["rewritten"],
                reason=state.get("grade_reason", "context did not match query intent"),
            )
            new_q = complete(
                state["query"],
                system=sys,
                model=_settings.grader_model,
                max_tokens=200,
            ).strip()
        else:
            new_q = complete(
                state["query"],
                system=REWRITE_SYSTEM,
                model=_settings.grader_model,
                max_tokens=200,
            ).strip()
        return {"rewritten": new_q}

    return rewrite_node


def _make_retrieve_node(retriever: Retriever):
    def retrieve_node(state: GraphState) -> GraphState:
        q = state.get("rewritten") or state["query"]
        results = retriever.hybrid_retrieve(q)
        return {"retrieval": [r.to_dict() for r in results]}

    return retrieve_node


def _make_grade_node():
    def grade_node(state: GraphState) -> GraphState:
        ctx = "\n\n".join(
            f"[{i+1}] ({r['doc_title']} > {r['section']})\n{r['text']}"
            for i, r in enumerate(state.get("retrieval", []))
        )
        prompt = f"Question: {state['query']}\n\nContext:\n{ctx}\n\nIs this context sufficient to answer?"
        raw = complete(
            prompt,
            system=GRADER_SYSTEM,
            model=_settings.grader_model,
            max_tokens=200,
        ).strip()
        # tolerant JSON parse
        try:
            parsed = json.loads(raw)
            verdict = parsed.get("verdict", "insufficient")
            reason = parsed.get("reason", "")
        except json.JSONDecodeError:
            verdict = "sufficient" if "sufficient" in raw.lower() and "insuff" not in raw.lower() else "insufficient"
            reason = raw[:200]
        return {"grade": verdict, "grade_reason": reason}

    return grade_node


def _make_generate_node():
    """Non-streaming generate — used in the sync graph. The API uses stream_answer below."""
    def generate_node(state: GraphState) -> GraphState:
        ctx = _format_context(state.get("retrieval", []))
        confidence_note = ""
        if state.get("grade") == "insufficient":
            confidence_note = "\n\nNote: the grader flagged the context as possibly insufficient. Answer best-effort and explicitly call out gaps."
        prompt = f"Question: {state['query']}\n\nContext:\n{ctx}{confidence_note}\n\nAnswer:"
        ans = complete(
            prompt,
            system=GENERATE_SYSTEM,
            model=_settings.answer_model,
            max_tokens=1024,
        ).strip()
        return {"answer": ans}

    return generate_node


# ---------- routing ----------

def _route_after_grade(state: GraphState) -> Literal["rewrite", "generate"]:
    if state.get("grade") == "sufficient":
        return "generate"
    if state.get("retries", 0) > 0:
        return "rewrite"
    return "generate"


def _decrement_retry_node(state: GraphState) -> GraphState:
    # Called only when routing back to rewrite after a failed grade
    return {"retries": state.get("retries", 0) - 1}


# ---------- helpers ----------

def _format_context(retrieval: list[dict]) -> str:
    return "\n\n".join(
        f"[#{i+1}] (source: {r.get('doc_title','?')} > {r.get('section','?')})\n{r['text']}"
        for i, r in enumerate(retrieval)
    )


# ---------- build graph ----------

def build_graph(retriever: Retriever):
    g = StateGraph(GraphState)
    g.add_node("rewrite", _make_rewrite_node(retriever))
    g.add_node("retrieve", _make_retrieve_node(retriever))
    g.add_node("grade", _make_grade_node())
    g.add_node("decrement_retry", _decrement_retry_node)
    g.add_node("generate", _make_generate_node())

    g.set_entry_point("rewrite")
    g.add_edge("rewrite", "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges(
        "grade",
        _route_after_grade,
        {"rewrite": "decrement_retry", "generate": "generate"},
    )
    g.add_edge("decrement_retry", "rewrite")
    g.add_edge("generate", END)
    return g.compile()


# ---------- API helpers ----------

def run_sync(retriever: Retriever, query: str) -> GraphState:
    """One-shot synchronous run. Returns the final state including answer + retrieval."""
    graph = build_graph(retriever)
    initial: GraphState = {"query": query, "retries": _settings.max_retries}
    return graph.invoke(initial)


async def stream_answer(
    retriever: Retriever, query: str
) -> AsyncIterator[dict]:
    """Run the graph synchronously up to (but not including) generate, then STREAM the final answer.

    Yields dicts: {'type': 'state', ...} for intermediate updates, {'type': 'token', 'text': ...} for the answer,
    and {'type': 'final', 'retrieval': [...]} at the end.
    """
    # Build a graph WITHOUT the generate node so we get retrieval + grading, then stream separately.
    g = StateGraph(GraphState)
    g.add_node("rewrite", _make_rewrite_node(retriever))
    g.add_node("retrieve", _make_retrieve_node(retriever))
    g.add_node("grade", _make_grade_node())
    g.add_node("decrement_retry", _decrement_retry_node)
    g.set_entry_point("rewrite")
    g.add_edge("rewrite", "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges(
        "grade",
        _route_after_grade,
        {"rewrite": "decrement_retry", "generate": END},
    )
    g.add_edge("decrement_retry", "rewrite")
    compiled = g.compile()

    state = compiled.invoke({"query": query, "retries": _settings.max_retries})

    yield {"type": "rewrite", "rewritten": state.get("rewritten", "")}
    yield {"type": "grade", "verdict": state.get("grade", ""), "reason": state.get("grade_reason", "")}

    ctx = _format_context(state.get("retrieval", []))
    confidence_note = ""
    if state.get("grade") == "insufficient":
        confidence_note = "\n\nNote: the grader flagged the context as possibly insufficient. Answer best-effort and explicitly call out gaps."
    prompt = f"Question: {query}\n\nContext:\n{ctx}{confidence_note}\n\nAnswer:"

    async for token in stream_complete(
        prompt,
        system=GENERATE_SYSTEM,
        model=_settings.answer_model,
        max_tokens=1024,
    ):
        yield {"type": "token", "text": token}

    yield {"type": "final", "retrieval": state.get("retrieval", [])}
