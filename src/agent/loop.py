"""Tool-calling agent loop.

The LLM reasons and decides which tools to call; the loop executes them, feeds
results back, and repeats until the model produces a final answer (or the step
limit is hit, in which case it is forced to answer without tools). Emits typed
events so the API can stream a live tool trace. Tool-deciding turns are
non-streaming; the final answer is chunked into `token` events (v1 limitation).
"""
from __future__ import annotations

import logging
from typing import AsyncIterator

from src.agent.prompts import AGENT_SYSTEM
from src.agent.tools import ToolRegistry
from src.citations import verify_citations
from src.config import get_settings
from src.guardrails import check_input
from src.observability import log_event, node_span
from src.providers.factory import get_provider
from src.providers.retry import call_with_retries

_logger = logging.getLogger("rag.agent")


def _chunk_tokens(text: str):
    """Split a finished answer into small pieces for token-style SSE."""
    for word in text.split(" "):
        yield word + " "


async def stream_agent(query, retriever, *, filters=None, settings=None, provider=None
                       ) -> AsyncIterator[dict]:
    s = settings or get_settings()
    provider = provider or get_provider()

    reason = check_input(query)
    if reason:
        yield {"type": "error", "message": f"query rejected: {reason}"}
        return

    registry = ToolRegistry(retriever, s)
    user_content = query
    if filters:
        user_content += f"\n\n(Apply these retrieval filters when searching: {filters})"
    messages = [{"role": "system", "content": AGENT_SYSTEM},
                {"role": "user", "content": user_content}]
    steps: list[dict] = []
    final_text = ""

    async def _chat(tools):
        async def make_call(override):
            return await provider.chat(messages, tools=tools,
                                       model=override or s.answer_model, max_tokens=1024)
        return await call_with_retries(make_call, max_attempts=s.llm_max_attempts)

    for step in range(s.agent_max_steps):
        with node_span(f"agent_step_{step}"):
            res = await _chat(registry.specs)
        if not res.wants_tools:
            final_text = res.text
            break
        messages.append({"role": "assistant", "content": res.text,
                         "tool_calls": res.tool_calls})
        for tc in res.tool_calls:
            yield {"type": "tool_call", "tool": tc.name, "args": tc.arguments}
            output = await registry.execute(tc.name, tc.arguments)
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "name": tc.name, "content": output})
            steps.append({"tool": tc.name, "args": tc.arguments})
            yield {"type": "tool_result", "tool": tc.name, "chars": len(output)}

    if not final_text:
        # step limit hit while still calling tools — force a final answer, no tools
        log_event(_logger, "agent_step_limit_forcing_answer", steps=len(steps))
        messages.append({"role": "user",
                         "content": "Provide your best final answer now using the "
                                    "information gathered. Cite sources as [#n]."})
        res = await _chat(None)
        final_text = res.text or "I could not complete the task within the step limit."

    for piece in _chunk_tokens(final_text):
        yield {"type": "token", "text": piece}

    citations = verify_citations(final_text, registry.num_sources)
    yield {"type": "final", "retrieval": list(registry.sources),
           "citations": citations, "steps": steps}


async def run_agent(query, retriever, *, filters=None, settings=None, provider=None) -> dict:
    """Collect the streamed events into a single result (for /ask and eval)."""
    answer_parts: list[str] = []
    final: dict | None = None
    error: str | None = None
    async for evt in stream_agent(query, retriever, filters=filters,
                                  settings=settings, provider=provider):
        t = evt["type"]
        if t == "token":
            answer_parts.append(evt["text"])
        elif t == "final":
            final = evt
        elif t == "error":
            error = evt["message"]
    if error is not None:
        return {"answer": "", "retrieval": [], "citations": {}, "steps": [], "error": error}
    return {
        "answer": "".join(answer_parts).strip(),
        "retrieval": final["retrieval"] if final else [],
        "citations": final["citations"] if final else {},
        "steps": final["steps"] if final else [],
    }
