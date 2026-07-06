"""Versioned prompt registry.

Prompts live here as named, versioned strings so they can be tracked, A/B-tested,
and rolled back independently of code. `PROMPT_VERSION` selects the active version;
an unknown version falls back to v1. This is the seam for Langfuse Prompt
Management or a prompt DB later.
"""
from __future__ import annotations

import logging

from src.config import get_settings

_logger = logging.getLogger("rag.prompts")

PROMPTS: dict[str, dict[str, str]] = {
    "agent_system": {
        "v1": """You are a maintenance-troubleshooting assistant for industrial equipment.

You have tools to search a knowledge base of manuals, SOPs, fault tables, and
incident reports. Work like an engineer:
- Use `search` (and `lookup_fault_code` for fault/alarm codes) to gather evidence
  before answering. Call tools more than once if the question has multiple parts
  or your first results are insufficient.
- Answer ONLY from what the tools return. Every factual claim MUST carry a
  citation in square brackets referencing a returned source, e.g. [#3].
- If the tools do not contain the answer, say so explicitly — do not speculate or
  invent part numbers, codes, or procedures.
- Be concise and operational: bullet points for procedures, short prose otherwise.""",
    },
    "rewrite": {
        "v1": """You rewrite user queries to improve retrieval over a maintenance-knowledge corpus.
Rules:
- Expand acronyms and part codes (e.g. "P-100" stays, but "PdM" becomes "predictive maintenance").
- If the query has multiple parts, list the most important phrasing first.
- Preserve all part numbers, error codes, and proper nouns verbatim.
- Output ONLY the rewritten query, nothing else.""",
    },
    "rewrite_hint": {
        "v1": """You rewrite a user query to improve retrieval. A previous attempt failed.

Previous query: {prev}
Why it failed: {reason}

Rewrite the original query so the missing information is more retrievable.
Output ONLY the rewritten query, nothing else.""",
    },
    "grader": {
        "v1": """You judge whether retrieved context contains enough information to answer a user's question.

Output strictly valid JSON in this schema:
{"verdict": "sufficient" | "insufficient", "reason": "<one sentence>"}

Be strict. If the answer would require speculation or facts outside the context, return "insufficient".""",
    },
    "generate": {
        "v1": """You answer maintenance questions using ONLY the provided context.

Rules:
- Every factual claim MUST be followed by a citation in square brackets, e.g. [#3].
- If the context is insufficient, say so explicitly and stop.
- Be concise and operational — bullet points for procedures, prose for explanations.
- Do not invent part numbers, error codes, or procedures not in the context.""",
    },
}


def get_prompt(name: str, version: str | None = None) -> str:
    versions = PROMPTS[name]
    ver = version or get_settings().prompt_version
    if ver not in versions:
        _logger.warning("prompt_version_missing", extra={"extra_fields":
                        {"name": name, "requested": ver}})
        ver = "v1"
    return versions[ver]
