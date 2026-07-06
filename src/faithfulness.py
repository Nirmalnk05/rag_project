"""Post-generation groundedness check (LLM-as-judge).

Citation verification checks that `[#n]` markers *exist*; this checks that the
answer's claims are actually *supported* by the retrieved sources. One extra LLM
call (gated by `enable_faithfulness_check`); non-blocking — the verdict is attached
to the response for the client/UI to act on.
"""
from __future__ import annotations

import json
import logging

from src.config import get_settings

_logger = logging.getLogger("rag.faithfulness")

JUDGE_SYSTEM = """You are a strict grader. Given an ANSWER and the SOURCES it must
be grounded in, decide whether every factual claim in the answer is supported by
the sources.

Output strictly valid JSON:
{"grounded": true|false, "unsupported": ["<claim>", ...]}

A claim is unsupported if the sources do not state it (even if it's plausible).
If the answer is a refusal or says the context is insufficient, it is grounded."""


def build_judge_prompt(answer: str, sources: list[dict]) -> str:
    ctx = "\n\n".join(
        f"[#{i+1}] {s.get('text', '')}" for i, s in enumerate(sources)
    )
    return f"SOURCES:\n{ctx}\n\nANSWER:\n{answer}\n\nIs the answer grounded?"


def parse_faithfulness(raw: str) -> dict:
    try:
        parsed = json.loads(raw)
        return {"grounded": bool(parsed.get("grounded", False)),
                "unsupported": list(parsed.get("unsupported", []))}
    except (json.JSONDecodeError, AttributeError):
        # tolerant fallback: assume grounded only if it clearly says so
        low = raw.lower()
        return {"grounded": "true" in low and "false" not in low, "unsupported": []}


async def check_faithfulness(answer: str, sources: list[dict], *, provider=None) -> dict:
    """Returns {grounded, unsupported}. If disabled or no sources, returns
    a benign 'grounded' verdict (nothing to contradict)."""
    s = get_settings()
    if not s.enable_faithfulness_check or not sources or not answer.strip():
        return {"grounded": True, "unsupported": [], "checked": False}
    from src.llm import complete
    try:
        raw = (await complete(build_judge_prompt(answer, sources),
                              system=JUDGE_SYSTEM, model=s.grader_model,
                              max_tokens=300)).text.strip()
    except Exception:  # judge failure must not break the answer
        _logger.warning("faithfulness_judge_failed", exc_info=True)
        return {"grounded": True, "unsupported": [], "checked": False}
    out = parse_faithfulness(raw)
    out["checked"] = True
    return out
