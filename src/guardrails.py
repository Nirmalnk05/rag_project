"""Heuristic guardrails: input validation, indirect-injection screening of
retrieved chunks, and PII redaction.

Deliberately regex-based. This module is the integration point for
NeMo Guardrails / LlamaGuard if you outgrow heuristics.
"""
from __future__ import annotations

import re

_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"ignore\s+(?:all\s+|any\s+)?(?:previous|prior|above)\s+(?:instructions|context|prompts?)",
        r"disregard\s+(?:the|your|all)\s+(?:system\s+prompt|instructions?)",
        r"forget\s+(?:everything|all\s+(?:previous\s+)?instructions?)",
        r"you\s+are\s+now\s+(?:a|an|in|DAN)\b",
        r"</?\s*system\s*>",
        r"\bDAN\s+mode\b",
        r"reveal\s+(?:your|the)\s+(?:system\s+)?prompt",
    ]
]

_PII_PATTERNS = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    # 8+ digits with optional +, spaces or dashes between — avoids part numbers
    # like MSC-100-A (letters) and P-100 (too few digits).
    "phone": re.compile(r"(?<![\w-])\+?\d[\d\s-]{6,}\d(?![\w-])"),
}


def check_input(query: str) -> str | None:
    """Return a rejection reason, or None if the query is acceptable."""
    if any(ord(c) < 32 and c not in "\n\r\t" for c in query):
        return "query contains control characters"
    for pat in _INJECTION_PATTERNS:
        if pat.search(query):
            return "query matches a prompt-injection pattern"
    return None


def screen_chunks(chunks: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split retrieved chunks into (clean, flagged) by injection heuristics."""
    clean, flagged = [], []
    for c in chunks:
        text = c.get("text", "")
        if any(p.search(text) for p in _INJECTION_PATTERNS):
            flagged.append(c)
        else:
            clean.append(c)
    return clean, flagged


def redact_pii(text: str) -> tuple[str, dict[str, int]]:
    """Redact emails/phones; returns (redacted_text, {kind: count})."""
    found: dict[str, int] = {}
    out = text
    for kind, pat in _PII_PATTERNS.items():
        out, n = pat.subn(f"[REDACTED-{kind.upper()}]", out)
        if n:
            found[kind] = n
    return out, found
