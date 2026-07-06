"""Safety escalation for high-risk maintenance topics.

Heuristic keyword detection over the question + answer. When a high-risk category
is present, a safety notice is surfaced (recommending the relevant procedure /
supervisor sign-off). This is advisory, not a substitute for site safety systems.
"""
from __future__ import annotations

import re

# category -> (compiled patterns, notice)
_RISK = {
    "electrical": (
        [r"\b(440|480|230|high[- ]?voltage|energis|energiz|live\s+circuit|earth\s+fault|busbar)\b",
         r"\belectric(al|ity)\b"],
        "This involves electrical work. Verify isolation and apply lockout/tagout "
        "(LOTO) before proceeding; confirm with a qualified electrician.",
    ),
    "lockout_tagout": (
        [r"\block\s?out\b", r"\btag\s?out\b", r"\bLOTO\b", r"\bisolat(e|ion)\b"],
        "Follow the lockout/tagout procedure and verify zero-energy state before work.",
    ),
    "pressure": (
        [r"\b(pressuris|pressuriz|high[- ]?pressure|hydraulic|pneumatic|accumulator)\b"],
        "Stored-pressure hazard: depressurize and verify before disassembly.",
    ),
    "confined_space": (
        [r"\bconfined\s+space\b", r"\btank\s+entry\b", r"\bvessel\s+entry\b"],
        "Confined-space entry requires a permit, atmosphere testing, and a standby attendant.",
    ),
    "refrigerant": (
        [r"\b(refrigerant|r-?410a|r-?134a|ammonia|freon)\b"],
        "Refrigerant hazard: risk of asphyxiation/frostbite. Ventilate and use PPE; "
        "evacuate confined areas on leak detection.",
    ),
}

_COMPILED = {cat: ([re.compile(p, re.IGNORECASE) for p in pats], notice)
             for cat, (pats, notice) in _RISK.items()}


def assess_risk(text: str) -> list[str]:
    """Return the high-risk categories present in the text."""
    hits = []
    for cat, (pats, _notice) in _COMPILED.items():
        if any(p.search(text) for p in pats):
            hits.append(cat)
    return hits


def safety_notice(categories: list[str]) -> str:
    """Assemble a safety notice for the detected categories ("" if none)."""
    if not categories:
        return ""
    lines = ["⚠ Safety notice:"]
    lines += [f"- {_COMPILED[c][1]}" for c in categories if c in _COMPILED]
    lines.append("- When in doubt, stop and confirm with your supervisor.")
    return "\n".join(lines)
