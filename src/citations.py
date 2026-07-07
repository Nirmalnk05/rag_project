"""Post-generation citation verification (non-blocking; clients decide policy)."""
from __future__ import annotations

import re

_CITE_RE = re.compile(r"\[#(\d+)\]")
# Heuristic: an answer longer than this that cites nothing is suspect,
# unless it's an explicit "insufficient context" style refusal.
_FACTUAL_LENGTH_THRESHOLD = 200
_REFUSAL_MARKERS = ("insufficient", "does not contain", "cannot answer", "no information")


def verify_citations(answer: str, num_sources: int) -> dict:
    refs = [int(m) for m in _CITE_RE.findall(answer)]
    used = sorted({n for n in refs if 1 <= n <= num_sources})
    orphaned = sorted({n for n in refs if n < 1 or n > num_sources})

    lower = answer.lower()
    is_refusal = any(m in lower for m in _REFUSAL_MARKERS)
    uncited_factual = (
        not used and not is_refusal and len(answer) > _FACTUAL_LENGTH_THRESHOLD
    )
    return {
        "valid": not orphaned and not uncited_factual,
        "used": used,
        "orphaned": orphaned,
    }


def ensure_citations(answer: str, num_sources: int) -> tuple[str, dict]:
    """Deterministic safety net for the 'uncited_factual' case only: real
    sources were retrieved and the answer is grounded-looking, but the model
    cited none of them. Appends a plain references trailer listing every
    retrieved source and re-verifies.

    Deliberately leaves the 'orphaned' case (a fabricated citation number)
    untouched — that must stay visibly flagged, not papered over.
    """
    citations = verify_citations(answer, num_sources)
    if not citations["valid"] and not citations["orphaned"] and num_sources > 0:
        trailer = " ".join(f"[#{n}]" for n in range(1, num_sources + 1))
        answer = f"{answer}\n\nSources: {trailer}"
        citations = verify_citations(answer, num_sources)
    return answer, citations
