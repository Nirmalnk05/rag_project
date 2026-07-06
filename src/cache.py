"""Semantic answer cache: reuse a prior answer when a new query is close enough.

Keyed on the query embedding (cosine similarity ≥ threshold). Bounded FIFO. Used
only when ACLs are off (otherwise different principals could share answers). The
embedding vector is supplied by the caller (Retriever.embed_query) so this module
stays model-free and unit-testable.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from src.config import Settings, get_settings


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class SemanticCache:
    def __init__(self, settings: Settings | None = None) -> None:
        self._s = settings or get_settings()
        self._entries: deque = deque(maxlen=self._s.semantic_cache_max)

    @property
    def enabled(self) -> bool:
        return self._s.enable_semantic_cache

    def get(self, query_vec) -> dict | None:
        if not self.enabled or not self._entries:
            return None
        v = np.asarray(query_vec, dtype=float)
        best, best_sim = None, -1.0
        for vec, entry in self._entries:
            sim = _cosine(v, vec)
            if sim > best_sim:
                best, best_sim = entry, sim
        if best is not None and best_sim >= self._s.semantic_cache_threshold:
            return best
        return None

    def put(self, query_vec, entry: dict) -> None:
        if not self.enabled:
            return
        self._entries.append((np.asarray(query_vec, dtype=float), entry))

    def clear(self) -> None:
        self._entries.clear()


_cache: SemanticCache | None = None


def get_cache(settings: Settings | None = None) -> SemanticCache:
    global _cache
    if _cache is None:
        _cache = SemanticCache(settings)
    return _cache


def set_cache(cache) -> None:  # test hook
    global _cache
    _cache = cache
