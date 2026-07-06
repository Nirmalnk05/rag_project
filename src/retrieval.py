"""Hybrid retrieval: BM25 + dense (Qdrant), RRF fusion, cross-encoder rerank.

data/chunks.json is the single source of truth; the BM25 index is rebuilt
in memory at startup (fine to ~100k chunks — beyond that, move sparse
search to OpenSearch/Qdrant sparse vectors).
"""
from __future__ import annotations

import asyncio
import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from src.config import get_settings

CHUNK_STORE_PATH = Path("data/chunks.json")
ALLOWED_FILTER_KEYS = {"source_file", "doc_title", "section", "element_type", "page"}
# keys that live under chunk["metadata"] rather than at the top level
METADATA_FILTER_KEYS = {"source_file", "element_type", "page"}


@dataclass
class Retrieved:
    chunk_id: str
    doc_id: str
    doc_title: str
    section: str
    text: str
    context: str
    score: float
    source: str  # 'bm25' | 'dense' | 'rrf' | 'rerank'

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id, "doc_id": self.doc_id,
            "doc_title": self.doc_title, "section": self.section,
            "text": self.text, "context": self.context,
            "score": self.score, "source": self.source,
        }


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _validate_filters(filters: dict | None) -> None:
    if filters:
        bad = set(filters) - ALLOWED_FILTER_KEYS
        if bad:
            raise ValueError(f"unsupported filter key(s): {sorted(bad)}")


def _matches_filters(chunk: dict, filters: dict | None) -> bool:
    if not filters:
        return True
    for k, v in filters.items():
        actual = chunk.get("metadata", {}).get(k) if k in METADATA_FILTER_KEYS else chunk.get(k)
        if str(actual) != str(v):  # coerce so page (int) matches "2" (str)
            return False
    return True


def _from_chunk(c: dict, score: float, source: str) -> Retrieved:
    return Retrieved(
        chunk_id=c["id"], doc_id=c["doc_id"], doc_title=c["doc_title"],
        section=c.get("section", ""), text=c["text"], context=c.get("context", ""),
        score=score, source=source,
    )


def _clone(r: Retrieved, *, score: float, source: str) -> Retrieved:
    return Retrieved(
        chunk_id=r.chunk_id, doc_id=r.doc_id, doc_title=r.doc_title,
        section=r.section, text=r.text, context=r.context,
        score=score, source=source,
    )


class Retriever:
    """Loads chunk store + models once; exposes hybrid retrieval.

    load_models=False / qdrant_client injection exist for unit tests.
    """

    def __init__(self, chunks_path: Path = CHUNK_STORE_PATH, *,
                 load_models: bool = True, qdrant_client: Any | None = None) -> None:
        s = get_settings()
        with Path(chunks_path).open("r", encoding="utf-8") as f:
            self._chunks: list[dict] = json.load(f)
        self._bm25 = BM25Okapi([_tokenize(c["embed_text"]) for c in self._chunks])

        self._embed_lock = threading.Lock()
        self._rerank_lock = threading.Lock()
        if load_models:
            from sentence_transformers import CrossEncoder, SentenceTransformer
            self._embedder = SentenceTransformer(s.embed_model)
            self._reranker = CrossEncoder(s.rerank_model)
        else:
            self._embedder = None
            self._reranker = None

        if qdrant_client is not None:
            self._qdrant = qdrant_client
        else:
            from qdrant_client import QdrantClient
            self._qdrant = QdrantClient(host=s.qdrant_host, port=s.qdrant_port)

    # ---------- single-source retrievers (sync; run via to_thread) ----------

    def dense_search(self, query: str, top_k: int, filters: dict | None = None,
                     access=None) -> list[Retrieved]:
        _validate_filters(filters)
        from src.access import acl_visible
        s = get_settings()
        with self._embed_lock:
            vec = self._embedder.encode(query, normalize_embeddings=True).tolist()

        qfilter = None
        if filters:
            from qdrant_client.models import FieldCondition, Filter, MatchValue
            conds = []
            for k, v in filters.items():
                key = f"metadata.{k}" if k in METADATA_FILTER_KEYS else k
                conds.append(FieldCondition(key=key, match=MatchValue(value=v)))
            qfilter = Filter(must=conds)

        res = self._qdrant.query_points(
            collection_name=s.qdrant_collection, query=vec,
            limit=top_k, query_filter=qfilter, with_payload=True,
        )
        out = []
        for h in res.points:
            p = h.payload or {}
            if not acl_visible(p.get("metadata", {}), access):
                continue
            out.append(Retrieved(
                chunk_id=p.get("id", ""), doc_id=p.get("doc_id", ""),
                doc_title=p.get("doc_title", ""), section=p.get("section", ""),
                text=p.get("text", ""), context=p.get("context", ""),
                score=float(h.score), source="dense",
            ))
        return out

    def sparse_search(self, query: str, top_k: int, filters: dict | None = None,
                      access=None) -> list[Retrieved]:
        _validate_filters(filters)
        from src.access import acl_visible
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out: list[Retrieved] = []
        for i in ranked:
            if scores[i] <= 0 or len(out) >= top_k:
                break
            c = self._chunks[i]
            if not _matches_filters(c, filters):
                continue
            if not acl_visible(c.get("metadata", {}), access):
                continue
            out.append(_from_chunk(c, float(scores[i]), "bm25"))
        return out

    # ---------- fusion ----------

    @staticmethod
    def reciprocal_rank_fusion(lists: list[list[Retrieved]], k: int = 60) -> list[Retrieved]:
        """Standard RRF: sum 1/(k+rank). k=60 is the Cormack et al. default."""
        scores: dict[str, float] = {}
        by_id: dict[str, Retrieved] = {}
        for lst in lists:
            for rank, r in enumerate(lst, start=1):
                scores[r.chunk_id] = scores.get(r.chunk_id, 0.0) + 1.0 / (k + rank)
                by_id[r.chunk_id] = r
        fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [_clone(by_id[cid], score=s, source="rrf") for cid, s in fused]

    # ---------- rerank ----------

    def rerank(self, query: str, candidates: list[Retrieved], top_k: int) -> list[Retrieved]:
        if not candidates:
            return []
        with self._rerank_lock:
            scores = self._reranker.predict([(query, c.text) for c in candidates]).tolist()
        scored = sorted(zip(candidates, scores), key=lambda t: t[1], reverse=True)[:top_k]
        return [_clone(c, score=float(sc), source="rerank") for c, sc in scored]

    # ---------- public ----------

    async def hybrid_retrieve(self, query: str, filters: dict | None = None,
                              access=None) -> list[Retrieved]:
        """sparse ∥ dense → RRF → rerank → top_k. Sparse+dense run in parallel.
        `access` (AccessContext) filters out chunks the principal may not see."""
        s = get_settings()
        sparse, dense = await asyncio.gather(
            asyncio.to_thread(self.sparse_search, query, s.top_k_sparse, filters, access),
            asyncio.to_thread(self.dense_search, query, s.top_k_dense, filters, access),
        )
        fused = self.reciprocal_rank_fusion([sparse, dense])
        head = fused[: max(s.top_k_sparse, s.top_k_dense)]
        return await asyncio.to_thread(self.rerank, query, head, s.top_k_rerank)
