"""Hybrid retrieval: BM25 + dense (Qdrant), RRF fusion, cross-encoder rerank.

data/chunks.json is the single source of truth; the BM25 index is rebuilt
in memory at startup. Measured on a synthetic corpus, 100k chunks costs ~1.8 GB
of process RAM (chunk store + BM25 doc_freqs), ~90s of startup inside the
FastAPI lifespan hook, and 300-730ms per sparse_search — so treat ~50k as the
practical ceiling for this shape and move sparse search to Qdrant sparse vectors
(or OpenSearch) beyond it.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from src.config import get_settings

_logger = logging.getLogger("rag.retrieval")

CHUNK_STORE_PATH = Path("data/chunks.json")
ALLOWED_FILTER_KEYS = {"source_file", "doc_title", "section", "element_type", "page"}
# keys that live under chunk["metadata"] rather than at the top level
METADATA_FILTER_KEYS = {"source_file", "element_type", "page"}
# payload paths the ACL filter reads; must match what ingestion writes
ACL_ROLES_KEY = "metadata.acl_roles"
ACL_SITES_KEY = "metadata.acl_sites"


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
    # Chunk metadata, carried so the boundary check in hybrid_retrieve can
    # re-authorize. Deliberately absent from to_dict(): that feeds the API
    # response and the LLM context, neither of which should see ACL labels.
    metadata: dict = field(default_factory=dict)

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


def _fresh(meta: dict, include_superseded: bool) -> bool:
    """Hide superseded document revisions unless explicitly requested."""
    if include_superseded:
        return True
    return meta.get("doc_status", "active") != "superseded"


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
        score=score, source=source, metadata=c.get("metadata", {}),
    )


def _clone(r: Retrieved, *, score: float, source: str) -> Retrieved:
    return Retrieved(
        chunk_id=r.chunk_id, doc_id=r.doc_id, doc_title=r.doc_title,
        section=r.section, text=r.text, context=r.context,
        score=score, source=source, metadata=r.metadata,
    )


def _acl_dimension(key: str, granted: frozenset[str]):
    """One ACL dimension: the chunk does not restrict it, OR the principal matches.

    Mirrors `acl_visible`'s per-dimension rule. `IsEmpty` covers a missing key,
    null, and an empty list — `_acl_for` omits the key entirely for public
    chunks, so the missing-key case is the common one.
    """
    from qdrant_client.models import (FieldCondition, Filter, IsEmptyCondition,
                                      MatchAny, PayloadField)
    should = [IsEmptyCondition(is_empty=PayloadField(key=key))]
    if granted:
        should.append(FieldCondition(key=key, match=MatchAny(any=sorted(granted))))
    return Filter(should=should)


def acl_filter(access=None):
    """Translate an AccessContext into a Qdrant filter, or None if unrestricted.

    Both dimensions are ANDed, each as its own should-group. Dropping either
    group fails *open*, so the differential test in tests/test_acl_enforcement.py
    checks this against `acl_visible` for every role/site combination.
    """
    if access is None or access.is_admin:
        return None
    from qdrant_client.models import Filter
    return Filter(must=[_acl_dimension(ACL_ROLES_KEY, access.roles),
                        _acl_dimension(ACL_SITES_KEY, access.sites)])


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

    def embed_query(self, query: str) -> list[float]:
        """Embed a query (for the semantic cache). Uses the same encoder as dense
        retrieval, guarded by the embed lock."""
        with self._embed_lock:
            return self._embedder.encode(query, normalize_embeddings=True).tolist()

    # ---------- single-source retrievers (sync; run via to_thread) ----------

    def dense_search(self, query: str, top_k: int, filters: dict | None = None,
                     access=None) -> list[Retrieved]:
        _validate_filters(filters)
        from src.access import acl_visible
        s = get_settings()
        with self._embed_lock:
            vec = self._embedder.encode(query, normalize_embeddings=True).tolist()

        # ACL goes into the query filter so top_k is filled with *permitted*
        # neighbours. Post-filtering here would truncate first and collapse
        # recall: 20 nearest, 15 inaccessible, 5 candidates left to rerank.
        conds = []
        if filters:
            from qdrant_client.models import FieldCondition, MatchValue
            for k, v in filters.items():
                key = f"metadata.{k}" if k in METADATA_FILTER_KEYS else k
                conds.append(FieldCondition(key=key, match=MatchValue(value=v)))
        aclf = acl_filter(access)
        if aclf is not None:
            conds.append(aclf)
        qfilter = None
        if conds:
            from qdrant_client.models import Filter
            qfilter = Filter(must=conds)

        res = self._qdrant.query_points(
            collection_name=s.qdrant_collection, query=vec,
            limit=top_k, query_filter=qfilter, with_payload=True,
        )
        out = []
        for h in res.points:
            p = h.payload or {}
            meta = p.get("metadata", {})
            # Kept as defence in depth: an index written before ACLs were
            # populated has no acl_* payload, so the filter above cannot gate it.
            if not acl_visible(meta, access) or not _fresh(meta, s.retrieve_superseded):
                continue
            out.append(Retrieved(
                chunk_id=p.get("id", ""), doc_id=p.get("doc_id", ""),
                doc_title=p.get("doc_title", ""), section=p.get("section", ""),
                text=p.get("text", ""), context=p.get("context", ""),
                score=float(h.score), source="dense", metadata=meta,
            ))
        return out

    def sparse_search(self, query: str, top_k: int, filters: dict | None = None,
                      access=None) -> list[Retrieved]:
        _validate_filters(filters)
        from src.access import acl_visible
        s = get_settings()
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out: list[Retrieved] = []
        for i in ranked:
            if scores[i] <= 0 or len(out) >= top_k:
                break
            c = self._chunks[i]
            if not _matches_filters(c, filters):
                continue
            meta = c.get("metadata", {})
            if not acl_visible(meta, access) or not _fresh(meta, s.retrieve_superseded):
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

    def _authorized(self, results: list[Retrieved], access) -> list[Retrieved]:
        """Final authorization gate for everything leaving the retrieval layer.

        Independent of the per-search checks on purpose: this is the one place
        every caller passes through, so a bug in filter translation, a stale
        Qdrant payload, or a future code path cannot leak a chunk into the LLM
        context *or* into the API response's `retrieval` field.
        """
        from src.access import acl_visible
        ok = [r for r in results if acl_visible(r.metadata, access)]
        if len(ok) != len(results):
            from src.observability import log_event
            log_event(_logger, "acl_boundary_removed_chunks",
                      removed=len(results) - len(ok), returned=len(ok))
        return ok

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
        ranked = await asyncio.to_thread(self.rerank, query, head, s.top_k_rerank)
        return self._authorized(ranked, access)
