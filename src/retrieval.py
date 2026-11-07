"""Hybrid retrieval: BM25 + dense (Qdrant), RRF fusion, cross-encoder rerank.

data/chunks.json is the single source of truth; the BM25 index is rebuilt
in memory at startup. It is a *snapshot*: ingestion updates Qdrant as it goes
and rewrites chunks.json only at the end, so while this process is behind the
index (loaded generation != the one in index_meta.json) the sparse leg resolves
each candidate's metadata from Qdrant instead of from memory. INDEX_RELOAD_POLL_S
then rebuilds the corpus off the request path.

Measured on a synthetic corpus, 100k chunks costs ~1.8 GB
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
# how many BM25 candidates to re-resolve per Qdrant round trip while this
# process is behind the index
LIVE_METADATA_BATCH = 64
# payload paths the ACL filter reads; must match what ingestion writes
ACL_ROLES_KEY = "metadata.acl_roles"
ACL_SITES_KEY = "metadata.acl_sites"


@dataclass(frozen=True)
class _Corpus:
    """The chunk store, the BM25 index built from it, and its generation.

    One immutable object so the two can only ever be swapped together: BM25
    returns positions into the list it was built from, so a half-swapped pair
    resolves them against the wrong documents.
    """
    chunks: list[dict]
    bm25: BM25Okapi
    generation: int


def _load_corpus(chunks_path: Path, generation: int | None = None) -> _Corpus:
    """Read chunks.json and build BM25 over it.

    When the caller supplies no generation it is read *before* the file, so a
    corpus is never labelled newer than the snapshot it was built from — an
    ingest that commits mid-load is picked up by the next poll instead.
    """
    from src.ingestion import committed_generation
    gen = committed_generation() if generation is None else generation
    with Path(chunks_path).open("r", encoding="utf-8") as f:
        chunks: list[dict] = json.load(f)
    return _Corpus(chunks=chunks,
                   bm25=BM25Okapi([_tokenize(c["embed_text"]) for c in chunks]),
                   generation=gen)


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


def is_fresh(meta: dict, include_superseded: bool) -> bool:
    """Hide superseded document revisions unless explicitly requested.

    Public because the agent's direct-fetch tools enforce the same rule: two
    definitions of "fresh" would drift apart.
    """
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
        self._chunks_path = Path(chunks_path)
        self._corpus = _load_corpus(self._chunks_path)

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

    # ---------- corpus / index generation ----------

    @property
    def _chunks(self) -> list[dict]:
        """The loaded chunk store (one read of the current corpus reference)."""
        return self._corpus.chunks

    @property
    def generation(self) -> int:
        """The index generation this process is serving."""
        return self._corpus.generation

    def index_has_moved(self) -> bool:
        """True while ingestion has claimed a generation past the loaded one.

        Compares generations, never timestamps: mtime granularity differs by
        filesystem and is not a reliable ordering on a shared RWX volume.
        """
        from src.ingestion import index_generation
        return index_generation() != self._corpus.generation

    def live_metadata(self, chunk_ids: list[str]) -> dict[str, dict]:
        """Current payload metadata for these chunks, straight from the index.

        Only chunks Qdrant still holds appear in the result, so a caller can
        tell "revoked" from "deleted" and fail closed on both.
        """
        if not chunk_ids:
            return {}
        from src.ingestion import point_id_for
        s = get_settings()
        wanted = [point_id_for(cid) for cid in chunk_ids]
        recs = self._qdrant.retrieve(collection_name=s.qdrant_collection,
                                     ids=wanted, with_payload=True)
        out: dict[str, dict] = {}
        for rec in recs or []:
            payload = getattr(rec, "payload", None) or {}
            cid = payload.get("id")
            if cid:
                out[cid] = payload.get("metadata", {}) or {}
        return out

    def reload_if_stale(self) -> bool:
        """Rebuild the corpus when the index has committed a newer generation.

        Runs off the request path (the poller calls it in a thread). Skipped
        while an ingest is in flight, because `committed_generation` still
        trails: BM25 built from a half-written chunks.json is worse than being
        a minute stale. Holds both corpora while it runs, so peak RAM is ~2x
        the chunk store. This reload is what makes NEW documents searchable.
        """
        from src.ingestion import committed_generation
        target = committed_generation()
        if target == self._corpus.generation:
            return False
        fresh = _load_corpus(self._chunks_path, generation=target)
        self._corpus = fresh      # ONE reference: chunks + BM25 swap together
        from src.observability import log_event
        log_event(_logger, "index_corpus_reloaded", generation=target,
                  chunks=len(fresh.chunks))
        return True

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
            if not acl_visible(meta, access) or not is_fresh(meta, s.retrieve_superseded):
                continue
            out.append(Retrieved(
                chunk_id=p.get("id", ""), doc_id=p.get("doc_id", ""),
                doc_title=p.get("doc_title", ""), section=p.get("section", ""),
                text=p.get("text", ""), context=p.get("context", ""),
                score=float(h.score), source="dense", metadata=meta,
            ))
        return out

    def _live_filtered(self, pending: list[tuple[dict, float]], access,
                       s) -> list[Retrieved]:
        """Authorize a batch of BM25 candidates against the live index.

        A chunk Qdrant no longer holds is dropped rather than served from the
        snapshot, so a deletion applies at once — fail closed.
        """
        from src.access import acl_visible
        live = self.live_metadata([c["id"] for c, _ in pending])
        kept: list[Retrieved] = []
        gone = 0
        for c, score in pending:
            meta = live.get(c["id"])
            if meta is None:
                gone += 1
                continue
            if not acl_visible(meta, access) or not is_fresh(meta, s.retrieve_superseded):
                continue
            r = _from_chunk(c, score, "bm25")
            # the boundary check re-authorizes on this, so it must be the
            # index's labels, not the snapshot's
            r.metadata = meta
            kept.append(r)
        if gone:
            from src.observability import log_event
            log_event(_logger, "sparse_dropped_chunks_absent_from_index", removed=gone)
        return kept

    def sparse_search(self, query: str, top_k: int, filters: dict | None = None,
                      access=None) -> list[Retrieved]:
        _validate_filters(filters)
        from src.access import acl_visible
        s = get_settings()
        corpus = self._corpus      # one read: a reload swaps the pair wholesale
        live = self.index_has_moved()
        scores = corpus.bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out: list[Retrieved] = []
        pending: list[tuple[dict, float]] = []
        for i in ranked:
            if scores[i] <= 0 or len(out) >= top_k:
                break
            c = corpus.chunks[i]
            if not _matches_filters(c, filters):
                continue
            if not live:
                meta = c.get("metadata", {})
                if not acl_visible(meta, access) or not is_fresh(meta, s.retrieve_superseded):
                    continue
                out.append(_from_chunk(c, float(scores[i]), "bm25"))
                continue
            pending.append((c, float(scores[i])))
            if len(pending) >= LIVE_METADATA_BATCH:
                out.extend(self._live_filtered(pending, access, s))
                pending = []
        if pending and len(out) < top_k:
            out.extend(self._live_filtered(pending, access, s))
        return out[:top_k]

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
