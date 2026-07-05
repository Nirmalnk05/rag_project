"""Hybrid retrieval: BM25 + dense, fused via reciprocal rank fusion, then cross-encoder reranked."""
from __future__ import annotations

import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer

from src.config import get_settings


_settings = get_settings()

BM25_INDEX_PATH = Path("data/bm25_index.pkl")
CHUNK_STORE_PATH = Path("data/chunks.json")


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
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "section": self.section,
            "text": self.text,
            "context": self.context,
            "score": self.score,
            "source": self.source,
        }


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class Retriever:
    """Loads indexes once, exposes hybrid retrieval + rerank."""

    def __init__(self) -> None:
        # Embedding model
        self._embedder = SentenceTransformer(_settings.embed_model)
        # Cross-encoder reranker
        self._reranker = CrossEncoder(_settings.rerank_model)
        # Qdrant
        self._qdrant = QdrantClient(host=_settings.qdrant_host, port=_settings.qdrant_port)
        # BM25 + chunk store
        with BM25_INDEX_PATH.open("rb") as f:
            self._bm25_blob = pickle.load(f)
        self._bm25 = self._bm25_blob["bm25"]
        with CHUNK_STORE_PATH.open("r", encoding="utf-8") as f:
            self._chunks: list[dict] = json.load(f)

    # ---------- single-source retrievers ----------

    def dense_search(self, query: str, top_k: int) -> list[Retrieved]:
        vec = self._embedder.encode(query, normalize_embeddings=True).tolist()
        hits = self._qdrant.search(
            collection_name=_settings.qdrant_collection,
            query_vector=vec,
            limit=top_k,
        )
        out: list[Retrieved] = []
        for h in hits:
            p = h.payload or {}
            out.append(
                Retrieved(
                    chunk_id=p.get("id", ""),
                    doc_id=p.get("doc_id", ""),
                    doc_title=p.get("doc_title", ""),
                    section=p.get("section", ""),
                    text=p.get("text", ""),
                    context=p.get("context", ""),
                    score=float(h.score),
                    source="dense",
                )
            )
        return out

    def sparse_search(self, query: str, top_k: int) -> list[Retrieved]:
        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)
        # top-k indices
        idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        out: list[Retrieved] = []
        for i in idx:
            if scores[i] <= 0:
                continue
            c = self._chunks[i]
            out.append(
                Retrieved(
                    chunk_id=c["id"],
                    doc_id=c["doc_id"],
                    doc_title=c["doc_title"],
                    section=c["section"],
                    text=c["text"],
                    context=c.get("context", ""),
                    score=float(scores[i]),
                    source="bm25",
                )
            )
        return out

    # ---------- fusion ----------

    @staticmethod
    def reciprocal_rank_fusion(
        lists: list[list[Retrieved]], k: int = 60
    ) -> list[Retrieved]:
        """Standard RRF: sum 1 / (k + rank) across all source rankings.

        k=60 is the Cormack et al. default. Lower k = stronger rank-1 bias.
        """
        scores: dict[str, float] = {}
        by_id: dict[str, Retrieved] = {}
        for lst in lists:
            for rank, r in enumerate(lst, start=1):
                scores[r.chunk_id] = scores.get(r.chunk_id, 0.0) + 1.0 / (k + rank)
                by_id[r.chunk_id] = r  # last writer wins; identity-equal anyway
        fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        out: list[Retrieved] = []
        for cid, s in fused:
            r = by_id[cid]
            out.append(
                Retrieved(
                    chunk_id=r.chunk_id,
                    doc_id=r.doc_id,
                    doc_title=r.doc_title,
                    section=r.section,
                    text=r.text,
                    context=r.context,
                    score=s,
                    source="rrf",
                )
            )
        return out

    # ---------- rerank ----------

    def rerank(self, query: str, candidates: list[Retrieved], top_k: int) -> list[Retrieved]:
        if not candidates:
            return []
        pairs = [(query, c.text) for c in candidates]
        scores = self._reranker.predict(pairs).tolist()
        scored = sorted(zip(candidates, scores), key=lambda t: t[1], reverse=True)[:top_k]
        return [
            Retrieved(
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                doc_title=c.doc_title,
                section=c.section,
                text=c.text,
                context=c.context,
                score=float(s),
                source="rerank",
            )
            for c, s in scored
        ]

    # ---------- public ----------

    def hybrid_retrieve(self, query: str) -> list[Retrieved]:
        """Full pipeline: sparse + dense → RRF → cross-encoder rerank → top_k."""
        sparse = self.sparse_search(query, _settings.top_k_sparse)
        dense = self.dense_search(query, _settings.top_k_dense)
        fused = self.reciprocal_rank_fusion([sparse, dense])
        reranked = self.rerank(query, fused[: max(_settings.top_k_sparse, _settings.top_k_dense)], _settings.top_k_rerank)
        return reranked
