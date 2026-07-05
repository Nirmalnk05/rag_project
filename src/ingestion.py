"""Ingestion pipeline: parse → chunk → contextually augment → dual-index.

Implements Anthropic's contextual retrieval pattern: each chunk gets a small
LLM-generated context blurb prepended before embedding/indexing, dramatically
improving recall on chunks that are meaningful only in document context.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.config import get_settings
from src.llm import complete


_settings = get_settings()

# Persisted artifacts (BM25 lives on disk; Qdrant is the vector store)
BM25_INDEX_PATH = Path("data/bm25_index.pkl")
CHUNK_STORE_PATH = Path("data/chunks.json")


# ---------- data model ----------

@dataclass
class Chunk:
    id: str
    doc_id: str
    doc_title: str
    text: str               # the raw chunk text (the part shown back to the LLM)
    context: str            # the LLM-generated context blurb
    embed_text: str         # the text that was actually embedded/indexed
    section: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------- parsing & chunking ----------

def _read_doc(path: Path) -> tuple[str, str]:
    """Return (title, body). Title comes from first H1 or filename."""
    body = path.read_text(encoding="utf-8")
    m = re.search(r"^#\s+(.+)$", body, flags=re.MULTILINE)
    title = m.group(1).strip() if m else path.stem
    return title, body


def _split_into_sections(body: str) -> list[tuple[str, str]]:
    """Split markdown by H2 headers; returns [(section_title, section_text), ...]."""
    parts: list[tuple[str, str]] = []
    current_title = "Preamble"
    current_buf: list[str] = []
    for line in body.splitlines():
        h2 = re.match(r"^##\s+(.+)$", line)
        if h2:
            if current_buf:
                parts.append((current_title, "\n".join(current_buf).strip()))
            current_title = h2.group(1).strip()
            current_buf = []
        else:
            current_buf.append(line)
    if current_buf:
        parts.append((current_title, "\n".join(current_buf).strip()))
    return [(t, b) for t, b in parts if b]


def _windowed_chunks(text: str, target: int = 600, overlap: int = 100) -> list[str]:
    """Word-windowed chunking. Production would use a tokenizer; words are good enough here."""
    words = text.split()
    if len(words) <= target:
        return [text]
    out: list[str] = []
    step = target - overlap
    for start in range(0, len(words), step):
        out.append(" ".join(words[start : start + target]))
        if start + target >= len(words):
            break
    return out


# ---------- contextual retrieval (Anthropic pattern) ----------

CONTEXT_AUG_PROMPT = """<document>
{doc}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Give a short, 1–2 sentence context that situates this chunk within the document for the purposes of improving search retrieval of the chunk. Output only the context, nothing else."""


def contextual_augment(doc_body: str, chunk_text: str) -> str:
    """One small-LLM call per chunk. Returns 1-2 sentence context blurb.

    Production: batch these, cache by chunk-hash, and run as a separate ingestion step.
    """
    # Truncate the doc to keep token costs sane; production would chunk-then-summarize.
    doc_snip = doc_body[:8000]
    prompt = CONTEXT_AUG_PROMPT.format(doc=doc_snip, chunk=chunk_text)
    return complete(
        prompt,
        model=_settings.context_aug_model,
        max_tokens=120,
        temperature=0.0,
    ).strip()


# ---------- indexing ----------

def _hash_id(*parts: str) -> str:
    return hashlib.sha256("||".join(parts).encode()).hexdigest()[:16]


def _tokenize(text: str) -> list[str]:
    """Cheap tokenizer for BM25. Lowercased alphanumeric words; production: spacy/nltk."""
    return re.findall(r"[a-z0-9]+", text.lower())


def build_chunks(docs_dir: Path, use_contextual: bool = True) -> list[Chunk]:
    chunks: list[Chunk] = []
    files = sorted(docs_dir.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No .md files in {docs_dir}")
    for path in files:
        title, body = _read_doc(path)
        doc_id = _hash_id(path.name)
        for section_title, section_body in _split_into_sections(body):
            for raw in _windowed_chunks(section_body):
                ctx = ""
                if use_contextual:
                    print(f"  augmenting chunk in '{section_title}'...", flush=True)
                    ctx = contextual_augment(body, raw)
                # The text that's embedded/indexed is context + chunk.
                # The text shown back to the LLM is just the raw chunk (with context as metadata).
                embed_text = f"{ctx}\n\n{raw}" if ctx else raw
                chunks.append(
                    Chunk(
                        id=_hash_id(doc_id, section_title, raw[:200]),
                        doc_id=doc_id,
                        doc_title=title,
                        text=raw,
                        context=ctx,
                        embed_text=embed_text,
                        section=section_title,
                        metadata={"source_file": path.name},
                    )
                )
    return chunks


def index_qdrant(chunks: list[Chunk], embedder: SentenceTransformer) -> None:
    client = QdrantClient(host=_settings.qdrant_host, port=_settings.qdrant_port)
    dim = embedder.get_sentence_embedding_dimension()

    # Recreate collection (idempotent)
    if client.collection_exists(_settings.qdrant_collection):
        client.delete_collection(_settings.qdrant_collection)
    client.create_collection(
        collection_name=_settings.qdrant_collection,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    texts = [c.embed_text for c in chunks]
    print(f"  embedding {len(texts)} chunks...", flush=True)
    vectors = embedder.encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    points = [
        PointStruct(
            id=i,  # qdrant needs int or uuid; we use array index and store the real id in payload
            vector=vec.tolist(),
            payload={**c.to_dict()},
        )
        for i, (c, vec) in enumerate(zip(chunks, vectors))
    ]
    # Batch upload
    BATCH = 64
    for i in range(0, len(points), BATCH):
        client.upsert(collection_name=_settings.qdrant_collection, points=points[i : i + BATCH])
    print(f"  indexed {len(points)} points into Qdrant '{_settings.qdrant_collection}'")


def index_bm25(chunks: list[Chunk]) -> None:
    BM25_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    tokenized = [_tokenize(c.embed_text) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    # Save the BM25 index + the chunk metadata (parallel arrays keyed by index)
    with BM25_INDEX_PATH.open("wb") as f:
        pickle.dump({"bm25": bm25, "tokenized_len": len(tokenized)}, f)
    with CHUNK_STORE_PATH.open("w", encoding="utf-8") as f:
        json.dump([c.to_dict() for c in chunks], f, ensure_ascii=False, indent=2)
    print(f"  indexed {len(chunks)} chunks into BM25 store")


# ---------- entry point ----------

def run_ingestion(docs_dir: Path = Path("data/docs"), use_contextual: bool = True) -> None:
    print(f"[ingest] building chunks from {docs_dir} (contextual={use_contextual})")
    chunks = build_chunks(docs_dir, use_contextual=use_contextual)
    print(f"[ingest] built {len(chunks)} chunks")

    print("[ingest] loading embedding model...")
    embedder = SentenceTransformer(_settings.embed_model)

    print("[ingest] indexing into Qdrant...")
    index_qdrant(chunks, embedder)

    print("[ingest] indexing into BM25...")
    index_bm25(chunks)

    print("[ingest] done.")
