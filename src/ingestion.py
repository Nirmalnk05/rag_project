"""Ingestion pipeline: parse → chunk → contextually augment → dual-index.

Implements Anthropic's contextual retrieval pattern: each chunk gets a small
LLM-generated context blurb prepended before embedding/indexing.

Ingestion is incremental: a content-hash manifest lets unchanged chunks skip
both LLM augmentation and re-embedding; stale chunks are deleted from Qdrant.
data/chunks.json is the single source of truth (BM25 is rebuilt from it at
Retriever startup — there is no separate BM25 artifact anymore).
"""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.config import get_settings

_settings = get_settings()

CHUNK_STORE_PATH = Path("data/chunks.json")
MANIFEST_PATH = Path("data/ingest_manifest.json")
CONTEXT_CACHE_PATH = Path("data/context_cache.json")


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
    """One small-LLM call per chunk (cache-checked by callers)."""
    doc_snip = doc_body[:8000]
    prompt = CONTEXT_AUG_PROMPT.format(doc=doc_snip, chunk=chunk_text)
    from src.llm import complete_sync
    return complete_sync(
        prompt, model=_settings.context_aug_model, max_tokens=120, temperature=0.0
    ).text.strip()


# ---------- identity & hashing ----------

def _hash_id(*parts: str) -> str:
    return hashlib.sha256("||".join(parts).encode()).hexdigest()[:16]


def chunk_content_hash(doc_title: str, section: str, raw_text: str) -> str:
    return hashlib.sha256(f"{doc_title}||{section}||{raw_text}".encode()).hexdigest()


def point_id_for(chunk_id: str) -> str:
    """Deterministic Qdrant point id so re-ingest upserts in place."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def diff_manifest(current: dict[str, str], previous: dict[str, str]) -> tuple[set, set, set]:
    """Returns (new_ids, changed_ids, stale_ids)."""
    new = {i for i in current if i not in previous}
    changed = {i for i in current if i in previous and current[i] != previous[i]}
    stale = {i for i in previous if i not in current}
    return new, changed, stale


def _load_json(path: Path, default):
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return default


def _save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ---------- chunk building (augmentation cache-aware) ----------

def build_chunks(docs_dir: Path, use_contextual: bool = True,
                 context_cache: dict[str, str] | None = None) -> list[Chunk]:
    context_cache = context_cache if context_cache is not None else {}
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
                    key = chunk_content_hash(title, section_title, raw)
                    if key in context_cache:
                        ctx = context_cache[key]
                    else:
                        print(f"  augmenting chunk in '{section_title}'...", flush=True)
                        ctx = contextual_augment(body, raw)
                        context_cache[key] = ctx
                        _save_json(CONTEXT_CACHE_PATH, context_cache)  # resumable
                embed_text = f"{ctx}\n\n{raw}" if ctx else raw
                chunks.append(Chunk(
                    id=_hash_id(doc_id, section_title, raw[:200]),
                    doc_id=doc_id, doc_title=title, text=raw, context=ctx,
                    embed_text=embed_text, section=section_title,
                    metadata={"source_file": path.name},
                ))
    return chunks


# ---------- indexing ----------

def index_qdrant_incremental(chunks: list[Chunk], embedder, *,
                             to_upsert: set[str], stale_ids: set[str],
                             rebuild: bool = False) -> None:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, PointStruct, VectorParams

    client = QdrantClient(host=_settings.qdrant_host, port=_settings.qdrant_port)
    dim = embedder.get_sentence_embedding_dimension()

    if rebuild and client.collection_exists(_settings.qdrant_collection):
        client.delete_collection(_settings.qdrant_collection)
    if not client.collection_exists(_settings.qdrant_collection):
        client.create_collection(
            collection_name=_settings.qdrant_collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

    if stale_ids:
        client.delete(
            collection_name=_settings.qdrant_collection,
            points_selector=[point_id_for(cid) for cid in stale_ids],
        )
        print(f"  deleted {len(stale_ids)} stale points")

    todo = [c for c in chunks if c.id in to_upsert]
    if not todo:
        print("  nothing new to embed/upsert")
        return
    print(f"  embedding {len(todo)} chunks...", flush=True)
    vectors = embedder.encode([c.embed_text for c in todo], batch_size=32,
                              normalize_embeddings=True, show_progress_bar=True)
    points = [
        PointStruct(id=point_id_for(c.id), vector=v.tolist(), payload=c.to_dict())
        for c, v in zip(todo, vectors)
    ]
    BATCH = 64
    for i in range(0, len(points), BATCH):
        client.upsert(collection_name=_settings.qdrant_collection,
                      points=points[i:i + BATCH])
    print(f"  upserted {len(points)} points into Qdrant '{_settings.qdrant_collection}'")


# ---------- entry point ----------

def run_ingestion(docs_dir: Path = Path("data/docs"), use_contextual: bool = True,
                  rebuild: bool = False) -> None:
    print(f"[ingest] building chunks from {docs_dir} (contextual={use_contextual}, rebuild={rebuild})")
    context_cache = {} if rebuild else _load_json(CONTEXT_CACHE_PATH, {})
    previous_manifest = {} if rebuild else _load_json(MANIFEST_PATH, {})

    chunks = build_chunks(docs_dir, use_contextual=use_contextual, context_cache=context_cache)
    current_manifest = {
        c.id: chunk_content_hash(c.doc_title, c.section, c.text) for c in chunks
    }
    new, changed, stale = diff_manifest(current_manifest, previous_manifest)
    to_upsert = new | changed if not rebuild else set(current_manifest)
    print(f"[ingest] {len(chunks)} chunks total — new={len(new)} changed={len(changed)} stale={len(stale)}")

    print("[ingest] loading embedding model...")
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(_settings.embed_model)

    index_qdrant_incremental(chunks, embedder, to_upsert=to_upsert,
                             stale_ids=stale, rebuild=rebuild)

    _save_json(CHUNK_STORE_PATH, [c.to_dict() for c in chunks])
    _save_json(MANIFEST_PATH, current_manifest)
    _save_json(CONTEXT_CACHE_PATH, context_cache)
    print("[ingest] done.")
