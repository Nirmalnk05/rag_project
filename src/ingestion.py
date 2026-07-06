"""Ingestion pipeline: parse (any format) → chunk → contextually augment → index.

The front-end delegates to `src/parsing` (Dispatcher) to turn PDF/scanned/Word/
Excel/Markdown into a normalized element stream. Elements are chunked by type:
text is window-split, tables are kept atomic, figures become their vision caption.
Everything then flows through the unchanged downstream: contextual augmentation,
incremental manifest, deterministic uuid5 point IDs, Qdrant upsert, chunks.json.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from src.config import get_settings
from src.parsing.dispatcher import SUPPORTED_EXT, Dispatcher
from src.parsing.markdown_parser import _split_sections as _split_into_sections  # re-export
from src.parsing.model import FigureElement, ParsedDocument, TableElement, TextElement

_settings = get_settings()

CHUNK_STORE_PATH = Path("data/chunks.json")
MANIFEST_PATH = Path("data/ingest_manifest.json")
CONTEXT_CACHE_PATH = Path("data/context_cache.json")
ACL_SIDECAR_PATH = Path("data/acl.json")   # {source_file: {roles: [...], sites: [...]}}
# {source_file: {version, effective_date, status: active|superseded, supersedes: [...]}}
VERSIONS_SIDECAR_PATH = Path("data/versions.json")


def _acl_for(source_file: str, acl_map: dict) -> dict:
    """Return acl_roles/acl_sites metadata for a file (public if unlisted)."""
    entry = acl_map.get(source_file) or {}
    meta = {}
    if entry.get("roles"):
        meta["acl_roles"] = list(entry["roles"])
    if entry.get("sites"):
        meta["acl_sites"] = list(entry["sites"])
    return meta


def _version_for(source_file: str, versions_map: dict) -> dict:
    """Return doc_version/doc_date/doc_status metadata (active if unlisted)."""
    entry = versions_map.get(source_file) or {}
    return {
        "doc_version": entry.get("version", ""),
        "doc_date": entry.get("effective_date", ""),
        "doc_status": entry.get("status", "active"),
    }

__all__ = ["Chunk", "build_chunks", "run_ingestion", "_split_into_sections",
           "_windowed_chunks", "chunk_content_hash", "diff_manifest", "point_id_for"]


# ---------- data model ----------

@dataclass
class Chunk:
    id: str
    doc_id: str
    doc_title: str
    text: str               # raw chunk text shown back to the LLM
    context: str            # LLM-generated context blurb
    embed_text: str         # text actually embedded/indexed
    section: str = ""
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------- chunking helpers ----------

def _windowed_chunks(text: str, target: int = 600, overlap: int = 100) -> list[str]:
    """Word-windowed chunking. Production would use a tokenizer; words suffice here."""
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


def _element_units(el) -> list[tuple[str, str, int | None]]:
    """Turn one parsed element into (chunk_text, element_type, page) units."""
    if isinstance(el, TextElement):
        return [(w, "text", el.page) for w in _windowed_chunks(el.text)]
    if isinstance(el, TableElement):
        return [(el.markdown, "table", el.page)]           # atomic — never split
    if isinstance(el, FigureElement):
        return [(el.caption, "figure", el.page)] if el.caption.strip() else []
    return []


def _doc_text(parsed: ParsedDocument) -> str:
    """Flatten a document for contextual-augmentation context."""
    parts = []
    for e in parsed.elements:
        if isinstance(e, TextElement):
            parts.append(e.text)
        elif isinstance(e, TableElement):
            parts.append(e.markdown)
        elif isinstance(e, FigureElement):
            parts.append(e.caption)
    return "\n\n".join(p for p in parts if p)


# ---------- contextual retrieval (Anthropic pattern) ----------

CONTEXT_AUG_PROMPT = """<document>
{doc}
</document>

Here is the chunk we want to situate within the whole document:
<chunk>
{chunk}
</chunk>

Give a short, 1–2 sentence context that situates this chunk within the document for the purposes of improving search retrieval of the chunk. Output only the context, nothing else."""


def contextual_augment(doc_text: str, chunk_text: str) -> str:
    """One small-LLM call per chunk (cache-checked by callers)."""
    prompt = CONTEXT_AUG_PROMPT.format(doc=doc_text[:8000], chunk=chunk_text)
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


# ---------- chunk building ----------

def build_chunks(docs_dir: Path, use_contextual: bool = True,
                 context_cache: dict[str, str] | None = None,
                 dispatcher: Dispatcher | None = None,
                 force_cloud: bool = False) -> list[Chunk]:
    context_cache = context_cache if context_cache is not None else {}
    dispatcher = dispatcher or Dispatcher()
    acl_map = _load_json(ACL_SIDECAR_PATH, {})
    versions_map = _load_json(VERSIONS_SIDECAR_PATH, {})
    files = sorted(p for p in docs_dir.iterdir()
                   if p.is_file() and p.suffix.lower() in SUPPORTED_EXT)
    if not files:
        raise FileNotFoundError(f"No supported documents in {docs_dir} "
                                f"(supported: {sorted(SUPPORTED_EXT)})")
    chunks: list[Chunk] = []
    for path in files:
        parsed = dispatcher.parse(path, force_cloud=force_cloud)
        doc_text = _doc_text(parsed)
        doc_id = _hash_id(path.name)
        for el in parsed.elements:
            for raw, element_type, page in _element_units(el):
                ctx = ""
                if use_contextual:
                    key = chunk_content_hash(parsed.title, el.section, raw)
                    if key in context_cache:
                        ctx = context_cache[key]
                    else:
                        print(f"  augmenting {element_type} in '{el.section}'...", flush=True)
                        ctx = contextual_augment(doc_text, raw)
                        context_cache[key] = ctx
                        _save_json(CONTEXT_CACHE_PATH, context_cache)  # resumable
                embed_text = f"{ctx}\n\n{raw}" if ctx else raw
                chunks.append(Chunk(
                    id=_hash_id(doc_id, el.section, element_type, raw[:200]),
                    doc_id=doc_id, doc_title=parsed.title, text=raw, context=ctx,
                    embed_text=embed_text, section=el.section,
                    metadata={"source_file": path.name,
                              "element_type": element_type, "page": page,
                              **_acl_for(path.name, acl_map),
                              **_version_for(path.name, versions_map)},
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
                  rebuild: bool = False, force_cloud: bool = False) -> None:
    print(f"[ingest] parsing documents in {docs_dir} "
          f"(contextual={use_contextual}, rebuild={rebuild}, force_cloud={force_cloud})")
    context_cache = {} if rebuild else _load_json(CONTEXT_CACHE_PATH, {})
    previous_manifest = {} if rebuild else _load_json(MANIFEST_PATH, {})

    chunks = build_chunks(docs_dir, use_contextual=use_contextual,
                          context_cache=context_cache, force_cloud=force_cloud)
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
