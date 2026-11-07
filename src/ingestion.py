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
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

from src.config import get_settings
from src.parsing.dispatcher import SUPPORTED_EXT, Dispatcher
from src.parsing.markdown_parser import _split_sections as _split_into_sections  # re-export
from src.parsing.model import FigureElement, ParsedDocument, TableElement, TextElement

_settings = get_settings()

CHUNK_STORE_PATH = Path("data/chunks.json")
MANIFEST_PATH = Path("data/ingest_manifest.json")
CONTEXT_CACHE_PATH = Path("data/context_cache.json")
INDEX_META_PATH = Path("data/index_meta.json")
# {embed_model, generation, committed_generation} — see index_generation()


def _meta_path(meta_path: Path | str | None) -> Path:
    """Resolve the meta path at CALL time.

    A `meta_path: Path = INDEX_META_PATH` default binds the module constant
    as it was at import time, so redirecting INDEX_META_PATH leaves readers
    pointing at the original file while writers follow the new one.
    """
    return INDEX_META_PATH if meta_path is None else Path(meta_path)


def embed_model_changed(current_model: str, meta_path: Path | None = None) -> bool:
    """True if the index was built with a different embedding model (needs re-embed)."""
    prev = _read_index_meta(meta_path).get("embed_model", "")
    return bool(prev) and prev != current_model


def _read_index_meta(meta_path: Path | None = None) -> dict:
    """index_meta.json, or {} when it is missing, unreadable or half-written."""
    try:
        with _meta_path(meta_path).open("r", encoding="utf-8") as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    return meta if isinstance(meta, dict) else {}


def _meta_int(meta_path: Path | None, key: str) -> int:
    try:
        return int(_read_index_meta(meta_path).get(key, 0))
    except (TypeError, ValueError):
        return 0


def index_generation(meta_path: Path | None = None) -> int:
    """The generation ingestion has *claimed*.

    Bumped before Qdrant is touched, so a serving process can tell that the
    index has moved past the chunks.json snapshot it loaded at startup.
    """
    return _meta_int(meta_path, "generation")


def committed_generation(meta_path: Path | None = None) -> int:
    """The generation chunks.json and the sidecars are known to match.

    While an ingest is in flight this trails `index_generation`, which is what
    stops a reloading process from rebuilding BM25 off a half-written snapshot
    and then claiming the new generation.
    """
    return _meta_int(meta_path, "committed_generation")


def _write_index_meta(generation: int, committed: int) -> None:
    _save_json(INDEX_META_PATH, {"embed_model": _settings.embed_model,
                                 "generation": generation,
                                 "committed_generation": committed})
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
           "_windowed_chunks", "chunk_content_hash", "chunk_index_hash",
           "diff_manifest", "point_id_for", "ensure_payload_indexes",
           "ACL_INDEXED_KEYS", "index_generation", "committed_generation",
           "count_tokens", "chunk_token_budget", "oversize_units"]


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

_WORD_RE = re.compile(r"\S+")
_SPECIAL_TOKENS = 2        # [CLS] / [SEP] around every embedded sequence


@lru_cache(maxsize=4)
def _hf_tokenizer(model_name: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name)


def count_tokens(text: str, model_name: str | None = None) -> int:
    """Token count under the EMBEDDING model's own tokenizer.

    The same model the chunks are embedded with, so this is the measurement
    that decides what gets truncated. Loaded lazily, and never reached for
    text that is already short enough (see `_fits_on_size_alone`).
    """
    tok = _hf_tokenizer(model_name or _settings.embed_model)
    # verbose=False: measuring an over-long text is the point here, and the
    # tokenizer's "longer than the maximum sequence length" warning would fire
    # on exactly the chunks this function exists to catch.
    return len(tok.encode(text, add_special_tokens=False, verbose=False))


def _default_counter(text: str) -> int:
    """The real tokenizer, resolved at call time so tests can inject instead."""
    return count_tokens(text)


def _fits_on_size_alone(text: str, max_tokens: int) -> bool:
    """True when the byte length alone proves the text fits.

    No tokenizer emits more than one token per UTF-8 byte: WordPiece works on
    characters, byte-level BPE on bytes. So this is a sound bound, not an
    estimate — and it keeps short content off the tokenizer entirely.
    """
    return len(text.encode("utf-8")) <= max_tokens


def chunk_token_budget(use_contextual: bool, settings=None) -> int:
    """How many tokens a chunk's raw text may occupy.

    embed_text is the contextual blurb plus the chunk, and the blurb runs to
    ~120 tokens, so when augmentation is on its share has to come out of the
    window — otherwise every full chunk silently loses its tail.
    """
    s = settings or _settings
    budget = s.chunk_max_tokens - _SPECIAL_TOKENS
    if use_contextual:
        budget -= s.chunk_context_reserve_tokens
    return max(budget, 1)


def _windowed_chunks(text: str, *, max_tokens: int | None = None,
                     overlap_tokens: int | None = None,
                     count_tokens=None) -> list[str]:
    """Token-windowed chunking, cut on word boundaries.

    Sized in tokens because that is what the embedder truncates on: a 600-word
    window measured 743 tokens on this corpus and lost 31% of every full chunk,
    invisibly — the text was still stored and still shown to the LLM, so only
    dense recall degraded, on the tail.

    Every window is a slice of the original string, so the chunk text stays
    byte-identical to the source (joining on single spaces would rewrite the
    newlines and tabs that carry meaning inside tables and code blocks).
    """
    counter = count_tokens or _default_counter
    limit = chunk_token_budget(False) if max_tokens is None else max_tokens
    overlap = (_settings.chunk_overlap_tokens if overlap_tokens is None
               else overlap_tokens)

    spans = [(m.start(), m.end()) for m in _WORD_RE.finditer(text)]
    if not spans:
        return [text] if text.strip() else []
    if _fits_on_size_alone(text, limit) or counter(text) <= limit:
        return [text]

    costs = [counter(text[a:b]) for a, b in spans]
    out: list[str] = []
    start, n = 0, len(spans)
    while start < n:
        total, end = 0, start
        while end < n and (end == start or total + costs[end] <= limit):
            total += costs[end]
            end += 1
        # per-word costs can under-count a joined window (byte-level BPE merges
        # across the space), so confirm the real window before emitting it
        while end - start > 1 and counter(text[spans[start][0]:spans[end - 1][1]]) > limit:
            end -= 1
        out.append(text[spans[start][0]:spans[end - 1][1]])
        if end >= n:
            break
        back, i = 0, end
        while i > start + 1 and back + costs[i - 1] <= overlap:
            back += costs[i - 1]
            i -= 1
        start = max(i, start + 1)      # always make progress
    return out


def _element_units(el, *, max_tokens: int | None = None,
                   overlap_tokens: int | None = None,
                   count_tokens=None) -> list[tuple[str, str, int | None]]:
    """Turn one parsed element into (chunk_text, element_type, page) units."""
    if isinstance(el, TextElement):
        windows = _windowed_chunks(el.text, max_tokens=max_tokens,
                                   overlap_tokens=overlap_tokens,
                                   count_tokens=count_tokens)
        return [(w, "text", el.page) for w in windows]
    if isinstance(el, TableElement):
        return [(el.markdown, "table", el.page)]           # atomic — never split
    if isinstance(el, FigureElement):
        return [(el.caption, "figure", el.page)] if el.caption.strip() else []
    return []


def oversize_units(units: list[tuple[str, str, int | None]], *, max_tokens: int,
                   count_tokens=None) -> list[dict]:
    """The units that still do not fit the embedding window.

    Tables are atomic and a single long token-dense run cannot be split, so some
    content genuinely cannot be made to fit. The embedder would truncate it in
    silence and only dense recall would suffer, so it is reported at ingest.
    """
    counter = count_tokens or _default_counter
    over: list[dict] = []
    for raw, element_type, page in units:
        if _fits_on_size_alone(raw, max_tokens):
            continue
        n = counter(raw)
        if n > max_tokens:
            over.append({"element_type": element_type, "page": page,
                         "tokens": n, "preview": raw[:60].replace("\n", " ")})
    return over


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
    """Content identity. Also the contextual-augmentation cache key, so it must
    NOT cover ACL/version metadata — an ACL edit would re-run every LLM call."""
    return hashlib.sha256(f"{doc_title}||{section}||{raw_text}".encode()).hexdigest()


# Metadata that changes what the *index* must contain, as opposed to chunk
# content. These belong in the manifest hash: without them, populating
# data/acl.json leaves every content hash unchanged, so to_upsert is empty and
# the new ACL reaches chunks.json (rewritten wholesale) but never the Qdrant
# payload — sparse then enforces ACLs while dense serves restricted chunks.
INDEX_META_KEYS = ("acl_roles", "acl_sites", "doc_status", "doc_version", "doc_date")


def chunk_index_hash(chunk: "Chunk") -> str:
    """Manifest hash: chunk content plus the metadata retrieval filters on."""
    meta = chunk.metadata or {}
    parts = [chunk_content_hash(chunk.doc_title, chunk.section, chunk.text)]
    for k in INDEX_META_KEYS:
        v = meta.get(k)
        if isinstance(v, list):
            v = sorted(str(x) for x in v)   # membership, not order, is what matters
        parts.append(f"{k}={v!r}")
    return hashlib.sha256("||".join(parts).encode()).hexdigest()


# Payload paths the retrieval filters read, with the schema Qdrant should index.
# Filtered HNSW over an unindexed payload field degrades badly as the collection
# grows, so these are part of the ACL change, not an optimisation after it.
ACL_INDEXED_KEYS: dict[str, str] = {
    "metadata.acl_roles": "keyword",
    "metadata.acl_sites": "keyword",
    "metadata.source_file": "keyword",
    "metadata.element_type": "keyword",
    "metadata.page": "integer",
    "doc_title": "keyword",
    "section": "keyword",
}


def ensure_payload_indexes(client, collection_name: str) -> None:
    """Create the payload indexes, idempotently, on every ingest.

    Deliberately not inside the `create_collection` branch: a collection built
    before these existed would otherwise never acquire them. Qdrant errors when
    an index is already present, which is the normal case and not a failure.
    """
    from qdrant_client.models import PayloadSchemaType
    for field_name, schema in ACL_INDEXED_KEYS.items():
        try:
            client.create_payload_index(
                collection_name=collection_name, field_name=field_name,
                field_schema=PayloadSchemaType(schema),
            )
        except Exception as e:  # noqa: BLE001 — "already exists" is expected
            print(f"  payload index {field_name}: {e}")


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
                 force_cloud: bool = False, count_tokens=None,
                 max_tokens: int | None = None) -> list[Chunk]:
    context_cache = context_cache if context_cache is not None else {}
    dispatcher = dispatcher or Dispatcher()
    budget = chunk_token_budget(use_contextual) if max_tokens is None else max_tokens
    oversize: list[dict] = []
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
            units = _element_units(el, max_tokens=budget,
                                   count_tokens=count_tokens)
            for over in oversize_units(units, max_tokens=budget,
                                       count_tokens=count_tokens):
                oversize.append({**over, "source_file": path.name,
                                 "section": el.section})
            for raw, element_type, page in units:
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
    _report_oversize(oversize, budget)
    return chunks


def _report_oversize(oversize: list[dict], budget: int) -> None:
    """Say what will be truncated. Silence here is how the 31% loss went unseen."""
    if not oversize:
        return
    print(f"[ingest] WARNING: {len(oversize)} chunk(s) exceed the {budget}-token "
          "embedding window and will be truncated by the embedder — dense recall "
          "degrades for these (BM25 still indexes the full text):")
    for item in oversize:
        page = f", p{item['page']}" if item.get("page") is not None else ""
        print(f"           {item['source_file']} > {item['section']} "
              f"({item['element_type']}{page}): {item['tokens']} tokens")


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
    ensure_payload_indexes(client, _settings.qdrant_collection)

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
    if embed_model_changed(_settings.embed_model) and not rebuild:
        print(f"[ingest] WARNING: embedding model changed to '{_settings.embed_model}'. "
              "Existing vectors were built with a different model — run with --rebuild "
              "to re-embed everything, or retrieval will be inconsistent.")
    context_cache = {} if rebuild else _load_json(CONTEXT_CACHE_PATH, {})
    previous_manifest = {} if rebuild else _load_json(MANIFEST_PATH, {})

    chunks = build_chunks(docs_dir, use_contextual=use_contextual,
                          context_cache=context_cache, force_cloud=force_cloud)
    current_manifest = {c.id: chunk_index_hash(c) for c in chunks}
    new, changed, stale = diff_manifest(current_manifest, previous_manifest)
    to_upsert = new | changed if not rebuild else set(current_manifest)
    print(f"[ingest] {len(chunks)} chunks total — new={len(new)} changed={len(changed)} stale={len(stale)}")

    print("[ingest] loading embedding model...")
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer(_settings.embed_model)

    # Claim the next generation BEFORE Qdrant changes. Qdrant is updated as we
    # go while chunks.json is rewritten only at the end, so for the whole run a
    # serving process holds ACL labels and document statuses the index has
    # already moved past. The claim is what lets it notice.
    generation = index_generation() + 1
    _write_index_meta(generation, committed_generation())

    index_qdrant_incremental(chunks, embedder, to_upsert=to_upsert,
                             stale_ids=stale, rebuild=rebuild)

    _save_json(CHUNK_STORE_PATH, [c.to_dict() for c in chunks])
    _save_json(MANIFEST_PATH, current_manifest)
    _save_json(CONTEXT_CACHE_PATH, context_cache)
    # Only now does the snapshot on disk match the claimed generation.
    _write_index_meta(generation, generation)
    print(f"[ingest] done (generation {generation}).")
