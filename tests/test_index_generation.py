"""The index generation must be claimed BEFORE Qdrant is touched.

Writing it only at the end leaves a window with Qdrant already updated and every
serving process still believing its chunks.json snapshot is current — which is
exactly the window a revocation lands in. Readers compare generations, never
timestamps: mtime granularity differs by filesystem and is not a reliable
ordering on a shared RWX volume.
"""
import json
from pathlib import Path

import pytest

from src.ingestion import (Chunk, committed_generation, index_generation,
                           run_ingestion)


@pytest.fixture
def index_dir(tmp_path, monkeypatch):
    """Point every ingestion artefact at a temp dir."""
    import src.ingestion as ing
    monkeypatch.setattr(ing, "CHUNK_STORE_PATH", tmp_path / "chunks.json")
    monkeypatch.setattr(ing, "MANIFEST_PATH", tmp_path / "manifest.json")
    monkeypatch.setattr(ing, "CONTEXT_CACHE_PATH", tmp_path / "context.json")
    monkeypatch.setattr(ing, "INDEX_META_PATH", tmp_path / "index_meta.json")
    return tmp_path


def _chunk(text="pump procedure"):
    return Chunk(id="c1", doc_id="d1", doc_title="Pump Manual", text=text,
                 context="", embed_text=text, section="Ops",
                 metadata={"source_file": "a.md", "element_type": "text", "page": 1})


class _StubEmbedder:
    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, texts, **kw):
        import numpy as np
        return np.zeros((len(texts), 3))


@pytest.fixture
def ingest(index_dir, monkeypatch):
    """run_ingestion with the parsers and the embedder stubbed out.

    index_qdrant_incremental is replaced by a spy that reads index_meta.json at
    the moment it is called, so the test observes what a *serving process* would
    see while Qdrant is being written.
    """
    import src.ingestion as ing
    monkeypatch.setattr(ing, "build_chunks", lambda *a, **k: [_chunk()])
    monkeypatch.setattr("sentence_transformers.SentenceTransformer",
                        lambda *a, **k: _StubEmbedder())
    seen: dict = {}

    def _spy(chunks, embedder, **kw):
        seen["generation"] = index_generation(ing.INDEX_META_PATH)
        seen["committed"] = committed_generation(ing.INDEX_META_PATH)
        seen["chunks_on_disk"] = Path(ing.CHUNK_STORE_PATH).exists()

    monkeypatch.setattr(ing, "index_qdrant_incremental", _spy)
    return seen


def test_generation_is_zero_before_any_ingest(index_dir):
    assert index_generation(index_dir / "index_meta.json") == 0
    assert committed_generation(index_dir / "index_meta.json") == 0


def test_first_ingest_claims_generation_one_before_touching_qdrant(index_dir, ingest):
    run_ingestion(docs_dir=index_dir, use_contextual=False)
    assert ingest["generation"] == 1


def test_the_claim_is_visible_while_chunks_json_is_still_absent(index_dir, ingest):
    """The whole point: Qdrant moves first, the snapshot is rewritten last."""
    run_ingestion(docs_dir=index_dir, use_contextual=False)
    assert ingest["chunks_on_disk"] is False
    assert ingest["committed"] == 0      # nothing committed at that instant


def test_generation_is_committed_once_the_sidecars_are_written(index_dir, ingest):
    run_ingestion(docs_dir=index_dir, use_contextual=False)
    meta_path = index_dir / "index_meta.json"
    assert index_generation(meta_path) == 1
    assert committed_generation(meta_path) == 1


def test_each_ingest_bumps_the_generation(index_dir, ingest):
    run_ingestion(docs_dir=index_dir, use_contextual=False)
    run_ingestion(docs_dir=index_dir, use_contextual=False)
    assert ingest["generation"] == 2                       # claimed before Qdrant again
    assert index_generation(index_dir / "index_meta.json") == 2


def test_a_second_ingest_claims_before_committing(index_dir, ingest):
    run_ingestion(docs_dir=index_dir, use_contextual=False)
    run_ingestion(docs_dir=index_dir, use_contextual=False)
    # during the 2nd run the previous generation was still the committed one
    assert ingest["committed"] == 1


def test_embed_model_survives_the_generation_writes(index_dir, ingest):
    from src.config import get_settings
    run_ingestion(docs_dir=index_dir, use_contextual=False)
    meta = json.loads((index_dir / "index_meta.json").read_text(encoding="utf-8"))
    assert meta["embed_model"] == get_settings().embed_model


def test_unparseable_meta_reads_as_generation_zero(index_dir):
    meta = index_dir / "index_meta.json"
    meta.write_text("{not json", encoding="utf-8")
    assert index_generation(meta) == 0
