"""Serving a corpus the index has moved past.

chunks.json is loaded once, at startup. Ingestion updates Qdrant as it goes and
rewrites chunks.json only at the end, so for the whole duration of an ingest a
serving process holds ACL labels and document statuses the index has already
changed. While its generation differs from the one on disk the sparse leg must
re-resolve metadata from Qdrant and drop what the index no longer holds, so a
revocation or a deletion applies at once rather than at the next restart.
"""
import json

import pytest

from src.access import AccessContext
from src.config import Settings
from src.ingestion import point_id_for
from src.retrieval import Retriever


class _Record:
    def __init__(self, payload):
        self.payload = payload


class _FakeQdrant:
    """Only `retrieve` matters here; dense search is covered elsewhere."""

    def __init__(self, payloads=None):
        self.payloads = payloads or {}          # chunk_id -> payload
        self.retrieve_calls = 0

    def retrieve(self, *, collection_name, ids, with_payload=True, **kw):
        self.retrieve_calls += 1
        wanted = set(ids)
        return [_Record(p) for cid, p in self.payloads.items()
                if point_id_for(cid) in wanted]

    def query_points(self, **kw):
        raise AssertionError("dense search should not run in these tests")


def _chunk(cid="c1", text="pump seal replacement procedure", meta=None):
    return {"id": cid, "doc_id": "d1", "doc_title": "Pump Manual", "section": "Ops",
            "text": text, "context": "", "embed_text": text,
            "metadata": {"source_file": "pump.md", "element_type": "text",
                         "page": 1, **(meta or {})}}


def _payload(cid="c1", meta=None, text="pump seal replacement procedure"):
    c = _chunk(cid, text)
    c["metadata"].update(meta or {})
    return c


@pytest.fixture
def meta_path(tmp_path, monkeypatch):
    """Redirect index_meta.json and start at generation 1, fully committed."""
    import src.ingestion as ing
    p = tmp_path / "index_meta.json"
    monkeypatch.setattr(ing, "INDEX_META_PATH", p)
    p.write_text(json.dumps({"embed_model": "m", "generation": 1,
                             "committed_generation": 1}), encoding="utf-8")
    return p


def _set_generations(meta_path, claimed, committed):
    meta_path.write_text(json.dumps({"embed_model": "m", "generation": claimed,
                                     "committed_generation": committed}),
                         encoding="utf-8")


# BM25Okapi floors the idf of a term that appears in every document, so a
# one-document corpus scores <= 0 and retrieves nothing. Filler with disjoint
# vocabulary keeps the idf of the query terms positive.
FILLER = [("f1", "valve calibration chart"),
          ("f2", "bearing lubrication schedule"),
          ("f3", "motor winding resistance")]


def _write_chunks(path, *chunks):
    body = list(chunks) + [_chunk(cid, text) for cid, text in FILLER]
    path.write_text(json.dumps(body), encoding="utf-8")


@pytest.fixture
def chunks_file(tmp_path):
    p = tmp_path / "chunks.json"
    _write_chunks(p, _chunk())
    return p


def _retriever(chunks_file, client):
    return Retriever(chunks_file, load_models=False, qdrant_client=client)


# ---------- the snapshot is trusted only while the generation matches ----------

def test_sparse_uses_the_snapshot_while_the_generation_matches(meta_path, chunks_file):
    client = _FakeQdrant()
    r = _retriever(chunks_file, client)
    out = r.sparse_search("pump seal", top_k=5)
    assert [x.chunk_id for x in out] == ["c1"]
    assert client.retrieve_calls == 0, "no reason to re-resolve a current snapshot"


def test_retriever_records_the_generation_it_loaded(meta_path, chunks_file):
    r = _retriever(chunks_file, _FakeQdrant())
    assert r.generation == 1
    assert r.index_has_moved() is False
    _set_generations(meta_path, claimed=2, committed=1)
    assert r.index_has_moved() is True


# ---------- while behind, metadata comes from the index ----------

def test_a_revoked_acl_applies_at_once_rather_than_at_the_next_restart(
        meta_path, chunks_file):
    """Snapshot says public; the index says eng-only. The index wins."""
    client = _FakeQdrant({"c1": _payload(meta={"acl_roles": ["eng"]})})
    r = _retriever(chunks_file, client)
    _set_generations(meta_path, claimed=2, committed=1)

    out = r.sparse_search("pump seal", top_k=5,
                          access=AccessContext(user_id="u", roles=frozenset({"ops"})))
    assert out == []
    assert client.retrieve_calls == 1


def test_a_chunk_the_index_no_longer_holds_is_dropped(meta_path, chunks_file):
    """Fail closed: a deletion applies immediately, not at the next restart."""
    client = _FakeQdrant({})          # Qdrant holds nothing
    r = _retriever(chunks_file, client)
    _set_generations(meta_path, claimed=2, committed=1)
    assert r.sparse_search("pump seal", top_k=5) == []


def test_a_superseded_revision_from_the_index_is_hidden(meta_path, chunks_file):
    client = _FakeQdrant({"c1": _payload(meta={"doc_status": "superseded"})})
    r = _retriever(chunks_file, client)
    _set_generations(meta_path, claimed=2, committed=1)
    assert r.sparse_search("pump seal", top_k=5) == []


def test_a_chunk_the_index_still_permits_is_served(meta_path, chunks_file):
    client = _FakeQdrant({"c1": _payload()})
    r = _retriever(chunks_file, client)
    _set_generations(meta_path, claimed=2, committed=1)
    out = r.sparse_search("pump seal", top_k=5)
    assert [x.chunk_id for x in out] == ["c1"]


def test_live_metadata_reaches_the_boundary_check(meta_path, chunks_file):
    """_authorized re-checks Retrieved.metadata, so it must carry the live
    labels rather than the snapshot's, which would re-open what the index closed."""
    client = _FakeQdrant({"c1": _payload(meta={"acl_roles": ["eng"]})})
    r = _retriever(chunks_file, client)
    _set_generations(meta_path, claimed=2, committed=1)
    out = r.sparse_search("pump seal", top_k=5, access=AccessContext.admin())
    assert out[0].metadata.get("acl_roles") == ["eng"]


# ---------- the reload ----------

def test_reload_is_a_noop_when_the_corpus_is_current(meta_path, chunks_file):
    r = _retriever(chunks_file, _FakeQdrant())
    assert r.reload_if_stale() is False


def test_reload_is_skipped_while_an_ingest_is_in_flight(meta_path, chunks_file):
    """Rebuilding from a half-written chunks.json is worse than being stale."""
    r = _retriever(chunks_file, _FakeQdrant())
    _write_chunks(chunks_file, _chunk(), _chunk("c2", "gearbox oil change"))
    _set_generations(meta_path, claimed=2, committed=1)      # claimed, not committed
    assert r.reload_if_stale() is False
    assert "c2" not in {c["id"] for c in r._chunks}


def test_reload_makes_new_documents_keyword_searchable(meta_path, chunks_file):
    r = _retriever(chunks_file, _FakeQdrant())
    assert r.sparse_search("gearbox oil", top_k=5) == []
    _write_chunks(chunks_file, _chunk(), _chunk("c2", "gearbox oil change"))
    _set_generations(meta_path, claimed=2, committed=2)
    assert r.reload_if_stale() is True
    assert [x.chunk_id for x in r.sparse_search("gearbox oil", top_k=5)] == ["c2"]
    assert r.generation == 2


def test_the_corpus_and_its_bm25_index_swap_as_one_reference(meta_path, chunks_file):
    """BM25 returns positions into the list it was built from, so a half-swapped
    pair resolves them against the wrong documents."""
    r = _retriever(chunks_file, _FakeQdrant())
    before = r._corpus
    _write_chunks(chunks_file, _chunk("c2", "gearbox oil change"))
    _set_generations(meta_path, claimed=2, committed=2)
    r.reload_if_stale()
    assert r._corpus is not before
    assert before.chunks[0]["id"] == "c1"        # the old pair is still coherent
    assert r._corpus.chunks[0]["id"] == "c2"


def test_reload_resolves_the_index_moved_state(meta_path, chunks_file):
    r = _retriever(chunks_file, _FakeQdrant())
    _set_generations(meta_path, claimed=2, committed=2)
    assert r.index_has_moved() is True
    r.reload_if_stale()
    assert r.index_has_moved() is False


# ---------- the poller has to actually run ----------

def _stub_startup(monkeypatch, poll_s):
    import src.api as api
    settings = Settings(_env_file=None, index_reload_poll_s=poll_s)
    monkeypatch.setattr(Settings, "validate_provider", lambda self: None)
    monkeypatch.setattr(Settings, "validate_access", lambda self, *a, **k: None)
    monkeypatch.setattr(api, "get_settings", lambda: settings)
    monkeypatch.setattr(api, "Orchestrator", lambda *a, **k: object())
    monkeypatch.setattr("src.retrieval.Retriever", lambda *a, **k: object())
    return api


async def test_lifespan_starts_the_reload_poller(monkeypatch):
    api = _stub_startup(monkeypatch, poll_s=60)
    async with api.lifespan(api.app):
        task = api.app.state.index_reload_task
        assert task is not None and not task.done()
    assert task.cancelled() or task.done(), "poller must be stopped on shutdown"


async def test_poller_is_disabled_when_the_interval_is_zero(monkeypatch):
    api = _stub_startup(monkeypatch, poll_s=0)
    async with api.lifespan(api.app):
        assert api.app.state.index_reload_task is None
