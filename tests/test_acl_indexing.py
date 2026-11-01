"""ACL metadata must reach the Qdrant payload, and the fields it filters on must
be indexed — on existing collections too, not only freshly created ones."""
from pathlib import Path

import pytest

from src.config import Settings
from src.ingestion import (ACL_INDEXED_KEYS, Chunk, chunk_content_hash, chunk_index_hash,
                           ensure_payload_indexes)


def _chunk(acl_roles=None, acl_sites=None, text="pump procedure"):
    meta = {"source_file": "a.md", "element_type": "text", "page": 1}
    if acl_roles is not None:
        meta["acl_roles"] = acl_roles
    if acl_sites is not None:
        meta["acl_sites"] = acl_sites
    return Chunk(id="c1", doc_id="d1", doc_title="Pump Manual", text=text,
                 context="", embed_text=text, section="Ops", metadata=meta)


# ---------- the manifest hash must notice an ACL-only change ----------

def test_index_hash_changes_when_acl_roles_change():
    """Populating acl.json must re-upsert, or Qdrant keeps serving public chunks."""
    assert chunk_index_hash(_chunk()) != chunk_index_hash(_chunk(acl_roles=["eng"]))


def test_index_hash_changes_when_acl_sites_change():
    assert chunk_index_hash(_chunk()) != chunk_index_hash(_chunk(acl_sites=["siteA"]))


def test_index_hash_changes_when_doc_status_changes():
    a = _chunk()
    b = _chunk()
    b.metadata["doc_status"] = "superseded"
    assert chunk_index_hash(a) != chunk_index_hash(b)


def test_index_hash_is_stable_and_order_independent():
    a = _chunk(acl_roles=["eng", "mgr"])
    b = _chunk(acl_roles=["mgr", "eng"])
    assert chunk_index_hash(a) == chunk_index_hash(a)
    assert chunk_index_hash(a) == chunk_index_hash(b)


def test_content_hash_is_unaffected_by_acl():
    """The context cache is keyed on content; an ACL edit must not invalidate it."""
    a, b = _chunk(), _chunk(acl_roles=["eng"])
    assert (chunk_content_hash(a.doc_title, a.section, a.text)
            == chunk_content_hash(b.doc_title, b.section, b.text))


# ---------- payload indexes ----------

class _FakeClient:
    def __init__(self):
        self.indexed: list[tuple[str, object]] = []

    def create_payload_index(self, *, collection_name, field_name, field_schema, **kw):
        self.indexed.append((field_name, field_schema))


def test_payload_indexes_cover_the_acl_fields():
    client = _FakeClient()
    ensure_payload_indexes(client, "coll")
    assert {"metadata.acl_roles", "metadata.acl_sites"} <= {f for f, _ in client.indexed}


def test_payload_indexes_cover_the_user_filter_fields():
    client = _FakeClient()
    ensure_payload_indexes(client, "coll")
    names = {f for f, _ in client.indexed}
    assert {"metadata.source_file", "metadata.element_type", "metadata.page",
            "doc_title", "section"} <= names
    assert names == set(ACL_INDEXED_KEYS)


def test_payload_index_errors_do_not_abort_ingestion():
    """Qdrant raises when an index already exists; that must not fail the run."""
    class _Raising(_FakeClient):
        def create_payload_index(self, **kw):
            raise RuntimeError("index already exists")

    ensure_payload_indexes(_Raising(), "coll")   # must not propagate


# ---------- fail-closed configuration gates ----------

def test_acl_sidecar_without_enable_acl_is_rejected(tmp_path):
    acl = tmp_path / "acl.json"
    acl.write_text('{"a.md": {"roles": ["eng"]}}', encoding="utf-8")
    s = Settings(enable_acl=False)
    with pytest.raises(ValueError, match="ENABLE_ACL"):
        s.validate_access(acl_path=acl)


def test_enable_acl_requires_explicit_identity_trust(tmp_path):
    s = Settings(enable_acl=True, trust_identity_headers=False)
    with pytest.raises(ValueError, match="TRUST_IDENTITY_HEADERS"):
        s.validate_access(acl_path=tmp_path / "missing.json")


def test_enable_acl_with_trusted_identity_is_accepted(tmp_path):
    s = Settings(enable_acl=True, trust_identity_headers=True)
    s.validate_access(acl_path=tmp_path / "missing.json")


def test_no_acl_sidecar_and_acl_disabled_is_accepted(tmp_path):
    Settings(enable_acl=False).validate_access(acl_path=tmp_path / "missing.json")


# ---------- the gate has to actually run at startup ----------

async def test_startup_runs_the_access_gate(monkeypatch):
    """A gate that is never called protects nothing."""
    import src.api as api

    called = []
    monkeypatch.setattr(Settings, "validate_access",
                        lambda self, *a, **k: called.append(True))
    monkeypatch.setattr(Settings, "validate_provider", lambda self: None)
    monkeypatch.setattr(api, "Orchestrator", lambda *a, **k: object())
    monkeypatch.setattr("src.retrieval.Retriever", lambda *a, **k: object())

    async with api.lifespan(api.app):
        pass
    assert called, "lifespan did not call Settings.validate_access"
