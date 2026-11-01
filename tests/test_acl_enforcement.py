"""End-to-end ACL invariant: trusted identity -> indexed metadata -> filtered
retrieval -> independent boundary check.

The differential test below interprets the *actual* Filter object
`src.retrieval.acl_filter` builds, resolving each condition's key as a payload
path. A wrong key (`acl_roles` instead of `metadata.acl_roles`) or a missing
dimension therefore makes it disagree with `acl_visible`, which is exactly the
fail-open shape we need to catch without a live Qdrant.
"""
import itertools
import json

import pytest
from qdrant_client.models import Filter, FieldCondition, IsEmptyCondition

from src.access import AccessContext, acl_visible
from src.retrieval import Retriever, acl_filter

ROLE_SETS = [None, [], ["eng"], ["eng", "mgr"], ["secret"]]
SITE_SETS = [None, [], ["siteA"], ["siteA", "siteB"], ["siteC"]]


def _payload(roles, sites):
    """A Qdrant payload. `None` means the key is absent (how `_acl_for` emits public)."""
    meta = {"source_file": "a.md"}
    if roles is not None:
        meta["acl_roles"] = roles
    if sites is not None:
        meta["acl_sites"] = sites
    return {"id": "c1", "metadata": meta}


def _resolve(payload: dict, key: str):
    """Resolve a dotted Qdrant payload path, or None when any segment is missing."""
    cur = payload
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _eval_condition(payload: dict, cond) -> bool:
    if isinstance(cond, Filter):
        return _eval_filter(payload, cond)
    if isinstance(cond, IsEmptyCondition):
        return not _resolve(payload, cond.is_empty.key)
    if isinstance(cond, FieldCondition):
        actual = _resolve(payload, cond.key)
        match = cond.match
        if getattr(match, "any", None) is not None:
            return bool(set(actual or []) & set(match.any))
        return str(actual) == str(match.value)
    raise AssertionError(f"filter uses a condition this test cannot evaluate: {cond!r}")


def _eval_filter(payload: dict, flt: Filter) -> bool:
    if all(_eval_condition(payload, c) for c in (flt.must or [])) is False:
        return False
    should = flt.should or []
    if should and not any(_eval_condition(payload, c) for c in should):
        return False
    for c in flt.must_not or []:
        if _eval_condition(payload, c):
            return False
    return True


# ---------- 1. the Qdrant translation must agree with acl_visible ----------

@pytest.mark.parametrize("roles,sites", list(itertools.product(ROLE_SETS, SITE_SETS)))
def test_acl_filter_agrees_with_acl_visible(roles, sites):
    access = AccessContext(user_id="u", roles=frozenset({"eng"}),
                           sites=frozenset({"siteA"}))
    payload = _payload(roles, sites)
    flt = acl_filter(access)
    assert _eval_filter(payload, flt) is acl_visible(payload["metadata"], access)


def test_acl_filter_is_none_when_unrestricted():
    assert acl_filter(None) is None
    assert acl_filter(AccessContext.admin()) is None


def test_acl_filter_denies_everything_restricted_for_principal_with_no_grants():
    """A principal with no roles/sites still sees public chunks, nothing gated."""
    access = AccessContext(user_id="u")
    flt = acl_filter(access)
    assert _eval_filter(_payload(None, None), flt) is True
    assert _eval_filter(_payload(["eng"], None), flt) is False
    assert _eval_filter(_payload(None, ["siteA"]), flt) is False


# ---------- 2. dense_search must send that filter to Qdrant ----------

class _FakeQdrant:
    """Captures query_filter and returns whatever points it is told to."""

    def __init__(self, points=()):
        self.points = list(points)
        self.last_filter = None

    def query_points(self, **kw):
        self.last_filter = kw.get("query_filter")
        return type("Res", (), {"points": self.points})()


def _point(cid, meta, score=0.9):
    payload = {"id": cid, "doc_id": "d", "doc_title": "T", "section": "S",
               "text": f"{cid} text", "context": "", "metadata": meta}
    return type("P", (), {"payload": payload, "score": score})()


@pytest.fixture
def retriever(tmp_path):
    def _c(cid, text, src):
        return {"id": cid, "doc_id": "d", "doc_title": "T", "section": "S",
                "text": text, "context": "", "embed_text": text,
                "metadata": {"source_file": src}}
    chunks = [_c("c1", "public pump text", "a.md"),
              # fillers so 'pump' stays discriminating (BM25 IDF positive)
              _c("f1", "chiller condenser cleaning", "f1.md"),
              _c("f2", "bearing alignment tolerance", "f2.md"),
              _c("f3", "earth fault cable gland", "f3.md")]
    path = tmp_path / "chunks.json"
    path.write_text(json.dumps(chunks), encoding="utf-8")

    def _make(client):
        r = Retriever(chunks_path=path, load_models=False, qdrant_client=client)
        r._embedder = type("E", (), {"encode": lambda self, q, **k: _Vec()})()
        return r
    return _make


class _Vec(list):
    def __init__(self):
        super().__init__([0.1, 0.2, 0.3])

    def tolist(self):
        return list(self)


def test_dense_search_pushes_acl_into_the_query_filter(retriever):
    client = _FakeQdrant()
    r = retriever(client)
    r.dense_search("pump", top_k=5,
                   access=AccessContext(roles=frozenset({"eng"}), sites=frozenset({"siteA"})))
    keys = json.dumps(client.last_filter.model_dump() if client.last_filter else {})
    assert "metadata.acl_roles" in keys
    assert "metadata.acl_sites" in keys


def test_dense_search_keeps_user_filters_alongside_acl(retriever):
    client = _FakeQdrant()
    r = retriever(client)
    r.dense_search("pump", top_k=5, filters={"source_file": "a.md"},
                   access=AccessContext(roles=frozenset({"eng"})))
    dumped = json.dumps(client.last_filter.model_dump())
    assert "metadata.source_file" in dumped and "metadata.acl_roles" in dumped


# ---------- 3. ACL metadata must survive to the boundary ----------

def test_retrieved_carries_acl_metadata_but_to_dict_does_not(retriever):
    client = _FakeQdrant([_point("c9", {"source_file": "b.md", "acl_roles": ["eng"]})])
    r = retriever(client)
    out = r.dense_search("pump", top_k=5, access=AccessContext(roles=frozenset({"eng"})))
    assert out[0].metadata.get("acl_roles") == ["eng"]
    assert "acl_roles" not in out[0].to_dict()
    assert "metadata" not in out[0].to_dict()


def test_sparse_search_carries_acl_metadata(retriever):
    r = retriever(_FakeQdrant())
    out = r.sparse_search("pump", top_k=5)
    assert out[0].metadata.get("source_file") == "a.md"


def test_clone_preserves_metadata():
    from src.retrieval import Retrieved, _clone
    r = Retrieved("c1", "d", "t", "s", "x", "", 1.0, "bm25", {"acl_roles": ["eng"]})
    assert _clone(r, score=2.0, source="rrf").metadata == {"acl_roles": ["eng"]}


# ---------- 4. the boundary re-check must catch a stale/incorrect index ----------

def test_authorized_removes_chunks_the_principal_may_not_see(retriever):
    from src.retrieval import Retrieved
    r = retriever(_FakeQdrant())
    results = [
        Retrieved("ok", "d", "t", "s", "x", "", 1.0, "rerank", {"source_file": "a.md"}),
        Retrieved("leak", "d", "t", "s", "y", "", 0.9, "rerank", {"acl_roles": ["secret"]}),
    ]
    kept = r._authorized(results, AccessContext(roles=frozenset({"eng"})))
    assert [k.chunk_id for k in kept] == ["ok"]


def test_authorized_is_a_no_op_when_acl_is_disabled(retriever):
    from src.retrieval import Retrieved
    r = retriever(_FakeQdrant())
    results = [Retrieved("x", "d", "t", "s", "x", "", 1.0, "rerank", {"acl_roles": ["secret"]})]
    assert r._authorized(results, None) == results


async def test_hybrid_retrieve_applies_the_boundary_check(retriever):
    """Stands in for a filter-translation bug or a future path that skips
    per-search filtering: rerank yields a restricted chunk, the boundary drops it."""
    r = retriever(_FakeQdrant())
    leaked = [__import__("src.retrieval", fromlist=["Retrieved"]).Retrieved(
        "leak", "d", "t", "s", "y", "", 1.0, "rerank", {"acl_roles": ["secret"]})]
    r.rerank = lambda query, candidates, top_k: leaked
    out = await r.hybrid_retrieve("pump", None, AccessContext(roles=frozenset({"eng"})))
    assert out == []
