"""The answer cache is PARTITIONED, not skipped.

It used to be bypassed whenever a principal was present — i.e. always, once
ENABLE_ACL is on, so the hit rate was 0% in exactly the deployments that need
it. The key is the principal's VISIBILITY (sorted roles + sites, not the user
id: those are what acl_visible decides on) plus the index generation, so two
users with the same grants share an answer, different visibility never does,
and an ingest invalidates everything.
"""
import json

import pytest

from src.access import AccessContext
from src.cache import SemanticCache, cache_scope
from src.config import Settings
from src.orchestrator import Orchestrator


def _settings(**kw):
    base = dict(_env_file=None, enable_semantic_cache=True,
                semantic_cache_threshold=0.95, semantic_cache_max=10)
    base.update(kw)
    return Settings(**base)


def _ctx(roles=(), sites=(), user_id="u"):
    return AccessContext(user_id=user_id, roles=frozenset(roles),
                         sites=frozenset(sites))


# ---------- what the scope is made of ----------

def test_same_visibility_shares_one_scope():
    assert cache_scope(_ctx(["eng"], ["siteA"], "alice"), 1) == \
           cache_scope(_ctx(["eng"], ["siteA"], "bob"), 1)


def test_the_user_id_does_not_partition_the_cache():
    """acl_visible decides on roles and sites, so keying on the user id would
    give every user a private cache and almost no hits."""
    a = cache_scope(_ctx(["eng"], [], "alice"), 1)
    b = cache_scope(_ctx(["eng"], [], "bob"), 1)
    assert a == b


def test_different_roles_are_different_scopes():
    assert cache_scope(_ctx(["eng"]), 1) != cache_scope(_ctx(["ops"]), 1)


def test_different_sites_are_different_scopes():
    assert cache_scope(_ctx([], ["siteA"]), 1) != cache_scope(_ctx([], ["siteB"]), 1)


def test_role_order_does_not_matter():
    assert cache_scope(_ctx(["eng", "mgr"]), 1) == cache_scope(_ctx(["mgr", "eng"]), 1)


def test_admin_is_its_own_scope():
    assert cache_scope(AccessContext.admin(), 1) != cache_scope(_ctx(["admin"]), 1)


def test_acl_disabled_is_its_own_scope():
    assert cache_scope(None, 1) != cache_scope(_ctx(), 1)


def test_an_ingest_invalidates_every_scope():
    """A cached answer must not outlive the documents it cited."""
    assert cache_scope(_ctx(["eng"]), 1) != cache_scope(_ctx(["eng"]), 2)
    assert cache_scope(None, 1) != cache_scope(None, 2)


# ---------- the cache honours the scope ----------

def test_an_entry_is_not_served_to_another_scope():
    c = SemanticCache(_settings())
    c.put([1.0, 0.0], {"answer": "eng only"}, scope="eng")
    assert c.get([1.0, 0.0], scope="ops") is None
    assert c.get([1.0, 0.0], scope="eng")["answer"] == "eng only"


def test_the_scope_is_checked_before_the_nearest_neighbour_search():
    """Filtering after the search lets an entry from another scope win and
    shadow a permitted one that would have matched."""
    c = SemanticCache(_settings(semantic_cache_threshold=0.9))
    c.put([1.0, 0.0], {"answer": "other scope, exact vector"}, scope="ops")
    c.put([1.0, 0.05], {"answer": "my scope, near vector"}, scope="eng")

    hit = c.get([1.0, 0.0], scope="eng")
    assert hit is not None, "the nearest neighbour in MY scope was shadowed"
    assert hit["answer"] == "my scope, near vector"


def test_eviction_still_applies_across_scopes():
    c = SemanticCache(_settings(semantic_cache_max=2))
    c.put([1.0, 0.0], {"answer": "a"}, scope="s1")
    c.put([0.0, 1.0], {"answer": "b"}, scope="s2")
    c.put([1.0, 1.0], {"answer": "c"}, scope="s3")
    assert c.get([1.0, 0.0], scope="s1") is None


# ---------- end to end through the orchestrator ----------

class _FakeRetriever:
    def __init__(self):
        self._chunks = []

    def embed_query(self, query):
        return [1.0, 0.0] if query == "same" else [0.0, 1.0]

    async def hybrid_retrieve(self, query, filters=None, access=None):
        return []


class _FakeGraph:
    def __init__(self):
        self.calls = 0

    async def ainvoke(self, initial):
        self.calls += 1
        return {**initial, "answer": f"answer {self.calls}", "retrieval": [],
                "citations": {"valid": True}, "grade": "sufficient", "rewritten": ""}


@pytest.fixture
def at_generation(tmp_path, monkeypatch):
    """A committed index generation the cache scope can read."""
    import src.ingestion as ing
    p = tmp_path / "index_meta.json"
    monkeypatch.setattr(ing, "INDEX_META_PATH", p)

    def _set(n):
        p.write_text(json.dumps({"embed_model": "m", "generation": n,
                                 "committed_generation": n}), encoding="utf-8")
    _set(1)
    return _set


def _orchestrator(cache=None, graph=None):
    return Orchestrator(_FakeRetriever(),
                        settings=_settings(agent_mode="pipeline"),
                        graph_full=graph or _FakeGraph(),
                        cache=cache if cache is not None else SemanticCache(_settings()))


async def test_a_principal_now_gets_cache_hits(at_generation):
    """The whole point: under ACLs the cache used to be dead weight."""
    orch = _orchestrator()
    access = _ctx(["eng"], ["siteA"])
    first = await orch.run("same", access=access)
    second = await orch.run("same", access=access)
    assert first["cached"] is False
    assert second["cached"] is True
    assert second["answer"] == first["answer"]


async def test_a_second_principal_with_the_same_grants_shares_the_answer(at_generation):
    orch = _orchestrator()
    await orch.run("same", access=_ctx(["eng"], ["siteA"], "alice"))
    out = await orch.run("same", access=_ctx(["eng"], ["siteA"], "bob"))
    assert out["cached"] is True


async def test_an_answer_is_never_served_to_a_different_visibility(at_generation):
    orch = _orchestrator()
    await orch.run("same", access=_ctx(["eng"], ["siteA"]))
    out = await orch.run("same", access=_ctx(["ops"], ["siteA"]))
    assert out["cached"] is False
    assert out["answer"] == "answer 2"


async def test_an_ingest_drops_the_cached_answer(at_generation):
    orch = _orchestrator()
    access = _ctx(["eng"])
    await orch.run("same", access=access)
    at_generation(2)                      # a new index generation is claimed
    out = await orch.run("same", access=access)
    assert out["cached"] is False


async def test_streaming_replays_a_cached_answer_for_the_same_scope(at_generation):
    orch = _orchestrator()
    access = _ctx(["eng"])
    await orch.run("same", access=access)
    events = [e async for e in orch.stream("same", access=access)]
    assert events[-1]["type"] == "final" and events[-1]["cached"] is True
