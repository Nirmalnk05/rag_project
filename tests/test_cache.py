from src.cache import SemanticCache
from src.config import Settings


def _settings(**kw):
    base = dict(_env_file=None, enable_semantic_cache=True,
                semantic_cache_threshold=0.95, semantic_cache_max=3)
    base.update(kw)
    return Settings(**base)


def test_disabled_never_hits():
    c = SemanticCache(_settings(enable_semantic_cache=False))
    c.put([1.0, 0.0], {"answer": "a"})
    assert c.get([1.0, 0.0]) is None


def test_exact_match_hits():
    c = SemanticCache(_settings())
    c.put([1.0, 0.0], {"answer": "cached"})
    assert c.get([1.0, 0.0])["answer"] == "cached"


def test_close_vector_hits_above_threshold():
    c = SemanticCache(_settings(semantic_cache_threshold=0.9))
    c.put([1.0, 0.0], {"answer": "cached"})
    # nearly parallel vector → cosine ~0.9997
    assert c.get([1.0, 0.02])["answer"] == "cached"


def test_distant_vector_misses():
    c = SemanticCache(_settings(semantic_cache_threshold=0.95))
    c.put([1.0, 0.0], {"answer": "cached"})
    assert c.get([0.0, 1.0]) is None  # orthogonal → cosine 0


def test_fifo_eviction():
    c = SemanticCache(_settings(semantic_cache_max=2))
    c.put([1.0, 0.0], {"answer": "a"})
    c.put([0.0, 1.0], {"answer": "b"})
    c.put([1.0, 1.0], {"answer": "c"})   # evicts "a"
    assert c.get([1.0, 0.0]) is None     # "a" gone (orthogonal-ish to survivors under thresh)
    assert c.get([0.0, 1.0])["answer"] == "b"
