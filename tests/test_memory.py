import json

from src.config import Settings
from src.memory import RedisSessionStore, SessionStore


def _settings(**kw):
    base = dict(_env_file=None, enable_memory=True, memory_max_turns=2, memory_ttl_s=100)
    base.update(kw)
    return Settings(**base)


def test_append_and_get_roundtrip():
    st = SessionStore(_settings())
    st.append("s1", "hi", "hello")
    hist = st.get_history("s1")
    assert hist == [{"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"}]


def test_history_capped_at_max_turns():
    st = SessionStore(_settings(memory_max_turns=2))
    for i in range(4):
        st.append("s1", f"q{i}", f"a{i}")
    hist = st.get_history("s1")
    assert len(hist) == 4  # 2 pairs
    assert hist[0]["content"] == "q2"  # oldest pairs evicted


def test_ttl_eviction():
    now = [1000.0]
    st = SessionStore(_settings(memory_ttl_s=50), clock=lambda: now[0])
    st.append("s1", "q", "a")
    now[0] = 1040.0
    assert st.get_history("s1")  # within ttl
    now[0] = 1100.0
    assert st.get_history("s1") == []  # expired


def test_disabled_is_passthrough():
    st = SessionStore(_settings(enable_memory=False))
    st.append("s1", "q", "a")
    assert st.get_history("s1") == []


def test_empty_session_id_ignored():
    st = SessionStore(_settings())
    st.append("", "q", "a")
    assert st.get_history("") == []


def test_reset_clears():
    st = SessionStore(_settings())
    st.append("s1", "q", "a")
    st.reset("s1")
    assert st.get_history("s1") == []


# ---------- redis backend ----------

class FakeRedis:
    def __init__(self):
        self.store = {}

    def get(self, k):
        return self.store.get(k)

    def set(self, k, v, ex=None):
        self.store[k] = v

    def delete(self, k):
        self.store.pop(k, None)


def test_redis_store_roundtrip_and_cap():
    r = FakeRedis()
    st = RedisSessionStore(r, _settings(memory_max_turns=2))
    for i in range(3):
        st.append("s1", f"q{i}", f"a{i}")
    hist = st.get_history("s1")
    assert len(hist) == 4 and hist[0]["content"] == "q1"
    # stored as JSON under a namespaced key
    assert "sess:s1" in r.store and isinstance(json.loads(r.store["sess:s1"]), list)


def test_redis_store_reset():
    r = FakeRedis()
    st = RedisSessionStore(r, _settings())
    st.append("s1", "q", "a")
    st.reset("s1")
    assert st.get_history("s1") == []
