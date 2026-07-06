import pytest
from fastapi import HTTPException

from src.security import check_rate_limit_redis, redis_allow


class FakeRedis:
    """Minimal in-memory stand-in for redis (INCR + EXPIRE)."""
    def __init__(self):
        self.store: dict[str, int] = {}

    def incr(self, key):
        self.store[key] = self.store.get(key, 0) + 1
        return self.store[key]

    def expire(self, key, ttl):
        return True


def test_redis_allow_counts_within_window():
    r = FakeRedis()
    # fixed now → same window
    results = [redis_allow(r, "clientA", 3, now=100.0) for _ in range(4)]
    assert results == [True, True, True, False]


def test_redis_allow_is_shared_across_callers():
    r = FakeRedis()  # one store = one shared limit (simulates replicas → same redis)
    assert redis_allow(r, "k", 2, now=0.0) is True
    assert redis_allow(r, "k", 2, now=0.0) is True
    assert redis_allow(r, "k", 2, now=0.0) is False


def test_redis_allow_new_window_resets():
    r = FakeRedis()
    assert redis_allow(r, "k", 1, now=0.0) is True
    assert redis_allow(r, "k", 1, now=0.0) is False   # same minute
    assert redis_allow(r, "k", 1, now=60.0) is True    # next minute → new key


def test_redis_allow_per_client_isolated():
    r = FakeRedis()
    assert redis_allow(r, "a", 1, now=0.0) is True
    assert redis_allow(r, "b", 1, now=0.0) is True     # different client id


def test_check_rate_limit_redis_raises_429():
    r = FakeRedis()
    check_rate_limit_redis(r, "c", rpm=1)
    with pytest.raises(HTTPException) as ei:
        check_rate_limit_redis(r, "c", rpm=1)
    assert ei.value.status_code == 429
