"""API-key auth and rate limiting.

Two rate-limit backends behind one interface:
- `memory` (default): in-process token bucket — correct for single-instance /
  air-gapped deploys.
- `redis`: atomic fixed-window counter shared across replicas — for horizontally
  scaled cloud deploys (RATE_LIMIT_BACKEND=redis + REDIS_URL).
"""
from __future__ import annotations

import time

from fastapi import HTTPException, Request

from src.config import get_settings


def check_api_key(*, provided: str | None, allowed: set[str]) -> None:
    if not allowed:
        return  # auth disabled (local dev)
    if provided is None or provided not in allowed:
        raise HTTPException(status_code=401, detail="invalid or missing API key")


class TokenBucket:
    def __init__(self, capacity: int, refill_per_s: float) -> None:
        self.capacity = capacity
        self.refill_per_s = refill_per_s
        self.tokens = float(capacity)
        self.last = time.monotonic()

    def allow(self) -> bool:
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.refill_per_s)
        self.last = now
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


_buckets: dict[str, TokenBucket] = {}


def _reset_buckets() -> None:  # test helper
    _buckets.clear()


def check_rate_limit(client_id: str, *, rpm: int) -> None:
    bucket = _buckets.get(client_id)
    if bucket is None or bucket.capacity != rpm:
        bucket = _buckets[client_id] = TokenBucket(capacity=rpm, refill_per_s=rpm / 60.0)
    if not bucket.allow():
        raise HTTPException(status_code=429, detail="rate limit exceeded")


# ---------- redis backend (shared across replicas) ----------

_redis_client = None


def _get_redis(url: str):
    """Lazily create a module-level redis client. Injectable for tests."""
    global _redis_client
    if _redis_client is None:
        import redis  # optional dependency; only imported when backend=redis
        _redis_client = redis.Redis.from_url(url)
    return _redis_client


def set_redis_client(client) -> None:  # test hook / explicit override
    global _redis_client
    _redis_client = client


def redis_allow(client, client_id: str, rpm: int, *, now: float | None = None) -> bool:
    """Atomic fixed-window limiter: rpm requests per rolling clock-minute."""
    window = int((now if now is not None else time.time()) // 60)
    key = f"rl:{client_id}:{window}"
    count = client.incr(key)
    if count == 1:
        client.expire(key, 60)
    return count <= rpm


def check_rate_limit_redis(client, client_id: str, *, rpm: int) -> None:
    if not redis_allow(client, client_id, rpm):
        raise HTTPException(status_code=429, detail="rate limit exceeded")


# ---------- FastAPI dependencies ----------

async def require_api_key(request: Request) -> None:
    s = get_settings()
    check_api_key(provided=request.headers.get("X-API-Key"), allowed=s.api_key_set)


async def rate_limiter(request: Request) -> None:
    s = get_settings()
    client_id = request.headers.get("X-API-Key") or (request.client.host if request.client else "anon")
    if s.rate_limit_backend == "redis":
        check_rate_limit_redis(_get_redis(s.redis_url), client_id, rpm=s.rate_limit_rpm)
    else:
        check_rate_limit(client_id, rpm=s.rate_limit_rpm)
