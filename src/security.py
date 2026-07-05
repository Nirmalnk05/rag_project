"""API-key auth and in-memory token-bucket rate limiting.

Single-instance only (state lives in this process). For multi-replica
deployments move both to a shared store (e.g. Redis).
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


# ---------- FastAPI dependencies ----------

async def require_api_key(request: Request) -> None:
    s = get_settings()
    check_api_key(provided=request.headers.get("X-API-Key"), allowed=s.api_key_set)


async def rate_limiter(request: Request) -> None:
    s = get_settings()
    client = request.headers.get("X-API-Key") or (request.client.host if request.client else "anon")
    check_rate_limit(client, rpm=s.rate_limit_rpm)
