import pytest
from fastapi import HTTPException

from src.security import TokenBucket, check_api_key, check_rate_limit, _reset_buckets


def test_no_keys_configured_auth_disabled():
    check_api_key(provided=None, allowed=set())  # no raise


def test_valid_key_passes_invalid_raises():
    check_api_key(provided="k1", allowed={"k1"})
    with pytest.raises(HTTPException) as ei:
        check_api_key(provided="nope", allowed={"k1"})
    assert ei.value.status_code == 401
    with pytest.raises(HTTPException):
        check_api_key(provided=None, allowed={"k1"})


def test_token_bucket_exhausts_and_refills():
    b = TokenBucket(capacity=2, refill_per_s=1000)
    assert b.allow() and b.allow() and not b.allow()
    import time
    time.sleep(0.01)  # 1000/s refill → replenished
    assert b.allow()


def test_check_rate_limit_429():
    _reset_buckets()
    for _ in range(3):
        check_rate_limit("client-a", rpm=3)
    with pytest.raises(HTTPException) as ei:
        check_rate_limit("client-a", rpm=3)
    assert ei.value.status_code == 429
    check_rate_limit("client-b", rpm=3)  # other clients unaffected
