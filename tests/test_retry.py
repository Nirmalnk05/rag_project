import pytest
from src.providers.base import LLMError, LLMResult, LLMRetryableError
from src.providers.retry import call_with_retries


async def test_success_first_try():
    async def make_call(model_override):
        return LLMResult(text="ok", model="m")

    r = await call_with_retries(make_call, max_attempts=3, base_delay=0.001)
    assert r.text == "ok"


async def test_retries_then_succeeds():
    calls = []

    async def make_call(model_override):
        calls.append(model_override)
        if len(calls) < 3:
            raise LLMRetryableError("boom")
        return LLMResult(text="ok", model="m")

    r = await call_with_retries(make_call, max_attempts=3, base_delay=0.001)
    assert r.text == "ok" and calls == [None, None, None]


async def test_fallback_model_used_after_exhaustion():
    calls = []

    async def make_call(model_override):
        calls.append(model_override)
        if model_override is None:
            raise LLMRetryableError("primary down")
        return LLMResult(text="fb", model=model_override)

    r = await call_with_retries(
        make_call, max_attempts=2, fallback_model="small", base_delay=0.001
    )
    assert r.text == "fb" and calls == [None, None, "small"]


async def test_raises_llm_error_when_all_fail():
    async def make_call(model_override):
        raise LLMRetryableError("down")

    with pytest.raises(LLMError):
        await call_with_retries(make_call, max_attempts=2, base_delay=0.001)


async def test_terminal_error_not_retried():
    calls = []

    async def make_call(model_override):
        calls.append(1)
        raise LLMError("bad request")

    with pytest.raises(LLMError):
        await call_with_retries(make_call, max_attempts=3, base_delay=0.001)
    assert len(calls) == 1
