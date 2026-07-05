from types import SimpleNamespace

import anthropic
import httpx
import pytest

from src.config import Settings
from src.providers.anthropic_provider import AnthropicProvider
from src.providers.base import LLMError, LLMRetryableError


def _settings():
    return Settings(_env_file=None, anthropic_api_key="test-key")


def _status_error(status: int) -> anthropic.APIStatusError:
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    resp = httpx.Response(status, request=req)
    return anthropic.APIStatusError(message=f"status {status}", response=resp, body=None)


class FakeMessages:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.last_kwargs = response, error, None

    async def create(self, **kwargs):
        self.last_kwargs = kwargs
        if self.error:
            raise self.error
        return self.response


def _fake_client(messages):
    return SimpleNamespace(messages=messages)


def _ok_response():
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text="hello")],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
        model="claude-x",
    )


async def test_complete_maps_result_and_uses_cache_control():
    fake = FakeMessages(response=_ok_response())
    p = AnthropicProvider(_settings(), client=_fake_client(fake))
    r = await p.complete("hi", system="sys", model="claude-x")
    assert r.text == "hello" and r.input_tokens == 10 and r.output_tokens == 5
    sysblock = fake.last_kwargs["system"][0]
    assert sysblock["cache_control"] == {"type": "ephemeral"}


async def test_complete_no_system_omits_key():
    fake = FakeMessages(response=_ok_response())
    p = AnthropicProvider(_settings(), client=_fake_client(fake))
    await p.complete("hi", model="claude-x")
    assert "system" not in fake.last_kwargs


async def test_429_is_retryable():
    fake = FakeMessages(error=_status_error(429))
    p = AnthropicProvider(_settings(), client=_fake_client(fake))
    with pytest.raises(LLMRetryableError):
        await p.complete("hi", model="claude-x")


async def test_400_is_terminal():
    fake = FakeMessages(error=_status_error(400))
    p = AnthropicProvider(_settings(), client=_fake_client(fake))
    with pytest.raises(LLMError) as ei:
        await p.complete("hi", model="claude-x")
    assert not isinstance(ei.value, LLMRetryableError)
