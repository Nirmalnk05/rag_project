import pytest

import src.faithfulness as fa
from src.config import Settings
from src.faithfulness import build_judge_prompt, check_faithfulness, parse_faithfulness
from src.providers.base import LLMResult


def test_build_judge_prompt_numbers_sources():
    p = build_judge_prompt("ans", [{"text": "src one"}, {"text": "src two"}])
    assert "[#1] src one" in p and "[#2] src two" in p and "ANSWER:\nans" in p


def test_parse_grounded_true():
    assert parse_faithfulness('{"grounded": true, "unsupported": []}') == {
        "grounded": True, "unsupported": []}


def test_parse_unsupported_list():
    out = parse_faithfulness('{"grounded": false, "unsupported": ["claim x"]}')
    assert out["grounded"] is False and out["unsupported"] == ["claim x"]


def test_parse_tolerant_fallback():
    assert parse_faithfulness("garbage")["grounded"] is False


async def test_disabled_returns_unchecked(monkeypatch):
    monkeypatch.setattr(fa, "get_settings",
                        lambda: Settings(_env_file=None, enable_faithfulness_check=False))
    out = await check_faithfulness("ans", [{"text": "s"}])
    assert out == {"grounded": True, "unsupported": [], "checked": False}


async def test_no_sources_short_circuits(monkeypatch):
    monkeypatch.setattr(fa, "get_settings",
                        lambda: Settings(_env_file=None, enable_faithfulness_check=True))
    out = await check_faithfulness("ans", [])
    assert out["checked"] is False


async def test_enabled_calls_judge(monkeypatch):
    monkeypatch.setattr(fa, "get_settings",
                        lambda: Settings(_env_file=None, enable_faithfulness_check=True))

    async def fake_complete(prompt, *, system=None, model=None, max_tokens=1024, temperature=0.0):
        return LLMResult(text='{"grounded": false, "unsupported": ["made-up code"]}', model="m")

    import src.llm
    monkeypatch.setattr(src.llm, "complete", fake_complete)
    out = await check_faithfulness("The code is 9Z [#1].", [{"text": "code 4F only"}])
    assert out["grounded"] is False and out["unsupported"] == ["made-up code"]
    assert out["checked"] is True
