import pytest

import src.prompts as prompts
from src.config import Settings
from src.prompts import get_prompt


def test_get_prompt_default_version(monkeypatch):
    monkeypatch.setattr(prompts, "get_settings", lambda: Settings(_env_file=None))
    assert "maintenance" in get_prompt("agent_system").lower()
    assert "rewrite" in get_prompt("rewrite").lower()


def test_explicit_version():
    assert get_prompt("grader", version="v1").startswith("You judge")


def test_unknown_version_falls_back_to_v1(monkeypatch):
    monkeypatch.setattr(prompts, "get_settings",
                        lambda: Settings(_env_file=None, prompt_version="v99"))
    # falls back to v1 rather than raising
    assert get_prompt("generate").startswith("You answer")


def test_unknown_prompt_name_raises():
    with pytest.raises(KeyError):
        get_prompt("does_not_exist")
