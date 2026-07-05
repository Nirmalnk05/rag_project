import pytest
from src.config import Settings


def _mk(**kw):
    # _env_file=None: ignore .env so tests are hermetic
    return Settings(_env_file=None, **kw)


def test_defaults_load_without_any_key():
    s = _mk()
    assert s.llm_provider == "anthropic"
    assert s.anthropic_api_key == ""
    assert s.llm_max_attempts == 3


def test_validate_provider_requires_anthropic_key():
    s = _mk(llm_provider="anthropic", anthropic_api_key="")
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        s.validate_provider()


def test_validate_provider_ollama_needs_no_key():
    s = _mk(llm_provider="ollama")
    s.validate_provider()  # no raise


def test_validate_provider_rejects_unknown():
    s = _mk(llm_provider="banana")
    with pytest.raises(ValueError, match="banana"):
        s.validate_provider()


def test_api_keys_parse():
    s = _mk(api_keys="k1, k2 ,k3")
    assert s.api_key_set == {"k1", "k2", "k3"}
    assert _mk(api_keys="").api_key_set == set()
