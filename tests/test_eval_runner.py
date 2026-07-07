import pytest

from src.config import Settings
from src.eval_runner import _build_ragas_llm_and_embeddings


class FakeHFEmbeddings:
    """Stand-in for langchain_huggingface.HuggingFaceEmbeddings — the real
    class downloads/loads an actual model at construction time, which would
    require network access and be slow in unit tests."""

    def __init__(self, *, model_name):
        self.model_name = model_name


@pytest.fixture(autouse=True)
def fake_hf_embeddings(monkeypatch):
    import langchain_huggingface
    monkeypatch.setattr(langchain_huggingface, "HuggingFaceEmbeddings", FakeHFEmbeddings)


def test_anthropic_provider_maps_to_chat_anthropic():
    from langchain_anthropic import ChatAnthropic
    s = Settings(_env_file=None, llm_provider="anthropic", anthropic_api_key="k",
                answer_model="claude-x")
    llm, emb = _build_ragas_llm_and_embeddings(s)
    assert isinstance(llm.langchain_llm, ChatAnthropic)
    assert llm.langchain_llm.model == "claude-x"


def test_ollama_provider_maps_to_chat_ollama():
    from langchain_ollama import ChatOllama
    s = Settings(_env_file=None, llm_provider="ollama", answer_model="llama3.1",
                ollama_base_url="http://x:11434")
    llm, emb = _build_ragas_llm_and_embeddings(s)
    assert isinstance(llm.langchain_llm, ChatOllama)
    assert llm.langchain_llm.model == "llama3.1"
    assert llm.langchain_llm.base_url == "http://x:11434"


def test_openai_compatible_maps_to_chat_openai():
    from langchain_openai import ChatOpenAI
    s = Settings(_env_file=None, llm_provider="openai_compatible",
                openai_base_url="http://x/v1", openai_api_key="", answer_model="qwen")
    llm, emb = _build_ragas_llm_and_embeddings(s)
    assert isinstance(llm.langchain_llm, ChatOpenAI)


def test_openai_compatible_uses_dummy_key_when_none_configured():
    s = Settings(_env_file=None, llm_provider="openai_compatible", openai_api_key="")
    llm, emb = _build_ragas_llm_and_embeddings(s)
    # SecretStr on the langchain object; just confirm it isn't empty (would
    # otherwise fail client-side validation before any request is made).
    key = llm.langchain_llm.openai_api_key.get_secret_value()
    assert key  # non-empty


def test_embeddings_always_local_regardless_of_provider():
    s = Settings(_env_file=None, llm_provider="ollama", embed_model="BAAI/bge-small-en-v1.5")
    llm, emb = _build_ragas_llm_and_embeddings(s)
    assert isinstance(emb.embeddings, FakeHFEmbeddings)
    assert emb.embeddings.model_name == "BAAI/bge-small-en-v1.5"


def test_unknown_provider_raises():
    s = Settings(_env_file=None, llm_provider="bogus")
    with pytest.raises(ValueError, match="bogus"):
        _build_ragas_llm_and_embeddings(s)
