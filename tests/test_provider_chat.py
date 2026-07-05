import json
from types import SimpleNamespace

import httpx
import pytest

from src.config import Settings
from src.providers.anthropic_provider import AnthropicProvider
from src.providers.base import ChatResult, LLMProvider, ToolCall, ToolSpec
from src.providers.ollama_provider import OllamaProvider
from src.providers.openai_compat_provider import OpenAICompatProvider

TOOLS = [ToolSpec(name="search", description="search docs",
                  parameters={"type": "object", "properties": {"query": {"type": "string"}}})]


# ---------- base default ----------

class _Bare(LLMProvider):
    async def complete(self, prompt, *, system=None, model, max_tokens=1024, temperature=0.0):
        raise NotImplementedError

    async def stream(self, prompt, *, system=None, model, max_tokens=2048, temperature=0.0):
        yield ""


async def test_base_chat_unsupported():
    from src.providers.base import LLMError
    with pytest.raises(LLMError, match="not supported"):
        await _Bare().chat([{"role": "user", "content": "hi"}], model="m")


# ---------- anthropic ----------

class FakeMessages:
    def __init__(self, response):
        self.response, self.last_kwargs = response, None

    async def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self.response


def _anthropic(response):
    return AnthropicProvider(Settings(_env_file=None, anthropic_api_key="k"),
                             client=SimpleNamespace(messages=FakeMessages(response)))


async def test_anthropic_chat_parses_tool_use():
    resp = SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", id="tu_1", name="search",
                                 input={"query": "pump"})],
        stop_reason="tool_use", model="claude-x",
        usage=SimpleNamespace(input_tokens=5, output_tokens=3),
    )
    p = _anthropic(resp)
    fake = p._client.messages
    out = await p.chat([{"role": "user", "content": "find pump"}], tools=TOOLS, model="claude-x")
    assert out.wants_tools and out.tool_calls[0].name == "search"
    assert out.tool_calls[0].arguments == {"query": "pump"}
    assert fake.last_kwargs["tools"][0]["input_schema"]["type"] == "object"


async def test_anthropic_chat_groups_tool_results_into_user_turn():
    resp = SimpleNamespace(content=[SimpleNamespace(type="text", text="done")],
                           stop_reason="end_turn", model="c",
                           usage=SimpleNamespace(input_tokens=1, output_tokens=1))
    p = _anthropic(resp)
    fake = p._client.messages
    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "",
         "tool_calls": [ToolCall(id="tu_1", name="search", arguments={"query": "x"})]},
        {"role": "tool", "tool_call_id": "tu_1", "name": "search", "content": "result text"},
    ]
    out = await p.chat(messages, tools=TOOLS, model="c")
    assert out.text == "done"
    native = fake.last_kwargs["messages"]
    # last native message is a user turn carrying a tool_result block
    assert native[-1]["role"] == "user"
    assert native[-1]["content"][0]["type"] == "tool_result"
    assert native[-1]["content"][0]["tool_use_id"] == "tu_1"


# ---------- openai-compatible ----------

def _openai(handler):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url="http://x/v1")
    return OpenAICompatProvider(
        Settings(_env_file=None, llm_provider="openai_compatible", openai_base_url="http://x/v1"),
        client=client)


async def test_openai_chat_parses_tool_calls_and_serializes_tools():
    def handler(request):
        body = json.loads(request.content)
        assert body["tools"][0]["type"] == "function"
        assert body["tools"][0]["function"]["name"] == "search"
        return httpx.Response(200, json={"choices": [{"finish_reason": "tool_calls",
            "message": {"content": None, "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "search", "arguments": '{"query": "pump"}'}}]}}]})

    out = await _openai(handler).chat([{"role": "user", "content": "q"}], tools=TOOLS, model="m")
    assert out.tool_calls[0].id == "c1" and out.tool_calls[0].arguments == {"query": "pump"}


async def test_openai_chat_final_text():
    def handler(request):
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop",
            "message": {"content": "the answer [#1]"}}]})

    out = await _openai(handler).chat([{"role": "user", "content": "q"}], model="m")
    assert out.text == "the answer [#1]" and not out.wants_tools


async def test_openai_serializes_assistant_toolcalls_and_tool_results():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    messages = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "",
         "tool_calls": [ToolCall(id="c1", name="search", arguments={"query": "x"})]},
        {"role": "tool", "tool_call_id": "c1", "name": "search", "content": "res"},
    ]
    await _openai(handler).chat(messages, model="m")
    msgs = seen["body"]["messages"]
    assert msgs[1]["tool_calls"][0]["function"]["arguments"] == '{"query": "x"}'
    assert msgs[2] == {"role": "tool", "tool_call_id": "c1", "content": "res"}


# ---------- ollama ----------

def _ollama(handler):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler),
                              base_url="http://localhost:11434")
    return OllamaProvider(Settings(_env_file=None, llm_provider="ollama"), client=client)


async def test_ollama_chat_parses_tool_calls():
    def handler(request):
        body = json.loads(request.content)
        assert request.url.path == "/api/chat"
        assert body["tools"][0]["function"]["name"] == "search"
        return httpx.Response(200, json={"message": {"content": "",
            "tool_calls": [{"function": {"name": "search", "arguments": {"query": "pump"}}}]}})

    out = await _ollama(handler).chat([{"role": "user", "content": "q"}], tools=TOOLS, model="llama3.1")
    assert out.tool_calls[0].name == "search"
    assert out.tool_calls[0].arguments == {"query": "pump"}
    assert out.tool_calls[0].id == "call_0"  # synthesized


async def test_ollama_chat_final_text():
    def handler(request):
        return httpx.Response(200, json={"message": {"content": "final answer"}})

    out = await _ollama(handler).chat([{"role": "user", "content": "q"}], model="llama3.1")
    assert out.text == "final answer" and not out.wants_tools
