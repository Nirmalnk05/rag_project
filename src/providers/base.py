"""Provider-neutral LLM interface and result/error types."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator


@dataclass
class LLMResult:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ToolSpec:
    """Provider-neutral tool definition (JSON-schema parameters)."""
    name: str
    description: str
    parameters: dict


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class ChatResult:
    """One assistant turn: either final text, or a request to call tools."""
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class LLMError(Exception):
    """Terminal LLM failure — do not retry."""


class LLMRetryableError(LLMError):
    """Transient failure (timeout, 429, 5xx) — safe to retry."""


class LLMProvider(ABC):
    @abstractmethod
    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> LLMResult: ...

    @abstractmethod
    def stream(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str,
        max_tokens: int = 2048,
        temperature: float = 0.0,
    ) -> AsyncIterator[str]: ...

    async def caption(
        self,
        *,
        image: bytes,
        media_type: str,
        prompt: str,
        model: str,
        max_tokens: int = 512,
    ) -> str:
        """Describe an image with a vision model. Providers that support vision
        override this; the default signals it is unavailable."""
        raise LLMError("vision captioning is not supported by this provider")

    async def chat(
        self,
        messages: list[dict],
        *,
        tools: list[ToolSpec] | None = None,
        model: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> ChatResult:
        """Multi-turn chat with optional tool-calling. Providers that support
        function-calling override this; the default signals it is unavailable."""
        raise LLMError("tool-calling chat is not supported by this provider")
