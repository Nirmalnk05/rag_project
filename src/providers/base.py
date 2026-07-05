"""Provider-neutral LLM interface and result/error types."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator


@dataclass
class LLMResult:
    text: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0


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
