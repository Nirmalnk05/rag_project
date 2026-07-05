"""Parser interface. Every concrete parser turns a file into a ParsedDocument."""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from src.parsing.model import ParsedDocument


class ParseError(Exception):
    """Raised when a document cannot be parsed by a given backend."""


class DocumentParser(ABC):
    @abstractmethod
    def parse(self, path: Path) -> ParsedDocument: ...


def rows_to_markdown(header: list[str], rows: list[list[str]]) -> str:
    """Serialize a header + rows into a GitHub-flavored Markdown table."""
    def cell(v) -> str:
        return "" if v is None else str(v).replace("|", "\\|").replace("\n", " ").strip()

    head = "| " + " | ".join(cell(h) for h in header) + " |"
    sep = "| " + " | ".join("---" for _ in header) + " |"
    body = ["| " + " | ".join(cell(c) for c in r) + " |" for r in rows]
    return "\n".join([head, sep, *body])
