"""Markdown → elements. Reused for (a) the existing .md sample docs and
(b) cloud-parser output (LlamaParse/Azure return Markdown).

Splits on H2 headings into sections, then within each section separates
contiguous Markdown table blocks (lines starting with '|') into TableElements
from surrounding prose (TextElements).
"""
from __future__ import annotations

import re
from pathlib import Path

from src.parsing.base import DocumentParser
from src.parsing.model import Element, ParsedDocument, TableElement, TextElement


def _split_sections(body: str) -> list[tuple[str, str]]:
    parts: list[tuple[str, str]] = []
    title, buf = "Preamble", []
    for line in body.splitlines():
        h2 = re.match(r"^##\s+(.+)$", line)
        if h2:
            if buf:
                parts.append((title, "\n".join(buf).strip()))
            title, buf = h2.group(1).strip(), []
        else:
            buf.append(line)
    if buf:
        parts.append((title, "\n".join(buf).strip()))
    return [(t, b) for t, b in parts if b]


def _is_table_line(line: str) -> bool:
    return line.lstrip().startswith("|")


def _is_heading_line(line: str) -> bool:
    return bool(re.match(r"^#{1,6}\s", line))


def markdown_to_elements(body: str) -> list[Element]:
    elements: list[Element] = []
    for section, text in _split_sections(body):
        buf_text: list[str] = []
        buf_table: list[str] = []

        def flush_text():
            joined = "\n".join(buf_text).strip()
            if joined:
                elements.append(TextElement(section=section, text=joined))
            buf_text.clear()

        def flush_table():
            if buf_table:
                elements.append(TableElement(section=section, markdown="\n".join(buf_table)))
            buf_table.clear()

        for line in text.splitlines():
            if _is_heading_line(line):
                continue  # structural, not content (title captured separately)
            if _is_table_line(line):
                if buf_text:
                    flush_text()
                buf_table.append(line)
            else:
                if buf_table:
                    flush_table()
                buf_text.append(line)
        flush_text()
        flush_table()
    return elements


def extract_title(body: str, fallback: str) -> str:
    m = re.search(r"^#\s+(.+)$", body, flags=re.MULTILINE)
    return m.group(1).strip() if m else fallback


class MarkdownParser(DocumentParser):
    def parse(self, path: Path) -> ParsedDocument:
        path = Path(path)
        body = path.read_text(encoding="utf-8")
        doc = ParsedDocument(title=extract_title(body, path.stem), source_file=path.name)
        doc.elements = markdown_to_elements(body)
        return doc
