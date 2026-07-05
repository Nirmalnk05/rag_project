"""Normalized document representation shared by all parsers.

Every parser (local Docling, Excel, cloud) emits a `ParsedDocument` — a title
plus an ordered list of typed elements. Downstream ingestion consumes only this
structure, so adding a new input format never touches chunking/indexing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Union

ElementType = Literal["text", "table", "figure"]


@dataclass
class TextElement:
    section: str
    text: str
    page: int | None = None
    type: ElementType = field(default="text", init=False)


@dataclass
class TableElement:
    section: str
    markdown: str            # table serialized as GitHub-flavored Markdown
    caption: str = ""
    page: int | None = None
    type: ElementType = field(default="table", init=False)


@dataclass
class FigureElement:
    section: str
    caption: str             # vision-model caption (searchable text)
    image_ref: str = ""      # path/id of the extracted image, if kept
    page: int | None = None
    type: ElementType = field(default="figure", init=False)


Element = Union[TextElement, TableElement, FigureElement]


@dataclass
class ParsedDocument:
    title: str
    source_file: str
    elements: list[Element] = field(default_factory=list)

    def by_type(self, t: ElementType) -> list[Element]:
        return [e for e in self.elements if e.type == t]
