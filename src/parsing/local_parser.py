"""Local parser built on Docling (PDF incl. scanned via OCR, Word).

Docling's API is version-sensitive, so the Docling-specific extraction lives in
`_extract_items` (a thin adapter, covered by manual e2e). Everything else —
mapping raw items to ParsedDocument and captioning figures — is pure logic and
unit-tested by overriding `_extract_items`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.config import Settings, get_settings
from src.parsing.base import DocumentParser
from src.parsing.model import FigureElement, ParsedDocument, TableElement, TextElement
from src.parsing.vision import FigureCaptioner


@dataclass
class RawItem:
    """Format-neutral item emitted by the Docling adapter."""
    kind: str                 # "text" | "table" | "figure"
    section: str = ""
    text: str = ""            # text body, or table markdown
    image: bytes | None = None
    media_type: str = "image/png"
    page: int | None = None


class LocalParser(DocumentParser):
    def __init__(self, *, settings: Settings | None = None,
                 captioner: FigureCaptioner | None = None) -> None:
        self._s = settings or get_settings()
        self._captioner = captioner or FigureCaptioner(settings=self._s)

    def parse(self, path: Path) -> ParsedDocument:
        path = Path(path)
        doc = ParsedDocument(title=path.stem, source_file=path.name)
        for item in self._extract_items(path):
            if item.kind == "text":
                if item.text.strip():
                    doc.elements.append(
                        TextElement(section=item.section, text=item.text, page=item.page))
            elif item.kind == "table":
                doc.elements.append(
                    TableElement(section=item.section, markdown=item.text, page=item.page))
            elif item.kind == "figure":
                caption = ""
                if item.image is not None:
                    caption = self._captioner.caption_sync(item.image, item.media_type)
                doc.elements.append(FigureElement(
                    section=item.section, caption=caption, page=item.page,
                    image_ref=f"{path.name}#p{item.page}"))
        return doc

    # ---------- Docling adapter (manual-e2e coverage) ----------

    def _extract_items(self, path: Path) -> list[RawItem]:  # pragma: no cover
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        dl = converter.convert(str(path)).document
        items: list[RawItem] = []
        for text in getattr(dl, "texts", []):
            items.append(RawItem(kind="text",
                                 section=_heading_of(text),
                                 text=getattr(text, "text", "") or "",
                                 page=_page_of(text)))
        for table in getattr(dl, "tables", []):
            md = table.export_to_markdown() if hasattr(table, "export_to_markdown") else ""
            items.append(RawItem(kind="table", text=md, page=_page_of(table)))
        for pic in getattr(dl, "pictures", []):
            img = _picture_bytes(pic, dl)
            items.append(RawItem(kind="figure", image=img, page=_page_of(pic)))
        return items


def _heading_of(item) -> str:  # pragma: no cover
    return getattr(item, "parent_heading", "") or ""


def _page_of(item) -> int | None:  # pragma: no cover
    prov = getattr(item, "prov", None)
    if prov:
        return getattr(prov[0], "page_no", None)
    return None


def _picture_bytes(pic, dl) -> bytes | None:  # pragma: no cover
    try:
        import io
        img = pic.get_image(dl)
        if img is None:
            return None
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return None


def pdf_text_density(path: Path) -> float:  # pragma: no cover
    """Average extractable chars per page — used to detect scanned PDFs."""
    import fitz  # PyMuPDF
    doc = fitz.open(str(path))
    try:
        pages = doc.page_count or 1
        total = sum(len(doc.load_page(i).get_text("text")) for i in range(pages))
        return total / pages
    finally:
        doc.close()
