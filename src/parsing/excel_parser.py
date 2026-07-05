"""Excel parser: each sheet → one or more table elements.

Troubleshooting knowledge in this domain often lives in spreadsheets (fault-code
matrices, PM checklists). The first row is treated as the header and **repeated
in every row-group chunk** so a lookup table stays interpretable after splitting.
"""
from __future__ import annotations

from pathlib import Path

from openpyxl import load_workbook

from src.parsing.base import DocumentParser, rows_to_markdown
from src.parsing.model import ParsedDocument, TableElement

DEFAULT_ROWS_PER_CHUNK = 50


class ExcelParser(DocumentParser):
    def __init__(self, rows_per_chunk: int = DEFAULT_ROWS_PER_CHUNK) -> None:
        self._rows_per_chunk = rows_per_chunk

    def parse(self, path: Path) -> ParsedDocument:
        path = Path(path)
        wb = load_workbook(path, read_only=True, data_only=True)
        doc = ParsedDocument(title=path.stem, source_file=path.name)
        for ws in wb.worksheets:
            rows = [list(r) for r in ws.iter_rows(values_only=True)]
            rows = [r for r in rows if any(c is not None for c in r)]  # drop blank rows
            if not rows:
                continue
            header, data = rows[0], rows[1:]
            if not data:
                # header-only sheet: emit it as a single table
                doc.elements.append(
                    TableElement(section=ws.title, markdown=rows_to_markdown(header, []))
                )
                continue
            for start in range(0, len(data), self._rows_per_chunk):
                group = data[start : start + self._rows_per_chunk]
                doc.elements.append(TableElement(
                    section=ws.title,
                    markdown=rows_to_markdown(header, group),
                    caption=f"{path.stem} — sheet '{ws.title}'",
                ))
        wb.close()
        return doc
