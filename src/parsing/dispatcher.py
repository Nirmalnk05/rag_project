"""Parser dispatch + hybrid local/cloud fallback.

Excel goes to the dedicated Excel parser. PDF/Word go to the local (Docling)
parser; under the `hybrid`/`cloud` backend a low-confidence local result (or
`force_cloud`) is retried on the configured cloud parser, when one is available.
"""
from __future__ import annotations

import logging
from pathlib import Path

from src.config import Settings, get_settings
from src.parsing.base import DocumentParser, ParseError
from src.parsing.excel_parser import ExcelParser
from src.parsing.model import ParsedDocument

_logger = logging.getLogger("rag.parsing")

EXCEL_EXT = {".xlsx", ".xls"}
LOCAL_EXT = {".pdf", ".docx", ".doc"}
SUPPORTED_EXT = EXCEL_EXT | LOCAL_EXT | {".md"}


class Dispatcher:
    def __init__(self, *, settings: Settings | None = None,
                 local: DocumentParser | None = None,
                 cloud: DocumentParser | None = None,
                 excel: DocumentParser | None = None) -> None:
        self._s = settings or get_settings()
        self._local = local
        self._cloud = cloud
        self._excel = excel or ExcelParser()

    def _local_parser(self) -> DocumentParser:
        if self._local is None:
            from src.parsing.local_parser import LocalParser
            self._local = LocalParser(settings=self._s)
        return self._local

    def _cloud_parser(self) -> DocumentParser | None:
        if self._cloud is not None:
            return self._cloud
        if not self._s.cloud_parse_configured:
            return None
        from src.parsing.cloud_parser import CloudParser
        self._cloud = CloudParser(settings=self._s)
        return self._cloud

    def _low_confidence(self, doc: ParsedDocument) -> bool:
        chars = sum(
            len(getattr(e, "text", "") or getattr(e, "markdown", "") or "")
            for e in doc.elements
        )
        return chars < self._s.cloud_parse_min_chars

    def parse(self, path: Path, *, force_cloud: bool = False) -> ParsedDocument:
        path = Path(path)
        ext = path.suffix.lower()
        if ext in EXCEL_EXT:
            return self._excel.parse(path)
        if ext not in LOCAL_EXT:
            raise ParseError(f"unsupported file type: {ext}")

        backend = self._s.parse_backend
        cloud = self._cloud_parser()

        # cloud-first (explicit backend or forced)
        if (backend == "cloud" or force_cloud) and cloud is not None:
            try:
                return cloud.parse(path)
            except Exception:  # noqa: BLE001 — fall back to local on cloud failure
                _logger.warning("cloud_parse_failed_falling_back_local", exc_info=True)

        doc = self._local_parser().parse(path)

        if backend == "hybrid" and cloud is not None and self._low_confidence(doc):
            _logger.info("local_parse_low_confidence_trying_cloud")
            try:
                return cloud.parse(path)
            except Exception:  # noqa: BLE001
                _logger.warning("cloud_parse_failed_keeping_local", exc_info=True)
        return doc
