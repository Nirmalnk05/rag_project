"""Cloud parser: LlamaParse or Azure Document Intelligence (env-gated, optional).

Both return Markdown, which is split into elements via markdown_to_elements. The
actual SDK call lives in `_fetch_markdown` (lazy import, manual-e2e coverage);
`parse` is pure and unit-tested by overriding it.
"""
from __future__ import annotations

from pathlib import Path

from src.config import Settings, get_settings
from src.parsing.base import DocumentParser, ParseError
from src.parsing.markdown_parser import extract_title, markdown_to_elements
from src.parsing.model import ParsedDocument


class CloudParser(DocumentParser):
    def __init__(self, *, settings: Settings | None = None) -> None:
        self._s = settings or get_settings()

    def parse(self, path: Path) -> ParsedDocument:
        path = Path(path)
        md = self._fetch_markdown(path)
        doc = ParsedDocument(title=extract_title(md, path.stem), source_file=path.name)
        doc.elements = markdown_to_elements(md)
        return doc

    def _fetch_markdown(self, path: Path) -> str:  # pragma: no cover
        provider = self._s.cloud_parse_provider
        if provider == "llamaparse":
            return self._llamaparse(path)
        if provider == "azure":
            return self._azure(path)
        raise ParseError(f"unknown cloud_parse_provider: {provider}")

    def _llamaparse(self, path: Path) -> str:  # pragma: no cover
        from llama_cloud_services import LlamaParse

        parser = LlamaParse(api_key=self._s.llamaparse_api_key, result_type="markdown")
        docs = parser.load_data(str(path))
        return "\n\n".join(d.text for d in docs)

    def _azure(self, path: Path) -> str:  # pragma: no cover
        from azure.ai.documentintelligence import DocumentIntelligenceClient
        from azure.core.credentials import AzureKeyCredential

        client = DocumentIntelligenceClient(
            endpoint=self._s.azure_di_endpoint,
            credential=AzureKeyCredential(self._s.azure_di_key),
        )
        with open(path, "rb") as f:
            poller = client.begin_analyze_document(
                "prebuilt-layout", body=f, output_content_format="markdown")
        return poller.result().content
