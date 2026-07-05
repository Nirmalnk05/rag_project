from src.config import Settings
from src.parsing.cloud_parser import CloudParser
from src.parsing.markdown_parser import MarkdownParser, markdown_to_elements


MD = """# Pump Manual

## Alarms

Some intro prose about alarms.

| code | meaning |
| --- | --- |
| 4F | low suction |

More prose after the table.

## Bearings

Alignment tolerance is 0.15mm.
"""


def test_markdown_splits_text_and_tables():
    els = markdown_to_elements(MD)
    kinds = [e.type for e in els]
    # Alarms: text, table, text ; Bearings: text
    assert kinds == ["text", "table", "text", "text"]
    table = next(e for e in els if e.type == "table")
    assert table.section == "Alarms" and "4F" in table.markdown
    assert els[0].section == "Alarms"
    assert els[-1].section == "Bearings"


def test_markdown_parser_reads_file(tmp_path):
    p = tmp_path / "pump.md"
    p.write_text(MD, encoding="utf-8")
    doc = MarkdownParser().parse(p)
    assert doc.title == "Pump Manual" and doc.source_file == "pump.md"
    assert len(doc.by_type("table")) == 1


def test_cloud_parser_parses_returned_markdown(tmp_path):
    class StubCloud(CloudParser):
        def _fetch_markdown(self, path):
            return MD

    doc = StubCloud(settings=Settings(_env_file=None)).parse(tmp_path / "scan.pdf")
    assert doc.title == "Pump Manual"
    assert len(doc.by_type("table")) == 1
    assert doc.source_file == "scan.pdf"
