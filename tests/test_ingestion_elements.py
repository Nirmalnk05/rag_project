from src.ingestion import _doc_text, _element_units, build_chunks
from src.parsing.dispatcher import Dispatcher
from src.parsing.model import FigureElement, ParsedDocument, TableElement, TextElement


def test_element_units_text_is_windowed():
    long_text = " ".join(f"w{i}" for i in range(1500))
    units = _element_units(TextElement("s", long_text))
    assert len(units) == 3 and all(u[1] == "text" for u in units)


def test_element_units_table_is_atomic():
    units = _element_units(TableElement("Alarms", "| a | b |\n" * 500))
    assert len(units) == 1 and units[0][1] == "table"


def test_element_units_figure_uses_caption():
    assert _element_units(FigureElement("s", "a pump diagram"))[0] == ("a pump diagram", "figure", None)
    assert _element_units(FigureElement("s", "   ")) == []  # empty caption skipped


def test_doc_text_flattens_all_types():
    doc = ParsedDocument("T", "f", [
        TextElement("s", "prose"), TableElement("s", "| x |"), FigureElement("s", "cap"),
    ])
    txt = _doc_text(doc)
    assert "prose" in txt and "| x |" in txt and "cap" in txt


class StubDispatcher(Dispatcher):
    def __init__(self, doc):
        self._doc = doc

    def parse(self, path, *, force_cloud=False):
        return self._doc


def test_build_chunks_tags_metadata(tmp_path):
    (tmp_path / "m.pdf").write_bytes(b"%PDF-1.4 fake")
    doc = ParsedDocument("Pump Manual", "m.pdf", [
        TextElement("Intro", "the pump operates at 50 bar", page=1),
        TableElement("Alarms", "| code | meaning |\n| 4F | low pressure |", page=2),
        FigureElement("Wiring", "a wiring diagram showing the P-100 loop", page=3),
    ])
    chunks = build_chunks(tmp_path, use_contextual=False, dispatcher=StubDispatcher(doc))
    types = {c.metadata["element_type"] for c in chunks}
    assert types == {"text", "table", "figure"}
    table_chunk = next(c for c in chunks if c.metadata["element_type"] == "table")
    assert "4F" in table_chunk.text and table_chunk.section == "Alarms"
    assert table_chunk.metadata["page"] == 2
    assert all(c.metadata["source_file"] == "m.pdf" for c in chunks)


def test_build_chunks_rejects_empty_dir(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        build_chunks(tmp_path, use_contextual=False)
