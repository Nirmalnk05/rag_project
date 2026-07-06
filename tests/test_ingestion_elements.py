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


def test_acl_for_maps_roles_and_sites():
    from src.ingestion import _acl_for
    acl_map = {"restricted.pdf": {"roles": ["eng"], "sites": ["siteA"]}}
    assert _acl_for("restricted.pdf", acl_map) == {"acl_roles": ["eng"], "acl_sites": ["siteA"]}
    assert _acl_for("public.pdf", acl_map) == {}  # unlisted = public


def test_build_chunks_attaches_acl_from_sidecar(tmp_path, monkeypatch):
    import src.ingestion as ing
    (tmp_path / "restricted.pdf").write_bytes(b"%PDF fake")
    monkeypatch.setattr(ing, "ACL_SIDECAR_PATH",
                        _write(tmp_path / "acl.json",
                               {"restricted.pdf": {"roles": ["eng"]}}))
    doc = ParsedDocument("Manual", "restricted.pdf",
                         [TextElement("Intro", "sensitive text", page=1)])
    chunks = build_chunks(tmp_path, use_contextual=False, dispatcher=StubDispatcher(doc))
    assert chunks[0].metadata["acl_roles"] == ["eng"]


def _write(path, obj):
    import json
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def test_version_for_maps_metadata():
    from src.ingestion import _version_for
    vm = {"m.pdf": {"version": "3", "effective_date": "2026-01-01", "status": "superseded"}}
    assert _version_for("m.pdf", vm) == {
        "doc_version": "3", "doc_date": "2026-01-01", "doc_status": "superseded"}
    # unlisted defaults to active
    assert _version_for("other.pdf", vm) == {
        "doc_version": "", "doc_date": "", "doc_status": "active"}


def test_build_chunks_attaches_version_from_sidecar(tmp_path, monkeypatch):
    import src.ingestion as ing
    (tmp_path / "m.pdf").write_bytes(b"%PDF fake")
    monkeypatch.setattr(ing, "VERSIONS_SIDECAR_PATH",
                        _write(tmp_path / "versions.json",
                               {"m.pdf": {"version": "2", "status": "active",
                                          "effective_date": "2026-02-02"}}))
    doc = ParsedDocument("Manual", "m.pdf", [TextElement("Intro", "text", page=1)])
    chunks = build_chunks(tmp_path, use_contextual=False, dispatcher=StubDispatcher(doc))
    assert chunks[0].metadata["doc_version"] == "2"
    assert chunks[0].metadata["doc_status"] == "active"
