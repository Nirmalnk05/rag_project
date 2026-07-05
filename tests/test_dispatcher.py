import pytest

from src.config import Settings
from src.parsing.base import DocumentParser, ParseError
from src.parsing.dispatcher import Dispatcher
from src.parsing.local_parser import LocalParser, RawItem
from src.parsing.model import FigureElement, ParsedDocument, TableElement, TextElement


# ---------- LocalParser mapping (override _extract_items) ----------

class FakeCaptioner:
    def __init__(self, caption="a wiring diagram"):
        self._c = caption
        self.calls = 0

    def caption_sync(self, image, media_type):
        self.calls += 1
        return self._c


class StubLocal(LocalParser):
    def __init__(self, items, **kw):
        super().__init__(**kw)
        self._items = items

    def _extract_items(self, path):
        return self._items


def _settings(**kw):
    return Settings(_env_file=None, **kw)


def test_local_parser_maps_items_and_captions_figures(tmp_path):
    cap = FakeCaptioner()
    items = [
        RawItem(kind="text", section="Intro", text="hello", page=1),
        RawItem(kind="table", text="| a | b |", page=2),
        RawItem(kind="figure", image=b"img", media_type="image/png", page=3),
        RawItem(kind="text", text="   ", page=4),  # blank text dropped
    ]
    doc = StubLocal(items, settings=_settings(), captioner=cap).parse(tmp_path / "m.pdf")
    kinds = [e.type for e in doc.elements]
    assert kinds == ["text", "table", "figure"]  # blank text dropped
    assert isinstance(doc.elements[2], FigureElement)
    assert doc.elements[2].caption == "a wiring diagram" and cap.calls == 1


def test_figure_without_image_gets_empty_caption(tmp_path):
    cap = FakeCaptioner()
    items = [RawItem(kind="figure", image=None, page=1)]
    doc = StubLocal(items, settings=_settings(), captioner=cap).parse(tmp_path / "m.pdf")
    assert doc.elements[0].caption == "" and cap.calls == 0


# ---------- Dispatcher routing ----------

class RecordingParser(DocumentParser):
    def __init__(self, tag, fail=False):
        self.tag, self.fail, self.calls = tag, fail, 0

    def parse(self, path):
        self.calls += 1
        if self.fail:
            raise RuntimeError("boom")
        return ParsedDocument(title=self.tag, source_file=str(path),
                              elements=[TextElement("s", self.tag * 100)])


def _dispatcher(settings, local, cloud):
    return Dispatcher(settings=settings, local=local, cloud=cloud)


def test_excel_routed_to_excel_parser(tmp_path):
    excel = RecordingParser("excel")
    d = Dispatcher(settings=_settings(), local=RecordingParser("local"),
                   cloud=None, excel=excel)
    d.parse(tmp_path / "x.xlsx")
    assert excel.calls == 1


def test_unsupported_extension_raises(tmp_path):
    d = Dispatcher(settings=_settings(), local=RecordingParser("local"), cloud=None)
    with pytest.raises(ParseError):
        d.parse(tmp_path / "note.txt")


def test_local_backend_never_calls_cloud(tmp_path):
    local, cloud = RecordingParser("local"), RecordingParser("cloud")
    d = _dispatcher(_settings(parse_backend="local"), local, cloud)
    doc = d.parse(tmp_path / "m.pdf")
    assert doc.title == "local" and cloud.calls == 0


def test_hybrid_low_confidence_falls_to_cloud(tmp_path):
    # local returns short text (< min_chars) → low confidence → cloud
    local = RecordingParser("s")            # "s"*100 = 100 chars < 200
    cloud = RecordingParser("cloud")
    d = _dispatcher(_settings(parse_backend="hybrid", cloud_parse_min_chars=200), local, cloud)
    doc = d.parse(tmp_path / "m.pdf")
    assert doc.title == "cloud" and cloud.calls == 1


def test_hybrid_high_confidence_keeps_local(tmp_path):
    local = RecordingParser("localx")       # "localx"*100 = 600 chars > 200
    cloud = RecordingParser("cloud")
    d = _dispatcher(_settings(parse_backend="hybrid", cloud_parse_min_chars=200), local, cloud)
    doc = d.parse(tmp_path / "m.pdf")
    assert doc.title == "localx" and cloud.calls == 0


def test_hybrid_low_confidence_no_cloud_keeps_local(tmp_path):
    local = RecordingParser("s")
    d = _dispatcher(_settings(parse_backend="hybrid", cloud_parse_min_chars=200), local, None)
    doc = d.parse(tmp_path / "m.pdf")
    assert doc.title == "s"  # cloud unavailable → local kept


def test_force_cloud_uses_cloud(tmp_path):
    local, cloud = RecordingParser("local"), RecordingParser("cloud")
    d = _dispatcher(_settings(parse_backend="hybrid"), local, cloud)
    doc = d.parse(tmp_path / "m.pdf", force_cloud=True)
    assert doc.title == "cloud" and local.calls == 0


def test_cloud_failure_falls_back_to_local(tmp_path):
    local = RecordingParser("localx")
    cloud = RecordingParser("cloud", fail=True)
    d = _dispatcher(_settings(parse_backend="cloud"), local, cloud)
    doc = d.parse(tmp_path / "m.pdf")
    assert doc.title == "localx" and local.calls == 1
