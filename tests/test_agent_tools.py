import pytest

from src.agent.tools import ToolRegistry, _coerce_top_k
from src.retrieval import Retrieved


class FakeRetriever:
    def __init__(self, results=None, chunks=None):
        self._results = results if results is not None else []
        self._chunks = chunks or []
        self.calls = []

    async def hybrid_retrieve(self, query, filters=None, access=None):
        self.calls.append((query, filters))
        return self._results


def _r(text, section="Alarms", title="Pump Manual", src="pump.pdf", etype="text"):
    return Retrieved("c1", "d1", title, section, text, "", 0.9, "rerank")


async def test_search_records_numbered_sources():
    reg = ToolRegistry(FakeRetriever(results=[_r("code 4F = low pressure"),
                                              _r("check the strainer")]))
    out = await reg.execute("search", {"query": "4F"})
    assert "[#1]" in out and "[#2]" in out
    assert reg.num_sources == 2


async def test_search_numbering_continues_across_calls():
    reg = ToolRegistry(FakeRetriever(results=[_r("a")]))
    await reg.execute("search", {"query": "x"})
    out2 = await reg.execute("search", {"query": "y"})
    assert "[#2]" in out2 and reg.num_sources == 2


def test_coerce_top_k():
    assert _coerce_top_k(5) == 5
    assert _coerce_top_k("5") == 5           # local models sometimes send a string
    assert _coerce_top_k("5.0") is None      # not parseable as int -> ignored, not raised
    assert _coerce_top_k(0) is None
    assert _coerce_top_k(-3) is None
    assert _coerce_top_k(None) is None
    assert _coerce_top_k("not a number") is None
    assert _coerce_top_k({"nested": 5}) is None


async def test_search_with_string_top_k_does_not_crash():
    """Regression test: the model can send top_k as a JSON string instead of
    an int (observed with a local Ollama model during eval). Must clip
    results, not raise TypeError on list slicing."""
    results = [_r(f"result {i}") for i in range(5)]
    reg = ToolRegistry(FakeRetriever(results=results))
    out = await reg.execute("search", {"query": "P-100 seal", "top_k": "2"})
    assert "Tool error" not in out
    assert reg.num_sources == 2  # clipped to top_k=2, not all 5


async def test_search_with_malformed_top_k_falls_back_to_unclipped():
    results = [_r(f"result {i}") for i in range(3)]
    reg = ToolRegistry(FakeRetriever(results=results))
    out = await reg.execute("search", {"query": "P-100 seal", "top_k": "not-a-number"})
    assert "Tool error" not in out
    assert reg.num_sources == 3  # malformed top_k ignored, all results returned


async def test_lookup_fault_code_filters_tables():
    fr = FakeRetriever(results=[_r("4F row")])
    reg = ToolRegistry(fr)
    await reg.execute("lookup_fault_code", {"code": "4F"})
    assert fr.calls[-1] == ("4F", {"element_type": "table"})


async def test_search_empty_results():
    reg = ToolRegistry(FakeRetriever(results=[]))
    assert "No results" in await reg.execute("search", {"query": "zzz"})


async def test_tool_error_is_returned_not_raised():
    class Boom(FakeRetriever):
        async def hybrid_retrieve(self, query, filters=None):
            raise ValueError("unsupported filter key(s): ['evil']")

    reg = ToolRegistry(Boom())
    out = await reg.execute("search", {"query": "x", "filters": {"evil": "1"}})
    assert out.startswith("Tool error:")


async def test_get_document_section():
    chunks = [
        {"text": "full startup text", "doc_title": "Pump", "section": "Startup",
         "metadata": {"source_file": "pump.pdf"}},
        {"text": "other", "doc_title": "Pump", "section": "Shutdown",
         "metadata": {"source_file": "pump.pdf"}},
    ]
    reg = ToolRegistry(FakeRetriever(chunks=chunks))
    out = await reg.execute("get_document_section",
                            {"source_file": "pump.pdf", "section": "Startup"})
    assert "full startup text" in out and reg.num_sources == 1


async def test_list_documents():
    chunks = [
        {"doc_title": "Pump Manual", "metadata": {"source_file": "pump.pdf"}},
        {"doc_title": "Chiller SOP", "metadata": {"source_file": "hvac.pdf"}},
    ]
    reg = ToolRegistry(FakeRetriever(chunks=chunks))
    out = await reg.execute("list_documents", {})
    assert "Pump Manual (pump.pdf)" in out and "Chiller SOP (hvac.pdf)" in out


async def test_unknown_tool():
    reg = ToolRegistry(FakeRetriever())
    assert "unknown tool" in await reg.execute("nope", {})


async def test_search_passes_access_to_retriever():
    from src.access import AccessContext

    class AccessRecordingRetriever(FakeRetriever):
        def __init__(self):
            super().__init__(results=[_r("x")])
            self.seen_access = "unset"

        async def hybrid_retrieve(self, query, filters=None, access=None):
            self.seen_access = access
            return self._results

    ctx = AccessContext(roles=frozenset({"tech"}))
    fr = AccessRecordingRetriever()
    reg = ToolRegistry(fr, access=ctx)
    await reg.execute("search", {"query": "pump"})
    assert fr.seen_access is ctx


async def test_get_section_and_list_respect_acl():
    from src.access import AccessContext
    chunks = [
        {"text": "public", "doc_title": "Pub", "section": "S",
         "metadata": {"source_file": "pub.md"}},
        {"text": "secret", "doc_title": "Sec", "section": "S",
         "metadata": {"source_file": "sec.md", "acl_roles": ["eng"]}},
    ]
    tech = ToolRegistry(FakeRetriever(chunks=chunks), access=AccessContext(roles=frozenset({"tech"})))
    docs = await tech.execute("list_documents", {})
    assert "pub.md" in docs and "sec.md" not in docs
    blocked = await tech.execute("get_document_section", {"source_file": "sec.md", "section": "S"})
    assert "No section" in blocked


def test_specs_exposed():
    reg = ToolRegistry(FakeRetriever())
    names = {t.name for t in reg.specs}
    assert names == {"search", "lookup_fault_code", "get_document_section", "list_documents"}


async def test_search_source_line_includes_revision():
    r = Retrieved("c1", "d", "Pump Manual", "Alarms", "4F text", "", 0.9, "rerank")
    # attach version metadata via a chunk dict path: use get_document_section
    chunks = [{"text": "4F procedure", "doc_title": "Pump Manual", "section": "Alarms",
               "metadata": {"source_file": "p.pdf", "doc_version": "3", "doc_date": "2026-02-02"}}]
    reg = ToolRegistry(FakeRetriever(chunks=chunks))
    out = await reg.execute("get_document_section", {"source_file": "p.pdf", "section": "Alarms"})
    assert "rev 3 2026-02-02" in out
