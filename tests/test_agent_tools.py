import pytest

from src.agent.tools import ToolRegistry
from src.retrieval import Retrieved


class FakeRetriever:
    def __init__(self, results=None, chunks=None):
        self._results = results if results is not None else []
        self._chunks = chunks or []
        self.calls = []

    async def hybrid_retrieve(self, query, filters=None):
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


def test_specs_exposed():
    reg = ToolRegistry(FakeRetriever())
    names = {t.name for t in reg.specs}
    assert names == {"search", "lookup_fault_code", "get_document_section", "list_documents"}
