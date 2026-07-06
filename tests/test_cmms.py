import pytest

from src.agent.cmms import MockCMMSConnector
from src.agent.tools import ToolRegistry


class FakeRetriever:
    def __init__(self):
        self._chunks = []

    async def hybrid_retrieve(self, query, filters=None, access=None):
        return []


def test_mock_connector_returns_asset_and_work_orders():
    c = MockCMMSConnector()
    assert c.get_asset("p-100")["status"] == "running"   # case-insensitive
    assert c.get_asset("nope") is None
    assert c.get_work_orders("P-100")[0]["id"] == "WO-4471"


def test_cmms_tool_registered_only_when_connector_present():
    without = ToolRegistry(FakeRetriever())
    assert "query_cmms" not in {t.name for t in without.specs}
    with_cmms = ToolRegistry(FakeRetriever(), cmms=MockCMMSConnector())
    assert "query_cmms" in {t.name for t in with_cmms.specs}


async def test_query_cmms_tool_formats_output():
    reg = ToolRegistry(FakeRetriever(), cmms=MockCMMSConnector())
    out = await reg.execute("query_cmms", {"asset_id": "P-100"})
    assert "Seawater cooling pump P-100" in out
    assert "WO-4471" in out and "status running" in out


async def test_query_cmms_unknown_asset():
    reg = ToolRegistry(FakeRetriever(), cmms=MockCMMSConnector())
    out = await reg.execute("query_cmms", {"asset_id": "ZZ-9"})
    assert "No asset" in out
