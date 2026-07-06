"""CMMS / system-of-record connector (pluggable).

Document RAG answers what *should* happen; the CMMS holds live asset status and
work-order history. This is the seam for a real SAP PM / Maximo / Fiix connector;
the mock returns canned data so the agent tool works offline and in tests.
"""
from __future__ import annotations

from src.config import Settings, get_settings


class CMMSConnector:
    def get_asset(self, asset_id: str) -> dict | None:
        raise NotImplementedError

    def get_work_orders(self, asset_id: str) -> list[dict]:
        raise NotImplementedError


class MockCMMSConnector(CMMSConnector):
    _ASSETS = {
        "P-100": {"name": "Seawater cooling pump P-100", "status": "running",
                  "last_service": "2026-05-12", "hours_since_service": 1180},
        "CH-200": {"name": "HVAC chiller CH-200", "status": "standby",
                   "last_service": "2026-06-01", "hours_since_service": 640},
    }
    _WORK_ORDERS = {
        "P-100": [{"id": "WO-4471", "type": "corrective", "status": "open",
                   "summary": "Alarm 4F intermittent — inspect suction strainer"}],
        "CH-200": [],
    }

    def get_asset(self, asset_id: str) -> dict | None:
        return self._ASSETS.get(asset_id.upper())

    def get_work_orders(self, asset_id: str) -> list[dict]:
        return self._WORK_ORDERS.get(asset_id.upper(), [])


def get_cmms_connector(settings: Settings | None = None) -> CMMSConnector:
    s = settings or get_settings()
    if s.cmms_connector == "mock":
        return MockCMMSConnector()
    raise ValueError(f"unknown CMMS connector: {s.cmms_connector}")
