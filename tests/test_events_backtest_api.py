"""HTTP tests for POST /backtest/events/simulate."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture()
def toy_prices() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2024-01-02", periods=10, tz="UTC")
    a = 100 * np.exp(rng.normal(0, 0.01, len(idx)).cumsum())
    b = 100 * np.exp(rng.normal(0, 0.01, len(idx)).cumsum())
    return pd.DataFrame({"AAA": a, "BBB": b}, index=idx)


def _rows_from_panel(panel: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for dt in panel.index:
        row: dict = {"date": dt.isoformat()}
        for c in panel.columns:
            row[str(c)] = float(panel.loc[dt, c])
        rows.append(row)
    return rows


def test_simulate_happy_path(client: TestClient, toy_prices: pd.DataFrame) -> None:
    body = {
        "events": [
            {
                "ts": "2024-01-01T00:00:00+00:00",
                "event_type": "rebalance",
                "payload": {"symbols": ["AAA", "BBB"]},
            }
        ],
        "price_rows": _rows_from_panel(toy_prices),
        "transaction_cost": 0.0,
    }
    r = client.post("/backtest/events/simulate", json=body)
    assert r.status_code == 200, r.text
    data = r.json()
    assert "dates" in data and "returns" in data
    assert len(data["dates"]) == len(data["returns"]) == len(toy_prices)


def test_simulate_rejects_duplicate_event_timestamps(
    client: TestClient, toy_prices: pd.DataFrame
) -> None:
    t = "2024-01-02T00:00:00+00:00"
    body = {
        "events": [
            {
                "ts": t,
                "event_type": "rebalance",
                "payload": {"symbols": ["AAA"]},
            },
            {
                "ts": t,
                "event_type": "rebalance",
                "payload": {"symbols": ["AAA", "BBB"]},
            },
        ],
        "price_rows": _rows_from_panel(toy_prices),
    }
    r = client.post("/backtest/events/simulate", json=body)
    assert r.status_code == 422


def test_simulate_rejects_empty_symbols_payload(
    client: TestClient, toy_prices: pd.DataFrame
) -> None:
    body = {
        "events": [
            {
                "ts": "2024-01-01T00:00:00+00:00",
                "event_type": "rebalance",
                "payload": {"symbols": []},
            }
        ],
        "price_rows": _rows_from_panel(toy_prices),
    }
    r = client.post("/backtest/events/simulate", json=body)
    assert r.status_code == 422


def test_simulate_rejects_bad_event_type(client: TestClient, toy_prices: pd.DataFrame) -> None:
    body = {
        "events": [
            {
                "ts": "2024-01-01T00:00:00+00:00",
                "event_type": "not_an_event",
                "payload": {"symbols": ["AAA"]},
            }
        ],
        "price_rows": _rows_from_panel(toy_prices),
    }
    r = client.post("/backtest/events/simulate", json=body)
    assert r.status_code == 422
