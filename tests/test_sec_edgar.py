"""Tests for SEC EDGAR client (mocked HTTP)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd

from core.data.sec.client import (
    build_filings_sample,
    fetch_company_tickers,
    fetch_recent_filings,
)


def test_fetch_company_tickers_maps_cik() -> None:
    session = MagicMock()
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc"},
        "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft"},
    }
    session.get.return_value = response

    mapping = fetch_company_tickers(session=session)
    assert mapping["AAPL"] == "0000320193"
    assert mapping["MSFT"] == "0000789019"


def test_fetch_recent_filings_filters_forms() -> None:
    session = MagicMock()
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {
        "filings": {
            "recent": {
                "form": ["8-K", "10-Q", "10-K"],
                "filingDate": ["2024-01-02", "2024-04-30", "2023-11-03"],
                "acceptanceDateTime": [
                    "2024-01-02T16:00:00.000Z",
                    "2024-04-30T17:15:00.000Z",
                    "2023-11-03T16:30:00.000Z",
                ],
                "accessionNumber": ["0001", "0002", "0003"],
            }
        }
    }
    session.get.return_value = response

    frame = fetch_recent_filings("320193", session=session, limit=10)
    assert list(frame["form"]) == ["10-Q", "10-K"]
    assert frame.iloc[0]["accepted_date"] == "2024-04-30"


def test_build_filings_sample_attaches_symbol() -> None:
    session = MagicMock()

    def _get(url: str, **kwargs):  # noqa: ANN003
        response = MagicMock()
        response.raise_for_status = MagicMock()
        if "company_tickers" in url:
            response.json.return_value = {
                "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple"}
            }
        else:
            response.json.return_value = {
                "filings": {
                    "recent": {
                        "form": ["10-K"],
                        "filingDate": ["2023-11-03"],
                        "acceptanceDateTime": ["2023-11-03T16:30:00.000Z"],
                        "accessionNumber": ["0003"],
                    }
                }
            }
        return response

    session.get.side_effect = _get
    frame = build_filings_sample(["AAPL"], session=session)
    assert isinstance(frame, pd.DataFrame)
    assert frame.iloc[0]["symbol"] == "AAPL"
    assert frame.iloc[0]["cik"] == "0000320193"
