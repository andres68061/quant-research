"""Fetch and parse FMP quarterly financial statements (point-in-time raw layer).

Endpoints: ``income-statement``, ``balance-sheet-statement``,
``cash-flow-statement`` with ``period=quarter``. Every row carries the three
dates that matter for point-in-time correctness:

- ``date``           -> reference_date (fiscal period end)
- ``filingDate``     -> publication_date (SEC filing)
- ``acceptedDate``   -> publication timestamp (EDGAR acceptance)

Raw files keep vendor fields verbatim; the derived PIT panel is built by
``core.data.factors.fundamentals``.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

from core.data.fmp.client import fmp_get
from core.exceptions import DataSchemaError

logger = logging.getLogger(__name__)

STATEMENT_ENDPOINTS = {
    "income_statement": "income-statement",
    "balance_sheet": "balance-sheet-statement",
    "cash_flow": "cash-flow-statement",
}

# ~30 years of quarters plus headroom; FMP caps per plan, extra is harmless.
_QUARTERS_LIMIT = 200


def parse_statement_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Convert raw statement JSON rows into a typed DataFrame (verbatim fields).

    Args:
        rows: JSON list from a statement endpoint (any order).

    Returns:
        DataFrame sorted by ``date`` ascending with ``date``, ``filingDate``
        parsed as datetimes and ``acceptedDate`` as a timestamp. Empty
        DataFrame when ``rows`` is empty. All vendor columns are preserved.

    Raises:
        DataSchemaError: If the payload lacks the point-in-time date fields.

    Example:
        >>> parse_statement_rows([{"date": "2024-03-30", "filingDate": "2024-05-02",
        ...     "acceptedDate": "2024-05-02 18:04:00", "symbol": "X", "revenue": 1.0}
        ... ])["revenue"].iloc[0]
        1.0
    """
    if not rows:
        return pd.DataFrame()

    missing = {"date", "filingDate", "acceptedDate"} - set(rows[0])
    if missing:
        raise DataSchemaError(f"FMP statement payload missing PIT fields: {missing}")

    statements = pd.DataFrame(rows)
    statements["date"] = pd.to_datetime(statements["date"])
    statements["filingDate"] = pd.to_datetime(statements["filingDate"])
    statements["acceptedDate"] = pd.to_datetime(statements["acceptedDate"])
    statements = statements.sort_values("date").reset_index(drop=True)
    return statements[~statements["date"].duplicated(keep="last")]


def fetch_quarterly_statement(
    symbol: str,
    statement: str,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch one quarterly statement type for one symbol.

    Args:
        symbol: Ticker as listed on FMP.
        statement: One of ``"income_statement"``, ``"balance_sheet"``, ``"cash_flow"``.
        api_key: Optional key override.

    Returns:
        Parsed DataFrame per :func:`parse_statement_rows`; empty if no coverage.
    """
    if statement not in STATEMENT_ENDPOINTS:
        raise DataSchemaError(f"Unknown statement type: {statement!r}")
    rows = fmp_get(
        STATEMENT_ENDPOINTS[statement],
        {"symbol": symbol, "period": "quarter", "limit": _QUARTERS_LIMIT},
        api_key=api_key,
    )
    if not isinstance(rows, list):
        raise DataSchemaError(f"Unexpected FMP payload for {symbol}/{statement}: {type(rows)}")
    return parse_statement_rows(rows)
