"""Parse and fetch FMP end-of-day price history.

The platform's canonical stock price layer (``data/factors/prices.parquet``) is a
wide panel of split+dividend adjusted closes with a tz-aware
(America/New_York) index. The matching FMP endpoint is
``historical-price-eod/dividend-adjusted`` (fields ``adjOpen/adjHigh/adjLow/adjClose/volume``).

The endpoint caps responses at 5000 rows, so full histories are fetched in
date-range chunks (default 5 calendar years ≈ 1260 rows per chunk).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

from core.data.fmp.client import fmp_get
from core.exceptions import DataSchemaError

logger = logging.getLogger(__name__)

PANEL_TIMEZONE = "America/New_York"
_CHUNK_YEARS = 5
_ROW_CAP = 5000

_COLUMN_RENAMES = {
    "adjOpen": "adj_open",
    "adjHigh": "adj_high",
    "adjLow": "adj_low",
    "adjClose": "adj_close",
}


def parse_dividend_adjusted_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Convert raw ``historical-price-eod/dividend-adjusted`` JSON rows to a DataFrame.

    Args:
        rows: JSON list from the endpoint (any order; typically newest-first).

    Returns:
        DataFrame indexed by tz-aware (America/New_York) ``date``, sorted
        ascending, deduplicated, with float64 columns ``adj_open``, ``adj_high``,
        ``adj_low``, ``adj_close`` and int64 ``volume``. Empty DataFrame with the
        same schema when ``rows`` is empty.

    Raises:
        DataSchemaError: If required fields are missing from the payload.

    Example:
        >>> parse_dividend_adjusted_rows(
        ...     [{"symbol": "X", "date": "2024-01-02", "adjOpen": 1.0,
        ...       "adjHigh": 1.2, "adjLow": 0.9, "adjClose": 1.1, "volume": 10}]
        ... )["adj_close"].iloc[0]
        1.1
    """
    schema_columns = ["adj_open", "adj_high", "adj_low", "adj_close", "volume"]
    if not rows:
        empty_index = pd.DatetimeIndex([], tz=PANEL_TIMEZONE, name="date")
        return pd.DataFrame(columns=schema_columns, index=empty_index)

    missing = {"date", "adjClose"} - set(rows[0])
    if missing:
        raise DataSchemaError(f"FMP dividend-adjusted payload missing fields: {missing}")

    daily_bars = pd.DataFrame(rows).rename(columns=_COLUMN_RENAMES)
    daily_bars["date"] = pd.to_datetime(daily_bars["date"]).dt.tz_localize(PANEL_TIMEZONE)
    daily_bars = (
        daily_bars.set_index("date")[schema_columns]
        .astype({c: "float64" for c in schema_columns[:-1]} | {"volume": "int64"})
        .sort_index()
    )
    # The API occasionally repeats a date at chunk boundaries; keep the last.
    return daily_bars[~daily_bars.index.duplicated(keep="last")]


def generate_date_chunks(
    start: pd.Timestamp,
    end: pd.Timestamp,
    years_per_chunk: int = _CHUNK_YEARS,
) -> list[tuple[str, str]]:
    """
    Split [start, end] into consecutive from/to string pairs for chunked fetching.

    Args:
        start: First date (inclusive).
        end: Last date (inclusive).
        years_per_chunk: Calendar years per chunk; 5 keeps each response
            (~252 * 5 = 1260 rows) far below the endpoint's 5000-row cap.

    Returns:
        List of ("YYYY-MM-DD", "YYYY-MM-DD") tuples covering the range.

    Example:
        >>> generate_date_chunks(pd.Timestamp("2020-01-01"), pd.Timestamp("2021-06-30"))
        [('2020-01-01', '2021-06-30')]
    """
    if start > end:
        raise DataSchemaError(f"start {start} is after end {end}")
    chunks: list[tuple[str, str]] = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(
            chunk_start + pd.DateOffset(years=years_per_chunk) - pd.Timedelta(days=1), end
        )
        chunks.append((chunk_start.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        chunk_start = chunk_end + pd.Timedelta(days=1)
    return chunks


def fetch_dividend_adjusted_history(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch the full dividend-adjusted daily history for one symbol, chunked.

    Args:
        symbol: Ticker as listed on FMP (e.g. ``"AAPL"``).
        start: First date (inclusive).
        end: Last date (inclusive).
        api_key: Optional key override (defaults to ``FMP_API_KEY`` from config).

    Returns:
        DataFrame per :func:`parse_dividend_adjusted_rows`; empty if FMP has no
        data for the symbol (e.g. delisted names absent from FMP's EOD coverage).
    """
    # Index symbols (^GSPC, ^VIX, ...) are rejected by the dividend-adjusted
    # endpoint (HTTP 402); they carry no dividend adjustment, so the plain
    # OHLC history is the correct equivalent.
    endpoint = (
        "historical-price-eod/full"
        if symbol.startswith("^")
        else "historical-price-eod/dividend-adjusted"
    )
    all_rows: list[dict[str, Any]] = []
    for chunk_from, chunk_to in generate_date_chunks(start, end):
        rows = fmp_get(
            endpoint,
            {"symbol": symbol, "from": chunk_from, "to": chunk_to},
            api_key=api_key,
        )
        if not isinstance(rows, list):
            raise DataSchemaError(f"Unexpected FMP payload for {symbol}: {type(rows)}")
        if len(rows) >= _ROW_CAP:
            logger.warning(
                "Chunk %s..%s for %s hit the %d-row cap; data may be truncated",
                chunk_from,
                chunk_to,
                symbol,
                _ROW_CAP,
            )
        all_rows.extend(rows)
    if endpoint.endswith("/full"):
        return _parse_full_rows_as_adjusted(all_rows)
    return parse_dividend_adjusted_rows(all_rows)


def _parse_full_rows_as_adjusted(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Map ``/full`` payload (open/high/low/close) onto the adj_* schema for indexes."""
    renamed = [
        {
            "symbol": r.get("symbol"),
            "date": r["date"],
            "adjOpen": r["open"],
            "adjHigh": r["high"],
            "adjLow": r["low"],
            "adjClose": r["close"],
            "volume": r.get("volume", 0),
        }
        for r in rows
    ]
    return parse_dividend_adjusted_rows(renamed)
