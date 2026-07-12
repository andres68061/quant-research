"""Fetch FMP historical market capitalization (true historical shares × price).

Endpoint: ``historical-market-capitalization``. Cap is 5000 rows per call, so
full histories are fetched in date-range chunks (same pattern as prices).

Raw files keep vendor fields; the derived panel is MultiIndex (date, symbol)
with a single ``market_cap`` column — the contract expected by
``load_market_cap`` / ``compute_fundamental_factors``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from core.data.fmp.client import fmp_get
from core.data.fmp.prices import PANEL_TIMEZONE, generate_date_chunks
from core.exceptions import DataSchemaError

logger = logging.getLogger(__name__)

_ROW_CAP = 5000
DEFAULT_START = pd.Timestamp("1985-01-01")


def parse_market_cap_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Convert raw historical-market-capitalization JSON into a typed DataFrame.

    Args:
        rows: JSON list (any order; typically newest-first).

    Returns:
        DataFrame indexed by tz-aware ``date`` (America/New_York), column
        ``market_cap`` (float64), sorted ascending, deduplicated.
    """
    if not rows:
        empty_index = pd.DatetimeIndex([], tz=PANEL_TIMEZONE, name="date")
        return pd.DataFrame(columns=["market_cap"], index=empty_index)

    missing = {"date", "marketCap"} - set(rows[0])
    if missing:
        raise DataSchemaError(f"FMP market-cap payload missing fields: {missing}")

    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(PANEL_TIMEZONE)
    frame = (
        frame.set_index("date")[["marketCap"]]
        .rename(columns={"marketCap": "market_cap"})
        .astype({"market_cap": "float64"})
        .sort_index()
    )
    return frame[~frame.index.duplicated(keep="last")]


def fetch_historical_market_cap(
    symbol: str,
    start: pd.Timestamp = DEFAULT_START,
    end: Optional[pd.Timestamp] = None,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch full historical market-cap series for one symbol, chunked.

    Args:
        symbol: Ticker as listed on FMP.
        start: First date (inclusive).
        end: Last date (inclusive); defaults to today.
        api_key: Optional key override.

    Returns:
        Parsed DataFrame per :func:`parse_market_cap_rows`; empty if no coverage.
    """
    fetch_end = end if end is not None else pd.Timestamp.now().normalize()
    all_rows: list[dict[str, Any]] = []
    for chunk_from, chunk_to in generate_date_chunks(start, fetch_end):
        rows = fmp_get(
            "historical-market-capitalization",
            {"symbol": symbol, "from": chunk_from, "to": chunk_to},
            api_key=api_key,
        )
        if not isinstance(rows, list):
            raise DataSchemaError(f"Unexpected FMP market-cap payload for {symbol}: {type(rows)}")
        if len(rows) >= _ROW_CAP:
            logger.warning(
                "Market-cap chunk %s..%s for %s hit the %d-row cap; data may be truncated",
                chunk_from,
                chunk_to,
                symbol,
                _ROW_CAP,
            )
        all_rows.extend(rows)
    return parse_market_cap_rows(all_rows)


def build_market_cap_panel(raw_dir: Path) -> pd.DataFrame:
    """
    Stack per-symbol raw market-cap files into a MultiIndex panel.

    Args:
        raw_dir: Directory of ``{SYMBOL}.parquet`` files (tz-aware date index,
            ``market_cap`` column).

    Returns:
        MultiIndex ``(date, symbol)`` DataFrame with column ``market_cap``.
    """
    raw_dir = Path(raw_dir)
    frames: list[pd.DataFrame] = []
    for path in sorted(raw_dir.glob("*.parquet")):
        series = pd.read_parquet(path)
        if series.empty or "market_cap" not in series.columns:
            continue
        symbol_frame = series[["market_cap"]].copy()
        symbol_frame["symbol"] = path.stem
        frames.append(symbol_frame)
    if not frames:
        return pd.DataFrame(columns=["market_cap"])
    panel = pd.concat(frames)
    panel.index.name = "date"
    return panel.set_index("symbol", append=True).sort_index()
