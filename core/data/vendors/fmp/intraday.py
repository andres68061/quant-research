"""Fetch FMP intraday bars (``historical-chart/{interval}``).

Two vendor behaviours drive this module's design, and both are easy to get wrong:

**1. The response is bar-capped, not range-capped.** Asking for seven months of
1-minute bars returns the most recent ~1170 bars and silently drops the rest. A
naive "request the whole history" call looks successful and returns three days of
data. Chunk sizes here are set below each interval's observed cap, and a chunk
that comes back at the cap is logged as suspect.

**2. Intraday bars are split-adjusted only as of the fetch date, and never
dividend-adjusted.** Measured (2026-08-11) on AAPL 4:1, NVDA 10:1, TSLA 3:1: the
vendor returns intraday history split-adjusted at request time, so our stored
files are a frozen snapshot — a split occurring AFTER the fetch leaves them
stale. Read through :func:`load_intraday_bars`, which detects and repairs
exactly those splits; applying a blanket adjustment instead double-adjusts and
manufactures a fake jump at every historical split. Dividends are never
adjusted: overnight returns across ex-dividend dates carry the dividend drop.

Cost: intraday is expensive per symbol-year. At 1-minute resolution a single
symbol-year costs ~85 calls, so a 500-name universe over 20 years is ~850k calls
(~28 hours at the client's ceiling). Prefer ``1hour`` for broad coverage and
reserve ``1min`` for a named shortlist.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from core.data.vendors.fmp.client import fmp_get
from core.exceptions import DataSchemaError

logger = logging.getLogger(__name__)

PANEL_TIMEZONE = "America/New_York"


class IntradayInterval:
    """Chunking parameters for one intraday interval."""

    def __init__(self, name: str, chunk_days: int, bar_cap: int) -> None:
        self.name = name
        self.chunk_days = chunk_days
        self.bar_cap = bar_cap


# chunk_days are calendar days chosen to stay comfortably below the observed cap.
INTRADAY_INTERVALS: dict[str, IntradayInterval] = {
    "1min": IntradayInterval("1min", chunk_days=3, bar_cap=1170),
    "5min": IntradayInterval("5min", chunk_days=8, bar_cap=624),
    "15min": IntradayInterval("15min", chunk_days=25, bar_cap=624),
    "30min": IntradayInterval("30min", chunk_days=45, bar_cap=500),
    "1hour": IntradayInterval("1hour", chunk_days=70, bar_cap=434),
    "4hour": IntradayInterval("4hour", chunk_days=140, bar_cap=249),
}

_BAR_COLUMNS = ["open", "high", "low", "close", "volume"]


def parse_intraday_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Convert raw intraday JSON rows into a typed, deduplicated DataFrame.

    Args:
        rows: JSON list from ``historical-chart/{interval}``.

    Returns:
        DataFrame indexed by tz-aware (America/New_York) ``date``, ascending,
        with float64 OHLC and int64 ``volume``. Empty input yields an empty frame
        with the same schema.

    Raises:
        DataSchemaError: If the payload lacks ``date`` or ``close``.
    """
    if not rows:
        empty_index = pd.DatetimeIndex([], tz=PANEL_TIMEZONE, name="date")
        return pd.DataFrame(columns=_BAR_COLUMNS, index=empty_index)

    if missing := {"date", "close"} - set(rows[0]):
        raise DataSchemaError(f"FMP intraday payload missing fields: {missing}")

    bars = pd.DataFrame(rows)
    # Vendor timestamps are US/Eastern wall clock without an offset.
    bars["date"] = pd.to_datetime(bars["date"]).dt.tz_localize(
        PANEL_TIMEZONE, ambiguous="NaT", nonexistent="NaT"
    )
    bars = bars.dropna(subset=["date"]).set_index("date")

    for column in _BAR_COLUMNS:
        if column not in bars.columns:
            bars[column] = float("nan")
    bars = bars[_BAR_COLUMNS].astype(
        {c: "float64" for c in _BAR_COLUMNS[:-1]} | {"volume": "int64"}
    )
    return bars.sort_index()[lambda df: ~df.index.duplicated(keep="last")]


def generate_intraday_chunks(
    start: pd.Timestamp,
    end: pd.Timestamp,
    chunk_days: int,
) -> list[tuple[str, str]]:
    """
    Split a date range into windows small enough to stay under the bar cap.

    Args:
        start: First date (inclusive).
        end: Last date (inclusive).
        chunk_days: Calendar days per chunk, from :data:`INTRADAY_INTERVALS`.

    Returns:
        List of ``("YYYY-MM-DD", "YYYY-MM-DD")`` covering the range.

    Example:
        >>> generate_intraday_chunks(pd.Timestamp("2024-01-01"),
        ...                          pd.Timestamp("2024-01-05"), 3)
        [('2024-01-01', '2024-01-03'), ('2024-01-04', '2024-01-05')]
    """
    if start > end:
        raise DataSchemaError(f"start {start} is after end {end}")
    chunks: list[tuple[str, str]] = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + pd.Timedelta(days=chunk_days - 1), end)
        chunks.append((chunk_start.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        chunk_start = chunk_end + pd.Timedelta(days=1)
    return chunks


def fetch_intraday_history(
    symbol: str,
    interval: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch one symbol's intraday bars over a date range, chunked under the bar cap.

    Args:
        symbol: Ticker as listed on FMP.
        interval: Key into :data:`INTRADAY_INTERVALS`.
        start: First date (inclusive).
        end: Last date (inclusive).
        api_key: Optional key override.

    Returns:
        DataFrame per :func:`parse_intraday_rows`; empty when FMP has no coverage.

    Raises:
        DataSchemaError: For an unknown interval or a non-list payload.
    """
    if interval not in INTRADAY_INTERVALS:
        raise DataSchemaError(f"Unknown interval {interval!r}; known: {sorted(INTRADAY_INTERVALS)}")
    spec = INTRADAY_INTERVALS[interval]

    all_rows: list[dict[str, Any]] = []
    for chunk_from, chunk_to in generate_intraday_chunks(start, end, spec.chunk_days):
        rows = fmp_get(
            f"historical-chart/{interval}",
            {"symbol": symbol, "from": chunk_from, "to": chunk_to},
            api_key=api_key,
        )
        if not isinstance(rows, list):
            raise DataSchemaError(f"Unexpected intraday payload for {symbol}: {type(rows)}")
        if len(rows) >= spec.bar_cap:
            logger.warning(
                "%s %s chunk %s..%s returned %d bars (cap %d) — data may be truncated; "
                "reduce chunk_days for this interval",
                symbol,
                interval,
                chunk_from,
                chunk_to,
                len(rows),
                spec.bar_cap,
            )
        all_rows.extend(rows)

    return parse_intraday_rows(all_rows)


def apply_split_adjustment(bars: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """
    Retroactively split-adjust raw intraday bars.

    Intraday bars come back unadjusted, so a split looks like a price collapse.
    Each bar before a split date is divided by the cumulative ratio of all splits
    that happened at or after it; volume is multiplied by the same factor.

    Args:
        bars: Output of :func:`parse_intraday_rows` (tz-aware index).
        splits: Rows from the ``splits`` dataset with ``date``, ``numerator``,
            ``denominator``.

    Returns:
        Adjusted copy of ``bars``. Returns ``bars`` unchanged when there are no
        splits in range.

    Example:
        A 4:1 split on 2020-08-31 divides every earlier price by 4 and
        multiplies every earlier volume by 4.
    """
    if bars.empty or splits.empty:
        return bars

    ratios = splits.copy()
    ratios["date"] = pd.to_datetime(ratios["date"])
    if ratios["date"].dt.tz is None:
        ratios["date"] = ratios["date"].dt.tz_localize(bars.index.tz)
    ratios["ratio"] = pd.to_numeric(ratios["numerator"], errors="coerce") / pd.to_numeric(
        ratios["denominator"], errors="coerce"
    )
    ratios = ratios.dropna(subset=["ratio"]).sort_values("date")
    ratios = ratios[(ratios["ratio"] > 0) & (ratios["ratio"] != 1.0)]
    if ratios.empty:
        return bars

    # Cumulative factor applied to a bar = product of every split strictly after it.
    factor = pd.Series(1.0, index=bars.index)
    for split_date, ratio in zip(ratios["date"], ratios["ratio"], strict=True):
        factor = factor.where(bars.index >= split_date, factor * ratio)

    adjusted = bars.copy()
    for column in ("open", "high", "low", "close"):
        adjusted[column] = adjusted[column] / factor
    adjusted["volume"] = (adjusted["volume"] * factor).round().astype("int64")
    return adjusted


INTRADAY_RAW_DIR = Path("data/raw/fmp/intraday")
SPLITS_RAW_DIR = Path("data/raw/fmp/splits")


def detect_unapplied_splits(bars: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """
    Find splits whose adjustment is missing from stored bars — by measurement.

    FMP returns intraday history split-adjusted *as of the request date* (verified
    empirically on AAPL 4:1, NVDA 10:1, TSLA 3:1 — all continuous in the raw
    layer). But our stored files are a frozen snapshot: a split that happens
    AFTER the fetch leaves stored history un-adjusted for it, and an incremental
    append of fresh bars would then sit 1/ratio away from the older rows.

    Rather than tracking fetch timestamps, this measures each split directly:
    compare the close-to-close move across the split date against the split
    ratio, and classify the split as unapplied when the observed move is closer
    (in log space) to ``1/ratio`` than to ``1``.

    Args:
        bars: Stored intraday bars for one symbol.
        splits: Rows from the ``splits`` dataset (``date``, ``numerator``,
            ``denominator``).

    Returns:
        The subset of ``splits`` that needs applying. Ambiguous cases (ratio so
        small the market move could explain it) are logged and NOT applied —
        refusing to guess beats silently double-adjusting.
    """
    if bars.empty or splits.empty:
        return splits.iloc[0:0]

    events = splits.copy()
    events["date"] = pd.to_datetime(events["date"])
    if events["date"].dt.tz is None:
        events["date"] = events["date"].dt.tz_localize(bars.index.tz)
    events["ratio"] = pd.to_numeric(events["numerator"], errors="coerce") / pd.to_numeric(
        events["denominator"], errors="coerce"
    )
    events = events.dropna(subset=["ratio"])
    events = events[(events["ratio"] > 0) & (events["ratio"] != 1.0)]

    unapplied_rows = []
    closes = bars["close"]
    for row in events.itertuples(index=False):
        before = closes[closes.index < row.date]
        after = closes[closes.index >= row.date]
        if before.empty or after.empty:
            continue
        observed = float(after.iloc[0]) / float(before.iloc[-1])
        distance_to_adjusted = abs(np.log(observed))
        distance_to_unadjusted = abs(np.log(observed * row.ratio))
        if distance_to_unadjusted < distance_to_adjusted:
            if row.ratio < 1.25:
                logger.warning(
                    "Split %.3g:1 on %s is too small to classify reliably "
                    "(observed move %.1f%%); NOT adjusting — refetch the symbol instead",
                    row.ratio,
                    row.date.date(),
                    (observed - 1) * 100,
                )
                continue
            unapplied_rows.append(row)

    if not unapplied_rows:
        return events.iloc[0:0]
    return pd.DataFrame(unapplied_rows)


def load_intraday_bars(
    symbol: str,
    interval: str = "1hour",
    repair_splits: bool = True,
    intraday_dir: Path = INTRADAY_RAW_DIR,
    splits_dir: Path = SPLITS_RAW_DIR,
) -> pd.DataFrame:
    """
    Load one symbol's stored intraday bars, repairing any post-fetch splits.

    THE way to read intraday data. What the raw layer actually holds (measured,
    not assumed — see ``detect_unapplied_splits``):

    - **Split-adjusted as of the fetch date.** Do NOT apply a blanket split
      adjustment on top: that double-adjusts and manufactures a fake jump at
      every historical split (+310% on AAPL 2020 if you do it).
    - **NOT dividend-adjusted** (unlike the daily ``adj_close`` layer). An
      overnight return across an ex-dividend date carries the dividend drop as
      a fake negative return; reconcile against the daily layer when that
      matters.
    - A split occurring AFTER our fetch leaves the stored snapshot stale; this
      loader detects and repairs exactly those splits when ``repair_splits``.

    Args:
        symbol: Ticker as stored (``BRK/B`` file stem handled here).
        interval: Interval directory name (``"1hour"``, ``"1min"``, ...).
        repair_splits: Detect-and-apply post-fetch splits (default True).
        intraday_dir: Root of the per-interval raw layout.
        splits_dir: Directory of per-symbol splits files.

    Returns:
        DataFrame indexed by tz-aware timestamp with ``open/high/low/close/volume``,
        ascending, deduplicated. Empty frame when nothing is stored.

    Example:
        >>> bars = load_intraday_bars("AAPL", "1hour")  # doctest: +SKIP
    """
    stem = symbol.replace("/", "-")
    symbol_dir = Path(intraday_dir) / interval / stem
    empty_index = pd.DatetimeIndex([], tz=PANEL_TIMEZONE, name="date")
    if not symbol_dir.is_dir():
        return pd.DataFrame(columns=_BAR_COLUMNS, index=empty_index)

    frames = [pd.read_parquet(path) for path in sorted(symbol_dir.glob("*.parquet"))]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=_BAR_COLUMNS, index=empty_index)

    bars = pd.concat(frames).sort_index()
    bars = bars[~bars.index.duplicated(keep="last")]
    if not repair_splits:
        return bars

    splits_path = Path(splits_dir) / f"{stem}.parquet"
    if not splits_path.exists():
        logger.warning(
            "No splits file for %s; cannot verify split continuity of intraday bars", symbol
        )
        return bars
    unapplied = detect_unapplied_splits(bars, pd.read_parquet(splits_path))
    if unapplied.empty:
        return bars
    logger.info(
        "%s: repairing %d post-fetch split(s): %s",
        symbol,
        len(unapplied),
        [str(d.date()) for d in unapplied["date"]],
    )
    return apply_split_adjustment(bars, unapplied)
