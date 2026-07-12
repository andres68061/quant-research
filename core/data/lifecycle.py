"""Symbol lifecycle windows: truncate ticker-reuse Frankenstein series.

A ticker can refer to different companies over time (STI was SunTrust until the
2019 Truist merger; later the symbol was recycled). The S&P membership filter
protects standard factor backtests, but raw panel consumers can still bridge
two entities. This module builds per-symbol ``valid_from`` / ``valid_to``
windows and applies them to the derived price panel (raw layer untouched).

Window sources (union of evidence):
- Price panel first/last non-null date (baseline)
- FMP ``delisted-companies`` ``delistedDate`` / ``ipoDate`` when present
- FMP ``symbol-change`` events: old symbol ends on the change date

Statuses are written to ``data/quality/symbol_lifecycle.parquet``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from config.settings import PROJECT_ROOT
from core.data.fmp.client import fmp_get
from core.exceptions import DataSchemaError

logger = logging.getLogger(__name__)

LIFECYCLE_PATH = Path("data/quality/symbol_lifecycle.parquet")


def fetch_all_symbol_changes(api_key: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch FMP ``/symbol-change`` (single capped response — page param is ignored).

    Returns:
        Columns ``date``, ``old_symbol``, ``new_symbol``, ``company_name``.
    """
    # Endpoint ignores ``page`` and repeats the same first page; raise ``limit`` instead.
    rows = fmp_get("symbol-change", {"limit": 10000}, api_key=api_key)
    if not isinstance(rows, list):
        raise DataSchemaError(f"Unexpected symbol-change payload: {type(rows)}")
    if not rows:
        return pd.DataFrame(columns=["date", "old_symbol", "new_symbol", "company_name"])
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime([r.get("date") for r in rows]),
            "old_symbol": [str(r.get("oldSymbol") or "").strip() for r in rows],
            "new_symbol": [str(r.get("newSymbol") or "").strip() for r in rows],
            "company_name": [r.get("companyName") or "" for r in rows],
        }
    )
    frame = frame[frame["old_symbol"].astype(bool)].drop_duplicates(
        subset=["date", "old_symbol", "new_symbol"]
    )
    return frame.sort_values("date").reset_index(drop=True)


def fetch_all_delisted(api_key: Optional[str] = None) -> pd.DataFrame:
    """Paginate FMP ``/delisted-companies`` (limit capped at 100 per page)."""
    rows: list[dict] = []
    for page in range(0, 500):
        chunk = fmp_get("delisted-companies", {"page": page, "limit": 100}, api_key=api_key)
        if not isinstance(chunk, list) or not chunk:
            break
        if page > 0 and rows and chunk[0] == rows[0]:
            logger.warning("delisted pagination repeated page 0 at page=%d; stopping", page)
            break
        rows.extend(chunk)
        if len(chunk) < 100:
            break
    if not rows:
        return pd.DataFrame(columns=["symbol", "ipo_date", "delisted_date", "company_name"])
    return pd.DataFrame(
        {
            "symbol": [str(r.get("symbol") or "").strip() for r in rows],
            "ipo_date": pd.to_datetime([r.get("ipoDate") for r in rows], errors="coerce"),
            "delisted_date": pd.to_datetime([r.get("delistedDate") for r in rows], errors="coerce"),
            "company_name": [r.get("companyName") or "" for r in rows],
        }
    )


def build_lifecycle_windows(
    prices: pd.DataFrame,
    symbol_changes: Optional[pd.DataFrame] = None,
    delisted: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build per-symbol validity windows for the price panel.

    Args:
        prices: Wide adjusted-close panel.
        symbol_changes: Output of :func:`fetch_all_symbol_changes` (optional).
        delisted: Output of :func:`fetch_all_delisted` (optional).

    Returns:
        DataFrame with columns ``symbol``, ``valid_from``, ``valid_to``,
        ``source_notes`` (semicolon-joined reasons for truncation).
    """
    change_end: dict[str, pd.Timestamp] = {}
    if symbol_changes is not None and not symbol_changes.empty:
        # Earliest rename of a symbol ends that symbol's validity.
        for row in symbol_changes.itertuples():
            if not row.old_symbol:
                continue
            end = pd.Timestamp(row.date)
            prev = change_end.get(row.old_symbol)
            if prev is None or end < prev:
                change_end[row.old_symbol] = end

    delisted_bounds: dict[str, tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]] = {}
    if delisted is not None and not delisted.empty:
        for row in delisted.itertuples():
            if not row.symbol:
                continue
            delisted_bounds[row.symbol] = (
                pd.Timestamp(row.ipo_date) if pd.notna(row.ipo_date) else None,
                pd.Timestamp(row.delisted_date) if pd.notna(row.delisted_date) else None,
            )

    tz = prices.index.tz
    records = []
    for symbol in prices.columns:
        series = prices[symbol].dropna()
        if series.empty:
            continue
        valid_from = series.index.min()
        valid_to = series.index.max()
        notes: list[str] = ["price_span"]

        if symbol in delisted_bounds:
            ipo, gone = delisted_bounds[symbol]
            if ipo is not None:
                ipo_ts = ipo.tz_localize(tz) if tz is not None and ipo.tzinfo is None else ipo
                # Only raise the floor when IPO is inside the observed span.
                if valid_from < ipo_ts <= valid_to:
                    valid_from = ipo_ts
                    notes.append("ipoDate")
            if gone is not None:
                gone_ts = gone.tz_localize(tz) if tz is not None and gone.tzinfo is None else gone
                # Trust delisting only when the series ends near that date.
                # Continuing prices long after = a different company reused the ticker
                # in FMP's delisted registry.
                days_after = (valid_to - gone_ts).days
                if gone_ts < valid_to and 0 <= days_after <= 30:
                    valid_to = gone_ts
                    notes.append("delistedDate")

        if symbol in change_end:
            end = change_end[symbol]
            end_ts = end.tz_localize(tz) if tz is not None and end.tzinfo is None else end
            # Trust a rename only when the price series actually ends near it.
            # False positives (symbol recycled as oldSymbol while still trading)
            # leave last_px far after the change date.
            days_after = (valid_to - end_ts).days
            if end_ts < valid_to and 0 <= days_after <= 90:
                valid_to = end_ts
                notes.append("symbol_change")

        # Multi-year price gap: keep the earliest contiguous segment (ticker reuse).
        gap_to = _earliest_segment_end(series, max_gap_days=252)
        if gap_to is not None and gap_to < valid_to:
            valid_to = gap_to
            notes.append("price_gap")

        records.append(
            {
                "symbol": symbol,
                "valid_from": valid_from,
                "valid_to": valid_to,
                "source_notes": ";".join(notes),
            }
        )
    windows = pd.DataFrame(records)
    # Drop inverted windows (bad registry rows); keep price_span-only fallback.
    if not windows.empty:
        bad = windows["valid_from"] > windows["valid_to"]
        if bad.any():
            logger.warning("Dropping %d inverted lifecycle windows", int(bad.sum()))
            windows = windows.loc[~bad].reset_index(drop=True)
    return windows


def _earliest_segment_end(series: pd.Series, max_gap_days: int = 252) -> Optional[pd.Timestamp]:
    """
    If the series has a calendar gap longer than ``max_gap_days``, return the
    last date of the first contiguous segment; otherwise ``None``.
    """
    if series.empty:
        return None
    dates = series.index
    gaps = dates.to_series().diff()
    big = gaps[gaps > pd.Timedelta(days=max_gap_days)]
    if big.empty:
        return None
    first_gap_loc = big.index[0]
    # End of the segment is the date just before the gap.
    pos = dates.get_loc(first_gap_loc)
    if isinstance(pos, slice):
        pos = pos.start
    if pos == 0:
        return None
    return dates[pos - 1]


def apply_lifecycle_to_panel(
    prices: pd.DataFrame,
    windows: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """
    NaN out prices outside each symbol's validity window (derived layer only).

    Returns:
        Tuple of (truncated panel, number of cells cleared).
    """
    if windows.empty:
        return prices.copy(), 0
    truncated = prices.copy()
    n_cleared = 0
    for row in windows.itertuples():
        if row.symbol not in truncated.columns:
            continue
        # Only truncate when the window is stricter than the observed price span.
        series = truncated[row.symbol]
        observed_from = series.first_valid_index()
        observed_to = series.last_valid_index()
        if observed_from is None:
            continue
        mask = (truncated.index < row.valid_from) | (truncated.index > row.valid_to)
        if not mask.any():
            continue
        # Skip no-ops that match the observed span exactly.
        if row.valid_from <= observed_from and row.valid_to >= observed_to:
            continue
        before = int(series.notna().sum())
        truncated.loc[mask, row.symbol] = pd.NA
        after = int(truncated[row.symbol].notna().sum())
        n_cleared += before - after
    return truncated, n_cleared


def write_lifecycle_windows(windows: pd.DataFrame, path: Path = LIFECYCLE_PATH) -> None:
    """Persist lifecycle windows under the project root."""
    out = PROJECT_ROOT / path if not path.is_absolute() else path
    out.parent.mkdir(parents=True, exist_ok=True)
    windows.to_parquet(out, index=False)
    logger.info("Wrote lifecycle windows: %d symbols → %s", len(windows), out)
