"""Raw-layer file helpers shared by the FMP fetch scripts.

Two concerns, both about surviving an interrupted multi-hour backfill:

- **Atomic writes.** A parquet written in place can be truncated if the process
  dies mid-write, and the next run — which decides what to skip by checking file
  existence — would treat the corpse as complete. Writing to a temp file and
  renaming makes the file appear all at once.
- **Per-symbol fetch windows.** A company that listed in 1998 and was delisted in
  2003 needs one 5-year chunk, not the nine that a 1985-today default would
  request. On a 9,000-symbol universe that is the difference between a 3-hour and
  a 6-hour run.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# FMP's earliest EOD coverage; earlier requests return empty.
EARLIEST_HISTORY = pd.Timestamp("1985-01-01")
# Delisted names keep a tail so the final days before the delisting are captured.
DELISTING_BUFFER_DAYS = 10


def write_atomic(frame: pd.DataFrame, path: Path) -> None:
    """
    Write a parquet via a temporary file so readers never see a partial file.

    Args:
        frame: Data to write.
        path: Final destination; parent directory must exist.
    """
    temporary = path.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary)
    os.replace(temporary, path)


def safe_filename(symbol: str) -> str:
    """Map a ticker to a filesystem-safe stem (``BRK/B`` would create a directory)."""
    return symbol.replace("/", "-")


def load_universe_symbols(universe_file: Path) -> list[str]:
    """
    Read the symbol column from a universe table.

    Args:
        universe_file: Parquet written by ``scripts/build/build_fmp_universe.py``.

    Returns:
        Sorted unique symbols.
    """
    universe = pd.read_parquet(universe_file)
    if "symbol" not in universe.columns:
        raise ValueError(f"{universe_file} has no 'symbol' column")
    return sorted(universe["symbol"].dropna().unique().tolist())


def load_fetch_windows(
    universe_file: Optional[Path],
    default_start: pd.Timestamp = EARLIEST_HISTORY,
    default_end: Optional[pd.Timestamp] = None,
) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Derive a per-symbol ``(start, end)`` fetch window from the universe table.

    A symbol's window is narrowed only where the table gives a reason to narrow
    it: ``ipo_date`` moves the start forward, ``delisted_date`` moves the end
    back. Missing dates fall back to the defaults, so a symbol is never silently
    under-fetched.

    Args:
        universe_file: Universe parquet, or None for "no windows known".
        default_start: Start used when no IPO date is available.
        default_end: End used when the symbol is still listed; defaults to today.

    Returns:
        ``{symbol: (start, end)}``. Empty dict when ``universe_file`` is None.
    """
    if universe_file is None:
        return {}

    default_end = default_end or pd.Timestamp.now().normalize()
    universe = pd.read_parquet(universe_file)
    windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}

    has_ipo = "ipo_date" in universe.columns
    has_delisted = "delisted_date" in universe.columns

    for row in universe.itertuples(index=False):
        symbol = getattr(row, "symbol", None)
        if not isinstance(symbol, str):
            continue

        start = default_start
        if has_ipo:
            ipo_date = getattr(row, "ipo_date", pd.NaT)
            if pd.notna(ipo_date):
                start = max(default_start, pd.Timestamp(ipo_date))

        end = default_end
        if has_delisted:
            delisted_date = getattr(row, "delisted_date", pd.NaT)
            if pd.notna(delisted_date):
                end = min(
                    default_end,
                    pd.Timestamp(delisted_date) + pd.Timedelta(days=DELISTING_BUFFER_DAYS),
                )

        # A nonsensical window (delisted before the default start) would raise in
        # the chunker; fall back to the full range and let the vendor return empty.
        windows[symbol] = (start, end) if start <= end else (default_start, default_end)

    return windows


def load_trading_calendar(
    canonical_panel: Path = Path("data/factors/prices.parquet"),
) -> pd.DatetimeIndex:
    """
    The NYSE trading calendar as observed by the canonical price panel.

    Expanded-universe raw files carry vendor bad prints on Sundays and US market
    holidays (975 Sunday bars found in the 2026-08 backfill). Panel builders
    intersect their date index with this calendar so those bars never enter a
    derived artifact.

    Args:
        canonical_panel: Wide panel whose index defines the calendar.

    Returns:
        tz-aware ascending DatetimeIndex of trading days.
    """
    return pd.read_parquet(canonical_panel, columns=[]).index
