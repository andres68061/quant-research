"""Incremental updates of the wide price panel from FMP.

Daily-update counterpart of the bulk downloader (``scripts/fetch_fmp_prices.py``):
fetches a short trailing window per symbol, appends to the per-symbol raw layer,
and merges the new rows into the wide adjusted-close panel.

The refetch window overlaps the last few sessions on purpose: FMP restates
recent rows (late corrections, dividend adjustments), and overlapping rows are
overwritten with the vendor's latest values rather than blindly appended.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from core.data.fmp.prices import fetch_dividend_adjusted_history

logger = logging.getLogger(__name__)

RAW_PRICES_DIR = Path("data/raw/fmp/prices")
RESTATEMENT_LOOKBACK_DAYS = 7


def _update_raw_symbol_file(symbol: str, new_history: pd.DataFrame, raw_dir: Path) -> None:
    """Merge freshly fetched rows into the symbol's raw parquet (vendor-latest wins)."""
    raw_path = raw_dir / f"{symbol}.parquet"
    if raw_path.exists():
        existing = pd.read_parquet(raw_path)
        combined = pd.concat([existing, new_history])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = new_history
    raw_dir.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(raw_path)


def add_symbol_to_fmp_panel(
    existing_panel: pd.DataFrame,
    symbol: str,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    raw_dir: Path = RAW_PRICES_DIR,
) -> pd.DataFrame:
    """
    Fetch one symbol from FMP, write its raw file, and add/replace the panel column.

    Args:
        existing_panel: Wide adjusted-close panel.
        symbol: FMP ticker (e.g. ``"BNY"``).
        start: First date to fetch (default 1985-01-01).
        end: Last date (default today).
        raw_dir: Per-symbol raw parquet directory.

    Returns:
        Panel with ``symbol`` column merged in (index unioned).
    """
    fetch_start = start if start is not None else pd.Timestamp("1985-01-01")
    fetch_end = end if end is not None else pd.Timestamp.now().normalize()
    history = fetch_dividend_adjusted_history(symbol, fetch_start, fetch_end)
    if history.empty:
        logger.warning("No FMP history for %s; panel unchanged", symbol)
        return existing_panel

    _update_raw_symbol_file(symbol, history, raw_dir)
    close = history["adj_close"].rename(symbol)
    if close.index.tz is None and existing_panel.index.tz is not None:
        close.index = close.index.tz_localize(existing_panel.index.tz)

    updated = existing_panel.copy()
    if symbol in updated.columns:
        updated = updated.drop(columns=[symbol])
    updated = updated.join(close, how="outer").sort_index()
    logger.info(
        "Added %s to panel: %d rows (%s → %s)",
        symbol,
        close.notna().sum(),
        close.index.min().date(),
        close.index.max().date(),
    )
    return updated


def update_panel_from_fmp(
    existing_panel: pd.DataFrame,
    raw_dir: Path = RAW_PRICES_DIR,
    lookback_days: int = RESTATEMENT_LOOKBACK_DAYS,
    end: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Fetch recent FMP data for every panel symbol and merge it in.

    Args:
        existing_panel: Wide adjusted-close panel (tz-aware date index, symbol columns).
        raw_dir: Raw layer directory (per-symbol parquet files, updated in place).
        lookback_days: Calendar days before the panel's last date to refetch,
            so recent vendor restatements overwrite stale values.
        end: Last date to fetch (defaults to today).

    Returns:
        Updated panel: same columns, index extended to new dates; overlapping
        cells replaced by the vendor's latest values. Symbols with no new data
        (delisted, no FMP coverage) are left untouched.

    Example:
        >>> panel = update_panel_from_fmp(pd.read_parquet("data/factors/prices.parquet"))
        ... # doctest: +SKIP
    """
    last_date = existing_panel.index.max()
    fetch_start = (last_date - pd.Timedelta(days=lookback_days)).tz_localize(None)
    fetch_end = end if end is not None else pd.Timestamp.now().normalize()

    fresh_closes: dict[str, pd.Series] = {}
    n_empty = 0
    for symbol in existing_panel.columns:
        history = fetch_dividend_adjusted_history(symbol, fetch_start, fetch_end)
        if history.empty:
            n_empty += 1
            continue
        _update_raw_symbol_file(symbol, history, raw_dir)
        fresh_closes[symbol] = history["adj_close"]

    logger.info(
        "FMP incremental fetch: %d symbols updated, %d without new data",
        len(fresh_closes),
        n_empty,
    )
    if not fresh_closes:
        return existing_panel

    new_wide = pd.DataFrame(fresh_closes)
    new_wide.index.name = existing_panel.index.name
    if new_wide.index.tz is None and existing_panel.index.tz is not None:
        new_wide.index = new_wide.index.tz_localize(existing_panel.index.tz)

    combined_index = existing_panel.index.union(new_wide.index)
    updated_panel = existing_panel.reindex(combined_index)
    # update() overwrites overlapping cells with the vendor's latest values.
    updated_panel.update(new_wide)
    return updated_panel.sort_index()
