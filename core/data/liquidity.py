"""Dollar ADV (average daily dollar volume) from FMP raw price files.

Used for liquidity-aware transaction costs. ADV is the trailing mean of
``volume * adj_close`` over ``window`` trading days — a simple capacity proxy,
not a full market-impact model.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from core.data.fmp.prices import PANEL_TIMEZONE

logger = logging.getLogger(__name__)

DEFAULT_RAW_DIR = Path("data/raw/fmp/prices")
DEFAULT_ADV_PATH = Path("data/factors/dollar_adv_21d.parquet")
ADV_WINDOW = 21

# One-way cost (decimal) by trailing dollar-ADV bucket.
# Rough large-cap / mid / small / micro schedule for US equities.
ADV_COST_SCHEDULE: tuple[tuple[float, float], ...] = (
    (100e6, 0.0005),  # >= $100M ADV → 5 bps
    (20e6, 0.0010),  # >= $20M  → 10 bps
    (5e6, 0.0020),  # >= $5M   → 20 bps
    (0.0, 0.0040),  # below    → 40 bps
)


def cost_bps_from_dollar_adv(dollar_adv: float) -> float:
    """
    Map a dollar-ADV level to a one-way transaction cost (decimal).

    Args:
        dollar_adv: Trailing average daily dollar volume (price × shares).

    Returns:
        Cost as a decimal (0.001 = 10 bps). NaN/non-positive ADV → illiquid bucket.
    """
    if not np.isfinite(dollar_adv) or dollar_adv <= 0:
        return ADV_COST_SCHEDULE[-1][1]
    for threshold, cost in ADV_COST_SCHEDULE:
        if dollar_adv >= threshold:
            return cost
    return ADV_COST_SCHEDULE[-1][1]


def build_dollar_adv_panel(
    raw_dir: Path = DEFAULT_RAW_DIR,
    window: int = ADV_WINDOW,
    symbols: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Build a wide dollar-ADV panel from per-symbol FMP raw files.

    Args:
        raw_dir: Directory of ``{SYMBOL}.parquet`` with ``adj_close`` and ``volume``.
        window: Trailing trading-day window for the mean.
        symbols: Optional subset; default = every parquet stem in ``raw_dir``.

    Returns:
        Wide DataFrame (tz-aware date index × symbol columns) of dollar ADV.
    """
    raw_dir = Path(raw_dir)
    paths = sorted(raw_dir.glob("*.parquet"))
    if symbols is not None:
        wanted = set(symbols)
        paths = [p for p in paths if p.stem in wanted]

    series_list: list[pd.Series] = []
    for path in paths:
        history = pd.read_parquet(path)
        if history.empty or "adj_close" not in history.columns or "volume" not in history.columns:
            continue
        dollar_volume = history["adj_close"].astype("float64") * history["volume"].astype("float64")
        dollar_adv = dollar_volume.rolling(window, min_periods=max(5, window // 2)).mean()
        dollar_adv.name = path.stem
        series_list.append(dollar_adv)

    if not series_list:
        empty_index = pd.DatetimeIndex([], tz=PANEL_TIMEZONE, name="date")
        return pd.DataFrame(index=empty_index)

    panel = pd.concat(series_list, axis=1).sort_index()
    if panel.index.tz is None:
        panel.index = panel.index.tz_localize(PANEL_TIMEZONE)
    panel.index.name = "date"
    logger.info(
        "Built dollar-ADV panel: %s symbols × %s dates (window=%d)",
        panel.shape[1],
        panel.shape[0],
        window,
    )
    return panel


def costs_for_date(
    dollar_adv_row: pd.Series,
    symbols: pd.Index,
    default_cost: float = 0.001,
) -> pd.Series:
    """
    Per-symbol one-way costs on one date from a dollar-ADV cross-section.

    Missing ADV falls back to ``default_cost``.
    """
    costs = pd.Series(default_cost, index=symbols, dtype="float64")
    for symbol in symbols:
        if symbol in dollar_adv_row.index and pd.notna(dollar_adv_row[symbol]):
            costs[symbol] = cost_bps_from_dollar_adv(float(dollar_adv_row[symbol]))
    return costs
