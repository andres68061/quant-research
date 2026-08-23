"""Cross-sectional (per-symbol) trailing path metrics.

``calculate_cid1_ratio`` in ``core.metrics.performance`` scores one strategy
curve over its whole history. This module scores *many symbols at many
evaluation dates* over a fixed trailing window so the ratio can be used as a
cross-sectional characteristic (ranked, regressed, sorted into quantiles).

Two deliberate departures from the strategy-level function (see ADR-0009):

1. A common trailing window (default 252 trading days) ending at each
   evaluation date, so the number is comparable across symbols.
2. The pain == 0 boundary maps to ``+inf`` (not 0.0): a path that never
   closed below its window-start value is the *best* outcome under this
   metric's ideology, not the worst. ``cid1_angle`` provides the finite,
   order-preserving transform for ranking and regression.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd

from core.exceptions import DataSchemaError

__all__ = ["calculate_trailing_cid1_cross_section"]

logger = logging.getLogger(__name__)


def calculate_trailing_cid1_cross_section(
    prices: pd.DataFrame,
    evaluation_dates: Sequence[pd.Timestamp] | pd.DatetimeIndex,
    window_days: int = 252,
) -> pd.DataFrame:
    """
    Trailing Cid-1 components for every symbol at each evaluation date.

    For each evaluation date, takes the ``window_days`` trading-day adj_close
    window ending at that date (inclusive), computes wealth relative to the
    window's first close, and derives per symbol:

    - ``total_return``: final wealth − 1 (decimal, over the window)
    - ``cost_basis_pain``: Σ max(0, 1 − wealth) over the window
    - ``cid1_ratio``: total_return / cost_basis_pain. Boundary convention:
      ``+inf`` when pain == 0 and total_return > 0; ``0.0`` only for the
      degenerate flat path (pain == 0 and total_return <= 0).
    - ``cid1_angle``: atan2(total_return, cost_basis_pain) — finite
      everywhere, identical ordering to ``cid1_ratio`` where pain > 0, and
      continuous at the pain → 0 boundary (pain-free paths map to π/2).

    Symbols with any missing close inside a window are excluded for that
    evaluation date (count logged, not silently absorbed).

    Args:
        prices: Wide adj_close panel (tz-aware date index × symbol columns).
        evaluation_dates: Dates at which to evaluate; each must be a trading
            day present in ``prices.index``.
        window_days: Trailing window length in trading days (>= 2).

    Returns:
        Long panel indexed by (date, symbol) with columns ``total_return``,
        ``cost_basis_pain``, ``cid1_ratio``, ``cid1_angle``.

    Raises:
        DataSchemaError: If ``window_days`` < 2, an evaluation date is not in
            the price index, or no evaluation date has enough history.
    """
    if window_days < 2:
        raise DataSchemaError(f"window_days must be >= 2, got {window_days}")

    frames: list[pd.DataFrame] = []
    skipped_short: list[pd.Timestamp] = []
    for date in pd.DatetimeIndex(evaluation_dates):
        try:
            pos = prices.index.get_loc(date)
        except KeyError as exc:
            raise DataSchemaError(f"Evaluation date {date} not in price index") from exc
        if not isinstance(pos, (int, np.integer)):
            raise DataSchemaError(f"Price index is not unique at {date}")
        if pos + 1 < window_days:
            skipped_short.append(date)
            continue

        window = prices.iloc[pos + 1 - window_days : pos + 1]
        base = window.iloc[0]
        valid = window.notna().all(axis=0) & (base > 0)
        n_dropped = int((~valid).sum())
        if n_dropped:
            logger.info(
                "cid1 cross-section %s: excluded %d/%d symbols with incomplete windows",
                date.date(),
                n_dropped,
                window.shape[1],
            )
        window = window.loc[:, valid]
        wealth = window / window.iloc[0]

        total_return = wealth.iloc[-1] - 1.0
        cost_basis_pain = (1.0 - wealth).clip(lower=0.0).sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(
                cost_basis_pain > 0,
                total_return / cost_basis_pain,
                np.where(total_return > 0, np.inf, 0.0),
            )
        angle = np.arctan2(total_return.to_numpy(), cost_basis_pain.to_numpy())

        frame = pd.DataFrame(
            {
                "total_return": total_return,
                "cost_basis_pain": cost_basis_pain,
                "cid1_ratio": ratio,
                "cid1_angle": angle,
            }
        )
        frame.index = pd.MultiIndex.from_product([[date], frame.index], names=["date", "symbol"])
        frames.append(frame)

    if skipped_short:
        logger.warning(
            "cid1 cross-section: skipped %d evaluation dates with < %d days of history "
            "(first: %s)",
            len(skipped_short),
            window_days,
            skipped_short[0].date(),
        )
    if not frames:
        raise DataSchemaError(f"No evaluation date had {window_days} trading days of history")
    return pd.concat(frames)
