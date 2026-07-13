"""Pairs-trading backtest runner (cointegration spread mean-reversion)."""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from core.signals.pairs import (
    align_pair_log_prices,
    engle_granger_test,
    pairs_position_from_zscore,
    rolling_hedge_ratio,
    rolling_spread_zscore,
    spread_from_hedge,
)

logger = logging.getLogger(__name__)

__all__ = ["run_pairs_cointegration_backtest"]


def run_pairs_cointegration_backtest(
    prices: pd.DataFrame,
    *,
    symbol_y: str,
    symbol_x: str,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    hedge_window: int = 252,
    zscore_window: int = 60,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    transaction_cost: float = 0.001,
    signal_lag_days: int = 1,
) -> dict[str, Any]:
    """
    Run a single-pair Engle–Granger style mean-reversion backtest.

    Position on day t (after lag) is applied to day-t simple returns of y and x.
    Notional: long-spread puts +1 on y and −hedge_ratio on x (share units of
    log-price hedge), then dollar-weights are normalized so |w_y| + |w_x| = 1
    when in a trade.

    Args:
        prices: Wide adj_close panel.
        symbol_y / symbol_x: Pair legs (y is the dependent OLS leg).
        start / end: Optional date slice on the price index.
        hedge_window: Rolling OLS lookback for the hedge ratio.
        zscore_window: Rolling window for spread z-score.
        entry_z / exit_z: Z-score thresholds (see ``pairs_position_from_zscore``).
        transaction_cost: One-way cost as a fraction of gross notional turnover.
        signal_lag_days: Trading days between signal and return application.

    Returns:
        Dict with ``net_returns``, ``gross_returns``, ``spread_z``, ``position``,
        ``hedge_ratio``, ``diagnostics`` (Engle–Granger full-sample test on the
        estimation window used for reporting only).
    """
    if signal_lag_days < 0:
        raise ValueError("signal_lag_days must be >= 0")

    panel = prices.sort_index()
    if start is not None:
        panel = panel.loc[panel.index >= start]
    if end is not None:
        panel = panel.loc[panel.index <= end]

    log_y, log_x = align_pair_log_prices(panel, symbol_y, symbol_x)
    eg = engle_granger_test(log_y, log_x)

    beta = rolling_hedge_ratio(log_y, log_x, window=hedge_window)
    spread = spread_from_hedge(log_y, log_x, beta)
    zscore = rolling_spread_zscore(spread, window=zscore_window)
    raw_pos = pairs_position_from_zscore(zscore, entry_z=entry_z, exit_z=exit_z)

    # Lag signal so today's close signal trades tomorrow's return.
    position = raw_pos.shift(signal_lag_days)
    beta_lag = beta.reindex(position.index).shift(signal_lag_days)

    idx = position.index
    py = panel[symbol_y].reindex(idx)
    px = panel[symbol_x].reindex(idx)
    ret_y = py.pct_change(fill_method=None)
    ret_x = px.pct_change(fill_method=None)
    zscore = zscore.reindex(idx)
    beta = beta.reindex(idx)

    # Share weights from hedge: w_y = pos, w_x = -pos * beta; then L1-normalize.
    w_y_raw = position
    w_x_raw = -position * beta_lag
    gross = w_y_raw.abs() + w_x_raw.abs()
    w_y = (w_y_raw / gross.replace(0.0, np.nan)).fillna(0.0)
    w_x = (w_x_raw / gross.replace(0.0, np.nan)).fillna(0.0)

    gross_returns = (w_y * ret_y + w_x * ret_x).fillna(0.0).rename("gross_returns")

    # Turnover on absolute weight changes (both legs).
    turnover = w_y.diff().abs().fillna(0.0) + w_x.diff().abs().fillna(0.0)
    costs = turnover * transaction_cost
    net_returns = (gross_returns - costs).rename("net_returns")

    # Drop warm-up NaNs from the first return row / incomplete windows.
    valid = (
        zscore.notna()
        & position.notna()
        & ret_y.notna()
        & ret_x.notna()
        & beta_lag.notna()
    )
    net_returns = net_returns.loc[valid]
    gross_returns = gross_returns.loc[valid]
    zscore = zscore.loc[valid]
    position = position.loc[valid]
    beta = beta.loc[valid]

    logger.info(
        "pairs backtest %s/%s: days=%d adf_p=%.4f hedge≈%.3f",
        symbol_y,
        symbol_x,
        len(net_returns),
        eg["adf_pvalue"],
        eg["hedge_ratio"],
    )

    return {
        "net_returns": net_returns,
        "gross_returns": gross_returns,
        "spread_z": zscore,
        "position": position,
        "hedge_ratio": beta,
        "diagnostics": {
            "symbol_y": symbol_y,
            "symbol_x": symbol_x,
            "engle_granger": eg,
            "hedge_window": hedge_window,
            "zscore_window": zscore_window,
            "entry_z": entry_z,
            "exit_z": exit_z,
            "transaction_cost": transaction_cost,
            "signal_lag_days": signal_lag_days,
            "n_days": len(net_returns),
            "pct_days_in_trade": float((position.fillna(0.0).abs() > 0).mean()),
        },
    }
