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

__all__ = ["run_pairs_cointegration_backtest", "run_pairs_holdout_backtest"]


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
    valid = zscore.notna() & position.notna() & ret_y.notna() & ret_x.notna() & beta_lag.notna()
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


def run_pairs_holdout_backtest(
    prices: pd.DataFrame,
    *,
    symbol_y: str,
    symbol_x: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    train_frac: float = 0.6,
    hedge_window: int = 252,
    zscore_window: int = 60,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    transaction_cost: float = 0.001,
    signal_lag_days: int = 1,
) -> dict[str, Any]:
    """
    Enforce train/held-out separation on a single-pair backtest.

    ``run_pairs_cointegration_backtest`` will happily compute a "backtest"
    over any ``[start, end]`` a caller supplies, including a range chosen
    *because* the caller already knows (from other analysis) that the pair
    performs well over it — selection bias, not signal leakage, but the
    reported number looks identical to a clean one either way. This
    function makes that impossible to do by accident: it splits
    ``[start, end]`` at ``train_frac``, computes the Engle-Granger
    cointegration diagnostic on the **train** slice only, and computes
    every PnL number (returns, equity curve, spread series) on the
    **held-out** slice only. The two are always returned separately and
    labeled — there is no blended number to misread.

    The held-out backtest is seeded with price history from before its
    start (same technique as ``pairs_index.run_pairs_stat_arb_index``) so
    the rolling hedge/z-score is warmed up by day one rather than eating
    into held-out days; that buffer only uses train-period prices, so nothing
    from the held-out window leaks backward into it.

    Args:
        prices: Wide adj_close panel.
        symbol_y / symbol_x: Pair legs.
        start / end: Overall date range to split.
        train_frac: Fraction of ``[start, end]`` used for the diagnostic-only
            train slice; the remainder is the held-out evaluation slice.
        hedge_window / zscore_window / entry_z / exit_z / transaction_cost /
            signal_lag_days: Passed through to the held-out backtest.

    Returns:
        Dict with ``train_start``, ``train_end``, ``held_out_start``,
        ``held_out_end`` (all ``pd.Timestamp``), ``train_diagnostics``
        (Engle-Granger dict), and ``held_out`` (the full
        ``run_pairs_cointegration_backtest`` result dict for the held-out
        slice only).
    """
    if not 0.2 <= train_frac <= 0.8:
        raise ValueError("train_frac must be in [0.2, 0.8]")
    if start >= end:
        raise ValueError("start must be before end")

    panel = prices.sort_index()
    panel = panel.loc[(panel.index >= start) & (panel.index <= end)]
    if len(panel) < 120:
        raise ValueError("Insufficient history in [start, end] for a train/held-out split")

    split_idx = int(len(panel) * train_frac)
    split_idx = max(60, min(split_idx, len(panel) - 40))
    train_panel = panel.iloc[:split_idx]
    held_out_start = panel.index[split_idx]

    log_y, log_x = align_pair_log_prices(train_panel, symbol_y, symbol_x)
    train_diagnostics = engle_granger_test(log_y, log_x)

    buffer_days = int((hedge_window + zscore_window) * 1.6) + 5
    warm_start = held_out_start - pd.DateOffset(days=buffer_days)
    held_out = run_pairs_cointegration_backtest(
        prices,
        symbol_y=symbol_y,
        symbol_x=symbol_x,
        start=warm_start,
        end=end,
        hedge_window=hedge_window,
        zscore_window=zscore_window,
        entry_z=entry_z,
        exit_z=exit_z,
        transaction_cost=transaction_cost,
        signal_lag_days=signal_lag_days,
    )
    for key in ("net_returns", "gross_returns", "spread_z", "position", "hedge_ratio"):
        held_out[key] = held_out[key].loc[held_out[key].index >= held_out_start]
    held_out["diagnostics"]["n_days"] = len(held_out["net_returns"])
    held_out["diagnostics"]["pct_days_in_trade"] = float(
        (held_out["position"].fillna(0.0).abs() > 0).mean()
    )

    logger.info(
        "pairs holdout %s/%s: train=%s..%s held_out=%s..%s train_adf_p=%.4f",
        symbol_y,
        symbol_x,
        train_panel.index[0],
        train_panel.index[-1],
        held_out_start,
        end,
        train_diagnostics["adf_pvalue"],
    )

    return {
        "train_start": train_panel.index[0],
        "train_end": train_panel.index[-1],
        "held_out_start": held_out_start,
        "held_out_end": end,
        "train_diagnostics": train_diagnostics,
        "held_out": held_out,
    }
