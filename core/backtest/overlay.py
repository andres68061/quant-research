"""Exposure overlay: scale a strategy's daily returns by a causal regime signal.

Applies an exposure series in [0, 1] (from an HMM regime model or a simple
baseline rule) to an existing strategy's net return stream, with a one-day
signal lag and transaction costs charged on exposure changes.
"""

from __future__ import annotations

import pandas as pd

__all__ = ["apply_exposure_overlay"]


def apply_exposure_overlay(
    net_returns: pd.Series,
    exposure: pd.Series,
    transaction_cost: float = 0.001,
    gross_leverage: float = 2.0,
    lag_days: int = 1,
) -> pd.DataFrame:
    """Scale a return stream by a causal exposure signal, net of scaling costs.

    The exposure observed at close(t - lag_days) multiplies the strategy
    return earned over close(t-1) → close(t), mirroring the ``signal_lag_days``
    convention in :func:`core.backtest.portfolio.create_signals_from_factor`.
    Changing exposure trades the whole book proportionally, so each change is
    charged ``|Δexposure| * gross_leverage / 2 * transaction_cost`` (one-way
    turnover convention, matching ``calculate_portfolio_returns``).

    Args:
        net_returns: Daily net return series of the underlying strategy.
        exposure: Exposure scale in [0, 1] indexed by date. May be on a
            different (e.g. tz-naive) index; it is aligned to ``net_returns``
            with forward-fill. Dates before the first exposure observation
            get exposure 0 (not yet actionable — no signal, no position).
        transaction_cost: One-way cost per unit turnover as a decimal.
        gross_leverage: Sum of absolute portfolio weights at full exposure
            (2.0 for a dollar-neutral long/short book, 1.0 for long-only).
        lag_days: Trading days between exposure observation and effect.

    Returns:
        DataFrame indexed like ``net_returns`` with columns:
        ``net_return`` (overlaid, net of scaling costs), ``exposure``
        (the lagged exposure actually applied), ``scaling_cost``.

    Example:
        >>> out = apply_exposure_overlay(book_returns, vix_exposure)
        >>> out["net_return"].add(1).prod()
    """
    if lag_days < 0:
        raise ValueError(f"lag_days must be >= 0, got {lag_days}")
    if float(exposure.min()) < 0.0 or float(exposure.max()) > 1.0:
        raise ValueError("exposure must lie in [0, 1]")

    aligned = exposure.copy().astype(float)
    if getattr(aligned.index, "tz", None) is None and net_returns.index.tz is not None:
        aligned.index = aligned.index.tz_localize(net_returns.index.tz)
    elif getattr(aligned.index, "tz", None) is not None and net_returns.index.tz is None:
        aligned.index = aligned.index.tz_localize(None)

    aligned = aligned.reindex(net_returns.index, method="ffill")
    applied = aligned.shift(lag_days).fillna(0.0)

    turnover = applied.diff().abs().fillna(applied.iloc[0] if len(applied) else 0.0)
    scaling_cost = turnover * gross_leverage / 2.0 * transaction_cost

    overlaid = applied * net_returns - scaling_cost
    return pd.DataFrame(
        {
            "net_return": overlaid,
            "exposure": applied,
            "scaling_cost": scaling_cost,
        }
    )
