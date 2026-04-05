"""
Shared factor cross-sectional backtest pipeline.

Centralizes date slicing, signal construction, and portfolio return simulation used by
HTTP backtest and replay endpoints.
"""

from __future__ import annotations

from typing import Callable, Optional

import pandas as pd

from core.backtest.portfolio import calculate_portfolio_returns, create_signals_from_factor


def run_factor_cross_section_backtest(
    factors: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    factor_col: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    top_pct: float = 0.20,
    bottom_pct: float = 0.20,
    long_only: bool = False,
    rebalance_freq: str = "ME",
    transaction_cost: float = 0.001,
    min_stocks: int = 20,
    universe_filter: Optional[Callable[[pd.Timestamp], set[str]]] = None,
    max_abs_value: Optional[float] = None,
) -> pd.Series:
    """
    Run a factor long/short (or long-only) backtest over [start, end].

    Args:
        factors: MultiIndex (date, symbol) with factor columns.
        prices: Wide panel, index = dates, columns = symbols.
        factor_col: Column in ``factors`` to rank on.
        start: Inclusive start timestamp for factor and price slices.
        end: Inclusive end timestamp for factor and price slices.
        top_pct: Fraction of names to go long each rebalance.
        bottom_pct: Fraction to short (ignored if long_only).
        long_only: If True, no short positions.
        rebalance_freq: Pandas offset for resampling (e.g. ME, QE, W, D; legacy M/Q normalized).
        transaction_cost: Per-trade cost as decimal (e.g. 0.001 = 10 bps).
        min_stocks: Minimum valid names per date for ranking.
        universe_filter: Optional point-in-time membership callable. See
            :func:`~core.backtest.portfolio.sp500_universe_filter`.
        max_abs_value: Passed to :func:`~core.backtest.portfolio.create_signals_from_factor`
            (``None`` infers factor-specific bounds).

    Returns:
        ``net_return`` daily series (aligned to portfolio return index).

    Example:
        >>> net = run_factor_cross_section_backtest(
        ...     factors, prices, factor_col="mom_12_1",
        ...     start=pd.Timestamp("2020-01-01"), end=pd.Timestamp("2023-01-01"),
        ... )
    """
    f_dates = factors.index.get_level_values("date")

    # Align start/end timezone to match the index so comparisons don't raise.
    # The factors index is tz-aware when built from yfinance prices; the API
    # passes tz-naive Timestamps, which pandas refuses to compare.
    def _align_tz(ts: pd.Timestamp, tz) -> pd.Timestamp:
        if tz is None:
            return ts.tz_localize(None) if ts.tzinfo is not None else ts
        return ts.tz_localize(tz) if ts.tzinfo is None else ts.tz_convert(tz)

    f_tz = getattr(f_dates, "tz", None)
    p_tz = getattr(prices.index, "tz", None)
    start_f, end_f = _align_tz(start, f_tz), _align_tz(end, f_tz)
    start_p, end_p = _align_tz(start, p_tz), _align_tz(end, p_tz)

    factors_slice = factors[(f_dates >= start_f) & (f_dates <= end_f)]
    prices_slice = prices[(prices.index >= start_p) & (prices.index <= end_p)]

    signals = create_signals_from_factor(
        factors_slice,
        factor_col,
        top_pct=top_pct,
        bottom_pct=bottom_pct,
        long_only=long_only,
        min_stocks=min_stocks,
        universe_filter=universe_filter,
        max_abs_value=max_abs_value,
    )

    results = calculate_portfolio_returns(
        signals,
        prices_slice,
        rebalance_freq=rebalance_freq,
        transaction_cost=transaction_cost,
        long_only=long_only,
    )
    return results["net_return"]
