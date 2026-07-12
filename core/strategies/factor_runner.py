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
    signal_lag_days: int = 1,
    dollar_adv: Optional[pd.DataFrame] = None,
) -> pd.Series:
    """
    Run a factor long/short (or long-only) backtest over [start, end].

    Args:
        factors: MultiIndex (date, symbol) with factor columns.
        prices: Wide panel, index = dates, columns = symbols.
        factor_col: Column in ``factors`` to rank on.
        start: Inclusive start timestamp for factor and price slices.
        end: Inclusive end timestamp for factor and price slices. See the
            "End-of-day handling" note below — the slice is inclusive through
            the end of trading on ``end``, not through 00:00 of that date.
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
        signal_lag_days: Trading-day lag between factor observation and execution
            (default 1). See :func:`~core.backtest.portfolio.create_signals_from_factor`.
        dollar_adv: Optional dollar-ADV panel for liquidity-scaled costs.

    Returns:
        ``net_return`` daily series (aligned to portfolio return index).

    End-of-day handling:
        When ``factors`` is tz-aware (e.g. UTC from yfinance), localising a
        naive date like ``'2024-06-28'`` to UTC yields 00:00 UTC, which is
        *before* the US equity close at 20:00 UTC and silently excludes the
        last bar. The slice below normalises ``end`` to 23:59:59.999 in the
        factor tz so the trailing day is included. See
        docs/FACTOR_BACKTEST_AUDIT.md §3 Bug 5.

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
    #
    # For `end`, we push to end-of-day so that the last trading bar of the
    # requested date (which is typically ~20:00 UTC for US equities) is kept.
    def _align_tz(ts: pd.Timestamp, tz, *, end_of_day: bool = False) -> pd.Timestamp:
        # Only normalise to end-of-day if the caller passed a pure date
        # (midnight). If they passed an explicit intraday time they meant it.
        if end_of_day and ts == ts.normalize():
            ts = ts + pd.Timedelta(hours=23, minutes=59, seconds=59, microseconds=999_999)
        if tz is None:
            return ts.tz_localize(None) if ts.tzinfo is not None else ts
        return ts.tz_localize(tz) if ts.tzinfo is None else ts.tz_convert(tz)

    f_tz = getattr(f_dates, "tz", None)
    p_tz = getattr(prices.index, "tz", None)
    start_f = _align_tz(start, f_tz)
    end_f = _align_tz(end, f_tz, end_of_day=True)
    start_p = _align_tz(start, p_tz)
    end_p = _align_tz(end, p_tz, end_of_day=True)

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
        signal_lag_days=signal_lag_days,
    )

    results = calculate_portfolio_returns(
        signals,
        prices_slice,
        rebalance_freq=rebalance_freq,
        transaction_cost=transaction_cost,
        long_only=long_only,
        dollar_adv=None if dollar_adv is None else dollar_adv.loc[start_p:end_p],
    )
    return results["net_return"]
