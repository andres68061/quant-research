"""
Shared factor cross-sectional backtest pipeline.

Centralizes date slicing, signal construction, and portfolio return simulation used by
HTTP backtest and replay endpoints.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, TypedDict

import pandas as pd

from core.backtest.portfolio import calculate_portfolio_returns, create_signals_from_factor
from core.metrics.coverage import summarize_invested_coverage


class FactorBacktestDetail(TypedDict):
    """Full factor backtest output used by API / replay."""

    net_return: pd.Series
    n_long: pd.Series
    n_short: pd.Series
    coverage: Dict[str, Any]


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

    Returns:
        ``net_return`` daily series. For position counts and invested-coverage
        disclosure, use :func:`run_factor_cross_section_backtest_detail`.
    """
    detail = run_factor_cross_section_backtest_detail(
        factors,
        prices,
        factor_col=factor_col,
        start=start,
        end=end,
        top_pct=top_pct,
        bottom_pct=bottom_pct,
        long_only=long_only,
        rebalance_freq=rebalance_freq,
        transaction_cost=transaction_cost,
        min_stocks=min_stocks,
        universe_filter=universe_filter,
        max_abs_value=max_abs_value,
        signal_lag_days=signal_lag_days,
        dollar_adv=dollar_adv,
    )
    return detail["net_return"]


def run_factor_cross_section_backtest_detail(
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
) -> FactorBacktestDetail:
    """
    Run a factor backtest and return returns plus invested-coverage diagnostics.

    When fewer than ``min_stocks`` names have a valid factor on a rebalance date,
    signals are zero and the portfolio stays in cash (0% return) until the next
    successful rebalance — see ``coverage["warning"]``.

    End-of-day handling:
        When ``factors`` is tz-aware, a naive ``end`` date is pushed to end-of-day
        in that tz so the last US equity bar is included (see
        docs/FACTOR_BACKTEST_AUDIT.md §3 Bug 5).
    """
    f_dates = factors.index.get_level_values("date")

    def _align_tz(ts: pd.Timestamp, tz: Any, *, end_of_day: bool = False) -> pd.Timestamp:
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
    n_long = results["n_long"]
    n_short = results["n_short"]
    return {
        "net_return": results["net_return"],
        "n_long": n_long,
        "n_short": n_short,
        "coverage": summarize_invested_coverage(n_long, n_short, min_stocks=min_stocks),
    }
