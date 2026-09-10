"""Quarterly-rebalanced, cap-weighted "simple top-N" index construction.

Same theory as the S&P 500 — hold the largest N companies, weighted by
market cap — but with the committee removed: membership is recomputed
mechanically at each calendar quarter's last trading day from point-in-time
market caps, with no float adjustment, no profitability screens, and no
grace periods for entrants/leavers.

Deliberate simplifications (disclose wherever results are shown):

- **Full-cap weights**, not float-adjusted (the real index uses float).
- **Universe = symbols with market-cap history in this repo's data**, which
  was collected from current + former S&P 500 membership. A large company
  the committee never admitted cannot appear here before it entered our
  data — this is closer to an S&P reconstruction than an independent index.
- **Delistings**: a position whose price series stops mid-quarter is frozen
  at its last close (ffill inside the segment only) and drops out at the
  next rebalance. This is a deliberate, documented exception to the
  "no forward-fill" rule.
- **Gross returns** — no transaction costs (cap-weight turnover is small).

All functions are pure: DataFrames in, DataFrames out. No I/O.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from core.exceptions import DataSchemaError

__all__ = [
    "CapWeightedIndexResult",
    "build_quarterly_rebalance_dates",
    "select_top_n_by_market_cap",
    "compute_cap_weighted_index",
]

logger = logging.getLogger(__name__)


@dataclass
class CapWeightedIndexResult:
    """Output of ``compute_cap_weighted_index``.

    Attributes:
        daily_returns: Gross daily index returns (starts the day after the
            first execution date).
        holdings: Per rebalance (selection) date: DataFrame indexed by symbol
            with ``market_cap`` and ``weight`` columns, sorted by weight desc.
        turnover: Per rebalance date, one-sided turnover vs the drifted
            previous portfolio (0.0 for the first).
        execution_dates: Trading day each selection was executed (selection
            date + ``execution_lag_days``).
    """

    daily_returns: pd.Series
    holdings: dict[pd.Timestamp, pd.DataFrame]
    turnover: pd.Series
    execution_dates: dict[pd.Timestamp, pd.Timestamp]


def build_quarterly_rebalance_dates(
    trading_days: pd.DatetimeIndex,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    max_quarter_end_gap_days: int = 7,
) -> pd.DatetimeIndex:
    """
    Last trading day of each calendar quarter within [start, end].

    A quarter only counts if its last observed trading day falls within
    ``max_quarter_end_gap_days`` calendar days of the true calendar quarter
    end — this drops the trailing, still-incomplete quarter of a live
    dataset instead of treating "latest day we have" as a quarter end.

    Args:
        trading_days: Tz-aware, monotonic, unique trading-day index.
        start: Inclusive lower bound (default: first trading day).
        end: Inclusive upper bound (default: last trading day).
        max_quarter_end_gap_days: Tolerance vs the calendar quarter end.

    Returns:
        DatetimeIndex of quarter-end trading days.
    """
    if not trading_days.is_monotonic_increasing or trading_days.has_duplicates:
        raise DataSchemaError("trading_days must be monotonic and unique")
    idx = trading_days
    if start is not None:
        idx = idx[idx >= start]
    if end is not None:
        idx = idx[idx <= end]
    if idx.empty:
        return idx

    frame = pd.DataFrame({"day": idx}, index=idx)
    frame["quarter_key"] = list(zip(idx.year, idx.quarter, strict=True))
    last_days = frame.groupby("quarter_key")["day"].max()

    kept: list[pd.Timestamp] = []
    for day in last_days:
        quarter_end = pd.Timestamp(
            pd.Period(f"{day.year}Q{day.quarter}").end_time.date(), tz=day.tz
        )
        if (quarter_end - day.normalize()).days <= max_quarter_end_gap_days:
            kept.append(day)
    return pd.DatetimeIndex(sorted(kept))


def select_top_n_by_market_cap(
    market_caps: pd.DataFrame,
    as_of_date: pd.Timestamp,
    top_n: int = 500,
    max_staleness_days: int = 10,
) -> pd.Series:
    """
    The largest ``top_n`` symbols by market cap as of a date.

    Uses the most recent market-cap row at or before ``as_of_date`` (market
    cap is price × known shares, so publication ≈ reference — same evening).

    Args:
        market_caps: Wide panel (date index × symbol) of market caps.
        as_of_date: Selection date (the backtest clock).
        top_n: Number of names to keep.
        max_staleness_days: Max trading days the latest cap row may lag
            ``as_of_date`` before raising.

    Returns:
        Series of market caps indexed by symbol, sorted descending,
        length <= ``top_n`` (shorter coverage is logged, not fatal).

    Raises:
        DataSchemaError: If no market-cap row exists within the staleness
            window at ``as_of_date``.
    """
    history = market_caps.loc[:as_of_date]
    if history.empty or len(market_caps.index) == 0:
        raise DataSchemaError(f"No market-cap data at or before {as_of_date}")
    row_date = history.index[-1]
    n_stale = len(market_caps.loc[row_date:as_of_date].index) - 1
    if n_stale > max_staleness_days:
        raise DataSchemaError(
            f"Market caps stale at {as_of_date}: latest row {row_date} " f"is {n_stale} rows behind"
        )
    row = history.iloc[-1].dropna()
    row = row[row > 0]
    if len(row) < top_n:
        logger.warning(
            "top-%d selection at %s: only %d symbols have market caps",
            top_n,
            as_of_date.date(),
            len(row),
        )
    return row.nlargest(top_n)


def compute_cap_weighted_index(
    prices: pd.DataFrame,
    market_caps: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    top_n: int = 500,
    execution_lag_days: int = 1,
) -> CapWeightedIndexResult:
    """
    Simulate the quarterly cap-weighted top-N index.

    At each rebalance (selection) date, membership and weights are fixed
    from that day's market caps; the portfolio is (re)established at the
    close ``execution_lag_days`` trading days later. Between rebalances the
    share counts are held fixed, so weights drift with prices — a
    cap-weighted portfolio is self-rebalancing and trades only at the
    quarterly reset.

    Args:
        prices: Wide adj_close panel (tz-aware date index × symbols).
        market_caps: Wide market-cap panel on the same calendar.
        rebalance_dates: Selection dates (quarter-end trading days).
        top_n: Names per portfolio.
        execution_lag_days: Trading days between selection and execution.

    Returns:
        CapWeightedIndexResult (gross daily returns, holdings, turnover).

    Raises:
        DataSchemaError: If no rebalance date can be executed within the
            price history.
    """
    if len(rebalance_dates) == 0:
        raise DataSchemaError("rebalance_dates is empty")

    segments: list[pd.Series] = []
    holdings: dict[pd.Timestamp, pd.DataFrame] = {}
    execution_dates: dict[pd.Timestamp, pd.Timestamp] = {}
    turnover_values: dict[pd.Timestamp, float] = {}
    prev_drifted: Optional[pd.Series] = None

    for i, selection_date in enumerate(rebalance_dates):
        exec_pos = prices.index.get_loc(selection_date) + execution_lag_days
        if exec_pos >= len(prices.index):
            logger.warning(
                "Rebalance %s cannot execute (price history ends); stopping",
                selection_date.date(),
            )
            break
        exec_date = prices.index[exec_pos]

        if i + 1 < len(rebalance_dates):
            next_exec_pos = min(
                prices.index.get_loc(rebalance_dates[i + 1]) + execution_lag_days,
                len(prices.index) - 1,
            )
        else:
            next_exec_pos = len(prices.index) - 1

        caps = select_top_n_by_market_cap(market_caps, selection_date, top_n)
        symbols = [s for s in caps.index if s in prices.columns]
        segment_prices = prices.iloc[exec_pos : next_exec_pos + 1][symbols]
        tradeable = segment_prices.iloc[0].notna()
        caps = caps[tradeable[tradeable].index]
        weights = caps / caps.sum()

        holdings[selection_date] = pd.DataFrame(
            {"market_cap": caps, "weight": weights}
        ).sort_values("weight", ascending=False)
        execution_dates[selection_date] = exec_date

        # Buy-and-hold within the segment: freeze delisted names at last close.
        relative = segment_prices[weights.index].ffill()
        relative = relative / relative.iloc[0]
        wealth = relative @ weights
        segments.append(wealth.pct_change().iloc[1:])

        if prev_drifted is None:
            turnover_values[selection_date] = 0.0
        else:
            aligned_new, aligned_old = weights.align(prev_drifted, fill_value=0.0)
            turnover_values[selection_date] = float(0.5 * (aligned_new - aligned_old).abs().sum())
        drifted = relative.iloc[-1] * weights
        prev_drifted = drifted / drifted.sum()

    if not segments:
        raise DataSchemaError("No rebalance date could be executed within price history")

    daily_returns = pd.concat(segments)
    daily_returns.name = f"top{top_n}_cap_weighted"
    return CapWeightedIndexResult(
        daily_returns=daily_returns,
        holdings=holdings,
        turnover=pd.Series(turnover_values, name="turnover"),
        execution_dates=execution_dates,
    )
