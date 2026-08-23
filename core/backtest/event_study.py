"""Event-time study: average abnormal return paths around announcement events.

The calendar-rebalanced factor runner asks "each month, how do stocks ranked by
X perform?" — the wrong clock for post-earnings announcement drift, where the
effect starts the day a company announces and decays over ~60 trading days. A
monthly rebalance holds a mix of day-2 and day-55 surprises and dilutes exactly
what it is trying to measure.

This module re-indexes returns onto **event time**: day 0 is each stock's own
first tradable day after its announcement, and performance is averaged across
events at each event-day offset. That is the Ball & Brown / Bernard & Thomas
design, and it answers the PEAD question directly: *after* the announcement day
jump, does the drift continue, and is it monotonic in the surprise?

Conventions:

- **Day 0 is excluded from the drift.** The announcement-day jump is not
  capturable by a strategy that learns the surprise from the announcement; the
  measured path starts at day +1 (buy at day-0 close).
- **Abnormal return** = stock return minus the equal-weighted mean return of all
  stocks trading that day (a simple market adjustment; no beta fit).
- **Quantile breakpoints are computed within each calendar quarter** of events,
  never over the full sample — full-sample breakpoints would rank a 1995 event
  against the 2020 surprise distribution, which was not knowable in 1995.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from core.data.returns import DEFAULT_MAX_ABS_RETURN, compute_abnormal_returns
from core.exceptions import DataSchemaError

logger = logging.getLogger(__name__)

DEFAULT_HORIZON_DAYS = 60
DEFAULT_N_QUANTILES = 5
MIN_EVENTS_PER_QUARTER = 10

_REQUIRED_EVENT_COLUMNS = {"symbol", "event_date", "signal"}


def extract_events_from_surprise_panel(
    surprise_panel: pd.DataFrame,
    signal_col: str = "sue_price_scaled",
) -> pd.DataFrame:
    """
    Recover one row per announcement from the dailyized surprise panel.

    The panel forward-fills each surprise for the drift window with
    ``days_since_earnings`` counting up; an announcement therefore starts
    wherever that counter resets (or a symbol's first row).

    Args:
        surprise_panel: MultiIndex (date, symbol) panel carrying ``signal_col``
            and ``days_since_earnings``.
        signal_col: Surprise definition to carry as the event signal.

    Returns:
        DataFrame with ``symbol``, ``event_date`` (first tradable day, tz-aware),
        ``signal``; rows without a signal value are dropped.
    """
    if "days_since_earnings" not in surprise_panel.columns:
        raise DataSchemaError("surprise panel lacks days_since_earnings")

    working = surprise_panel[[signal_col, "days_since_earnings"]].reset_index()
    working = working.sort_values(["symbol", "date"])
    counter = working["days_since_earnings"]
    same_symbol = working["symbol"].eq(working["symbol"].shift())
    is_event_start = ~same_symbol | (counter < counter.shift())

    events = working[is_event_start & working[signal_col].notna()]
    return events.rename(columns={"date": "event_date", signal_col: "signal"})[
        ["symbol", "event_date", "signal"]
    ].reset_index(drop=True)


def assign_quarterly_quantiles(events: pd.DataFrame, n_quantiles: int) -> pd.Series:
    """
    Rank each event's signal against the other events of its calendar quarter.

    Args:
        events: Event table with ``event_date`` and ``signal``.
        n_quantiles: Number of buckets (5 = quintiles). Quantile 1 is the most
            negative signal, ``n_quantiles`` the most positive.

    Returns:
        Int Series aligned to ``events`` (NaN where the quarter has fewer than
        :data:`MIN_EVENTS_PER_QUARTER` events — a 3-event quarter cannot
        support quintiles).
    """
    quarter = events["event_date"].dt.tz_localize(None).dt.to_period("Q")

    def _bucket(group: pd.Series) -> pd.Series:
        if len(group) < MIN_EVENTS_PER_QUARTER:
            return pd.Series(np.nan, index=group.index)
        ranks = group.rank(pct=True, method="first")
        return np.ceil(ranks * n_quantiles).clip(1, n_quantiles)

    return events.groupby(quarter)["signal"].transform(_bucket)


def run_event_study(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    n_quantiles: int = DEFAULT_N_QUANTILES,
    max_abs_return: float | None = DEFAULT_MAX_ABS_RETURN,
    min_price: float | None = None,
) -> dict[str, object]:
    """
    Average abnormal-return paths in event time, by signal quantile.

    Args:
        events: One row per event: ``symbol``, ``event_date``, ``signal``.
        prices: Wide adjusted-close panel covering the events.
        horizon_days: Trading days of drift to measure after day 0.
        n_quantiles: Signal buckets (quarterly breakpoints).
        max_abs_return: Reject daily observations beyond this absolute return as
            vendor bad prints (see :mod:`core.data.returns`).
        min_price: Optional price floor for return eligibility; the conventional
            screen against bid-ask-bounce-dominated sub-dollar stocks.

    Returns:
        Dict with:
        - ``car_paths``: DataFrame indexed by event day (1..horizon), one column
          per quantile — average cumulative abnormal return, in decimal;
        - ``spread_path``: Series, top-quantile CAR minus bottom-quantile CAR;
        - ``event_counts``: events per quantile;
        - ``spread_t_stat``: Welch t on per-event horizon CARs, top vs bottom;
        - ``horizon_days``, ``n_quantiles``.

    Raises:
        DataSchemaError: On a malformed event table.
    """
    if missing := _REQUIRED_EVENT_COLUMNS - set(events.columns):
        raise DataSchemaError(f"events table missing columns: {missing}")
    if events.empty:
        raise DataSchemaError("no events supplied")

    # Bad-print rejection is not optional at full-universe scale: the panel
    # contains vendor defects up to +100,000,000% (un-adjusted reverse splits on
    # delisted micro-caps), and a single one destroys both the cross-sectional
    # benchmark and every cumulative path built from it.
    abnormal = compute_abnormal_returns(prices, max_abs_return=max_abs_return, min_price=min_price)

    working = events.copy()
    working["quantile"] = assign_quarterly_quantiles(working, n_quantiles)
    working = working.dropna(subset=["quantile"])
    working = working[working["symbol"].isin(prices.columns)]

    index_positions = prices.index.get_indexer(working["event_date"])
    working = working[index_positions >= 0]
    positions = index_positions[index_positions >= 0]

    # Event x event-day matrix of abnormal returns, day 1..horizon after day 0.
    abnormal_values = abnormal.to_numpy()
    n_days = len(prices.index)
    column_lookup = {s: i for i, s in enumerate(prices.columns)}

    paths = np.full((len(working), horizon_days), np.nan)
    for row, (position, symbol) in enumerate(zip(positions, working["symbol"], strict=True)):
        start = position + 1
        stop = min(start + horizon_days, n_days)
        if start >= n_days:
            continue
        paths[row, : stop - start] = abnormal_values[start:stop, column_lookup[symbol]]

    car_by_event = np.nancumsum(paths, axis=1)
    quantiles = working["quantile"].to_numpy()

    car_paths: dict[str, np.ndarray] = {}
    event_counts: dict[str, int] = {}
    for q in range(1, n_quantiles + 1):
        mask = quantiles == q
        label = f"Q{q}"
        event_counts[label] = int(mask.sum())
        with np.errstate(invalid="ignore"):
            car_paths[label] = np.nanmean(car_by_event[mask], axis=0)

    car_frame = pd.DataFrame(car_paths, index=pd.RangeIndex(1, horizon_days + 1, name="event_day"))
    top, bottom = f"Q{n_quantiles}", "Q1"
    spread_path = car_frame[top] - car_frame[bottom]

    top_cars = car_by_event[quantiles == n_quantiles, -1]
    bottom_cars = car_by_event[quantiles == 1, -1]
    top_cars = top_cars[np.isfinite(top_cars)]
    bottom_cars = bottom_cars[np.isfinite(bottom_cars)]
    if len(top_cars) > 2 and len(bottom_cars) > 2:
        pooled_se = np.sqrt(
            top_cars.var(ddof=1) / len(top_cars) + bottom_cars.var(ddof=1) / len(bottom_cars)
        )
        spread_t = float((top_cars.mean() - bottom_cars.mean()) / pooled_se)
    else:
        spread_t = float("nan")

    return {
        "car_paths": car_frame,
        "spread_path": spread_path,
        "event_counts": event_counts,
        "spread_t_stat": spread_t,
        "horizon_days": horizon_days,
        "n_quantiles": n_quantiles,
    }
