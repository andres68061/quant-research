"""Turn the post-earnings drift into a portfolio you could actually hold.

The event study (``core.backtest.event_study``) answers a research question:
*after a company reports a surprise, does its price keep drifting?* It measures
that by lining every announcement up at day 0 and averaging. The answer on our
data is yes — but an average abnormal path is not a strategy. Nobody can hold
"the average event"; announcements arrive on different days, and a real book holds
many overlapping positions at once.

This module builds the tradable form: an **overlapping portfolio**. Each
announcement opens a position the next day and holds it for a fixed window
(default 60 trading days). Because roughly 1/60th of the book turns over daily,
the portfolio is always holding ~60 overlapping cohorts. That is the standard
Jegadeesh-Titman construction, and it converts an event-time result into a
calendar-time return series that can be compared with every other strategy in the
repo.

Two things the event study could not tell us, and this can:

1. **Costs.** The event study is gross. Here every entry and exit is charged, and
   the charge can scale with each name's liquidity (dollar-ADV), because
   small-cap drift is exactly where a flat basis-point assumption lies most.
2. **Capacity shape.** Position sizing is explicit, so the return series reflects
   holding many small positions rather than an idealised average.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
import pandas as pd

from core.data.factors.liquidity import cost_bps_from_dollar_adv
from core.data.factors.returns import DEFAULT_MAX_ABS_RETURN, compute_clean_returns
from core.exceptions import DataSchemaError

logger = logging.getLogger(__name__)

DEFAULT_HOLD_DAYS = 60
DEFAULT_QUANTILES = 5
DEFAULT_COST_BPS = 10.0

_REQUIRED_EVENT_COLUMNS = {"symbol", "event_date", "signal"}


def assign_event_quantiles(
    events: pd.DataFrame,
    n_quantiles: int = DEFAULT_QUANTILES,
    min_events_per_quarter: int = 10,
) -> pd.Series:
    """
    Rank each announcement's surprise against others from the same quarter.

    Breakpoints must be point-in-time: ranking a 1998 surprise against the
    2020 distribution uses information that did not exist, and the surprise
    distribution genuinely shifts over decades.

    Args:
        events: ``symbol``, ``event_date``, ``signal`` rows.
        n_quantiles: Buckets; 5 = quintiles, 1 = most negative surprise.
        min_events_per_quarter: Quarters thinner than this yield NaN rather than
            quintiles computed from a handful of events.

    Returns:
        Float Series aligned to ``events``; NaN where the quarter was too thin.
    """
    quarter = events["event_date"].dt.tz_localize(None).dt.to_period("Q")

    def bucket(group: pd.Series) -> pd.Series:
        if len(group) < min_events_per_quarter:
            return pd.Series(np.nan, index=group.index)
        ranks = group.rank(pct=True, method="first")
        return np.ceil(ranks * n_quantiles).clip(1, n_quantiles)

    return events.groupby(quarter)["signal"].transform(bucket)


def build_target_weights(
    events: pd.DataFrame,
    trading_index: pd.DatetimeIndex,
    symbols: pd.Index,
    hold_days: int = DEFAULT_HOLD_DAYS,
    n_quantiles: int = DEFAULT_QUANTILES,
    long_only: bool = False,
) -> pd.DataFrame:
    """
    Build the daily target weight matrix for the overlapping PEAD book.

    Each qualifying announcement contributes an equal-sized position from the day
    after the announcement for ``hold_days`` trading days: long for the top
    surprise quantile, short for the bottom. Overlapping cohorts accumulate, and
    each day's weights are normalized so gross exposure is 1.0 (or 1.0 long when
    ``long_only``), which keeps leverage constant as event counts vary.

    Args:
        events: ``symbol``, ``event_date``, ``signal`` rows.
        trading_index: Trading calendar to build weights on.
        symbols: Column universe for the weight matrix.
        hold_days: Trading days each position is held.
        n_quantiles: Surprise buckets.
        long_only: Drop the short leg.

    Returns:
        Date x symbol weight matrix; rows sum to 0 (long/short, gross 1) or 1
        (long-only).

    Raises:
        DataSchemaError: If the event table is malformed.
    """
    if missing := _REQUIRED_EVENT_COLUMNS - set(events.columns):
        raise DataSchemaError(f"events missing columns: {missing}")

    working = events.copy()
    working["quantile"] = assign_event_quantiles(working, n_quantiles)
    working = working.dropna(subset=["quantile"])
    working = working[working["symbol"].isin(symbols)]

    top, bottom = float(n_quantiles), 1.0
    traded = working[working["quantile"].isin([top, bottom])]
    if traded.empty:
        return pd.DataFrame(0.0, index=trading_index, columns=symbols)

    positions = np.zeros((len(trading_index), len(symbols)), dtype="float64")
    symbol_position = {symbol: i for i, symbol in enumerate(symbols)}
    date_position = pd.Series(np.arange(len(trading_index)), index=trading_index)

    entry_rows = date_position.reindex(traded["event_date"]).to_numpy()
    for entry, symbol, quantile in zip(
        entry_rows, traded["symbol"], traded["quantile"], strict=True
    ):
        if np.isnan(entry):
            continue
        # Enter the day AFTER the announcement: the surprise is only knowable
        # once it is announced, and same-close execution is not available.
        start = int(entry) + 1
        stop = min(start + hold_days, len(trading_index))
        if start >= len(trading_index):
            continue
        side = 1.0 if quantile == top else -1.0
        if long_only and side < 0:
            continue
        positions[start:stop, symbol_position[symbol]] += side

    weights = pd.DataFrame(positions, index=trading_index, columns=symbols)
    gross = weights.abs().sum(axis=1)
    normalized = weights.div(gross.replace(0.0, np.nan), axis=0).fillna(0.0)

    active_days = int((gross > 0).sum())
    logger.info(
        "PEAD book: %s events traded, active on %s of %s days, median %.0f positions",
        f"{len(traded):,}",
        f"{active_days:,}",
        f"{len(trading_index):,}",
        float((weights != 0).sum(axis=1).replace(0, np.nan).median() or 0),
    )
    return normalized


def simulate_pead_portfolio(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    hold_days: int = DEFAULT_HOLD_DAYS,
    n_quantiles: int = DEFAULT_QUANTILES,
    long_only: bool = False,
    cost_bps: float = DEFAULT_COST_BPS,
    dollar_adv: Optional[pd.DataFrame] = None,
    min_price: Optional[float] = 1.0,
    max_abs_return: Optional[float] = DEFAULT_MAX_ABS_RETURN,
    universe_filter: Optional[Callable[[pd.Timestamp], set[str]]] = None,
) -> dict[str, object]:
    """
    Simulate the overlapping PEAD book and return calendar-time net returns.

    Args:
        events: ``symbol``, ``event_date``, ``signal`` rows.
        prices: Wide adjusted-close panel.
        hold_days: Trading days each position is held.
        n_quantiles: Surprise buckets; top is bought, bottom is sold.
        long_only: Drop the short leg.
        cost_bps: Flat one-way cost in basis points, used when ``dollar_adv`` is
            None. A flat charge flatters illiquid names, which is precisely where
            this strategy trades — prefer the ADV schedule.
        dollar_adv: Optional date x symbol trailing dollar-volume panel. When
            given, each name's cost comes from
            :func:`core.data.factors.liquidity.cost_bps_from_dollar_adv` instead, so
            micro-caps are charged what they actually cost.
        min_price: Price floor for return eligibility.
        max_abs_return: Bad-print rejection bound.
        universe_filter: Optional ``date -> eligible symbols``.

    Returns:
        Dict with ``gross_return``, ``net_return``, ``turnover`` (daily, one-way
        fraction of gross), ``cost_drag`` (annualized), ``positions`` (daily
        count) and ``weights_sum`` diagnostics.
    """
    returns = compute_clean_returns(prices, max_abs_return=max_abs_return, min_price=min_price)
    symbols = prices.columns
    weights = build_target_weights(events, prices.index, symbols, hold_days, n_quantiles, long_only)

    if universe_filter is not None:
        eligible_mask = pd.DataFrame(False, index=weights.index, columns=symbols)
        for date, positions in weights.groupby(level=0).indices.items():
            allowed = universe_filter(date)
            if allowed:
                eligible_mask.iloc[positions] = symbols.isin(allowed)
        weights = weights.where(eligible_mask, 0.0)
        gross = weights.abs().sum(axis=1)
        weights = weights.div(gross.replace(0.0, np.nan), axis=0).fillna(0.0)

    # Yesterday's weights earn today's return: a position opened on day t
    # captures the move from t's close to t+1's close.
    held = weights.shift(1).fillna(0.0)
    gross_return = (held * returns.reindex_like(held).fillna(0.0)).sum(axis=1)

    traded = (weights - held).abs()
    turnover = traded.sum(axis=1)

    if dollar_adv is not None:
        adv = dollar_adv.reindex(index=weights.index, columns=symbols).ffill(limit=5)
        # Vectorized bucket lookup: apply the schedule's thresholds directly
        # rather than calling the scalar helper 27M times.
        cost_rate = pd.DataFrame(
            cost_bps_from_dollar_adv(float("nan")), index=weights.index, columns=symbols
        )
        for threshold, rate in sorted(
            [(t, r) for t, r in _adv_schedule()], key=lambda pair: pair[0]
        ):
            cost_rate = cost_rate.mask(adv >= threshold, rate)
        costs = (traded * cost_rate).sum(axis=1)
    else:
        costs = turnover * (cost_bps / 10_000.0)

    net_return = gross_return - costs
    annualization = 252
    return {
        "gross_return": gross_return,
        "net_return": net_return,
        "turnover": turnover,
        "cost_drag": float(costs.mean() * annualization),
        "positions": (weights != 0).sum(axis=1),
        "cost_model": "dollar_adv_schedule" if dollar_adv is not None else f"flat_{cost_bps}bps",
    }


def _adv_schedule() -> tuple[tuple[float, float], ...]:
    """The dollar-ADV cost buckets, imported lazily to keep the dependency light."""
    from core.data.factors.liquidity import ADV_COST_SCHEDULE

    return ADV_COST_SCHEDULE
