"""
Minimal event-driven simulator (v0): equal-weight holdings between rebalance events.

Uses only ``EventType.REBALANCE`` events with payload ``{"symbols": [str, ...]}``.
Daily portfolio return is the dot product of weights and per-asset returns on ``prices``.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd

from core.exceptions import DataSchemaError

from .log import EventLog
from .types import Event, EventType


def simulate_equal_weight_rebalances(
    prices: pd.DataFrame,
    events: Sequence[Event],
    *,
    transaction_cost: float = 0.0,
) -> pd.Series:
    """
    Walk rebalance events and compute daily portfolio returns (equal weight within payload).

    Between two rebalance dates (inclusive of start, exclusive of next rebalance's day for
    return attribution), weights are held constant. The first bar used for returns is the
    first trading day **after** each rebalance timestamp that exists in ``prices.index``.

    Args:
        prices: Wide panel of levels, DatetimeIndex (tz-aware recommended), columns = symbols.
        events: Rebalance events; others ignored. Each rebalance payload must contain
            ``symbols``: list of column names present in ``prices``.
        transaction_cost: Optional per-rebalance turnover cost as a fraction (0 = none).

    Returns:
        Daily net return series aligned to ``prices.index`` (zeros before first effective rebalance).

    Raises:
        DataSchemaError: From :class:`EventLog` if timestamps invalid; if payload missing ``symbols``.

    Example:
        >>> # See tests for a full example.
    """
    if prices.empty:
        raise DataSchemaError("prices must be non-empty")
    if not isinstance(prices.index, pd.DatetimeIndex):
        raise DataSchemaError("prices.index must be a DatetimeIndex")

    log = EventLog([e for e in events if e.event_type == EventType.REBALANCE])
    if not log.events:
        raise DataSchemaError("Need at least one REBALANCE event")

    returns = prices.pct_change()
    out = pd.Series(0.0, index=prices.index, dtype=np.float64)

    rebalance_events: List[Event] = list(log.events)
    price_index = prices.index

    for i, ev in enumerate(rebalance_events):
        payload = dict(ev.payload) if ev.payload is not None else {}
        symbols = payload.get("symbols")
        if not isinstance(symbols, list) or not symbols:
            raise DataSchemaError(
                f"REBALANCE payload must include non-empty 'symbols' list; got {payload!r}"
            )
        missing = [s for s in symbols if s not in prices.columns]
        if missing:
            raise DataSchemaError(f"Unknown symbols in rebalance payload: {missing}")

        start_idx = price_index.searchsorted(ev.ts, side="right")
        if start_idx >= len(price_index):
            continue
        start_date = price_index[start_idx]

        if i + 1 < len(rebalance_events):
            next_ts = rebalance_events[i + 1].ts
            end_idx = price_index.searchsorted(next_ts, side="left")
            hold_dates = price_index[start_idx:end_idx]
        else:
            hold_dates = price_index[start_idx:]

        if len(hold_dates) == 0:
            continue

        n = len(symbols)
        w = pd.Series(1.0 / n, index=symbols, dtype=np.float64)

        sub = returns.loc[hold_dates, symbols]
        port_ret = (sub * w).sum(axis=1)

        if transaction_cost > 0.0 and len(hold_dates) > 0:
            # Simplified: charge once at first day of segment (turnover from cash to invested)
            port_ret = port_ret.copy()
            port_ret.iloc[0] -= transaction_cost

        out.loc[hold_dates] = port_ret.values

    return out
