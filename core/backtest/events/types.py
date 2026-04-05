"""Event types and records for discrete-event simulation (v0)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Optional

import pandas as pd


class EventType(str, Enum):
    """Kinds of events in the event-driven backtest log."""

    BAR = "bar"
    SIGNAL = "signal"
    REBALANCE = "rebalance"
    FILL = "fill"
    CUSTOM = "custom"


@dataclass(frozen=True)
class Event:
    """
    A single event in the simulation.

    Args:
        ts: Event time; must be timezone-aware (validated by EventLog).
        event_type: Category of event.
        symbol: Optional instrument; required for bar/fill per symbol.
        payload: Arbitrary JSON-like dict (e.g. ``{"symbols": ["AAPL", "MSFT"]}`` for rebalance).

    Example:
        >>> import pandas as pd
        >>> Event(
        ...     ts=pd.Timestamp("2024-01-02", tz="UTC"),
        ...     event_type=EventType.REBALANCE,
        ...     symbol=None,
        ...     payload={"symbols": ["AAPL", "MSFT"]},
        ... )
    """

    ts: pd.Timestamp
    event_type: EventType
    symbol: Optional[str]
    payload: Mapping[str, Any]
