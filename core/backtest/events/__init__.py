"""
Discrete-event backtesting (v0): validated event logs and a minimal rebalance simulator.

Public API is intentionally small; see :mod:`core.backtest.events.simulator`.
"""

from core.backtest.events.log import EventLog, validate_and_sort_events
from core.backtest.events.simulator import simulate_equal_weight_rebalances
from core.backtest.events.types import Event, EventType

__all__ = [
    "Event",
    "EventLog",
    "EventType",
    "simulate_equal_weight_rebalances",
    "validate_and_sort_events",
]
