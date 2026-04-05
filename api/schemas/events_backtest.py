"""
Pydantic models for event-driven backtest simulation over HTTP.

Maps JSON payloads to :class:`core.backtest.events.types.Event` and accepts a
wide daily price panel as row-oriented records.
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator

from core.backtest.events.types import EventType


class EventIn(BaseModel):
    """
    Serializable event for ``POST /backtest/events/simulate``.

    Attributes:
        ts: ISO 8601 datetime (timezone required or UTC assumed — see route).
        event_type: One of :class:`core.backtest.events.types.EventType` values
            (e.g. ``rebalance``).
        symbol: Optional instrument key for per-symbol events.
        payload: JSON object; for ``rebalance`` must include ``symbols`` (list of str).
    """

    ts: str
    event_type: str
    symbol: Optional[str] = None
    payload: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def rebalance_requires_symbols(self) -> EventIn:
        if self.event_type == EventType.REBALANCE.value:
            syms = self.payload.get("symbols")
            if not isinstance(syms, list) or not syms:
                raise ValueError(
                    "REBALANCE events require payload['symbols'] as a non-empty list of strings"
                )
            if not all(isinstance(s, str) for s in syms):
                raise ValueError("payload['symbols'] must be a list of strings")
        return self


class EventSimulateRequest(BaseModel):
    """
    Request body for equal-weight event simulation.

    Attributes:
        events: Ordered list of events (sorted server-side; must be strictly increasing ts).
        price_rows: Each row is a mapping with a ``date`` key (ISO string) and one float column
            per symbol (wide panel, one row per calendar date).
        transaction_cost: Optional per-rebalance cost fraction passed to the simulator.
    """

    events: list[EventIn]
    price_rows: list[dict[str, Any]]
    transaction_cost: float = 0.0

    @model_validator(mode="after")
    def non_empty(self) -> EventSimulateRequest:
        if not self.events:
            raise ValueError("events must be non-empty")
        if not self.price_rows:
            raise ValueError("price_rows must be non-empty")
        return self


class EventSimulateResponse(BaseModel):
    """Daily portfolio returns aligned to the price index after simulation."""

    dates: list[str]
    returns: list[float]
