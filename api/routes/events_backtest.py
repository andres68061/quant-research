"""
Event-driven backtest simulation API (v0).

Exposes :func:`core.backtest.events.simulator.simulate_equal_weight_rebalances` over HTTP.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException

from api.schemas.events_backtest import EventIn, EventSimulateRequest, EventSimulateResponse
from core.backtest.events import Event, EventType, simulate_equal_weight_rebalances
from core.exceptions import DataSchemaError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/backtest/events", tags=["backtest-events"])


def _parse_ts(raw: str) -> pd.Timestamp:
    """Parse ISO timestamp; naive values are interpreted as UTC."""
    ts = pd.Timestamp(raw)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _dto_to_event(dto: EventIn) -> Event:
    """Map Pydantic DTO to core Event."""
    try:
        et = EventType(dto.event_type)
    except ValueError as exc:
        raise DataSchemaError(f"Invalid event_type: {dto.event_type!r}") from exc
    return Event(
        ts=_parse_ts(dto.ts),
        event_type=et,
        symbol=dto.symbol,
        payload=dto.payload,
    )


def _prices_from_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a wide price DataFrame: index = tz-aware dates, columns = symbols."""
    df = pd.DataFrame(rows)
    if "date" not in df.columns:
        raise DataSchemaError("Each price row must include a 'date' field")
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], utc=True)
    if df["date"].duplicated().any():
        raise DataSchemaError("Duplicate dates in price_rows")
    df = df.set_index("date").sort_index()
    if df.shape[1] == 0:
        raise DataSchemaError("price_rows need at least one price column in addition to 'date'")
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df.isna().any().any():
        raise DataSchemaError("price_rows contain invalid or missing numeric values")
    return df


@router.post("/simulate", response_model=EventSimulateResponse)
def simulate_events(body: EventSimulateRequest) -> EventSimulateResponse:
    """
    Run equal-weight rebalance simulation on a supplied price panel and event list.

    Inputs must satisfy :class:`~core.backtest.events.log.EventLog` rules (strictly
    increasing tz-aware timestamps). Only ``rebalance`` events with ``symbols`` in
    the payload are used by the v0 simulator.
    """
    try:
        prices = _prices_from_rows(body.price_rows)
        events = [_dto_to_event(e) for e in body.events]
        out = simulate_equal_weight_rebalances(
            prices,
            events,
            transaction_cost=body.transaction_cost,
        )
    except DataSchemaError as exc:
        logger.info("event_simulate_schema_error", extra={"detail": str(exc)})
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    idx = out.index
    dates = [t.isoformat() for t in idx]
    returns = [float(x) for x in out.values]
    return EventSimulateResponse(dates=dates, returns=returns)
