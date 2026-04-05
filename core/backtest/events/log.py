"""Event log validation: monotonic tz-aware timestamps."""

from __future__ import annotations

from typing import List, Sequence

import pandas as pd

from core.exceptions import DataSchemaError

from .types import Event


class EventLog:
    """
    Ordered, append-only view of events with validation.

    Timestamps must be **strictly increasing** (no duplicates) and **timezone-aware**.
    """

    __slots__ = ("_events",)

    def __init__(self, events: Sequence[Event]) -> None:
        self._events: tuple[Event, ...] = tuple(validate_and_sort_events(events))

    @property
    def events(self) -> tuple[Event, ...]:
        """Immutable sequence of events in time order."""
        return self._events

    def __len__(self) -> int:
        return len(self._events)


def validate_and_sort_events(events: Sequence[Event]) -> List[Event]:
    """
    Sort events by timestamp and enforce strictly increasing, tz-aware ``ts``.

    Args:
        events: Input events (may be unsorted).

    Returns:
        New list sorted by ``ts``.

    Raises:
        DataSchemaError: If any ``ts`` is naive, or timestamps are not strictly increasing.
    """
    if not events:
        return []

    sorted_ev = sorted(events, key=lambda e: e.ts)
    prev: pd.Timestamp | None = None
    for ev in sorted_ev:
        ts = ev.ts
        if ts.tzinfo is None:
            raise DataSchemaError(
                "Event.ts must be timezone-aware; got naive timestamp "
                f"for event_type={ev.event_type!r}"
            )
        if prev is not None and ts <= prev:
            raise DataSchemaError(
                f"Events must have strictly increasing ts; saw {ts} after {prev}"
            )
        prev = ts
    return sorted_ev
