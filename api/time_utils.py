"""Helpers for comparing API date strings against DataFrame indexes."""

from __future__ import annotations

from typing import Optional

import pandas as pd


def bound_timestamp(value: str, index: pd.DatetimeIndex) -> pd.Timestamp:
    """
    Convert a date string to a Timestamp aligned with ``index`` timezone.

    API query/body dates arrive as ``YYYY-MM-DD`` strings. Comparing them
    directly to a tz-aware ``DatetimeIndex`` is fragile (element-wise
    comparisons raise; naive ``pd.Timestamp`` raises ``TypeError``). Always
    localize/convert to the index tz before ``>=`` / ``<=``.
    """
    ts = pd.Timestamp(value)
    if index.tz is None:
        return ts.tz_localize(None) if ts.tzinfo is not None else ts
    if ts.tzinfo is None:
        return ts.tz_localize(index.tz)
    return ts.tz_convert(index.tz)


def slice_by_dates(
    frame: pd.DataFrame | pd.Series,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> pd.DataFrame | pd.Series:
    """Inclusive date filter for a Series/DataFrame with a DatetimeIndex."""
    out = frame
    if start:
        out = out[out.index >= bound_timestamp(start, out.index)]
    if end:
        out = out[out.index <= bound_timestamp(end, out.index)]
    return out
