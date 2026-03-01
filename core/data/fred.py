"""
FRED (Federal Reserve Economic Data) integration.

Provides a typed indicator catalog, series fetching with caching,
and NBER recession period extraction.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

INDICATOR_CATALOG: Dict[str, List[Dict[str, str]]] = {
    "Interest Rates": [
        {"id": "DFF", "name": "Federal Funds Rate", "unit": "%", "frequency": "daily"},
        {"id": "DGS10", "name": "10-Year Treasury Yield", "unit": "%", "frequency": "daily"},
        {"id": "DGS2", "name": "2-Year Treasury Yield", "unit": "%", "frequency": "daily"},
        {"id": "T10Y2Y", "name": "10Y-2Y Spread", "unit": "%", "frequency": "daily"},
    ],
    "Inflation": [
        {"id": "CPIAUCSL", "name": "CPI (All Urban)", "unit": "index", "frequency": "monthly"},
        {"id": "PCEPI", "name": "PCE Price Index", "unit": "index", "frequency": "monthly"},
    ],
    "GDP & Growth": [
        {"id": "GDP", "name": "Nominal GDP", "unit": "B USD", "frequency": "quarterly"},
        {"id": "GDPC1", "name": "Real GDP", "unit": "B 2017 USD", "frequency": "quarterly"},
    ],
    "Employment": [
        {"id": "UNRATE", "name": "Unemployment Rate", "unit": "%", "frequency": "monthly"},
        {"id": "PAYEMS", "name": "Non-Farm Payrolls", "unit": "thousands", "frequency": "monthly"},
    ],
}


def _get_fred():
    """Lazy-load the Fred client."""
    from config.settings import FRED_API_KEY
    from fredapi import Fred

    if not FRED_API_KEY:
        raise RuntimeError("FRED_API_KEY not configured")
    return Fred(api_key=FRED_API_KEY)


def get_fred_series(
    series_ids: List[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    yoy: bool = False,
) -> pd.DataFrame:
    """Fetch multiple FRED series and return a DataFrame.

    Parameters
    ----------
    series_ids : list of FRED series identifiers (e.g. ["DFF", "DGS10"])
    start, end : date strings for filtering
    yoy        : if True, compute YoY percentage change for non-rate series

    Returns
    -------
    DataFrame with DatetimeIndex and one column per series id.
    """
    fred = _get_fred()
    frames = []
    for sid in series_ids:
        try:
            s = fred.get_series(sid)
            s.index = pd.to_datetime(s.index)
            s = s.rename(sid)
            if yoy:
                s = s.pct_change(12, fill_method=None) * 100
            frames.append(s)
        except Exception:
            logger.warning("Failed to fetch FRED series %s", sid)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, axis=1).sort_index().ffill()
    if start:
        df = df[df.index >= start]
    if end:
        df = df[df.index <= end]
    return df


def get_recession_periods(
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Return NBER recession date ranges from USREC.

    Returns
    -------
    List of {"start": "YYYY-MM-DD", "end": "YYYY-MM-DD"} dicts.
    """
    fred = _get_fred()
    rec = fred.get_series("USREC")
    rec.index = pd.to_datetime(rec.index)
    if start:
        rec = rec[rec.index >= start]
    if end:
        rec = rec[rec.index <= end]

    periods: List[Dict[str, str]] = []
    in_recession = False
    rec_start = None

    for date, val in rec.items():
        if val == 1 and not in_recession:
            in_recession = True
            rec_start = date
        elif val == 0 and in_recession:
            in_recession = False
            if rec_start is not None:
                periods.append({
                    "start": str(rec_start.date()),
                    "end": str(date.date()),
                })

    if in_recession and rec_start is not None:
        periods.append({
            "start": str(rec_start.date()),
            "end": str(rec.index[-1].date()),
        })

    return periods
