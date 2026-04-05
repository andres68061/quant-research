"""
Download and persist Fama-French 5 factor daily returns from the Kenneth French data library.

Data source: `mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html`
accessed via ``pandas_datareader``. No API key required.

Columns (decimal returns, not percent):
    mkt_rf, smb, hml, rmw, cma, rf
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import pandas_datareader.data as web  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

_DATASET = "F-F_Research_Data_5_Factors_2x3_daily"

_COLUMN_MAP = {
    "Mkt-RF": "mkt_rf",
    "SMB": "smb",
    "HML": "hml",
    "RMW": "rmw",
    "CMA": "cma",
    "RF": "rf",
}

_MAX_RETRIES = 3
_BACKOFF_BASE = 2.0

# Kenneth French daily 5-factor file begins 1963-07-01. ``pandas_datareader`` uses
# a ~5-year default when ``start`` is omitted, so we must pass an explicit date.
_DEFAULT_FULL_HISTORY_START = "1963-07-01"


def fetch_ff5_daily(start: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch Fama-French 5 factor daily returns from the Kenneth French data library.

    Parameters
    ----------
    start : str or None
        Earliest date to keep (ISO format). ``None`` uses 1963-07-01 (full
        Kenneth French daily history); required because ``pandas_datareader``
        defaults to only ~5 years when ``start`` is omitted.

    Returns
    -------
    pd.DataFrame
        ``DatetimeIndex`` named ``date``; columns
        ``['mkt_rf', 'smb', 'hml', 'rmw', 'cma', 'rf']`` in **decimal** returns.

    Raises
    ------
    RuntimeError
        If all retry attempts fail.

    Examples
    --------
    >>> df = fetch_ff5_daily(start="2020-01-01")  # doctest: +SKIP
    >>> sorted(df.columns.tolist())  # doctest: +SKIP
    ['cma', 'hml', 'mkt_rf', 'rf', 'rmw', 'smb']
    """
    effective_start = start if start is not None else _DEFAULT_FULL_HISTORY_START
    last_exc: Optional[Exception] = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            tables = web.DataReader(_DATASET, "famafrench", start=effective_start)
            break
        except Exception as exc:
            last_exc = exc
            wait = _BACKOFF_BASE**attempt
            logger.warning(
                "FF5 fetch attempt %d/%d failed: %s — retrying in %.1fs",
                attempt,
                _MAX_RETRIES,
                exc,
                wait,
            )
            time.sleep(wait)
    else:
        raise RuntimeError(
            f"Failed to fetch {_DATASET} after {_MAX_RETRIES} attempts"
        ) from last_exc

    df: pd.DataFrame = tables[0]  # daily table is key 0

    df = df.rename(columns=_COLUMN_MAP)
    expected = set(_COLUMN_MAP.values())
    missing = expected - set(df.columns)
    if missing:
        raise RuntimeError(f"Kenneth French dataset missing expected columns: {missing}")
    df = df[list(_COLUMN_MAP.values())]

    df = df / 100.0

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    df = df.sort_index()
    return df


def load_ff5_parquet(path: Path) -> Optional[pd.DataFrame]:
    """
    Load cached Fama-French 5 factors from Parquet.

    Returns ``None`` if the file does not exist.
    """
    if not path.exists():
        logger.warning("FF5 Parquet not found at %s", path)
        return None
    df = pd.read_parquet(path)
    return df


def update_ff5_parquet(path: Path, start: Optional[str] = None) -> pd.DataFrame:
    """
    Fetch FF5 daily returns, merge with existing Parquet if present, and write back.

    Deduplicates on the date index (keeps latest fetch for overlapping dates).

    Parameters
    ----------
    path : Path
        Output Parquet path.
    start : str or None
        Passed to :func:`fetch_ff5_daily`.

    Returns
    -------
    pd.DataFrame
        The merged and deduplicated frame (also written to *path*).
    """
    fresh = fetch_ff5_daily(start=start)
    existing = load_ff5_parquet(path)

    if existing is not None and not existing.empty:
        combined = pd.concat([existing, fresh])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined = combined.sort_index()
    else:
        combined = fresh

    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(path)
    logger.info("Wrote FF5 factors: %s rows to %s", len(combined), path)
    return combined
