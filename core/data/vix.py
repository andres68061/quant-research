"""VIX index data fetcher with local parquet cache."""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CACHE = _PROJECT_ROOT / "data" / "factors" / "vix.parquet"


def load_vix(
    start: str = "2000-01-01",
    cache_path: Optional[Path] = None,
    force_refresh: bool = False,
) -> pd.Series:
    """Load daily VIX close, fetching from Yahoo Finance if not cached.

    Args:
        start: Earliest date to fetch (ISO format).
        cache_path: Where to store/read parquet cache.
            Defaults to ``data/factors/vix.parquet``.
        force_refresh: Re-download even if cache exists.

    Returns:
        Series named ``"VIX"`` with tz-naive DatetimeIndex (business-day freq).

    Example:
        >>> vix = load_vix()
        >>> vix.head()
    """
    cache_path = Path(cache_path) if cache_path else _DEFAULT_CACHE

    if cache_path.exists() and not force_refresh:
        logger.info("Loading VIX from cache: %s", cache_path)
        series = pd.read_parquet(cache_path).squeeze()
        series.index = pd.to_datetime(series.index).tz_localize(None)
        series.name = "VIX"
        return series

    logger.info("Fetching VIX from Yahoo Finance (start=%s)", start)
    raw = yf.download("^VIX", start=start, auto_adjust=True, progress=False)

    if raw.empty:
        logger.warning("No VIX data returned from Yahoo Finance")
        return pd.Series(dtype=float, name="VIX")

    close = raw["Close"].squeeze()
    close.index = pd.to_datetime(close.index).tz_localize(None)
    close.name = "VIX"

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    close.to_frame().to_parquet(cache_path)
    logger.info("Cached VIX data to %s (%d rows)", cache_path, len(close))

    return close
