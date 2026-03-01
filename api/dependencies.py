"""
Shared FastAPI dependencies: data loaders and singletons.

Loaded once at startup via the ``lifespan`` context manager in ``main.py``.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from config.settings import PROJECT_ROOT

logger = logging.getLogger(__name__)

_factors: Optional[pd.DataFrame] = None
_prices: Optional[pd.DataFrame] = None
_sectors: Optional[pd.DataFrame] = None


def load_data() -> None:
    """Load core datasets into module-level caches."""
    global _factors, _prices, _sectors

    data_dir = PROJECT_ROOT / "data"
    factors_path = data_dir / "factors" / "factors_price.parquet"
    prices_path = data_dir / "factors" / "prices.parquet"
    sectors_path = data_dir / "sectors" / "sector_classifications.parquet"

    if factors_path.exists():
        _factors = pd.read_parquet(factors_path)
        logger.info("Loaded factors: %s", _factors.shape)
    else:
        logger.warning("Factors file not found: %s", factors_path)

    if prices_path.exists():
        _prices = pd.read_parquet(prices_path)
        logger.info("Loaded prices: %s", _prices.shape)
    else:
        logger.warning("Prices file not found: %s", prices_path)

    if sectors_path.exists():
        _sectors = pd.read_parquet(sectors_path)
        logger.info("Loaded sectors: %s", _sectors.shape)


def get_factors() -> Optional[pd.DataFrame]:
    return _factors


def get_prices() -> Optional[pd.DataFrame]:
    return _prices


def get_sectors() -> Optional[pd.DataFrame]:
    return _sectors
