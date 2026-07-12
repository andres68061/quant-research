"""
Shared FastAPI dependencies: data loaders and singletons.

Loaded once at startup via the ``lifespan`` context manager in ``main.py``.
"""

import logging
from typing import Optional

import pandas as pd

from config.settings import PROJECT_ROOT
from core.data.quality import QUARANTINE_PATH, load_quarantined_symbols

logger = logging.getLogger(__name__)

_factors: Optional[pd.DataFrame] = None
_prices: Optional[pd.DataFrame] = None
_sectors: Optional[pd.DataFrame] = None
_dollar_adv: Optional[pd.DataFrame] = None
_quarantined: set[str] = set()


def load_data() -> None:
    """Load core datasets into module-level caches, excluding quarantined symbols."""
    global _factors, _prices, _sectors, _dollar_adv, _quarantined

    data_dir = PROJECT_ROOT / "data"
    factors_all_path = data_dir / "factors" / "factors_all.parquet"
    factors_price_path = data_dir / "factors" / "factors_price.parquet"
    factors_path = factors_all_path if factors_all_path.exists() else factors_price_path
    prices_path = data_dir / "factors" / "prices.parquet"
    dollar_adv_path = data_dir / "factors" / "dollar_adv_21d.parquet"
    sectors_path = data_dir / "sectors" / "sector_classifications.parquet"

    _quarantined = load_quarantined_symbols(PROJECT_ROOT / QUARANTINE_PATH)
    if _quarantined:
        logger.info("Quarantine list: excluding %d symbols from loaded data", len(_quarantined))

    if factors_path.exists():
        _factors = pd.read_parquet(factors_path)
        fundamentals_path = data_dir / "factors" / "factors_fundamental.parquet"
        if fundamentals_path.exists():
            fundamental_factors = pd.read_parquet(fundamentals_path)
            overlap = [c for c in fundamental_factors.columns if c in _factors.columns]
            _factors = _factors.drop(columns=overlap, errors="ignore").join(
                fundamental_factors, how="left"
            )
        if _quarantined:
            symbol_level = _factors.index.get_level_values("symbol")
            _factors = _factors[~symbol_level.isin(_quarantined)]
        logger.info("Loaded factors: %s columns=%s", _factors.shape, list(_factors.columns))
    else:
        logger.warning("Factors file not found: %s", factors_path)

    if prices_path.exists():
        _prices = pd.read_parquet(prices_path)
        if _quarantined:
            _prices = _prices.drop(columns=[s for s in _quarantined if s in _prices.columns])
        logger.info("Loaded prices: %s", _prices.shape)
    else:
        logger.warning("Prices file not found: %s", prices_path)

    if sectors_path.exists():
        _sectors = pd.read_parquet(sectors_path)
        logger.info("Loaded sectors: %s", _sectors.shape)

    if dollar_adv_path.exists():
        _dollar_adv = pd.read_parquet(dollar_adv_path)
        if _quarantined:
            _dollar_adv = _dollar_adv.drop(
                columns=[s for s in _quarantined if s in _dollar_adv.columns]
            )
        logger.info("Loaded dollar ADV: %s", _dollar_adv.shape)
    else:
        _dollar_adv = None
        logger.warning("Dollar ADV file not found: %s", dollar_adv_path)


def get_factors() -> Optional[pd.DataFrame]:
    return _factors


def get_prices() -> Optional[pd.DataFrame]:
    return _prices


def get_sectors() -> Optional[pd.DataFrame]:
    return _sectors


def get_dollar_adv() -> Optional[pd.DataFrame]:
    return _dollar_adv


def get_quarantined_symbols() -> set[str]:
    return _quarantined
