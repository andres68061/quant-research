"""
Shared FastAPI dependencies: data loaders and singletons.

Loaded once at startup via the ``lifespan`` context manager in ``main.py``.
"""

import logging
from typing import Optional

import pandas as pd

from config.settings import PROJECT_ROOT
from core.data.api_universe import select_api_symbols
from core.data.factor_store import FactorStore
from core.data.quality import QUARANTINE_PATH, load_quarantined_symbols

logger = logging.getLogger(__name__)

_factors: Optional[pd.DataFrame] = None
_factor_store: Optional[FactorStore] = None
_universe_disclosure: dict = {}
_prices: Optional[pd.DataFrame] = None
_sectors: Optional[pd.DataFrame] = None
_dollar_adv: Optional[pd.DataFrame] = None
_quarantined: set[str] = set()


def load_data() -> None:
    """Load core datasets into module-level caches, excluding quarantined symbols."""
    global _factors, _factor_store, _prices, _sectors, _dollar_adv, _quarantined

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

    # NOTE: the factor panels are deliberately NOT loaded eagerly here.
    # Post-cutover they total ~7 GB (26.7M-row price factors + ~21M-row
    # fundamentals), and no consumer needs the wide frame — every route wants
    # either the list of factor names or one column. FactorStore (built below,
    # after prices define the universe) reads metadata now and columns on demand.
    # Cross-sectional composites (value_quality, *_sn) are attached by
    # scripts/build_fundamentals_panel.py at build time, not here: computing a
    # cross-sectional z-score during a load is both slow and silently dependent
    # on whichever universe happened to be loaded. See ADR 0014.
    if not factors_path.exists():
        logger.warning("Factors file not found: %s", factors_path)

    if prices_path.exists():
        full_panel = pd.read_parquet(prices_path)
        if _quarantined:
            full_panel = full_panel.drop(
                columns=[s for s in _quarantined if s in full_panel.columns]
            )
        # ADR 0013: the canonical panel is now the whole US market (~8,900
        # symbols). Loading all of it plus every factor panel costs ~7 GB, so the
        # API loads a policy-selected subset and DISCLOSES which one.
        selected, _universe_disclosure_local = select_api_symbols(full_panel)
        _universe_disclosure.clear()
        _universe_disclosure.update(_universe_disclosure_local)
        _prices = full_panel[[c for c in selected if c in full_panel.columns]]
        del full_panel
        logger.info(
            "Loaded prices under %r policy: %s (from %s panel symbols)",
            _universe_disclosure.get("policy"),
            _prices.shape,
            _universe_disclosure.get("panel_symbols"),
        )
    else:
        logger.warning("Prices file not found: %s", prices_path)

    _factor_store = FactorStore(
        data_dir / "factors",
        symbols=set(_prices.columns) if _prices is not None else None,
    )

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
    """
    The eagerly-loaded factor frame, when one is available.

    Prefer :func:`get_factor_frame` for backtests: it reads a single column on
    demand (~43 MB) instead of holding every panel in memory (~7 GB post-cutover).
    This getter remains for callers that genuinely need the wide frame.
    """
    return _factors


def get_factor_store() -> Optional[FactorStore]:
    """Lazy per-column access to every factor panel."""
    return _factor_store


def get_factor_frame(factor_col: str) -> pd.DataFrame:
    """
    Load one factor as a ``(date, symbol)`` single-column frame.

    Args:
        factor_col: Factor name from :meth:`FactorStore.available_factors`.

    Returns:
        One-column DataFrame ready for the cross-section runner.

    Raises:
        RuntimeError: If no factor store was loaded.
        KeyError: If the factor is unknown.
    """
    if _factor_store is None:
        raise RuntimeError("Factor store not loaded")
    return _factor_store.load_factor(factor_col)


def get_universe_disclosure() -> dict:
    """Which universe policy the API loaded, with per-step counts and the reason."""
    return dict(_universe_disclosure)


def get_prices() -> Optional[pd.DataFrame]:
    return _prices


def get_sectors() -> Optional[pd.DataFrame]:
    return _sectors


def get_dollar_adv() -> Optional[pd.DataFrame]:
    return _dollar_adv


def get_quarantined_symbols() -> set[str]:
    return _quarantined
