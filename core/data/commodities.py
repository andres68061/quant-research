"""Commodity price series from FMP (ETFs + futures).

Internal column names (``GLD``, ``WTI``, …) stay stable for the API and
frontend. Each maps to an FMP symbol via ``COMMODITIES_CONFIG["fmp_symbol"]``.

FMP note: several futures symbols return HTTP 402 on ``/historical-price-eod/full``
under Premium, but ``/historical-price-eod/dividend-adjusted`` works (commodities
have no dividends, so adj close ≡ close). We fetch through
:func:`core.data.fmp.prices.fetch_dividend_adjusted_history`.

Series discontinuity: energy/ag/industrial columns previously came from Alpha
Vantage *spot* series; they now use FMP *futures* (``CLUSD``, ``BZUSD``, …).
Precious metals remain ETF proxies (``GLD``/``SLV``/``PPLT``/``PALL``) for
continuity with the existing Metals Analytics panel.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from core.data.fmp.prices import fetch_dividend_adjusted_history

logger = logging.getLogger(__name__)

DEFAULT_START = pd.Timestamp("2000-01-01")
RAW_COMMODITIES_DIR = Path("data/raw/fmp/commodities")

# Internal key → metadata. ``fmp_symbol`` is what we request from FMP.
COMMODITIES_CONFIG: dict[str, dict[str, str]] = {
    "GLD": {
        "name": "Gold (GLD ETF)",
        "source": "fmp",
        "fmp_symbol": "GLD",
        "unit": "USD",
        "category": "precious_metals",
    },
    "SLV": {
        "name": "Silver (SLV ETF)",
        "source": "fmp",
        "fmp_symbol": "SLV",
        "unit": "USD",
        "category": "precious_metals",
    },
    "PPLT": {
        "name": "Platinum (PPLT ETF)",
        "source": "fmp",
        "fmp_symbol": "PPLT",
        "unit": "USD",
        "category": "precious_metals",
    },
    "PALL": {
        "name": "Palladium (PALL ETF)",
        "source": "fmp",
        "fmp_symbol": "PALL",
        "unit": "USD",
        "category": "precious_metals",
    },
    "WTI": {
        "name": "Crude Oil (WTI futures)",
        "source": "fmp",
        "fmp_symbol": "CLUSD",
        "unit": "USD/barrel",
        "category": "energy",
    },
    "BRENT": {
        "name": "Crude Oil (Brent futures)",
        "source": "fmp",
        "fmp_symbol": "BZUSD",
        "unit": "USD/barrel",
        "category": "energy",
    },
    "NATURAL_GAS": {
        "name": "Natural Gas futures",
        "source": "fmp",
        "fmp_symbol": "NGUSD",
        "unit": "USD/MMBtu",
        "category": "energy",
    },
    "COPPER": {
        "name": "Copper futures",
        "source": "fmp",
        "fmp_symbol": "HGUSD",
        "unit": "USD/lb",
        "category": "industrial",
    },
    "ALUMINUM": {
        "name": "Aluminum futures",
        "source": "fmp",
        "fmp_symbol": "ALIUSD",
        "unit": "USD/ton",
        "category": "industrial",
    },
    "WHEAT": {
        "name": "Wheat futures",
        "source": "fmp",
        "fmp_symbol": "KEUSX",
        "unit": "USD/bushel",
        "category": "agricultural",
    },
    "CORN": {
        "name": "Corn futures",
        "source": "fmp",
        "fmp_symbol": "ZCUSX",
        "unit": "USD/bushel",
        "category": "agricultural",
    },
    "COFFEE": {
        "name": "Coffee futures",
        "source": "fmp",
        "fmp_symbol": "KCUSX",
        "unit": "USD/lb",
        "category": "agricultural",
    },
    "COTTON": {
        "name": "Cotton futures",
        "source": "fmp",
        "fmp_symbol": "CTUSX",
        "unit": "USD/lb",
        "category": "agricultural",
    },
    "SUGAR": {
        "name": "Sugar futures",
        "source": "fmp",
        "fmp_symbol": "SBUSX",
        "unit": "USD/lb",
        "category": "agricultural",
    },
}


def fetch_commodity_close(
    internal_symbol: str,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
) -> pd.Series:
    """
    Fetch one commodity close series from FMP.

    Args:
        internal_symbol: Key in ``COMMODITIES_CONFIG`` (e.g. ``"WTI"``).
        start: Inclusive start (default 2000-01-01).
        end: Inclusive end (default today).

    Returns:
        tz-naive Series named ``internal_symbol`` (calendar dates), empty if
        FMP returns nothing.
    """
    if internal_symbol not in COMMODITIES_CONFIG:
        raise KeyError(f"Unknown commodity: {internal_symbol}")
    fmp_symbol = COMMODITIES_CONFIG[internal_symbol]["fmp_symbol"]
    start_ts = start if start is not None else DEFAULT_START
    end_ts = end if end is not None else pd.Timestamp.now().normalize()
    history = fetch_dividend_adjusted_history(fmp_symbol, start_ts, end_ts)
    if history.empty:
        logger.warning("No FMP history for %s (%s)", internal_symbol, fmp_symbol)
        return pd.Series(dtype="float64", name=internal_symbol)
    series = history["adj_close"].copy()
    series.index = series.index.tz_localize(None)
    series.name = internal_symbol
    series.index.name = "date"
    return series.astype("float64")


class CommodityDataFetcher:
    """Fetch/store commodity panel under ``data/commodities/prices.parquet``."""

    def __init__(self, data_dir: Optional[Path] = None) -> None:
        if data_dir is None:
            self.data_dir = Path(__file__).parents[2] / "data" / "commodities"
        else:
            self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.prices_file = self.data_dir / "prices.parquet"
        self.raw_dir = Path(__file__).parents[2] / RAW_COMMODITIES_DIR

    def fetch_commodity(self, symbol: str) -> pd.Series:
        """Fetch full history for one internal commodity key."""
        series = fetch_commodity_close(symbol)
        if not series.empty:
            logger.info(
                "Fetched %s: %d points (%s → %s)",
                symbol,
                len(series),
                series.index[0].date(),
                series.index[-1].date(),
            )
            self._write_raw(symbol, series)
        return series

    def _write_raw(self, symbol: str, series: pd.Series) -> None:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        frame = series.to_frame(name="adj_close")
        frame.to_parquet(self.raw_dir / f"{symbol}.parquet")

    def fetch_all_commodities(self) -> pd.DataFrame:
        """Fetch every configured commodity into a wide panel."""
        columns: dict[str, pd.Series] = {}
        for symbol in COMMODITIES_CONFIG:
            series = self.fetch_commodity(symbol)
            if not series.empty:
                columns[symbol] = series
        if not columns:
            return pd.DataFrame()
        panel = pd.concat(columns, axis=1).sort_index()
        panel.index.name = "date"
        return panel

    def save_prices(self, prices: pd.DataFrame) -> None:
        """Persist the derived commodity panel."""
        if prices.empty:
            logger.warning("No commodity data to save")
            return
        prices.to_parquet(self.prices_file, engine="pyarrow", compression="snappy")
        logger.info("Saved %d rows × %d cols to %s", *prices.shape, self.prices_file)

    def load_prices(self) -> pd.DataFrame:
        """Load the derived commodity panel (empty if missing)."""
        if not self.prices_file.exists():
            logger.warning("No commodity panel at %s", self.prices_file)
            return pd.DataFrame()
        prices = pd.read_parquet(self.prices_file)
        prices.index = pd.to_datetime(prices.index).tz_localize(None)
        return prices

    def update_commodity(
        self,
        symbol: str,
        existing_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Incrementally refresh one commodity column.

        Refetches from ``last_valid_index - 7d`` so recent FMP restatements overwrite.
        """
        if existing_df is None:
            existing_df = self.load_prices()

        start = DEFAULT_START
        if symbol in existing_df.columns:
            last = existing_df[symbol].last_valid_index()
            if last is not None:
                start = pd.Timestamp(last) - pd.Timedelta(days=7)

        new_series = fetch_commodity_close(symbol, start=start)
        if new_series.empty:
            logger.warning("No new FMP data for %s", symbol)
            return existing_df

        if existing_df.empty:
            updated = pd.DataFrame({symbol: new_series}).sort_index()
        else:
            updated = existing_df.copy()
            if symbol in updated.columns:
                combined = pd.concat([updated[symbol], new_series])
                combined = combined[~combined.index.duplicated(keep="last")].sort_index()
                updated[symbol] = combined
            else:
                updated[symbol] = new_series
            updated = updated.sort_index()

        full_col = updated[symbol].dropna()
        self._write_raw(symbol, full_col.rename(symbol))
        return updated

    def update_all_commodities(self) -> pd.DataFrame:
        """Refresh every configured commodity into the on-disk panel."""
        updated = self.load_prices()
        for symbol in COMMODITIES_CONFIG:
            updated = self.update_commodity(symbol, existing_df=updated)
        return updated
