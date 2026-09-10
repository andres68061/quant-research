"""The FRED series the platform tracks, with the metadata that makes them usable.

One entry per project series. The project id (the dict key) is what appears as
a column in ``data/factors/macro.parquet``; the FRED id is what is fetched.

**Publication lag is measured from the FRED reference date.** FRED stamps a
monthly value on the *first day of the reference month*: June CPI sits on
``2026-06-01`` but is released around July 12. The lag therefore has to cover
the rest of the reference month *plus* the release delay - roughly 31 days
plus the release day of the following month. A lag of 30 would make June CPI
visible on July 1, eleven days before anyone could know it. Weekly series are
stamped on the week-ending date; daily series on the observation date.

Lags are deliberately conservative (a few days past the typical release) so a
holiday-shifted release does not leak. They are fixed calendar-day offsets, not
ALFRED vintages - see ``docs/MACRO_VINTAGES.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

Transform = Literal["level", "yoy_pct"]
Frequency = Literal["daily", "weekly", "monthly"]
Group = Literal["rates", "curve", "inflation", "credit", "labor", "activity", "money", "markets"]


@dataclass(frozen=True)
class FredSeriesSpec:
    """Everything needed to fetch, lag and label one FRED series.

    Attributes:
        fred_id: FRED series identifier.
        name: Short human label.
        group: Which family the series belongs to (drives page grouping).
        frequency: Native FRED frequency.
        lag_days: Calendar days from the FRED reference date to the first day
            the value could have been known (see module docstring).
        transform: ``level`` stores FRED's value; ``yoy_pct`` stores the
            12-period percentage change (index levels -> inflation rates).
        unit: Unit of the stored value, after transform.
    """

    fred_id: str
    name: str
    group: Group
    frequency: Frequency
    lag_days: int
    transform: Transform = "level"
    unit: str = "%"


def _daily(fred_id: str, name: str, group: Group, unit: str = "%", lag: int = 1) -> FredSeriesSpec:
    return FredSeriesSpec(fred_id, name, group, "daily", lag, "level", unit)


def _monthly(
    fred_id: str, name: str, group: Group, lag: int, transform: Transform = "level", unit: str = "%"
) -> FredSeriesSpec:
    return FredSeriesSpec(fred_id, name, group, "monthly", lag, transform, unit)


# Release-day anchors used below (day of the month following the reference month):
#   Employment Situation ~ first Friday (<= 7th)  -> 31 + 7 + 2 slack = 40
#   CPI ~ 10th-15th                                 -> 31 + 15 + 2       = 48
#   Retail sales ~ 15th-17th, INDPRO ~ 15th-17th    -> 31 + 17 + 2       = 50
#   Housing starts ~ 17th-19th                      -> 31 + 19 + 3       = 53
#   PCE ~ last business day                         -> 31 + 31 + 2       = 64
#   M2 ~ fourth Tuesday                             -> 31 + 28 + 2       = 61
#   FEDFUNDS monthly average: knowable once the month ends -> 31 + 2     = 33
#   UMCSENT: FRED receives it ~one month after the university publishes -> 62
FRED_SERIES_CATALOG: Dict[str, FredSeriesSpec] = {
    # --- policy and money-market rates ---------------------------------------
    "fed_funds": _monthly("FEDFUNDS", "Fed funds (monthly avg)", "rates", 33),
    "dff": _daily("DFF", "Fed funds effective (daily)", "rates"),
    "sofr": _daily("SOFR", "SOFR", "rates"),
    # --- Treasury constant-maturity curve --------------------------------------
    "dgs1mo": _daily("DGS1MO", "Treasury 1M", "curve"),
    "dgs3mo": _daily("DGS3MO", "Treasury 3M", "curve"),
    "dgs6mo": _daily("DGS6MO", "Treasury 6M", "curve"),
    "dgs1": _daily("DGS1", "Treasury 1Y", "curve"),
    "dgs2": _daily("DGS2", "Treasury 2Y", "curve"),
    "dgs3": _daily("DGS3", "Treasury 3Y", "curve"),
    "dgs5": _daily("DGS5", "Treasury 5Y", "curve"),
    "dgs7": _daily("DGS7", "Treasury 7Y", "curve"),
    "dgs10": _daily("DGS10", "Treasury 10Y", "curve"),
    "dgs20": _daily("DGS20", "Treasury 20Y", "curve"),
    "dgs30": _daily("DGS30", "Treasury 30Y", "curve"),
    "t10y2y": _daily("T10Y2Y", "10Y-2Y spread", "curve"),
    "t10y3m": _daily("T10Y3M", "10Y-3M spread", "curve"),
    # --- real rates and breakevens ---------------------------------------------
    "dfii5": _daily("DFII5", "TIPS 5Y real yield", "rates"),
    "dfii10": _daily("DFII10", "TIPS 10Y real yield", "rates"),
    "t5yie": _daily("T5YIE", "5Y breakeven inflation", "inflation"),
    "t10yie": _daily("T10YIE", "10Y breakeven inflation", "inflation"),
    # --- credit ------------------------------------------------------------------
    "baa10y": _daily("BAA10Y", "Baa corporate - 10Y Treasury", "credit"),
    "hy_oas": _daily("BAMLH0A0HYM2", "US high-yield OAS", "credit"),
    "ig_oas": _daily("BAMLC0A0CM", "US investment-grade OAS", "credit"),
    # --- inflation ----------------------------------------------------------------
    "cpi_yoy": _monthly("CPIAUCSL", "CPI, YoY", "inflation", 48, "yoy_pct"),
    "core_cpi_yoy": _monthly("CPILFESL", "Core CPI, YoY", "inflation", 48, "yoy_pct"),
    "pce_yoy": _monthly("PCEPI", "PCE price index, YoY", "inflation", 64, "yoy_pct"),
    # --- labor --------------------------------------------------------------------
    "unrate": _monthly("UNRATE", "Unemployment rate", "labor", 40),
    "payems": _monthly("PAYEMS", "Nonfarm payrolls", "labor", 40, unit="thousands"),
    "initial_claims": FredSeriesSpec(
        "ICSA", "Initial jobless claims", "labor", "weekly", 5, "level", "persons"
    ),
    # --- activity -----------------------------------------------------------------
    "indpro_yoy": _monthly("INDPRO", "Industrial production, YoY", "activity", 50, "yoy_pct"),
    "retail_sales_yoy": _monthly("RSAFS", "Retail sales, YoY", "activity", 50, "yoy_pct"),
    "housing_starts": _monthly("HOUST", "Housing starts (SAAR)", "activity", 53, unit="thousands"),
    "umcsent": _monthly("UMCSENT", "Consumer sentiment (Michigan)", "activity", 62, unit="index"),
    # --- money and Fed balance sheet ----------------------------------------------
    "m2_yoy": _monthly("M2SL", "M2 money stock, YoY", "money", 61, "yoy_pct"),
    "fed_assets": FredSeriesSpec(
        "WALCL", "Fed total assets", "money", "weekly", 2, "level", "USD millions"
    ),
    # --- market reference series --------------------------------------------------
    "dollar_broad": _daily("DTWEXBGS", "Broad dollar index", "markets", "index", lag=7),
    "vix": _daily("VIXCLS", "VIX close", "markets", "index"),
    # EIA posts the daily spot price weekly, about a week in arrears.
    "wti": _daily("DCOILWTICO", "WTI crude spot", "markets", "USD/bbl", lag=8),
}

# Compatibility views used by the derivation code and tests.
DEFAULT_FRED_SERIES_MAP: Dict[str, str] = {k: v.fred_id for k, v in FRED_SERIES_CATALOG.items()}
MACRO_PUBLICATION_LAGS_DAYS: Dict[str, int] = {
    k: v.lag_days for k, v in FRED_SERIES_CATALOG.items()
}

# Tenors of the constant-maturity curve, in years, in curve order.
TREASURY_CURVE_TENORS_YEARS: Dict[str, float] = {
    "dgs1mo": 1 / 12,
    "dgs3mo": 0.25,
    "dgs6mo": 0.5,
    "dgs1": 1.0,
    "dgs2": 2.0,
    "dgs3": 3.0,
    "dgs5": 5.0,
    "dgs7": 7.0,
    "dgs10": 10.0,
    "dgs20": 20.0,
    "dgs30": 30.0,
}

__all__ = [
    "FredSeriesSpec",
    "FRED_SERIES_CATALOG",
    "DEFAULT_FRED_SERIES_MAP",
    "MACRO_PUBLICATION_LAGS_DAYS",
    "TREASURY_CURVE_TENORS_YEARS",
]
