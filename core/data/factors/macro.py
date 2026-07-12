"""Macro data layer: raw FRED store plus derived publication-lagged panels.

Two layers:

1. Raw layer (``data/raw/macro_fred.parquet``): long-format ``(reference_date,
   series_id, value)`` at the native FRED frequency (monthly for
   CPI/UNRATE/FEDFUNDS, daily for DGS10/T10Y2Y), with no publication lag,
   no business-day forward-fill, and no standardisation.
2. Derived layer (``data/factors/macro.parquet``): wide business-day panel
   with publication lag applied. Built from the raw layer via
   :func:`derive_macro_panel_from_raw`.

**Vintages:** this module uses *fixed* calendar-day publication lags
(``MACRO_PUBLICATION_LAGS_DAYS``), not ALFRED real-time revision histories.
See ``docs/MACRO_VINTAGES.md``. ``MACRO_USES_TRUE_VINTAGES`` is False until
an ALFRED (or equivalent) vintage store lands.

The notebook research flow loads from the raw layer and applies pub-lag /
ffill / z-score visibly inline.
"""

from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from fredapi import Fred

from config.settings import FRED_API_KEY, PROJECT_ROOT

# Fixed lags approximate first-release availability. Not ALFRED vintages.
MACRO_USES_TRUE_VINTAGES: bool = False

MACRO_PUBLICATION_LAGS_DAYS: Dict[str, int] = {
    "cpi_yoy": 30,
    "unrate": 10,
    "fed_funds": 5,
    "dgs10": 1,
    "t10y2y": 1,
}

DEFAULT_FRED_SERIES_MAP: Dict[str, str] = {
    "cpi_yoy": "CPIAUCSL",
    "unrate": "UNRATE",
    "fed_funds": "FEDFUNDS",
    "dgs10": "DGS10",
    "t10y2y": "T10Y2Y",
}

RAW_MACRO_PARQUET = PROJECT_ROOT / "data" / "raw" / "macro_fred.parquet"

RAW_LONG_COLUMNS = ("reference_date", "series_id", "value")


def fetch_raw_fred_series(series_map: Dict[str, str]) -> pd.DataFrame:
    """Fetch FRED series as a long-format raw panel.

    Returns one row per ``(reference_date, series_id, value)`` triple with no
    publication lag, no business-day forward-fill, and no standardisation.
    Each series stays at its native FRED frequency.

    The single project-specific transform applied here is the YoY pct_change
    used to translate ``CPIAUCSL`` (a price index level) into ``cpi_yoy`` (a
    YoY change). The reference date is preserved from FRED.

    Args:
        series_map: Mapping ``{series_id: fred_id}``. ``series_id`` is the
            project-internal name (e.g. ``"cpi_yoy"``); ``fred_id`` is the
            FRED ticker (e.g. ``"CPIAUCSL"``).

    Returns:
        Long-format DataFrame with columns
        ``["reference_date", "series_id", "value"]`` sorted by
        ``(series_id, reference_date)``.

    Raises:
        RuntimeError: If ``FRED_API_KEY`` is not set.

    Example:
        >>> raw = fetch_raw_fred_series({"unrate": "UNRATE"})
        >>> sorted(raw.columns.tolist())
        ['reference_date', 'series_id', 'value']
    """
    if not FRED_API_KEY:
        raise RuntimeError(
            "FRED_API_KEY is not set; cannot fetch raw macro series. "
            "Add it to your .env or skip this step."
        )

    fred = Fred(api_key=FRED_API_KEY)
    parts: list[pd.DataFrame] = []
    for series_id, fred_id in series_map.items():
        raw_series = fred.get_series(fred_id)
        raw_series.index = pd.to_datetime(raw_series.index)

        if series_id == "cpi_yoy" and fred_id == "CPIAUCSL":
            values = raw_series.pct_change(12, fill_method=None).dropna()
        else:
            values = raw_series.dropna()

        if values.empty:
            continue

        parts.append(
            pd.DataFrame(
                {
                    "reference_date": values.index,
                    "series_id": series_id,
                    "value": values.astype("float64").to_numpy(),
                }
            )
        )

    if not parts:
        return pd.DataFrame(columns=list(RAW_LONG_COLUMNS))

    raw_long = pd.concat(parts, ignore_index=True)
    raw_long["reference_date"] = pd.to_datetime(raw_long["reference_date"])
    raw_long = raw_long.sort_values(["series_id", "reference_date"]).reset_index(drop=True)
    return raw_long


def load_raw_macro_default() -> pd.DataFrame:
    """Fetch the project's canonical macro raw layer from FRED.

    Returns:
        Long-format DataFrame for the five canonical series defined in
        :data:`DEFAULT_FRED_SERIES_MAP`.
    """
    return fetch_raw_fred_series(DEFAULT_FRED_SERIES_MAP)


def read_raw_macro_parquet(path: Optional[Path] = None) -> pd.DataFrame:
    """Read the raw FRED long-format panel from disk.

    Args:
        path: Optional override. Defaults to :data:`RAW_MACRO_PARQUET`.

    Returns:
        Long-format DataFrame with columns
        ``["reference_date", "series_id", "value"]``.

    Raises:
        FileNotFoundError: If the raw parquet does not exist.
    """
    resolved = path or RAW_MACRO_PARQUET
    if not resolved.exists():
        raise FileNotFoundError(
            f"Raw macro parquet not found at {resolved}. "
            "Run scripts/fetch_raw_macro.py to backfill it."
        )
    raw_long = pd.read_parquet(resolved)
    raw_long["reference_date"] = pd.to_datetime(raw_long["reference_date"])
    return raw_long.sort_values(["series_id", "reference_date"]).reset_index(drop=True)


def apply_macro_publication_lag(
    macro_df: pd.DataFrame,
    publication_lags_days: Optional[Dict[str, int]] = None,
) -> pd.DataFrame:
    """Shift macro reference dates to conservative observable dates.

    FRED indexes many monthly series by the economic reference period, not
    the public release date. This helper applies a conservative calendar-day
    lag to each column before any daily forward-fill is performed, preventing
    CPI or unemployment values from appearing before they could have been
    known.

    Args:
        macro_df: Wide DataFrame indexed by FRED reference dates.
        publication_lags_days: Optional ``{column: lag_days}`` override.

    Returns:
        DataFrame with values indexed by conservative as-of dates.

    Example:
        >>> raw = pd.DataFrame({"unrate": [4.0]}, index=[pd.Timestamp("2020-01-01")])
        >>> apply_macro_publication_lag(raw).index[0]
        Timestamp('2020-01-11 00:00:00')
    """
    if macro_df is None or macro_df.empty:
        return pd.DataFrame()

    lags = publication_lags_days or MACRO_PUBLICATION_LAGS_DAYS
    shifted_parts: list[pd.Series] = []
    for column in macro_df.columns:
        series = macro_df[column].dropna().copy()
        if series.empty:
            continue
        lag_days = lags.get(column, 0)
        series.index = pd.to_datetime(series.index) + pd.to_timedelta(lag_days, unit="D")
        series.name = column
        shifted_parts.append(series)

    if not shifted_parts:
        return pd.DataFrame()

    shifted = pd.concat(shifted_parts, axis=1).sort_index()
    shifted.index.name = macro_df.index.name or "date"
    return shifted


def derive_macro_panel_from_raw(raw_long: pd.DataFrame) -> pd.DataFrame:
    """Build the publication-lagged business-day panel from the raw long table.

    Pipeline (each step is a deterministic function of ``raw_long``):

    1. Pivot ``raw_long`` to wide with one column per ``series_id``.
    2. Apply :func:`apply_macro_publication_lag`.
    3. Reindex to business-day frequency and forward-fill.

    Args:
        raw_long: Long-format DataFrame as produced by
            :func:`fetch_raw_fred_series` or
            :func:`read_raw_macro_parquet`.

    Returns:
        Wide DataFrame indexed by business-day ``date`` with one column per
        ``series_id``. Empty DataFrame if ``raw_long`` is empty.
    """
    if raw_long is None or raw_long.empty:
        return pd.DataFrame()

    wide = (
        raw_long.pivot(index="reference_date", columns="series_id", values="value")
        .sort_index()
    )
    wide.index = pd.to_datetime(wide.index)
    wide.index.name = "date"
    wide.columns.name = None

    lagged = apply_macro_publication_lag(wide)
    daily = lagged.asfreq("B").ffill()
    daily.index.name = "date"
    return daily


def load_default_macro() -> pd.DataFrame:
    """Return the publication-lagged macro panel derived from raw FRED.

    Equivalent to ``derive_macro_panel_from_raw(load_raw_macro_default())``.
    Returns an empty DataFrame if no FRED API key is configured.
    """
    if not FRED_API_KEY:
        return pd.DataFrame()
    return derive_macro_panel_from_raw(load_raw_macro_default())


def compute_macro_zscores(macro_df: pd.DataFrame, window_days: int = 252 * 5) -> pd.DataFrame:
    """Rolling z-score per series to standardise macro signals.

    If insufficient history is available, the rolling window automatically
    falls back to ``max(30, window_days // 12)`` minimum periods, matching the
    in-notebook standardisation used by the regime HMM.
    """
    if macro_df is None or macro_df.empty:
        return pd.DataFrame()
    out = pd.DataFrame(index=macro_df.index)
    min_periods = max(30, window_days // 12)
    for col in macro_df.columns:
        series = macro_df[col].astype(float)
        roll = series.rolling(window_days, min_periods=min_periods)
        mu = roll.mean()
        sd = roll.std(ddof=0)
        z = (series - mu) / (sd.replace(0, pd.NA))
        out[f"macro_z_{col}"] = z
    return out
