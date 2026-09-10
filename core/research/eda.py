"""Exploratory statistics for monitoring a single series or a rate curve.

Every function takes a Series/DataFrame and returns plain data (dataclasses,
dicts, DataFrames). No I/O, no plotting. The questions answered here are the
ones a statistician asks before trusting a series, not alpha questions:

- *What does this look like historically, and where is today in that?*
  :func:`summarize_distribution`, :func:`histogram`, :func:`rolling_profile`.
- *Does it repeat within the year?*  :func:`seasonality_table`,
  :func:`annual_paths`.
- *What shape is the curve, and how has that shape moved?*
  :func:`yield_curve_snapshots`, :func:`curve_shape_history`.
- *Is the series still arriving?*  :func:`staleness`.

Transform vocabulary (used by the API and page):
``level`` the stored value; ``diff`` first difference (rates, spreads);
``pct_change`` simple return (prices); ``log_return`` log return.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd

Transform = Literal["level", "diff", "pct_change", "log_return"]
StaleStatus = Literal["fresh", "late", "stale", "empty"]

EPS = 1e-10
MIN_OBS_FOR_MOMENTS = 20
MIN_OBS_FOR_ADF = 50

__all__ = [
    "Transform",
    "DistributionSummary",
    "StalenessReport",
    "apply_transform",
    "summarize_distribution",
    "histogram",
    "rolling_profile",
    "seasonality_table",
    "annual_paths",
    "yield_curve_snapshots",
    "curve_shape_history",
    "staleness",
]


# --------------------------------------------------------------------------- transforms
def apply_transform(values: pd.Series, transform: Transform) -> pd.Series:
    """Turn a level series into the quantity whose distribution is of interest.

    Args:
        values: Level series with a DatetimeIndex. NaNs are dropped first so a
            gap does not create a spurious multi-day change.
        transform: See module docstring.

    Returns:
        Transformed float64 series, NaNs dropped. Empty input -> empty output.
    """
    clean = values.dropna().astype("float64")
    if clean.empty:
        return clean
    if transform == "level":
        return clean
    if transform == "diff":
        return clean.diff().dropna()
    if transform == "pct_change":
        return clean.pct_change(fill_method=None).dropna()
    if transform == "log_return":
        return np.log(np.maximum(clean, EPS)).diff().dropna()
    raise ValueError(f"Unknown transform: {transform!r}")


# --------------------------------------------------------------------------- distribution
@dataclass(frozen=True)
class DistributionSummary:
    """Moments and position-of-today for one transformed series.

    ``percentile_of_last`` is the share of history at or below the latest value
    (0-100). ``adf_pvalue`` is the augmented Dickey-Fuller p-value on the
    transformed series (small = rejects a unit root); None when the sample is
    too short. ``kurtosis`` is excess kurtosis (normal = 0).
    """

    n: int
    start: Optional[str]
    end: Optional[str]
    last_value: Optional[float]
    mean: Optional[float]
    std: Optional[float]
    min: Optional[float]
    p05: Optional[float]
    p25: Optional[float]
    median: Optional[float]
    p75: Optional[float]
    p95: Optional[float]
    max: Optional[float]
    skew: Optional[float]
    kurtosis: Optional[float]
    autocorr_lag1: Optional[float]
    percentile_of_last: Optional[float]
    zscore_of_last: Optional[float]
    adf_pvalue: Optional[float]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _adf_pvalue(values: pd.Series) -> Optional[float]:
    if len(values) < MIN_OBS_FOR_ADF or values.std() < EPS:
        return None
    from statsmodels.tsa.stattools import adfuller

    try:
        return float(adfuller(values.to_numpy(), autolag="AIC")[1])
    except (ValueError, np.linalg.LinAlgError):
        return None


def summarize_distribution(values: pd.Series, run_adf: bool = True) -> DistributionSummary:
    """Describe a (transformed) series and locate its latest value in history.

    Args:
        values: Already-transformed series (see :func:`apply_transform`).
        run_adf: Compute the ADF stationarity test (costs ~10 ms per call).

    Returns:
        :class:`DistributionSummary`; all statistics None when ``n`` is below
        :data:`MIN_OBS_FOR_MOMENTS`.
    """
    clean = values.dropna().astype("float64")
    n = int(len(clean))
    empty = DistributionSummary(
        n,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )
    if n < MIN_OBS_FOR_MOMENTS:
        return empty

    last = float(clean.iloc[-1])
    std = float(clean.std())
    q = clean.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    return DistributionSummary(
        n=n,
        start=str(clean.index.min().date()),
        end=str(clean.index.max().date()),
        last_value=last,
        mean=float(clean.mean()),
        std=std,
        min=float(clean.min()),
        p05=float(q.loc[0.05]),
        p25=float(q.loc[0.25]),
        median=float(q.loc[0.5]),
        p75=float(q.loc[0.75]),
        p95=float(q.loc[0.95]),
        max=float(clean.max()),
        skew=float(clean.skew()),
        kurtosis=float(clean.kurt()),
        autocorr_lag1=float(clean.autocorr(lag=1)) if n > 2 else None,
        percentile_of_last=float((clean <= last).mean() * 100.0),
        zscore_of_last=float((last - clean.mean()) / max(std, EPS)),
        adf_pvalue=_adf_pvalue(clean) if run_adf else None,
    )


def histogram(values: pd.Series, bins: int = 50, mark: Optional[float] = None) -> Dict[str, object]:
    """Bin a series; optionally report which bin a marked value falls in.

    Args:
        values: Transformed series.
        bins: Number of equal-width bins.
        mark: Value to locate (typically the latest observation).

    Returns:
        ``{"edges": [...], "counts": [...], "mark": mark, "mark_bin": int | None}``
        where ``len(edges) == len(counts) + 1``. Empty input -> empty lists.
    """
    clean = values.dropna().astype("float64")
    if clean.empty:
        return {"edges": [], "counts": [], "mark": mark, "mark_bin": None}
    counts, edges = np.histogram(clean.to_numpy(), bins=bins)
    mark_bin: Optional[int] = None
    if mark is not None and np.isfinite(mark):
        idx = int(np.searchsorted(edges, mark, side="right") - 1)
        mark_bin = min(max(idx, 0), len(counts) - 1)
    return {
        "edges": [float(e) for e in edges],
        "counts": [int(c) for c in counts],
        "mark": mark,
        "mark_bin": mark_bin,
    }


def rolling_profile(values: pd.Series, window: int = 252, n_std: float = 2.0) -> pd.DataFrame:
    """Level with a rolling mean, band and z-score - the 'is this unusual' chart.

    Args:
        values: Level series.
        window: Observations in the rolling window (252 ~ one trading year).
        n_std: Band half-width in standard deviations.

    Returns:
        DataFrame ``[level, rolling_mean, upper, lower, zscore]`` on the input
        index; NaN until the window fills.
    """
    clean = values.dropna().astype("float64")
    if clean.empty:
        return pd.DataFrame(columns=["level", "rolling_mean", "upper", "lower", "zscore"])
    mean = clean.rolling(window, min_periods=max(window // 2, 2)).mean()
    std = clean.rolling(window, min_periods=max(window // 2, 2)).std()
    return pd.DataFrame(
        {
            "level": clean,
            "rolling_mean": mean,
            "upper": mean + n_std * std,
            "lower": mean - n_std * std,
            "zscore": (clean - mean) / std.clip(lower=EPS),
        }
    )


# --------------------------------------------------------------------------- seasonality
def seasonality_table(values: pd.Series, transform: Transform = "pct_change") -> Dict[str, object]:
    """Year x month table of the transformed series, plus per-month statistics.

    For ``pct_change``/``log_return`` the monthly cell is the compounded/summed
    month; for ``diff`` it is the month's total change; for ``level`` the
    month's mean. This is the "stacked by calendar month" view that shows
    whether a series repeats within the year.

    Args:
        values: Level series.
        transform: How to turn levels into the quantity being compared.

    Returns:
        ``years``: list of ints; ``matrix``: rows aligned with ``years``, 12
        entries each (None where missing); ``month_stats``: 12 dicts with
        ``month, mean, median, hit_rate, n`` where hit_rate is the share of
        years the month was positive (None for ``level``).
    """
    clean = values.dropna().astype("float64")
    if clean.empty:
        return {"years": [], "matrix": [], "month_stats": [], "transform": transform}

    if transform == "level":
        monthly = clean.resample("ME").mean()
    elif transform == "diff":
        monthly = clean.resample("ME").last().diff().dropna()
    elif transform == "pct_change":
        monthly = clean.resample("ME").last().pct_change(fill_method=None).dropna()
    elif transform == "log_return":
        monthly = np.log(np.maximum(clean, EPS)).resample("ME").last().diff().dropna()
    else:
        raise ValueError(f"Unknown transform: {transform!r}")

    # A month that has no observation at all must not inherit a value from
    # resample's last()/mean() over an empty bin.
    monthly = monthly.dropna()
    frame = pd.DataFrame(
        {"value": monthly, "year": monthly.index.year, "month": monthly.index.month}
    )
    pivot = frame.pivot_table(index="year", columns="month", values="value", aggfunc="first")
    pivot = pivot.reindex(columns=range(1, 13))

    years = [int(y) for y in pivot.index]
    matrix = [[None if pd.isna(v) else float(v) for v in row] for row in pivot.to_numpy()]

    month_stats: List[Dict[str, object]] = []
    for month in range(1, 13):
        col = pivot[month].dropna()
        month_stats.append(
            {
                "month": month,
                "n": int(len(col)),
                "mean": float(col.mean()) if len(col) else None,
                "median": float(col.median()) if len(col) else None,
                "hit_rate": (
                    float((col > 0).mean()) if len(col) and transform != "level" else None
                ),
            }
        )
    return {"years": years, "matrix": matrix, "month_stats": month_stats, "transform": transform}


def annual_paths(
    values: pd.Series, normalize: bool = True, max_years: int = 15
) -> Dict[str, object]:
    """Each calendar year as its own path on a common day-of-year axis.

    Overlaying years is the other seasonality view: instead of one cell per
    month, the whole path. With ``normalize`` each year is rebased to 100 at
    its first observation, so price-like series compare; leave it off for
    rates and spreads.

    Args:
        values: Level series.
        normalize: Rebase each year to 100 at its first observation.
        max_years: Most recent years to keep (older ones only add clutter).

    Returns:
        ``{"years": [...], "paths": {year: [{"doy": int, "value": float}, ...]}}``
    """
    clean = values.dropna().astype("float64")
    if clean.empty:
        return {"years": [], "paths": {}, "normalized": normalize}
    years = sorted(set(clean.index.year))[-max_years:]
    paths: Dict[int, List[Dict[str, float]]] = {}
    for year in years:
        year_values = clean[clean.index.year == year]
        if year_values.empty:
            continue
        if normalize:
            base = float(year_values.iloc[0])
            year_values = year_values / max(abs(base), EPS) * 100.0 if base != 0 else year_values
        paths[int(year)] = [
            {"doy": int(ts.dayofyear), "value": float(v)} for ts, v in year_values.items()
        ]
    return {"years": [int(y) for y in paths], "paths": paths, "normalized": normalize}


# --------------------------------------------------------------------------- curves
def yield_curve_snapshots(
    curve_panel: pd.DataFrame,
    tenors_years: Dict[str, float],
    dates: List[pd.Timestamp],
) -> List[Dict[str, object]]:
    """The term structure on each requested date (nearest prior observation).

    Args:
        curve_panel: Wide frame, DatetimeIndex, one column per tenor id.
        tenors_years: ``{column: tenor in years}`` in curve order.
        dates: Dates to snapshot. Each resolves to the last row at or before
            it; dates before the panel starts are skipped.

    Returns:
        One dict per resolved date: ``{"requested", "date", "points": [{"id",
        "tenor_years", "yield"}]}`` with None for tenors missing on that date.
    """
    cols = [c for c in tenors_years if c in curve_panel.columns]
    if not cols or curve_panel.empty:
        return []
    panel = curve_panel[cols].sort_index()
    out: List[Dict[str, object]] = []
    for requested in dates:
        prior = panel.loc[:requested]
        if prior.empty:
            continue
        row = prior.iloc[-1]
        # Skip a date where the whole curve is missing (holiday row from ffill gaps).
        if row.isna().all():
            continue
        out.append(
            {
                "requested": str(pd.Timestamp(requested).date()),
                "date": str(prior.index[-1].date()),
                "points": [
                    {
                        "id": c,
                        "tenor_years": float(tenors_years[c]),
                        "yield": None if pd.isna(row[c]) else float(row[c]),
                    }
                    for c in cols
                ],
            }
        )
    return out


def curve_shape_history(curve_panel: pd.DataFrame) -> pd.DataFrame:
    """Standard summaries of curve shape through time, in percentage points.

    Columns (each only when its inputs exist):
    ``spread_2s10s`` (10Y-2Y), ``spread_3m10y`` (10Y-3M), ``spread_5s30s``
    (30Y-5Y), ``butterfly_2_5_10`` (2*5Y - 2Y - 10Y), ``level`` (mean of
    2Y, 5Y, 10Y). Positive butterfly means the belly is cheap relative to wings.
    """
    if curve_panel.empty:
        return pd.DataFrame()
    c = curve_panel
    out: Dict[str, pd.Series] = {}
    if {"dgs2", "dgs10"} <= set(c.columns):
        out["spread_2s10s"] = c["dgs10"] - c["dgs2"]
    if {"dgs3mo", "dgs10"} <= set(c.columns):
        out["spread_3m10y"] = c["dgs10"] - c["dgs3mo"]
    if {"dgs5", "dgs30"} <= set(c.columns):
        out["spread_5s30s"] = c["dgs30"] - c["dgs5"]
    if {"dgs2", "dgs5", "dgs10"} <= set(c.columns):
        out["butterfly_2_5_10"] = 2 * c["dgs5"] - c["dgs2"] - c["dgs10"]
        out["level"] = (c["dgs2"] + c["dgs5"] + c["dgs10"]) / 3.0
    return pd.DataFrame(out).dropna(how="all")


# --------------------------------------------------------------------------- freshness
@dataclass(frozen=True)
class StalenessReport:
    """Whether a series is still arriving on schedule.

    ``expected_max_gap_days`` is how old the last observation may be before
    the series counts as late (cadence plus publication lag plus slack);
    ``stale`` is twice that. ``empty`` means there is nothing to judge.
    """

    series_id: str
    last_date: Optional[str]
    days_since_last: Optional[int]
    expected_max_gap_days: int
    status: StaleStatus

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def staleness(
    values: pd.Series,
    series_id: str,
    as_of: pd.Timestamp,
    expected_max_gap_days: int,
) -> StalenessReport:
    """Judge whether a series has stopped arriving.

    Args:
        values: Level series (NaNs ignored).
        series_id: Label for the report.
        as_of: The clock to measure against (tz-naive calendar date).
        expected_max_gap_days: Age at which the series is 'late'.

    Returns:
        :class:`StalenessReport`. ``fresh`` within the expected gap, ``late``
        up to twice it, ``stale`` beyond, ``empty`` with no observations.
    """
    clean = values.dropna()
    if clean.empty:
        return StalenessReport(series_id, None, None, expected_max_gap_days, "empty")
    last = pd.Timestamp(clean.index.max())
    if last.tzinfo is not None:
        last = last.tz_convert("UTC").tz_localize(None)
    days = int((pd.Timestamp(as_of).normalize() - last.normalize()).days)
    if days <= expected_max_gap_days:
        status: StaleStatus = "fresh"
    elif days <= 2 * expected_max_gap_days:
        status = "late"
    else:
        status = "stale"
    return StalenessReport(series_id, str(last.date()), days, expected_max_gap_days, status)
