"""The data-monitor catalog: every series the Data Monitor page can inspect.

This is the glue between two catalogs that live with their vendors -
``COMMODITIES_CONFIG`` (FMP) and ``FRED_SERIES_CATALOG`` - and the EDA
functions in :mod:`core.research.eda`. It decides, per series, *how* to look
at it: which transform makes the distribution meaningful, and how old the
latest point may be before the series counts as late.

Pure: every function takes Series/DataFrames and returns plain data. Loading
the panels is the caller's job (the API route, a notebook, the watchdog).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Literal, Optional

import pandas as pd

from core.data.factors.macro_catalog import FRED_SERIES_CATALOG, TREASURY_CURVE_TENORS_YEARS
from core.data.vendors.commodities import COMMODITIES_CONFIG
from core.research import eda

Source = Literal["fmp", "fred"]

# Display order for groups; anything not listed sorts last.
GROUP_ORDER: tuple[str, ...] = (
    "commodities",
    "curve",
    "rates",
    "credit",
    "inflation",
    "labor",
    "activity",
    "money",
    "markets",
)

GROUP_LABELS: Dict[str, str] = {
    "commodities": "Commodities",
    "curve": "Treasury curve",
    "rates": "Policy & real rates",
    "credit": "Credit spreads",
    "inflation": "Inflation",
    "labor": "Labor",
    "activity": "Activity",
    "money": "Money & Fed",
    "markets": "Market reference",
}

# Slack added to the publication lag before a series is called 'late'. Daily
# series get a long weekend plus a holiday; weekly one missed print; monthly one
# full cycle, because the lag already covers the normal release delay.
_CADENCE_SLACK_DAYS: Dict[str, int] = {"daily": 6, "weekly": 14, "monthly": 31}

# Rolling-window length (observations) used by the level chart, by cadence.
_ROLLING_WINDOW: Dict[str, int] = {"daily": 252, "weekly": 52, "monthly": 36}


@dataclass(frozen=True)
class MonitoredSeries:
    """One series the monitor knows how to display.

    Attributes:
        id: Column name in the panel it comes from.
        name: Human label.
        group: One of :data:`GROUP_ORDER`.
        source: Vendor.
        unit: Unit of the level series.
        frequency: Native cadence.
        default_transform: Transform whose distribution is the informative one
            (returns for prices, changes for rates, the level for monthly rates
            like unemployment or YoY inflation).
        lag_days: Publication lag applied downstream (0 for prices).
        expected_max_gap_days: Age of the last observation at which the series
            is 'late'; twice this is 'stale'.
        rolling_window: Observations in the rolling band on the level chart.
    """

    id: str
    name: str
    group: str
    source: Source
    unit: str
    frequency: str
    default_transform: eda.Transform
    lag_days: int
    expected_max_gap_days: int
    rolling_window: int

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _fred_default_transform(group: str, frequency: str) -> eda.Transform:
    if frequency != "daily":
        return "level"
    if group == "markets":
        # VIX and the dollar index are level-meaningful; WTI is a price.
        return "level"
    return "diff"


def monitored_series_catalog() -> List[MonitoredSeries]:
    """Every commodity and FRED series, in display order."""
    out: List[MonitoredSeries] = []
    for symbol, cfg in COMMODITIES_CONFIG.items():
        out.append(
            MonitoredSeries(
                id=symbol,
                name=str(cfg.get("name", symbol)),
                group="commodities",
                source="fmp",
                unit=str(cfg.get("unit", "USD")),
                frequency="daily",
                default_transform="log_return",
                lag_days=0,
                expected_max_gap_days=_CADENCE_SLACK_DAYS["daily"],
                rolling_window=_ROLLING_WINDOW["daily"],
            )
        )
    for series_id, spec in FRED_SERIES_CATALOG.items():
        transform = _fred_default_transform(spec.group, spec.frequency)
        if series_id == "wti":
            transform = "log_return"
        out.append(
            MonitoredSeries(
                id=series_id,
                name=spec.name,
                group=spec.group,
                source="fred",
                unit=spec.unit,
                frequency=spec.frequency,
                default_transform=transform,
                lag_days=spec.lag_days,
                expected_max_gap_days=spec.lag_days + _CADENCE_SLACK_DAYS[spec.frequency],
                rolling_window=_ROLLING_WINDOW[spec.frequency],
            )
        )
    rank = {g: i for i, g in enumerate(GROUP_ORDER)}
    out.sort(key=lambda s: (rank.get(s.group, len(rank)), s.id))
    return out


def find_series(series_id: str) -> Optional[MonitoredSeries]:
    """Catalog entry by id, or None."""
    return next((s for s in monitored_series_catalog() if s.id == series_id), None)


def pivot_raw_macro(raw_long: pd.DataFrame) -> pd.DataFrame:
    """Raw FRED long table -> wide panel at native frequency, indexed by reference_date.

    This is the right input for distributions and seasonality: the derived
    ``macro.parquet`` is forward-filled to business days, which would count
    every monthly value ~21 times and make every monthly series look 'fresh'.
    """
    if raw_long.empty:
        return pd.DataFrame()
    wide = raw_long.pivot(index="reference_date", columns="series_id", values="value").sort_index()
    wide.index = pd.to_datetime(wide.index)
    wide.index.name = "reference_date"
    wide.columns.name = None
    return wide


def _downsample(frame: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(frame) <= max_points:
        return frame
    step = int(-(-len(frame) // max_points))  # ceil
    return pd.concat([frame.iloc[::step], frame.iloc[[-1]]]).loc[
        lambda f: ~f.index.duplicated(keep="last")
    ]


def series_report(
    values: pd.Series,
    spec: MonitoredSeries,
    as_of: pd.Timestamp,
    transform: Optional[eda.Transform] = None,
    max_points: int = 1500,
    bins: int = 50,
) -> Dict[str, object]:
    """Everything the monitor page shows for one series, as plain data.

    Args:
        values: Level series at native frequency.
        spec: Catalog entry.
        as_of: Clock for the staleness check.
        transform: Override the catalog default.
        max_points: Cap on level-chart points (uniformly thinned, last kept).
        bins: Histogram bins.

    Returns:
        Dict with ``series`` (catalog entry), ``transform``, ``summary``,
        ``histogram``, ``levels`` (rolling profile rows), ``seasonality``,
        ``annual_paths``, ``staleness`` and ``methodology``.
    """
    used: eda.Transform = transform or spec.default_transform
    transformed = eda.apply_transform(values, used)
    summary = eda.summarize_distribution(transformed)
    profile = _downsample(eda.rolling_profile(values, window=spec.rolling_window), max_points)
    levels = [
        {
            "date": str(ts.date()),
            **{k: (None if pd.isna(v) else float(v)) for k, v in row.items()},
        }
        for ts, row in profile.iterrows()
    ]
    normalize = spec.default_transform in ("pct_change", "log_return")
    return {
        "series": spec.to_dict(),
        "transform": used,
        "summary": summary.to_dict(),
        "histogram": eda.histogram(transformed, bins=bins, mark=summary.last_value),
        "levels": levels,
        "seasonality": eda.seasonality_table(values, used),
        "annual_paths": eda.annual_paths(values, normalize=normalize),
        "staleness": eda.staleness(values, spec.id, as_of, spec.expected_max_gap_days).to_dict(),
        "methodology": {
            "transform": used,
            "distribution_sample": (
                f"{summary.n} {spec.frequency} observations"
                + (f" from {summary.start} to {summary.end}" if summary.start else "")
            ),
            "rolling_window": f"{spec.rolling_window} {spec.frequency} observations, ±2σ band",
            "seasonality_cell": {
                "level": "mean of the month's observations",
                "diff": "month-end to month-end change",
                "pct_change": "month-end to month-end simple return",
                "log_return": "month-end to month-end log return",
            }[used],
            "annual_paths": (
                "each year rebased to 100 at its first observation"
                if normalize
                else "raw levels, one line per year"
            ),
            "stationarity": "augmented Dickey-Fuller on the transformed series, AIC lag selection",
            "freshness": (
                f"late after {spec.expected_max_gap_days} days without a new observation "
                f"(publication lag {spec.lag_days}d + cadence slack); stale after twice that"
            ),
            "dates": (
                "FRED values are shown on their reference date (the period they describe), "
                "not the publication date"
                if spec.source == "fred"
                else "close prices on their trading date"
            ),
        },
    }


def staleness_board(
    panels: Dict[str, pd.DataFrame], as_of: pd.Timestamp
) -> List[Dict[str, object]]:
    """Freshness of every catalogued series, worst first.

    Args:
        panels: ``{"fmp": commodity panel, "fred": raw macro pivot}`` - wide
            frames keyed by source. A series whose panel or column is missing
            reports ``empty``.
        as_of: The clock.
    """
    rows: List[Dict[str, object]] = []
    for spec in monitored_series_catalog():
        panel = panels.get(spec.source)
        values = (
            panel[spec.id]
            if panel is not None and spec.id in panel.columns
            else pd.Series(dtype="float64")
        )
        report = eda.staleness(values, spec.id, as_of, spec.expected_max_gap_days)
        rows.append(
            {
                **report.to_dict(),
                "name": spec.name,
                "group": spec.group,
                "source": spec.source,
                "frequency": spec.frequency,
            }
        )
    order = {"stale": 0, "empty": 1, "late": 2, "fresh": 3}
    rows.sort(key=lambda r: (order[str(r["status"])], -(r["days_since_last"] or 0)))
    return rows


def curve_report(raw_macro_wide: pd.DataFrame, dates: List[pd.Timestamp]) -> Dict[str, object]:
    """Yield-curve snapshots and shape history from the raw FRED pivot."""
    tenors = {k: v for k, v in TREASURY_CURVE_TENORS_YEARS.items() if k in raw_macro_wide.columns}
    curve = raw_macro_wide[list(tenors)] if tenors else pd.DataFrame()
    shape = eda.curve_shape_history(curve)
    shape = _downsample(shape, 1500)
    return {
        "tenors": [
            {"id": k, "tenor_years": v, "name": FRED_SERIES_CATALOG[k].name}
            for k, v in tenors.items()
        ],
        "snapshots": eda.yield_curve_snapshots(curve, tenors, dates),
        "shape_history": [
            {
                "date": str(ts.date()),
                **{k: (None if pd.isna(v) else float(v)) for k, v in row.items()},
            }
            for ts, row in shape.iterrows()
        ],
        "methodology": {
            "source": "FRED constant-maturity Treasury yields (DGS*), percent, on the observation date",
            "snapshot_rule": "each requested date resolves to the last observation at or before it",
            "spreads": "2s10s = 10Y-2Y; 3m10y = 10Y-3M; 5s30s = 30Y-5Y; butterfly = 2×5Y-2Y-10Y",
        },
    }


__all__ = [
    "GROUP_ORDER",
    "GROUP_LABELS",
    "MonitoredSeries",
    "monitored_series_catalog",
    "find_series",
    "pivot_raw_macro",
    "series_report",
    "staleness_board",
    "curve_report",
]
