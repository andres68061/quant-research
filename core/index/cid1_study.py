"""Cid-1 relevance study on the quarterly top-N cap-weighted universe.

Answers "is trailing Cid-1 a relevant characteristic?" with the standard
factor-evaluation toolkit, not cohort anecdotes:

1. **Persistence** — cross-sectional Spearman rank autocorrelation of the
   characteristic between consecutive rebalance dates (a prerequisite: a
   characteristic that reshuffles randomly cannot be traded on).
2. **Information Coefficient** — per rebalance date, Spearman correlation
   between Cid-1 (trailing window ending at the selection date) and the
   *forward* quarter return; mean IC tested with a Newey-West t-stat.
3. **Fama-MacBeth** — per-date cross-sectional OLS of forward returns on
   z-scored ranks of Cid-1 plus controls (``mom_12_1``, ``vol_60d``,
   ``log_market_cap``), so a "Cid-1 effect" that is just momentum × low-vol
   in disguise shows up as such.
4. **Quantile sort** — mean forward return by Cid-1 quantile and the
   top-minus-bottom spread, Newey-West tested.

Plus a start-year sensitivity table (annual cadence per the repo's
walk-forward re-evaluation rule) so the conclusion is not an artifact of
one lucky window. Diagnostic analysis only — not a tradeable backtest; a
promotion to a strategy would go through ``run_factor_cross_section_backtest``.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from core.exceptions import DataSchemaError
from core.index.top500 import build_quarterly_rebalance_dates, select_top_n_by_market_cap
from core.metrics.cross_section import calculate_trailing_cid1_cross_section
from core.metrics.factor_regression import default_hac_lags

__all__ = ["compute_forward_returns", "run_cid1_relevance_study"]

logger = logging.getLogger(__name__)

CONTROL_COLUMNS = ["mom_12_1", "vol_60d", "log_market_cap"]
MIN_SYMBOLS_PER_DATE = 50


def _newey_west_mean(values: pd.Series) -> dict[str, float]:
    """Mean of a (possibly autocorrelated) series with a Newey-West t-stat."""
    import statsmodels.api as sm

    clean = values.dropna()
    n = len(clean)
    if n < 8:
        return {"mean": float("nan"), "tstat": float("nan"), "pvalue": float("nan"), "n_obs": n}
    lags = default_hac_lags(n)
    fit = sm.OLS(clean.to_numpy(dtype=float), np.ones((n, 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": lags}
    )
    return {
        "mean": float(fit.params[0]),
        "tstat": float(fit.tvalues[0]),
        "pvalue": float(fit.pvalues[0]),
        "n_obs": n,
    }


def _zscore_ranks(values: pd.Series) -> pd.Series:
    """Cross-sectional z-score of average ranks (robust to fat tails / infs)."""
    ranks = values.rank(method="average")
    std = ranks.std()
    if std == 0 or np.isnan(std):
        return ranks * 0.0
    return (ranks - ranks.mean()) / std


def _quantile_buckets(cid1: pd.DataFrame, n_quantiles: int) -> pd.Series:
    """Bucket 0..n-1 by cid1_angle, ties (pain-free paths) broken by return."""
    order = np.lexsort((cid1["total_return"].to_numpy(), cid1["cid1_angle"].to_numpy()))
    positions = np.empty(len(order), dtype=int)
    positions[order] = np.arange(len(order))
    buckets = np.minimum((positions * n_quantiles) // len(order), n_quantiles - 1)
    return pd.Series(buckets, index=cid1.index)


def compute_forward_returns(
    prices: pd.DataFrame,
    rebalance_dates: pd.DatetimeIndex,
    execution_lag_days: int = 1,
    ffill_limit_days: int = 63,
) -> pd.DataFrame:
    """
    Forward per-symbol returns between consecutive execution dates.

    The factor is observed at the selection (quarter-end) close; the return
    it must predict runs from the close ``execution_lag_days`` later to the
    next rebalance's execution close. Prices forward-fill at most
    ``ffill_limit_days`` trading days so a mid-quarter delisting is marked
    at its last close (matching index construction) without resurrecting
    long-dead series.

    Returns:
        Wide DataFrame indexed by selection date (all but the last) with one
        forward-quarter decimal return per symbol (NaN if unpriceable).
    """
    filled = prices.ffill(limit=ffill_limit_days)
    exec_positions: list[int] = []
    selection_dates: list[pd.Timestamp] = []
    for date in rebalance_dates:
        pos = prices.index.get_loc(date) + execution_lag_days
        if pos < len(prices.index):
            exec_positions.append(pos)
            selection_dates.append(date)
    if len(exec_positions) < 2:
        raise DataSchemaError("Need at least 2 executable rebalance dates for forward returns")
    exec_prices = filled.iloc[exec_positions]
    forward = exec_prices.shift(-1) / exec_prices - 1.0
    forward = forward.iloc[:-1]
    forward.index = pd.DatetimeIndex(selection_dates[:-1], name="date")
    return forward


def run_cid1_relevance_study(
    prices: pd.DataFrame,
    market_caps: pd.DataFrame,
    factors: pd.DataFrame,
    start: pd.Timestamp,
    end: Optional[pd.Timestamp] = None,
    top_n: int = 500,
    window_days: int = 252,
    n_quantiles: int = 5,
    execution_lag_days: int = 1,
) -> dict[str, Any]:
    """
    Run the full Cid-1 relevance study on the quarterly top-N universe.

    Args:
        prices: Wide adj_close panel (tz-aware NY index).
        market_caps: Wide market-cap panel on the same calendar.
        factors: Long panel (date, symbol) containing ``CONTROL_COLUMNS``.
        start: First selection date considered (backtest clock start).
        end: Last date considered (default: end of price history).
        top_n: Universe size at each rebalance.
        window_days: Trailing Cid-1 window in trading days.
        n_quantiles: Buckets for the quantile sort.
        execution_lag_days: Signal lag between selection and execution.

    Returns:
        Dict with ``per_date`` (DataFrame), ``quantile_mean_returns``,
        ``ic_summary``, ``persistence_summary``, ``quantile_spread_summary``,
        ``fama_macbeth`` (univariate + multivariate), ``start_year_sensitivity``
        (DataFrame) and ``config``.
    """
    import statsmodels.api as sm

    rebalance_dates = build_quarterly_rebalance_dates(prices.index, start=start, end=end)
    if len(rebalance_dates) < 8:
        raise DataSchemaError(
            f"Only {len(rebalance_dates)} rebalance dates in [{start}, {end}] — "
            "too few cross-sections for inference"
        )
    forward_returns = compute_forward_returns(prices, rebalance_dates, execution_lag_days)
    cid1_panel = calculate_trailing_cid1_cross_section(prices, forward_returns.index, window_days)

    per_date_rows: dict[pd.Timestamp, dict[str, float]] = {}
    quantile_rows: dict[pd.Timestamp, pd.Series] = {}
    fmb_uni: dict[pd.Timestamp, float] = {}
    fmb_multi: dict[pd.Timestamp, dict[str, float]] = {}
    prev_angle: Optional[pd.Series] = None

    for date in forward_returns.index:
        universe = select_top_n_by_market_cap(market_caps, date, top_n).index
        cid1_date = cid1_panel.xs(date, level="date")
        cid1_date = cid1_date[cid1_date.index.isin(universe)]
        fwd = forward_returns.loc[date].reindex(cid1_date.index).dropna()
        cid1_date = cid1_date.loc[fwd.index]
        if len(cid1_date) < MIN_SYMBOLS_PER_DATE:
            logger.warning(
                "cid1 study %s: only %d usable symbols — skipping date",
                date.date(),
                len(cid1_date),
            )
            continue

        angle = cid1_date["cid1_angle"]
        row: dict[str, float] = {
            "n_symbols": float(len(cid1_date)),
            "median_cid1_angle": float(angle.median()),
            "share_pain_free": float((cid1_date["cost_basis_pain"] == 0).mean()),
        }
        row["ic"] = float(spearmanr(angle, fwd)[0])
        if prev_angle is not None:
            common = angle.index.intersection(prev_angle.index)
            if len(common) >= MIN_SYMBOLS_PER_DATE:
                row["persistence_vs_prev"] = float(spearmanr(angle[common], prev_angle[common])[0])
        prev_angle = angle

        buckets = _quantile_buckets(cid1_date, n_quantiles)
        bucket_means = fwd.groupby(buckets).mean()
        quantile_rows[date] = bucket_means
        if 0 in bucket_means.index and n_quantiles - 1 in bucket_means.index:
            row["quantile_spread"] = float(bucket_means[n_quantiles - 1] - bucket_means[0])

        controls = factors.xs(date, level="date")[CONTROL_COLUMNS]
        design = pd.DataFrame({"cid1": _zscore_ranks(angle)})
        for col in CONTROL_COLUMNS:
            design[col] = _zscore_ranks(controls[col].reindex(design.index))
        design = design.dropna()
        y = fwd.loc[design.index]
        if len(design) >= MIN_SYMBOLS_PER_DATE:
            x_uni = sm.add_constant(design[["cid1"]].to_numpy(dtype=float))
            fmb_uni[date] = float(sm.OLS(y.to_numpy(dtype=float), x_uni).fit().params[1])
            x_multi = sm.add_constant(design.to_numpy(dtype=float))
            params = sm.OLS(y.to_numpy(dtype=float), x_multi).fit().params[1:]
            fmb_multi[date] = {
                name: float(p) for name, p in zip(design.columns, params, strict=True)
            }
        per_date_rows[date] = row

    per_date = pd.DataFrame.from_dict(per_date_rows, orient="index").sort_index()
    per_date.index.name = "date"
    if per_date.empty:
        raise DataSchemaError("No rebalance date had enough usable symbols")
    quantile_means = pd.DataFrame(quantile_rows).T.sort_index()
    fmb_multi_frame = pd.DataFrame.from_dict(fmb_multi, orient="index").sort_index()

    sensitivity_rows: list[dict[str, Any]] = []
    for start_year in range(per_date.index[0].year, per_date.index[-1].year - 4):
        subset = per_date[per_date.index.year >= start_year]
        if len(subset) < 12:
            continue
        ic_stats = _newey_west_mean(subset["ic"])
        spread_stats = _newey_west_mean(subset["quantile_spread"])
        sensitivity_rows.append(
            {
                "start_year": start_year,
                "n_quarters": int(len(subset)),
                "mean_ic": ic_stats["mean"],
                "ic_tstat": ic_stats["tstat"],
                "mean_spread": spread_stats["mean"],
                "spread_tstat": spread_stats["tstat"],
            }
        )

    return {
        "per_date": per_date,
        "quantile_mean_returns": quantile_means,
        "ic_summary": _newey_west_mean(per_date["ic"]),
        "persistence_summary": _newey_west_mean(
            per_date.get("persistence_vs_prev", pd.Series(dtype=float))
        ),
        "quantile_spread_summary": _newey_west_mean(
            per_date.get("quantile_spread", pd.Series(dtype=float))
        ),
        "fama_macbeth": {
            "univariate_cid1": _newey_west_mean(pd.Series(fmb_uni)),
            "multivariate": {
                col: _newey_west_mean(fmb_multi_frame[col]) for col in fmb_multi_frame.columns
            },
        },
        "start_year_sensitivity": pd.DataFrame(sensitivity_rows),
        "config": {
            "top_n": top_n,
            "window_days": window_days,
            "n_quantiles": n_quantiles,
            "execution_lag_days": execution_lag_days,
            "start": str(rebalance_dates[0].date()),
            "end": str(rebalance_dates[-1].date()),
            "n_rebalances": int(len(rebalance_dates)),
            "controls": CONTROL_COLUMNS,
        },
    }
