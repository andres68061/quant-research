"""Composite accounting scores: Piotroski F-score and Altman Z-score.

Both are published, fully specified scoring rules — no fitted parameters, no
cross-sectional standardisation. They are computed from the same statement metric
frame as the ratio factors, so they inherit its point-in-time alignment.

The F-score is deliberately built from **nine independent binary tests** rather
than a z-scored composite: that is the published construction, and it is what
makes the score comparable across sectors and decades.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

PIOTROSKI_SIGNAL_COLUMNS: tuple[str, ...] = (
    "f_roa_positive",
    "f_cfo_positive",
    "f_roa_improving",
    "f_accrual_quality",
    "f_leverage_falling",
    "f_liquidity_improving",
    "f_no_dilution",
    "f_margin_improving",
    "f_turnover_improving",
)

MIN_SIGNALS_FOR_SCORE = 7


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Ratio that is NaN wherever the denominator is not strictly positive."""
    denom = denominator.astype("float64")
    return numerator.astype("float64") / denom.where(denom > 0)


def compute_piotroski_signals(metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate the nine Piotroski (2000) F-score tests.

    Each signal is 1.0 when the test passes, 0.0 when it fails, and NaN when the
    inputs are missing — a missing input is not a failed test, and scoring it as
    zero would systematically penalise companies with short histories.

    Args:
        metrics: Frame carrying the statement metrics and their ``*_lag4``
            counterparts (see
            :mod:`core.data.factors.statement_metrics`).

    Returns:
        DataFrame indexed like ``metrics`` with :data:`PIOTROSKI_SIGNAL_COLUMNS`,
        each float64 in {0.0, 1.0, NaN}.
    """
    # Piotroski scales flow items by total assets at the START of the year, not by
    # contemporaneous assets. Using contemporaneous assets makes a company that
    # grew its balance sheet look like it lost profitability, which flips the
    # ROA-improving and turnover-improving tests on fast-growing firms.
    opening_assets = metrics["total_assets_lag4"]
    prior_opening_assets = metrics["total_assets_lag8"]

    roa = _safe_ratio(metrics["net_income_ttm"], opening_assets)
    prior_roa = _safe_ratio(metrics["net_income_ttm_lag4"], prior_opening_assets)
    cfo_scaled = _safe_ratio(metrics["cfo_ttm"], opening_assets)

    # Leverage change is scaled by average total assets in the original.
    average_assets = (metrics["total_assets"] + opening_assets) / 2.0
    prior_average_assets = (opening_assets + prior_opening_assets) / 2.0
    leverage = _safe_ratio(metrics["long_term_debt"], average_assets)
    prior_leverage = _safe_ratio(metrics["long_term_debt_lag4"], prior_average_assets)

    current_ratio = _safe_ratio(
        metrics["total_current_assets"], metrics["total_current_liabilities"]
    )
    prior_current_ratio = _safe_ratio(
        metrics["total_current_assets_lag4"], metrics["total_current_liabilities_lag4"]
    )
    gross_margin = _safe_ratio(metrics["gross_profit_ttm"], metrics["revenue_ttm"])
    prior_gross_margin = _safe_ratio(metrics["gross_profit_ttm_lag4"], metrics["revenue_ttm_lag4"])
    asset_turnover = _safe_ratio(metrics["revenue_ttm"], opening_assets)
    prior_asset_turnover = _safe_ratio(metrics["revenue_ttm_lag4"], prior_opening_assets)

    def indicator(condition: pd.Series, *inputs: pd.Series) -> pd.Series:
        """1/0 where every input is present, NaN where any is missing."""
        observed = pd.concat(inputs, axis=1).notna().all(axis=1)
        return condition.astype("float64").where(observed)

    signals = pd.DataFrame(index=metrics.index)
    signals["f_roa_positive"] = indicator(roa > 0, roa)
    signals["f_cfo_positive"] = indicator(metrics["cfo_ttm"] > 0, metrics["cfo_ttm"])
    signals["f_roa_improving"] = indicator(roa > prior_roa, roa, prior_roa)
    signals["f_accrual_quality"] = indicator(cfo_scaled > roa, cfo_scaled, roa)
    # Falling leverage and rising liquidity are the "good" directions.
    signals["f_leverage_falling"] = indicator(leverage < prior_leverage, leverage, prior_leverage)
    signals["f_liquidity_improving"] = indicator(
        current_ratio > prior_current_ratio, current_ratio, prior_current_ratio
    )
    signals["f_no_dilution"] = indicator(
        metrics["shares_diluted"] <= metrics["shares_diluted_lag4"],
        metrics["shares_diluted"],
        metrics["shares_diluted_lag4"],
    )
    signals["f_margin_improving"] = indicator(
        gross_margin > prior_gross_margin, gross_margin, prior_gross_margin
    )
    signals["f_turnover_improving"] = indicator(
        asset_turnover > prior_asset_turnover, asset_turnover, prior_asset_turnover
    )
    return signals[list(PIOTROSKI_SIGNAL_COLUMNS)]


def compute_piotroski_f_score(
    metrics: pd.DataFrame,
    min_signals: int = MIN_SIGNALS_FOR_SCORE,
) -> pd.Series:
    """
    Sum the nine F-score tests into a 0–9 score.

    Args:
        metrics: Statement metric frame (see :func:`compute_piotroski_signals`).
        min_signals: Minimum number of evaluable tests required to report a score.
            Below this the row is NaN: a "3" built from four available tests is
            not comparable with a "3" built from nine.

    Returns:
        float64 Series named ``piotroski_f``, values 0–9 or NaN.
    """
    signals = compute_piotroski_signals(metrics)
    evaluable = signals.notna().sum(axis=1)
    score = signals.sum(axis=1, min_count=1).where(evaluable >= min_signals)
    return score.rename("piotroski_f")


def compute_altman_z_score(pit_panel: pd.DataFrame, market_cap: pd.Series) -> pd.Series:
    """
    Altman (1968) Z-score for publicly traded manufacturers.

    ``Z = 1.2·X1 + 1.4·X2 + 3.3·X3 + 0.6·X4 + 1.0·X5`` where X1 is working capital
    to assets, X2 retained earnings to assets, X3 EBIT to assets, X4 market equity
    to total liabilities, and X5 sales to assets. Below ~1.8 is the distress zone.

    Args:
        pit_panel: Dailyized MultiIndex (date, symbol) panel with
            ``total_assets``, ``total_current_assets``, ``total_current_liabilities``,
            ``retained_earnings``, ``ebit_ttm``, ``total_liabilities``, ``revenue_ttm``.
        market_cap: MultiIndex (date, symbol) market capitalisation.

    Returns:
        float64 Series named ``altman_z``.

    Notes:
        The coefficients were fitted on manufacturers. The score is routinely
        applied more broadly, but it is not meaningful for banks and insurers —
        filter by sector before ranking on it.
    """
    total_assets = pit_panel["total_assets"].astype("float64")
    scale = total_assets.where(total_assets > 0)
    cap = market_cap.reindex(pit_panel.index).astype("float64")
    liabilities = pit_panel["total_liabilities"].astype("float64")

    working_capital = pit_panel["total_current_assets"] - pit_panel["total_current_liabilities"]
    z_score = (
        1.2 * (working_capital / scale)
        + 1.4 * (pit_panel["retained_earnings"] / scale)
        + 3.3 * (pit_panel["ebit_ttm"] / scale)
        + 0.6 * (cap / liabilities.where(liabilities > 0))
        + 1.0 * (pit_panel["revenue_ttm"] / scale)
    )
    return z_score.rename("altman_z")
