"""Gatev–Goetzmann–Rouwenhorst distance pairs formation.

Formation: normalize each price series to 1.0 at the start of the formation
window (cumulative total-return index style), rank unordered pairs by sum of
squared deviations (SSD), keep the closest ``top_n``.

Trading: run the platform's rolling-hedge / z-score pairs backtest on the
**subsequent** trading window only (no formation-window PnL).
"""

from __future__ import annotations

import itertools
import logging
from typing import Any, Optional

import numpy as np
import pandas as pd

from core.metrics.performance import calculate_performance_metrics
from core.strategies.pairs_runner import run_pairs_cointegration_backtest

logger = logging.getLogger(__name__)

__all__ = [
    "normalize_price_index",
    "pair_sum_squared_deviations",
    "form_pairs_by_distance",
    "resolve_liquid_symbols",
    "screen_pairs_gatev",
]


def normalize_price_index(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize each column to 1.0 at the first non-NaN observation.

    Gatev et al. form pairs in cumulative total-return index space; with
    adjusted closes this is equivalent to ``P_t / P_0``.
    """
    if prices.empty:
        return prices.copy()
    first = prices.apply(lambda col: col.dropna().iloc[0] if col.notna().any() else np.nan)
    return prices.divide(first.replace(0.0, np.nan), axis=1)


def pair_sum_squared_deviations(
    norm_a: pd.Series,
    norm_b: pd.Series,
) -> float:
    """SSD between two normalized price series on their common dates."""
    aligned = pd.concat([norm_a, norm_b], axis=1).dropna()
    if len(aligned) < 2:
        return float("inf")
    diff = aligned.iloc[:, 0].to_numpy(dtype=float) - aligned.iloc[:, 1].to_numpy(dtype=float)
    return float(np.dot(diff, diff))


def form_pairs_by_distance(
    formation_prices: pd.DataFrame,
    symbols: list[str],
    *,
    top_n: int = 20,
    min_overlap: int = 60,
) -> list[dict[str, Any]]:
    """
    Rank all unordered pairs by formation-period SSD (ascending).

    Returns list of ``{symbol_y, symbol_x, formation_ssd, n_overlap}``.
    """
    if top_n < 1:
        raise ValueError("top_n must be >= 1")
    syms = [s for s in symbols if s in formation_prices.columns]
    if len(syms) < 2:
        return []

    panel = formation_prices[syms].dropna(how="all")
    ok = [s for s in syms if int(panel[s].notna().sum()) >= min_overlap]
    if len(ok) < 2:
        return []

    norm = normalize_price_index(panel[ok])
    ranked: list[dict[str, Any]] = []
    for a, b in itertools.combinations(ok, 2):
        pair = norm[[a, b]].dropna()
        if len(pair) < min_overlap:
            continue
        ssd = pair_sum_squared_deviations(pair[a], pair[b])
        if not np.isfinite(ssd):
            continue
        ranked.append(
            {
                "symbol_y": a,
                "symbol_x": b,
                "formation_ssd": ssd,
                "n_overlap": int(len(pair)),
            }
        )
    ranked.sort(key=lambda r: r["formation_ssd"])
    return ranked[:top_n]


def resolve_liquid_symbols(
    sectors: pd.DataFrame,
    sector_name: str,
    *,
    price_columns: list[str],
    dollar_adv: Optional[pd.DataFrame] = None,
    max_symbols: int = 12,
) -> list[str]:
    """
    Same-sector universe ranked by mean dollar ADV when available.

    Falls back to alphabetical order only if ADV is missing (discouraged).
    """
    if max_symbols < 2:
        raise ValueError("max_symbols must be >= 2")
    price_set = set(price_columns)
    mask = sectors["sector"].astype(str).str.casefold() == sector_name.casefold()
    if "quoteType" in sectors.columns:
        mask &= sectors["quoteType"].astype(str).str.upper().isin({"EQUITY", "STOCK", ""})
    candidates = sorted(
        {
            str(s).upper()
            for s in sectors.loc[mask, "symbol"].tolist()
            if str(s).upper() in price_set
        }
    )
    if not candidates:
        return []
    if dollar_adv is None or dollar_adv.empty:
        logger.warning("No dollar_adv for liquid universe; using alphabetical sector cap")
        return candidates[:max_symbols]

    cols = [c for c in candidates if c in dollar_adv.columns]
    if not cols:
        return candidates[:max_symbols]
    mean_adv = dollar_adv[cols].mean(skipna=True).sort_values(ascending=False)
    return [str(s) for s in mean_adv.head(max_symbols).index.tolist()]


def screen_pairs_gatev(
    prices: pd.DataFrame,
    symbols: list[str],
    *,
    formation_frac: float = 0.67,
    top_n: int = 15,
    min_overlap: int = 60,
    hedge_window: int = 252,
    zscore_window: int = 60,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    transaction_cost: float = 0.001,
    signal_lag_days: int = 1,
) -> dict[str, Any]:
    """
    Gatev distance formation on the first ``formation_frac`` of the panel,
    then OOS pairs backtest on the remainder for the top-SSD pairs.
    """
    if not 0.4 <= formation_frac <= 0.85:
        raise ValueError("formation_frac must be in [0.4, 0.85]")
    syms = [s.strip().upper() for s in symbols]
    missing = [s for s in syms if s not in prices.columns]
    if missing:
        raise KeyError(f"Symbols missing from price panel: {missing}")

    panel = prices[syms].sort_index()
    if len(panel) < min_overlap * 2:
        raise ValueError("Insufficient history for Gatev formation + trading")

    split_idx = int(len(panel) * formation_frac)
    split_idx = min(max(split_idx, min_overlap), len(panel) - 40)
    formation = panel.iloc[:split_idx]
    trading = panel.iloc[split_idx:]
    split_date = panel.index[split_idx]

    formed = form_pairs_by_distance(formation, syms, top_n=top_n, min_overlap=min_overlap)

    results: list[dict[str, Any]] = []
    for row in formed:
        try:
            out = run_pairs_cointegration_backtest(
                trading,
                symbol_y=row["symbol_y"],
                symbol_x=row["symbol_x"],
                hedge_window=min(hedge_window, max(60, len(trading) // 3)),
                zscore_window=min(zscore_window, max(20, len(trading) // 5)),
                entry_z=entry_z,
                exit_z=exit_z,
                transaction_cost=transaction_cost,
                signal_lag_days=signal_lag_days,
            )
        except (ValueError, KeyError) as exc:
            logger.info(
                "Gatev OOS skip %s/%s: %s",
                row["symbol_y"],
                row["symbol_x"],
                exc,
            )
            continue
        net = out["net_returns"]
        if len(net) < 20:
            continue
        metrics = calculate_performance_metrics(net)
        results.append(
            {
                "symbol_y": row["symbol_y"],
                "symbol_x": row["symbol_x"],
                "formation_ssd": float(row["formation_ssd"]),
                "train_corr": None,
                "train_adf_pvalue": None,
                "train_hedge_ratio": None,
                "oos_sharpe": float(metrics["sharpe_ratio"]),
                "oos_annualized_return": float(metrics["annualized_return"]),
                "oos_max_drawdown": float(metrics["max_drawdown"]),
                "oos_n_days": int(len(net)),
                "oos_pct_days_in_trade": float(out["diagnostics"]["pct_days_in_trade"]),
            }
        )

    results.sort(key=lambda r: r["oos_sharpe"], reverse=True)
    logger.info(
        "Gatev screen: symbols=%d formed=%d oos_scored=%d split=%s",
        len(syms),
        len(formed),
        len(results),
        split_date,
    )
    return {
        "symbols": syms,
        "split_date": str(pd.Timestamp(split_date).date()),
        "train_frac": formation_frac,
        "method": "gatev",
        "n_pairs_tested": int(len(syms) * (len(syms) - 1) // 2),
        "n_pairs_passed_train": len(formed),
        "results": results,
    }
