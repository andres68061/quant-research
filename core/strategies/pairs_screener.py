"""Walk-forward pairs universe screener.

Candidates are filtered on a **train** window (return correlation + Engle–Granger
ADF), then ranked by **out-of-sample** pairs-backtest Sharpe on a held-out
window. This avoids fitting and evaluating on the same dates.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

from core.metrics.performance import calculate_performance_metrics
from core.signals.pairs import find_cointegrated_candidates
from core.strategies.pairs_runner import run_pairs_cointegration_backtest

logger = logging.getLogger(__name__)

__all__ = ["resolve_sector_symbols", "screen_pairs_walk_forward"]


def resolve_sector_symbols(
    sectors: pd.DataFrame,
    sector_name: str,
    *,
    price_columns: list[str],
    max_symbols: int = 12,
) -> list[str]:
    """
    Return up to ``max_symbols`` equity tickers in ``sector_name`` that exist
    in the price panel.

    Args:
        sectors: Sector classification table with ``symbol`` and ``sector``.
        sector_name: Exact sector label (e.g. ``Consumer Defensive``).
        price_columns: Available price-panel symbols.
        max_symbols: Cap to keep pair count tractable (C(n,2) grows fast).
    """
    if max_symbols < 2:
        raise ValueError("max_symbols must be >= 2")
    price_set = set(price_columns)
    mask = sectors["sector"].astype(str).str.casefold() == sector_name.casefold()
    if "quoteType" in sectors.columns:
        mask &= sectors["quoteType"].astype(str).str.upper().isin({"EQUITY", "STOCK", ""})
    symbols = [
        str(s).upper() for s in sectors.loc[mask, "symbol"].tolist() if str(s).upper() in price_set
    ]
    # Stable order for reproducibility.
    symbols = sorted(set(symbols))[:max_symbols]
    return symbols


def screen_pairs_walk_forward(
    prices: pd.DataFrame,
    symbols: list[str],
    *,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    train_frac: float = 0.6,
    min_train_corr: float = 0.5,
    max_train_adf_pvalue: float = 0.05,
    max_oos_backtests: int = 15,
    hedge_window: int = 252,
    zscore_window: int = 60,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    transaction_cost: float = 0.001,
    signal_lag_days: int = 1,
) -> dict[str, Any]:
    """
    Screen unordered pairs with train-only cointegration, score on OOS PnL.

    Pipeline per pair ``(y, x)``:
      1. Split the common calendar at ``train_frac``.
      2. Train filter: |corr(returns)| >= ``min_train_corr`` and Engle–Granger
         ADF p-value <= ``max_train_adf_pvalue``.
      3. Among passers (best ADF first), run the standard pairs backtest on the
         **test** window only; record OOS Sharpe / return / drawdown.

    Returns:
        Dict with ``split_date``, ``symbols``, ``n_pairs_tested``,
        ``n_pairs_passed_train``, and ``results`` (list of row dicts, best OOS
        Sharpe first).
    """
    if not 0.2 <= train_frac <= 0.8:
        raise ValueError("train_frac must be in [0.2, 0.8]")
    if len(symbols) < 2:
        raise ValueError("Need at least 2 symbols to screen pairs")

    syms = [s.strip().upper() for s in symbols]
    missing = [s for s in syms if s not in prices.columns]
    if missing:
        raise KeyError(f"Symbols missing from price panel: {missing}")

    panel = prices[syms].sort_index()
    if start is not None:
        panel = panel.loc[panel.index >= start]
    if end is not None:
        panel = panel.loc[panel.index <= end]
    if len(panel) < hedge_window + zscore_window + 60:
        raise ValueError("Insufficient history for walk-forward pairs screen")

    split_idx = int(len(panel) * train_frac)
    split_idx = min(max(split_idx, hedge_window + zscore_window), len(panel) - 40)
    split_date = panel.index[split_idx]
    train_panel = panel.iloc[:split_idx]
    test_panel = panel.iloc[split_idx:]

    n_tested = len(syms) * (len(syms) - 1) // 2
    raw_candidates = find_cointegrated_candidates(
        train_panel,
        syms,
        min_corr=min_train_corr,
        max_adf_pvalue=max_train_adf_pvalue,
        min_obs=hedge_window,
    )
    candidates = [
        {
            "symbol_y": c["symbol_y"],
            "symbol_x": c["symbol_x"],
            "train_corr": c["corr"],
            "train_adf_pvalue": c["adf_pvalue"],
            "train_hedge_ratio": c["hedge_ratio"],
        }
        for c in raw_candidates
    ]
    to_backtest = candidates[:max_oos_backtests]

    results: list[dict[str, Any]] = []
    for row in to_backtest:
        try:
            out = run_pairs_cointegration_backtest(
                test_panel,
                symbol_y=row["symbol_y"],
                symbol_x=row["symbol_x"],
                hedge_window=min(hedge_window, max(60, len(test_panel) // 3)),
                zscore_window=min(zscore_window, max(20, len(test_panel) // 5)),
                entry_z=entry_z,
                exit_z=exit_z,
                transaction_cost=transaction_cost,
                signal_lag_days=signal_lag_days,
            )
        except (ValueError, KeyError) as exc:
            logger.info(
                "OOS backtest skipped for %s/%s: %s",
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
                **row,
                "oos_sharpe": float(metrics["sharpe_ratio"]),
                "oos_annualized_return": float(metrics["annualized_return"]),
                "oos_max_drawdown": float(metrics["max_drawdown"]),
                "oos_n_days": int(len(net)),
                "oos_pct_days_in_trade": float(out["diagnostics"]["pct_days_in_trade"]),
            }
        )

    results.sort(key=lambda r: r["oos_sharpe"], reverse=True)
    logger.info(
        "pairs screen: symbols=%d tested=%d train_pass=%d oos_scored=%d split=%s",
        len(syms),
        n_tested,
        len(candidates),
        len(results),
        split_date,
    )
    return {
        "symbols": syms,
        "split_date": str(pd.Timestamp(split_date).date()),
        "train_frac": train_frac,
        "n_pairs_tested": n_tested,
        "n_pairs_passed_train": len(candidates),
        "results": results,
    }
