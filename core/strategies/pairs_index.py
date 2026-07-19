"""Rolling multi-pair statistical-arbitrage long/short index.

A single cointegrated pair (see ``pairs_runner.py``) is a noisy bet: our own
validated XOM/CVX signal swings from Sharpe +0.89 to -1.11 depending on the
exact window and lookback (see notebook 17). The classic fix from the pairs
literature (Gatev, Goetzmann & Rouwenhorst, 2006) is not a better single
pair — it is to trade a *diversified basket* of many pairs at once and let
idiosyncratic pair risk average out, re-forming the basket on a rolling
schedule so it never trades on stale cointegration.

Formation ranking is Gatev's **distance (SSD)** method, not Engle-Granger
p-value ranking. An earlier version of this module ranked candidates by
formation-window ADF p-value and the resulting basket lost money (real-data
Sharpe -0.48, see notebooks/18): picking the "most significant" p-value out
of 40-60 candidates per period is exactly the multiple-comparisons trap this
platform's own pairs notebook (17) warns about, and it kept selecting
degenerate matches (e.g. GOOGL/GOOG, the same company's two share classes)
that are "cointegrated" by construction but have no tradeable spread left
after costs. SSD requires the *entire price path* to have tracked closely
throughout formation, which is a stronger and less overfit-prone criterion.

Rolling schedule (walk-forward, no lookahead):
    for each non-overlapping period:
        1. FORMATION [t, t+formation_months): rank same-sector pairs by
           sum-of-squared-deviations of normalized prices (Gatev SSD).
        2. Keep the top ``top_n_pairs`` (smallest SSD) across all sectors.
        3. TRADING [t+formation_months, t+formation_months+trading_months):
           run the standard rolling-hedge/z-score backtest for each selected
           pair, then equal-weight blend their net returns into one index
           return for that period.
        4. Roll forward by ``trading_months`` and repeat.

The concatenation of blended period returns is the index. Formation and
trading windows never overlap in time, so no period's pair selection ever
sees the returns it is later scored on.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

from core.metrics.performance import calculate_performance_metrics
from core.signals.pairs import align_pair_log_prices, engle_granger_test
from core.strategies.pairs_gatev import form_pairs_by_distance, resolve_liquid_symbols
from core.strategies.pairs_runner import run_pairs_cointegration_backtest

logger = logging.getLogger(__name__)

__all__ = ["build_pairs_universe", "run_pairs_stat_arb_index"]


def build_pairs_universe(
    sectors: pd.DataFrame,
    sector_names: list[str],
    *,
    price_columns: list[str],
    dollar_adv: Optional[pd.DataFrame] = None,
    max_symbols_per_sector: int = 12,
) -> list[str]:
    """
    Pool ADV-ranked liquid symbols across several sectors (deduplicated).

    Pairs are only ever tested within a sector (see ``run_pairs_stat_arb_index``),
    so pooling here just controls how many sectors' worth of candidates enter
    the formation search per period.
    """
    pooled: list[str] = []
    seen: set[str] = set()
    for sector_name in sector_names:
        for sym in resolve_liquid_symbols(
            sectors,
            sector_name,
            price_columns=price_columns,
            dollar_adv=dollar_adv,
            max_symbols=max_symbols_per_sector,
        ):
            if sym not in seen:
                seen.add(sym)
                pooled.append(sym)
    return pooled


def run_pairs_stat_arb_index(
    prices: pd.DataFrame,
    sectors: pd.DataFrame,
    *,
    sector_names: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    dollar_adv: Optional[pd.DataFrame] = None,
    formation_months: int = 12,
    trading_months: int = 6,
    top_n_pairs: int = 10,
    max_symbols_per_sector: int = 12,
    hedge_window: int = 252,
    zscore_window: int = 60,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    transaction_cost: float = 0.001,
    signal_lag_days: int = 1,
    min_formation_obs: int = 120,
) -> dict[str, Any]:
    """
    Roll a same-sector pairs basket forward and concatenate blended returns.

    Args:
        prices: Wide adj_close panel (date x symbol), tz-aware index.
        sectors: Sector classification table (``symbol``, ``sector``).
        sector_names: Sectors to search for candidates each formation period.
            Pairs are only formed within a sector (economically related legs).
        start: First formation window's start. The index itself only starts
            producing returns at ``start + formation_months`` (warm-up).
        end: Last date in scope; the final trading window is truncated to it.
        dollar_adv: Optional dollar-ADV panel for liquidity ranking.
        formation_months: Trailing lookback used to find candidates.
        trading_months: Holding period before the basket is re-formed.
        top_n_pairs: Max pairs held per trading period (smallest formation
            SSD first, pooled across all sectors); fewer are held if fewer
            sectors produce candidates.
        max_symbols_per_sector: Liquidity cap per sector (keeps C(n,2) sane).
        hedge_window / zscore_window: Rolling lookbacks for each pair's
            trade signal. Seeded from price history before ``trading_start``
            so the lookback is fully warmed up on day one of trading (the
            warm-up period's returns are discarded, not counted in the index).
        entry_z / exit_z / transaction_cost / signal_lag_days: Passed through
            to each pair's backtest.
        min_formation_obs: Minimum overlapping observations required in the
            formation window for a pair to be testable at all.

    Returns:
        Dict with ``net_returns`` (the concatenated index series),
        ``periods`` (per-period diagnostics: window dates, candidates found,
        selected pairs, and each pair's own trading-period Sharpe), and
        ``universe`` (pooled symbols considered).
    """
    if formation_months < 3:
        raise ValueError("formation_months must be >= 3")
    if trading_months < 1:
        raise ValueError("trading_months must be >= 1")
    if top_n_pairs < 1:
        raise ValueError("top_n_pairs must be >= 1")
    if start >= end:
        raise ValueError("start must be before end")
    if formation_months * 20 < min_formation_obs:
        raise ValueError(
            f"formation_months={formation_months} (~{formation_months * 20} trading days) "
            f"is too short for min_formation_obs={min_formation_obs}; every period would "
            "silently form zero candidates. Raise formation_months or lower min_formation_obs."
        )

    panel = prices.sort_index()
    universe_by_sector = {
        sector_name: [
            s
            for s in resolve_liquid_symbols(
                sectors,
                sector_name,
                price_columns=list(panel.columns),
                dollar_adv=dollar_adv,
                max_symbols=max_symbols_per_sector,
            )
        ]
        for sector_name in sector_names
    }

    period_returns: list[pd.Series] = []
    periods: list[dict[str, Any]] = []

    formation_start = start
    while True:
        formation_end = formation_start + pd.DateOffset(months=formation_months)
        trading_start = formation_end
        trading_end = min(trading_start + pd.DateOffset(months=trading_months), end)
        if trading_start >= end or trading_end <= trading_start:
            break

        candidates: list[dict[str, Any]] = []
        for sector_name, syms in universe_by_sector.items():
            if len(syms) < 2:
                continue
            formation_panel = panel.loc[formation_start:formation_end, syms].dropna(how="all")
            if len(formation_panel) < min_formation_obs:
                continue
            found = form_pairs_by_distance(
                formation_panel, syms, top_n=top_n_pairs, min_overlap=min_formation_obs
            )
            for c in found:
                c["sector"] = sector_name
            candidates.extend(found)

        candidates.sort(key=lambda r: r["formation_ssd"])
        selected = candidates[:top_n_pairs]

        # Half-open [trading_start, trading_end) so consecutive periods never
        # double-count the boundary date (trading_end_i == trading_start_{i+1}
        # by construction, and pandas .loc slicing is inclusive on both ends).
        trading_index = panel.index[(panel.index >= trading_start) & (panel.index < trading_end)]
        # Seed the rolling hedge/z-score from history *before* trading_start
        # (a live trader would too) rather than shrinking the lookback to
        # fit inside a short trading window — a 6mo trading window can't
        # otherwise fit a 252d hedge warm-up.
        buffer_days = int((hedge_window + zscore_window) * 1.6) + 5
        warm_start = trading_start - pd.DateOffset(days=buffer_days)

        pair_returns: dict[str, pd.Series] = {}
        selected_rows: list[dict[str, Any]] = []
        for c in selected:
            y, x = c["symbol_y"], c["symbol_x"]
            # EG cointegration is reported for transparency only — SSD (above)
            # drives selection; a NaN p-value here just means the formation
            # window was too short for a reliable ADF fit, not a rejection.
            try:
                log_y, log_x = align_pair_log_prices(panel.loc[formation_start:formation_end], y, x)
                formation_adf_pvalue = engle_granger_test(log_y, log_x)["adf_pvalue"]
            except (ValueError, KeyError):
                formation_adf_pvalue = float("nan")
            try:
                out = run_pairs_cointegration_backtest(
                    panel,
                    symbol_y=y,
                    symbol_x=x,
                    start=warm_start,
                    end=trading_end,
                    hedge_window=hedge_window,
                    zscore_window=zscore_window,
                    entry_z=entry_z,
                    exit_z=exit_z,
                    transaction_cost=transaction_cost,
                    signal_lag_days=signal_lag_days,
                )
            except (ValueError, KeyError) as exc:
                logger.info("Index period skip %s/%s: %s", y, x, exc)
                continue
            net = out["net_returns"]
            net = net.loc[net.index.isin(trading_index)]
            if len(net) < 10:
                continue
            pair_returns[f"{y}/{x}"] = net
            selected_rows.append(
                {
                    "symbol_y": y,
                    "symbol_x": x,
                    "sector": c["sector"],
                    "formation_ssd": c["formation_ssd"],
                    "formation_adf_pvalue": formation_adf_pvalue,
                    "period_sharpe": float(calculate_performance_metrics(net)["sharpe_ratio"]),
                    "period_n_days": int(len(net)),
                }
            )

        if pair_returns:
            blended = pd.DataFrame(pair_returns).mean(axis=1, skipna=True)
            active_pairs = pd.DataFrame(pair_returns).notna().sum(axis=1)
        else:
            blended = pd.Series(0.0, index=trading_index)
            active_pairs = pd.Series(0, index=trading_index)

        period_returns.append(blended)
        periods.append(
            {
                "formation_start": str(pd.Timestamp(formation_start).date()),
                "formation_end": str(pd.Timestamp(formation_end).date()),
                "trading_start": str(pd.Timestamp(trading_start).date()),
                "trading_end": str(pd.Timestamp(trading_end).date()),
                "n_candidates_formed": len(candidates),
                "n_pairs_selected": len(selected_rows),
                "avg_active_pairs": float(active_pairs.mean()) if len(active_pairs) else 0.0,
                "blended_sharpe": (
                    float(calculate_performance_metrics(blended)["sharpe_ratio"])
                    if blended.notna().any()
                    else float("nan")
                ),
                "selected_pairs": selected_rows,
            }
        )

        formation_start = formation_start + pd.DateOffset(months=trading_months)

    net_returns = (
        pd.concat(period_returns).sort_index() if period_returns else pd.Series(dtype=float)
    )
    logger.info(
        "pairs index: sectors=%d periods=%d total_days=%d",
        len(sector_names),
        len(periods),
        len(net_returns),
    )
    return {
        "net_returns": net_returns,
        "periods": periods,
        "universe": sorted({s for syms in universe_by_sector.values() for s in syms}),
        "params": {
            "formation_months": formation_months,
            "trading_months": trading_months,
            "top_n_pairs": top_n_pairs,
            "hedge_window": hedge_window,
            "zscore_window": zscore_window,
            "entry_z": entry_z,
            "exit_z": exit_z,
            "transaction_cost": transaction_cost,
            "signal_lag_days": signal_lag_days,
        },
    }
