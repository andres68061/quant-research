"""Cointegration-persistence pairs index: trade a pair until it stops cointegrating.

Corrects the flawed premise of ``pairs_index.py`` (see
``docs/FAILED_STRATEGIES_LOG.md``): ranking candidates by how tightly two
prices track each other (Gatev SSD, or ADF significance as a proxy for the
same idea) selects pairs with too little deviation to profit from after
costs — exactly the GOOGL/GOOG failure mode (same company's two share
classes, near-zero gross edge). A pair worth trading must actually
*deviate and revert*: its normalized cumulative-return paths should cross
each other repeatedly, not sit on top of one another.

This module differs from ``pairs_index.py`` in two ways:

1. **Candidate filter** requires genuine Engle-Granger cointegration *and*
   a minimum number of sign-changes in the normalized cumulative-return
   difference over the formation window (real oscillation with tradeable
   amplitude).
2. **Trading duration is event-driven, not a fixed calendar window**: each
   selected pair trades from the moment it's chosen until a rolling
   Engle-Granger monitor shows it is no longer cointegrated for
   ``persistence_checks`` consecutive checks (or the backtest ends) — not
   for a synchronized 6-month block regardless of what's actually
   happening to the spread. New formation rounds only fill *free* slots
   left by pairs that have already stopped; a pair that is still working
   keeps trading past its formation round.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

from core.signals.pairs import (
    align_pair_log_prices,
    count_cumulative_return_crossings,
    engle_granger_test,
    find_cointegrated_candidates,
)
from core.strategies.pairs_gatev import normalize_price_index, resolve_liquid_symbols
from core.strategies.pairs_runner import run_pairs_cointegration_backtest

logger = logging.getLogger(__name__)

__all__ = [
    "find_crossing_cointegrated_candidates",
    "run_pair_until_broken",
    "run_pairs_persistent_index",
]


def find_crossing_cointegrated_candidates(
    formation_panel: pd.DataFrame,
    symbols: list[str],
    *,
    min_corr: float = 0.5,
    max_adf_pvalue: float = 0.05,
    min_crossings: int = 3,
    min_obs: int = 120,
) -> list[dict[str, Any]]:
    """
    Cointegrated candidates (see ``find_cointegrated_candidates``), further
    restricted to pairs whose normalized price paths cross at least
    ``min_crossings`` times over the formation window.

    Ranked by ADF p-value ascending among survivors — SSD/distance is not
    used anywhere in this filter, by design.

    Returns:
        List of ``{symbol_y, symbol_x, corr, adf_pvalue, hedge_ratio,
        crossings}``, sorted by ``adf_pvalue`` ascending.
    """
    cointegrated = find_cointegrated_candidates(
        formation_panel, symbols, min_corr=min_corr, max_adf_pvalue=max_adf_pvalue, min_obs=min_obs
    )
    if not cointegrated:
        return []
    norm = normalize_price_index(formation_panel[symbols])

    viable = []
    for c in cointegrated:
        y, x = c["symbol_y"], c["symbol_x"]
        crossings = count_cumulative_return_crossings(norm[y], norm[x])
        if crossings >= min_crossings:
            row = dict(c)
            row["crossings"] = crossings
            viable.append(row)
    viable.sort(key=lambda r: r["adf_pvalue"])
    return viable


def run_pair_until_broken(
    prices: pd.DataFrame,
    *,
    symbol_y: str,
    symbol_x: str,
    start: pd.Timestamp,
    max_end: pd.Timestamp,
    hedge_window: int = 252,
    zscore_window: int = 60,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    transaction_cost: float = 0.001,
    signal_lag_days: int = 1,
    monitor_window: int = 252,
    check_every_days: int = 21,
    max_pvalue: float = 0.10,
    persistence_checks: int = 4,
) -> dict[str, Any]:
    """
    Trade ``[start, max_end]``, but stop the first time a rolling
    Engle-Granger monitor shows the pair is no longer cointegrated for
    ``persistence_checks`` consecutive checks (spaced ``check_every_days``
    apart, each looking back ``monitor_window`` observations).

    The persistence requirement (not a single failed check) exists so one
    noisy month doesn't stop a pair that's still genuinely cointegrated —
    matching the finding (notebook 17) that cointegration significance is
    naturally noisy even for a real, validated pair.

    Returns:
        The usual ``run_pairs_cointegration_backtest`` result dict
        (``net_returns``, ``gross_returns``, ``spread_z``, ``position``,
        ``hedge_ratio``, ``diagnostics``), truncated at the stop point if
        one was found, plus ``stop_date`` (``pd.Timestamp`` or ``None``)
        and ``stopped_early`` (``bool``).
    """
    # Seed the rolling hedge/z-score from history *before* start (same
    # technique as pairs_index.py / run_pairs_holdout_backtest) so trading
    # begins right at `start` instead of ~hedge_window+zscore_window days
    # later -- otherwise most of the "trading" window is silently consumed
    # by warm-up and the persistence monitor ends up looking at a pair that
    # has already drifted well past its formation date.
    trade_buffer_days = int((hedge_window + zscore_window) * 1.6) + 5
    warm_start = start - pd.DateOffset(days=trade_buffer_days)
    out = run_pairs_cointegration_backtest(
        prices,
        symbol_y=symbol_y,
        symbol_x=symbol_x,
        start=warm_start,
        end=max_end,
        hedge_window=hedge_window,
        zscore_window=zscore_window,
        entry_z=entry_z,
        exit_z=exit_z,
        transaction_cost=transaction_cost,
        signal_lag_days=signal_lag_days,
    )
    for key in ("net_returns", "gross_returns", "spread_z", "position", "hedge_ratio"):
        out[key] = out[key].loc[out[key].index >= start]

    idx = out["net_returns"].index
    if len(idx) == 0:
        return {**out, "stop_date": None, "stopped_early": False}

    panel = prices.sort_index()
    buffer_days = int(monitor_window * 1.6) + 5
    consecutive_failures = 0
    stop_date: Optional[pd.Timestamp] = None

    for ts in idx[::check_every_days]:
        window_start = ts - pd.DateOffset(days=buffer_days)
        window_panel = panel.loc[(panel.index > window_start) & (panel.index <= ts)]
        try:
            log_y, log_x = align_pair_log_prices(window_panel, symbol_y, symbol_x)
            eg = engle_granger_test(log_y, log_x)
            broken = eg["adf_pvalue"] > max_pvalue
        except (ValueError, KeyError):
            broken = True

        consecutive_failures = consecutive_failures + 1 if broken else 0
        if consecutive_failures >= persistence_checks:
            stop_date = ts
            break

    stopped_early = stop_date is not None
    if stopped_early:
        for key in ("net_returns", "gross_returns", "spread_z", "position", "hedge_ratio"):
            out[key] = out[key].loc[out[key].index <= stop_date]

    return {**out, "stop_date": stop_date, "stopped_early": stopped_early}


def run_pairs_persistent_index(
    prices: pd.DataFrame,
    sectors: pd.DataFrame,
    *,
    sector_names: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    dollar_adv: Optional[pd.DataFrame] = None,
    formation_months: int = 12,
    top_n_pairs: int = 10,
    max_symbols_per_sector: int = 12,
    min_corr: float = 0.5,
    max_adf_pvalue: float = 0.05,
    min_crossings: int = 3,
    hedge_window: int = 252,
    zscore_window: int = 60,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    transaction_cost: float = 0.001,
    signal_lag_days: int = 1,
    monitor_window: int = 252,
    check_every_days: int = 21,
    max_pvalue: float = 0.10,
    persistence_checks: int = 4,
    min_formation_obs: int = 120,
) -> dict[str, Any]:
    """
    Rolling formation that only tops up *free* slots, each pair trading
    until its own cointegration breaks (event-driven), not a synchronized
    calendar window.

    Every ``formation_months``, count currently-active pairs (selected
    previously and not yet stopped as of this formation date); if fewer
    than ``top_n_pairs`` are active, screen for new crossing+cointegrated
    candidates (excluding any pair ever selected before, win or lose) and
    fill the free slots with the most significant ones. Each newly
    selected pair then trades via ``run_pair_until_broken`` from this
    formation date all the way to ``end`` (its own persistence check
    decides when it actually stops, independent of the next formation
    round). The index return each day is the equal-weight average of
    whichever pairs are still active that day.

    Returns:
        Dict with ``net_returns`` (the blended index), ``formations`` (one
        entry per formation round: date, free slots, candidates found,
        newly selected pairs with their formation stats), and
        ``pair_history`` (one entry per pair ever selected: formation
        date, trading start/stop dates, own Sharpe).
    """
    if formation_months < 3:
        raise ValueError("formation_months must be >= 3")
    if top_n_pairs < 1:
        raise ValueError("top_n_pairs must be >= 1")
    if start >= end:
        raise ValueError("start must be before end")
    if formation_months * 20 < min_formation_obs:
        raise ValueError(
            f"formation_months={formation_months} is too short for "
            f"min_formation_obs={min_formation_obs}"
        )

    panel = prices.sort_index()
    universe_by_sector = {
        s: resolve_liquid_symbols(
            sectors,
            s,
            price_columns=list(panel.columns),
            dollar_adv=dollar_adv,
            max_symbols=max_symbols_per_sector,
        )
        for s in sector_names
    }

    ever_selected: set[tuple[str, str]] = set()
    pair_history: list[dict[str, Any]] = []
    formations: list[dict[str, Any]] = []
    pair_returns: dict[str, pd.Series] = {}

    formation_start = start
    while formation_start < end:
        formation_end = min(formation_start + pd.DateOffset(months=formation_months), end)
        if formation_end <= formation_start:
            break

        active_count = sum(
            1
            for h in pair_history
            if h["trading_start"] < formation_end
            and (h["stop_date"] is None or h["stop_date"] > formation_end)
        )
        free_slots = max(0, top_n_pairs - active_count)

        candidates: list[dict[str, Any]] = []
        if free_slots > 0:
            for sector_name, syms in universe_by_sector.items():
                if len(syms) < 2:
                    continue
                fp = panel.loc[
                    (panel.index >= formation_start) & (panel.index < formation_end), syms
                ]
                fp = fp.dropna(how="all")
                if len(fp) < min_formation_obs:
                    continue
                found = find_crossing_cointegrated_candidates(
                    fp,
                    syms,
                    min_corr=min_corr,
                    max_adf_pvalue=max_adf_pvalue,
                    min_crossings=min_crossings,
                    min_obs=min_formation_obs,
                )
                for c in found:
                    c["sector"] = sector_name
                candidates.extend(found)
        candidates.sort(key=lambda r: r["adf_pvalue"])

        selected: list[dict[str, Any]] = []
        for c in candidates:
            if len(selected) >= free_slots:
                break
            key = (c["symbol_y"], c["symbol_x"])
            if key in ever_selected:
                continue
            ever_selected.add(key)
            selected.append(c)

        for c in selected:
            y, x = c["symbol_y"], c["symbol_x"]
            try:
                out = run_pair_until_broken(
                    panel,
                    symbol_y=y,
                    symbol_x=x,
                    start=formation_end,
                    max_end=end,
                    hedge_window=hedge_window,
                    zscore_window=zscore_window,
                    entry_z=entry_z,
                    exit_z=exit_z,
                    transaction_cost=transaction_cost,
                    signal_lag_days=signal_lag_days,
                    monitor_window=monitor_window,
                    check_every_days=check_every_days,
                    max_pvalue=max_pvalue,
                    persistence_checks=persistence_checks,
                )
            except (ValueError, KeyError) as exc:
                logger.info("Persistent index skip %s/%s: %s", y, x, exc)
                continue
            net = out["net_returns"]
            if len(net) < 10:
                continue
            pair_returns[f"{y}/{x}@{formation_end.date()}"] = net
            pair_history.append(
                {
                    "symbol_y": y,
                    "symbol_x": x,
                    "sector": c["sector"],
                    "formation_adf_pvalue": c["adf_pvalue"],
                    "formation_crossings": c["crossings"],
                    "trading_start": formation_end,
                    "stop_date": out["stop_date"],
                    "stopped_early": out["stopped_early"],
                    "n_days": len(net),
                }
            )

        formations.append(
            {
                "formation_start": str(pd.Timestamp(formation_start).date()),
                "formation_end": str(pd.Timestamp(formation_end).date()),
                "active_before": active_count,
                "free_slots": free_slots,
                "n_candidates_found": len(candidates),
                "n_selected": len(selected),
            }
        )

        formation_start = formation_start + pd.DateOffset(months=formation_months)

    if pair_returns:
        combined = pd.DataFrame(pair_returns)
        net_returns = combined.mean(axis=1, skipna=True).sort_index()
    else:
        net_returns = pd.Series(dtype=float)

    logger.info(
        "persistent pairs index: sectors=%d formations=%d pairs_ever=%d total_days=%d",
        len(sector_names),
        len(formations),
        len(pair_history),
        len(net_returns),
    )
    return {
        "net_returns": net_returns,
        "formations": formations,
        "pair_history": pair_history,
    }
