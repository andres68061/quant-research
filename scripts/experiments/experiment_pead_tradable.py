#!/usr/bin/env python3
"""
ROADMAP item: is the post-earnings drift actually *tradable* after real costs?

The event study established the effect exists (Q5-Q1 spread +1.70% over 60 days,
t=10.05 across 267k announcements). That is a research finding measured in event
time and gross of costs — it says nothing about whether a portfolio can harvest
it.

This builds the tradable form (``core.strategies.pead``): an overlapping book
that opens a position the day after each announcement and holds it 60 trading
days, so ~1/60th of the book turns over daily. Then it charges costs two ways:

- **flat 10 bps** — the assumption used everywhere else in this repo
- **liquidity-scaled** — each name charged by its trailing dollar-ADV bucket
  (``core.data.factors.liquidity``), which is the honest model for a strategy that
  trades small caps

The gap between those two numbers is the whole question. Variants are included
because the obvious response to "costs ate it" is "trade less, or trade cheaper
names" — so those are tested rather than assumed, and all of them are reported.

Usage:
    /opt/anaconda3/envs/quant/bin/python scripts/experiments/experiment_pead_tradable.py
"""

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.backtest.event_study import extract_events_from_surprise_panel
from core.data.universe.filters import load_non_operating_symbols
from core.metrics.performance import calculate_performance_metrics
from core.strategies.pead import simulate_pead_portfolio

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("experiment_pead_tradable")

FACTORS_DIR = ROOT / "data" / "factors"
QUALITY_DIR = ROOT / "data" / "quality"

# Liquidity floor for the "trade only liquid names" variant. $20M trailing daily
# dollar volume is roughly the point where the ADV cost schedule drops to 10 bps.
LIQUID_ADV_FLOOR = 20e6


def score(label: str, result: dict, extra: dict | None = None) -> dict:
    """Score one simulated book and log a one-line summary."""
    gross = calculate_performance_metrics(result["gross_return"])
    net = calculate_performance_metrics(result["net_return"])
    row = {
        "variant": label,
        "cost_model": result["cost_model"],
        "gross_sharpe": round(gross["sharpe_ratio"], 3),
        "net_sharpe": round(net["sharpe_ratio"], 3),
        "net_ann_return": round(net["annualized_return"], 4),
        "hit_rate": round(net["hit_rate"], 3),
        "win_loss_ratio": round(net["win_loss_ratio"], 2),
        "max_drawdown": round(net["max_drawdown"], 3),
        "avg_daily_turnover": round(float(result["turnover"].mean()), 4),
        "cost_drag_annual": round(float(result["cost_drag"]), 4),
        "median_positions": int(result["positions"].replace(0, pd.NA).median() or 0),
        **(extra or {}),
    }
    logger.info(
        "%-34s gross %5.2f -> net %5.2f | ann %6.1f%% | turnover %4.1f%%/d | cost %4.1f%%/yr",
        label,
        row["gross_sharpe"],
        row["net_sharpe"],
        row["net_ann_return"] * 100,
        row["avg_daily_turnover"] * 100,
        row["cost_drag_annual"] * 100,
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Is PEAD tradable after costs?")
    parser.add_argument("--hold-days", type=int, default=60)
    args = parser.parse_args()

    panel = pd.read_parquet(
        FACTORS_DIR / "factors_earnings_surprise.parquet",
        columns=["sue_price_scaled", "days_since_earnings"],
    )
    prices = pd.read_parquet(FACTORS_DIR / "prices.parquet")
    dollar_adv = pd.read_parquet(FACTORS_DIR / "dollar_adv_21d.parquet")

    events = extract_events_from_surprise_panel(panel)
    events = events[~events["symbol"].isin(load_non_operating_symbols())]
    logger.info(
        "Events: %s across %s symbols", f"{len(events):,}", f"{events['symbol'].nunique():,}"
    )

    results = [
        score(
            "A flat 10bps (repo default)",
            simulate_pead_portfolio(events, prices, hold_days=args.hold_days, cost_bps=10.0),
        ),
        score(
            "B liquidity-scaled costs",
            simulate_pead_portfolio(
                events, prices, hold_days=args.hold_days, dollar_adv=dollar_adv
            ),
        ),
    ]

    # Can it be salvaged? Two obvious responses to "costs ate it", both tested
    # rather than assumed. Reported whether they work or not.
    liquid_symbols = set(
        dollar_adv.columns[(dollar_adv.ffill(limit=5).median() >= LIQUID_ADV_FLOOR).fillna(False)]
    )
    liquid_events = events[events["symbol"].isin(liquid_symbols)]
    logger.info(
        "Liquid subset: %s of %s events (ADV >= $%.0fM)",
        f"{len(liquid_events):,}",
        f"{len(events):,}",
        LIQUID_ADV_FLOOR / 1e6,
    )
    results.append(
        score(
            "C liquid names only, ADV costs",
            simulate_pead_portfolio(
                liquid_events, prices, hold_days=args.hold_days, dollar_adv=dollar_adv
            ),
            {"n_events": len(liquid_events)},
        )
    )
    results.append(
        score(
            "D 120-day hold, ADV costs",
            simulate_pead_portfolio(events, prices, hold_days=120, dollar_adv=dollar_adv),
            {"hold_days": 120},
        )
    )
    results.append(
        score(
            "E long-only top quintile, ADV",
            simulate_pead_portfolio(
                events, prices, hold_days=args.hold_days, dollar_adv=dollar_adv, long_only=True
            ),
        )
    )
    # THE control for E. A long-only book's Sharpe is dominated by market beta,
    # so E beating the long/short variants proves nothing on its own. This holds
    # EVERY announcer long with the surprise signal replaced by a constant: same
    # universe, same overlapping construction, same costs, no signal. If E does
    # not beat F, the surprise added nothing and E is just "own equities that
    # recently reported".
    flat_signal_events = events.assign(signal=1.0)
    results.append(
        score(
            "F long-only ALL announcers (control)",
            simulate_pead_portfolio(
                flat_signal_events,
                prices,
                hold_days=args.hold_days,
                dollar_adv=dollar_adv,
                long_only=True,
                n_quantiles=1,
            ),
        )
    )

    by_variant = {r["variant"][0]: r for r in results}
    verdict = {
        "gross_effect_is_real": bool(by_variant["A"]["gross_sharpe"] > 0.5),
        "survives_flat_costs": bool(by_variant["A"]["net_sharpe"] > 0.3),
        "survives_realistic_costs": bool(by_variant["B"]["net_sharpe"] > 0.3),
        "long_only_signal_beats_no_signal": bool(
            by_variant["E"]["net_sharpe"] > by_variant["F"]["net_sharpe"]
        ),
        "long_only_edge_over_control": round(
            by_variant["E"]["net_sharpe"] - by_variant["F"]["net_sharpe"], 3
        ),
        "best_realistic_variant": max(
            (r for r in results if r["cost_model"] == "dollar_adv_schedule"),
            key=lambda r: r["net_sharpe"],
        )["variant"],
        "note": (
            "The gap between A and B is the finding. A flat basis-point cost "
            "assumption is not conservative for a strategy whose edge lives in "
            "small caps — it is the assumption that makes it look tradable. Five "
            "variants were tested; the best of six is inflated by selection. E "
            "(long-only) must be read against F, not against the long/short "
            "variants: a long-only book's Sharpe is mostly market beta."
        ),
    }

    payload = {
        "run_date": str(date.today()),
        "construction": (
            f"overlapping book, position opened the day after each announcement and held "
            f"{args.hold_days} trading days, top/bottom surprise quintile, gross exposure "
            "normalized to 1.0 daily, shells/SPACs excluded, returns require price >= $1 "
            "and |ret| <= 300%"
        ),
        "results": results,
        "verdict": verdict,
    }
    QUALITY_DIR.mkdir(parents=True, exist_ok=True)
    out_path = QUALITY_DIR / f"experiment_pead_tradable_{date.today():%Y%m%d}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote %s", out_path)
    logger.info(
        "VERDICT: gross effect real=%s | survives flat costs=%s | survives REAL costs=%s",
        verdict["gross_effect_is_real"],
        verdict["survives_flat_costs"],
        verdict["survives_realistic_costs"],
    )


if __name__ == "__main__":
    main()
