#!/usr/bin/env python3
"""
Build the expanded, survivorship-bias-free US equity universe table.

Writes:

    data/raw/fmp/universe/us_equity_universe.parquet

This table is the input to every backfill script (`--universe-file`). It is
deliberately a separate step from downloading: deciding *what* to download is a
research decision worth reviewing before spending hours of API budget on it.

The market-cap floor is applied to **today's** market cap, so it is a download
filter, not a point-in-time universe. Point-in-time eligibility comes from the
market-cap panel at backtest time.

Usage:
    /opt/anaconda3/envs/quant/bin/python scripts/build_fmp_universe.py
    /opt/anaconda3/envs/quant/bin/python scripts/build_fmp_universe.py --min-market-cap 2e9
    /opt/anaconda3/envs/quant/bin/python scripts/build_fmp_universe.py --no-delisted
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data.fmp.universe import US_EXCHANGES, build_universe, fetch_live_universe

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("build_fmp_universe")

UNIVERSE_DIR = ROOT / "data" / "raw" / "fmp" / "universe"
UNIVERSE_OUT = UNIVERSE_DIR / "us_equity_universe.parquet"
EXISTING_PANEL = ROOT / "data" / "factors" / "prices.parquet"

DEFAULT_MIN_MARKET_CAP = 300e6


def existing_panel_symbols() -> list[str]:
    """Symbols already in the price panel — never dropped by an expansion."""
    if not EXISTING_PANEL.exists():
        return []
    return sorted(pd.read_parquet(EXISTING_PANEL).columns)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the FMP US equity universe table")
    parser.add_argument("--min-market-cap", type=float, default=DEFAULT_MIN_MARKET_CAP)
    parser.add_argument("--no-delisted", action="store_true", help="Live names only (unsafe)")
    parser.add_argument("--exchanges", type=str, default=",".join(US_EXCHANGES))
    args = parser.parse_args()

    exchanges = tuple(e.strip().upper() for e in args.exchanges.split(","))
    keep = existing_panel_symbols()
    logger.info("Preserving %d symbols already in the price panel", len(keep))

    if args.no_delisted:
        logger.warning("Building WITHOUT delisted names — the result has survivorship bias")
        universe = fetch_live_universe(args.min_market_cap, exchanges)
        universe = universe[~universe["symbol"].isin(keep)]
        universe = pd.concat([universe, pd.DataFrame({"symbol": keep})], ignore_index=True)
    else:
        universe = build_universe(args.min_market_cap, exchanges, extra_symbols=keep)

    UNIVERSE_DIR.mkdir(parents=True, exist_ok=True)
    universe.to_parquet(UNIVERSE_OUT, index=False)

    n_live = int((universe["is_delisted"] == False).sum())  # noqa: E712 - NA-aware
    n_dead = int((universe["is_delisted"] == True).sum())  # noqa: E712 - NA-aware
    logger.info(
        "Wrote %s: %d symbols (%d live, %d delisted, %d carried over)",
        UNIVERSE_OUT,
        len(universe),
        n_live,
        n_dead,
        len(universe) - n_live - n_dead,
    )
    by_exchange = universe["exchange"].value_counts(dropna=False).head(6).to_dict()
    logger.info("By exchange: %s", by_exchange)
    logger.info(
        "Backfill cost estimate: prices ~%s calls, fundamentals ~%s calls",
        f"{len(universe) * 9:,}",
        f"{len(universe) * 3:,}",
    )


if __name__ == "__main__":
    main()
