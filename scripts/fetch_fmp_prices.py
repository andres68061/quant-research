#!/usr/bin/env python3
"""
Download full dividend-adjusted price histories from FMP into the raw layer.

Raw layer layout (one file per symbol, immutable source of truth):

    data/raw/fmp/prices/{SYMBOL}.parquet   # tz-aware date index, adj OHLC + volume

The script is resumable: symbols that already have a raw file are skipped
(unless --refresh), so it can be interrupted and rerun freely. A summary of
missing/empty symbols is written to data/raw/fmp/prices/_fetch_report.csv.

Usage:
    /opt/anaconda3/envs/quant/bin/python scripts/fetch_fmp_prices.py                # all panel symbols
    /opt/anaconda3/envs/quant/bin/python scripts/fetch_fmp_prices.py --symbols AAPL,MSFT
    /opt/anaconda3/envs/quant/bin/python scripts/fetch_fmp_prices.py --refresh      # refetch everything

After downloading, assemble the wide close panel for validation:
    /opt/anaconda3/envs/quant/bin/python scripts/fetch_fmp_prices.py --build-panel
    # -> data/factors/prices_fmp.parquet (does NOT overwrite prices.parquet)
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data.fmp.prices import PANEL_TIMEZONE, fetch_dividend_adjusted_history

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("fetch_fmp_prices")

RAW_PRICES_DIR = ROOT / "data" / "raw" / "fmp" / "prices"
EXISTING_PANEL = ROOT / "data" / "factors" / "prices.parquet"
FMP_PANEL_OUT = ROOT / "data" / "factors" / "prices_fmp.parquet"
DEFAULT_START = pd.Timestamp("1985-01-01")


def load_universe() -> list[str]:
    """Symbols to fetch: the columns of the existing wide price panel."""
    existing_prices = pd.read_parquet(EXISTING_PANEL)
    return sorted(existing_prices.columns)


def fetch_all(symbols: list[str], start: pd.Timestamp, end: pd.Timestamp, refresh: bool) -> None:
    """Fetch each symbol's history to its own raw parquet; skip existing files."""
    RAW_PRICES_DIR.mkdir(parents=True, exist_ok=True)
    report_rows = []
    n_done = n_skipped = n_empty = n_failed = 0

    for i, symbol in enumerate(symbols, 1):
        out_path = RAW_PRICES_DIR / f"{symbol}.parquet"
        if out_path.exists() and not refresh:
            n_skipped += 1
            continue
        try:
            history = fetch_dividend_adjusted_history(symbol, start, end)
        except Exception:
            logger.exception("FAILED %s", symbol)
            report_rows.append({"symbol": symbol, "status": "failed", "rows": 0})
            n_failed += 1
            continue

        if history.empty:
            logger.warning("EMPTY %s (no FMP coverage)", symbol)
            report_rows.append({"symbol": symbol, "status": "empty", "rows": 0})
            n_empty += 1
            # Write the empty frame too so reruns don't refetch known gaps.
            history.to_parquet(out_path)
            continue

        history.to_parquet(out_path)
        report_rows.append(
            {
                "symbol": symbol,
                "status": "ok",
                "rows": len(history),
                "first": str(history.index.min().date()),
                "last": str(history.index.max().date()),
            }
        )
        n_done += 1
        if i % 25 == 0 or i == len(symbols):
            logger.info(
                "[%d/%d] ok=%d skipped=%d empty=%d failed=%d (last: %s)",
                i,
                len(symbols),
                n_done,
                n_skipped,
                n_empty,
                n_failed,
                symbol,
            )

    if report_rows:
        report = pd.DataFrame(report_rows)
        report_path = RAW_PRICES_DIR / "_fetch_report.csv"
        # Append to any previous report, keeping the latest row per symbol.
        if report_path.exists():
            previous = pd.read_csv(report_path)
            report = pd.concat([previous, report]).drop_duplicates("symbol", keep="last")
        report.to_csv(report_path, index=False)
        logger.info("Report written to %s", report_path)

    logger.info("DONE ok=%d skipped=%d empty=%d failed=%d", n_done, n_skipped, n_empty, n_failed)


def build_panel() -> None:
    """Assemble raw per-symbol files into a wide adj_close panel (prices_fmp.parquet)."""
    files = sorted(RAW_PRICES_DIR.glob("*.parquet"))
    if not files:
        raise SystemExit(f"No raw files in {RAW_PRICES_DIR}; run the fetch step first.")

    close_series = {}
    for path in files:
        raw_history = pd.read_parquet(path)
        if raw_history.empty:
            continue
        close_series[path.stem] = raw_history["adj_close"]

    panel = pd.DataFrame(close_series).sort_index()
    panel.index.name = "date"
    if panel.index.tz is None:
        panel.index = panel.index.tz_localize(PANEL_TIMEZONE)
    panel.to_parquet(FMP_PANEL_OUT)
    logger.info(
        "Wrote %s: %d dates x %d symbols (%s -> %s)",
        FMP_PANEL_OUT,
        len(panel),
        panel.shape[1],
        panel.index.min().date(),
        panel.index.max().date(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch FMP dividend-adjusted price histories")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated subset")
    parser.add_argument("--start", type=str, default=str(DEFAULT_START.date()))
    parser.add_argument("--end", type=str, default=str(pd.Timestamp.now().date()))
    parser.add_argument("--refresh", action="store_true", help="Refetch even if raw file exists")
    parser.add_argument("--build-panel", action="store_true", help="Assemble wide panel and exit")
    args = parser.parse_args()

    if args.build_panel:
        build_panel()
        return

    symbols = args.symbols.split(",") if args.symbols else load_universe()
    fetch_all(symbols, pd.Timestamp(args.start), pd.Timestamp(args.end), refresh=args.refresh)


if __name__ == "__main__":
    main()
