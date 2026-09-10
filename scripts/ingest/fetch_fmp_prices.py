#!/usr/bin/env python3
"""
Download full dividend-adjusted price histories from FMP into the raw layer.

Raw layer layout (one file per symbol, immutable source of truth):

    data/raw/fmp/prices/{SYMBOL}.parquet   # tz-aware date index, adj OHLC + volume

The script is resumable: symbols that already have a raw file are skipped
(unless --refresh), so it can be interrupted and rerun freely. A summary of
missing/empty symbols is written to data/raw/fmp/prices/_fetch_report.csv.

Usage:
    /opt/anaconda3/envs/quant/bin/python scripts/ingest/fetch_fmp_prices.py                # all panel symbols
    /opt/anaconda3/envs/quant/bin/python scripts/ingest/fetch_fmp_prices.py --symbols AAPL,MSFT
    /opt/anaconda3/envs/quant/bin/python scripts/ingest/fetch_fmp_prices.py --refresh      # refetch everything

After downloading, assemble the wide close panel for validation:
    /opt/anaconda3/envs/quant/bin/python scripts/ingest/fetch_fmp_prices.py --build-panel
    # -> data/factors/prices_fmp.parquet (does NOT overwrite prices.parquet)
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data.quality.validation import assert_panel_valid
from core.data.vendors.fmp.prices import PANEL_TIMEZONE, fetch_dividend_adjusted_history
from core.data.vendors.fmp.storage import (
    EARLIEST_HISTORY,
    load_fetch_windows,
    load_trading_calendar,
    load_universe_symbols,
    safe_filename,
    write_atomic,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("fetch_fmp_prices")

RAW_PRICES_DIR = ROOT / "data" / "raw" / "fmp" / "prices"
EXISTING_PANEL = ROOT / "data" / "factors" / "prices.parquet"
FMP_PANEL_OUT = ROOT / "data" / "factors" / "prices_fmp.parquet"
DEFAULT_START = EARLIEST_HISTORY

_PROGRESS_EVERY = 50


def load_universe() -> list[str]:
    """Symbols to fetch: the columns of the existing wide price panel."""
    existing_prices = pd.read_parquet(EXISTING_PANEL)
    return sorted(existing_prices.columns)


def fetch_all(
    symbols: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    refresh: bool,
    windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]] | None = None,
    merge_recent: bool = False,
) -> None:
    """
    Fetch each symbol's history to its own raw parquet; skip symbols already done.

    Args:
        symbols: Tickers to fetch.
        start: Default first date, used when the symbol has no known window.
        end: Default last date.
        refresh: Refetch even when the raw file exists.
        windows: Optional ``{symbol: (start, end)}`` from the universe table, so a
            short-lived delisted name costs one chunk instead of nine.
        merge_recent: Fetch ``[start, end]`` for every symbol and MERGE it into the
            existing raw file (vendor-latest wins) instead of skipping symbols that
            already have data. This is the "bring the whole universe current"
            mode — it ignores per-symbol windows, since the point is the tail.
    """
    RAW_PRICES_DIR.mkdir(parents=True, exist_ok=True)
    windows = windows or {}
    report_rows = []
    n_done = n_skipped = n_empty = n_failed = 0
    started = time.monotonic()

    for i, symbol in enumerate(symbols, 1):
        out_path = RAW_PRICES_DIR / f"{safe_filename(symbol)}.parquet"
        if out_path.exists() and not refresh and not merge_recent:
            n_skipped += 1
            continue

        symbol_start, symbol_end = (
            (start, end) if merge_recent else windows.get(symbol, (start, end))
        )
        try:
            history = fetch_dividend_adjusted_history(symbol, symbol_start, symbol_end)
        except Exception:
            logger.exception("FAILED %s", symbol)
            report_rows.append({"symbol": symbol, "status": "failed", "rows": 0})
            n_failed += 1
            continue

        if merge_recent and out_path.exists():
            existing = pd.read_parquet(out_path)
            if not history.empty:
                # Vendor-latest wins on overlapping dates (restatements).
                history = pd.concat([existing, history])
                history = history[~history.index.duplicated(keep="last")].sort_index()
            else:
                history = existing

        if history.empty:
            report_rows.append({"symbol": symbol, "status": "empty", "rows": 0})
            n_empty += 1
            # Write the empty frame too so reruns don't refetch known gaps.
            write_atomic(history, out_path)
            continue

        write_atomic(history, out_path)
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
        if i % _PROGRESS_EVERY == 0 or i == len(symbols):
            attempted = n_done + n_empty + n_failed
            rate = attempted / max(time.monotonic() - started, 1e-9)
            remaining = (len(symbols) - i) / rate / 60.0 if rate > 0 else float("nan")
            logger.info(
                "[%d/%d] ok=%d skipped=%d empty=%d failed=%d (~%.0f min left)",
                i,
                len(symbols),
                n_done,
                n_skipped,
                n_empty,
                n_failed,
                remaining,
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
    """Assemble raw per-symbol files into a wide adj_close panel (prices_fmp.parquet).

    The index is intersected with the canonical trading calendar: expanded-universe
    raw files carry vendor bars on Sundays and US market holidays, and a return
    computed across such a bar is wrong for every symbol involved.
    """
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

    # A stock cannot trade at exactly $0.00: a zero (or negative) close is a
    # vendor defect, and it is a *poisonous* one — pct_change across it yields
    # inf, and one inf in a cross-sectional mean makes every symbol's abnormal
    # return infinite for that date. Null them so they read as "no data".
    non_positive = (panel <= 0).sum().sum()
    if non_positive:
        affected = int(((panel <= 0).any()).sum())
        logger.warning(
            "Nulling %d non-positive prices across %d symbols (vendor defect)",
            non_positive,
            affected,
        )
        panel = panel.mask(panel <= 0)

    trading_calendar = load_trading_calendar(EXISTING_PANEL)
    non_trading = panel.index.difference(trading_calendar)
    if len(non_trading):
        logger.warning(
            "Dropping %d non-trading dates (vendor bad prints, e.g. %s)",
            len(non_trading),
            [str(d.date()) for d in non_trading[:3]],
        )
    panel = panel.reindex(panel.index.intersection(trading_calendar))

    # Structural gate: publishing a panel that violates an invariant is worse
    # than failing here, because the artifact looks complete and the error
    # surfaces days later inside a research result.
    assert_panel_valid(panel, name=FMP_PANEL_OUT.name)
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
    parser.add_argument(
        "--universe-file",
        type=Path,
        default=None,
        help="Universe parquet from build_fmp_universe.py; enables per-symbol windows",
    )
    parser.add_argument("--start", type=str, default=str(DEFAULT_START.date()))
    parser.add_argument("--end", type=str, default=str(pd.Timestamp.now().date()))
    parser.add_argument("--refresh", action="store_true", help="Refetch even if raw file exists")
    parser.add_argument(
        "--refresh-recent",
        action="store_true",
        help="Fetch [--start, --end] for every symbol and merge into existing raw files "
        "(brings the whole universe current without a full refetch)",
    )
    parser.add_argument("--build-panel", action="store_true", help="Assemble wide panel and exit")
    args = parser.parse_args()

    if args.build_panel:
        build_panel()
        return

    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    if args.symbols:
        symbols = args.symbols.split(",")
    elif args.universe_file:
        symbols = load_universe_symbols(args.universe_file)
    else:
        symbols = load_universe()

    windows = load_fetch_windows(args.universe_file, default_start=start, default_end=end)
    logger.info(
        "Fetching %d symbols (%d with narrowed windows)%s",
        len(symbols),
        len(windows),
        " [merge-recent mode]" if args.refresh_recent else "",
    )
    fetch_all(
        symbols,
        start,
        end,
        refresh=args.refresh,
        windows=windows,
        merge_recent=args.refresh_recent,
    )


if __name__ == "__main__":
    main()
