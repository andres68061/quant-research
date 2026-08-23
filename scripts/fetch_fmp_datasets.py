#!/usr/bin/env python3
"""
Download per-symbol FMP datasets into the raw layer. Resumable by construction.

Layout (one file per symbol per dataset, immutable source of truth):

    data/raw/fmp/{dataset}/{SYMBOL}.parquet

**Checkpointing.** A symbol that already has a file is skipped, so an interrupted
run resumes exactly where it stopped — no state file to corrupt. Writes go to a
temporary file and are then atomically renamed, so a process killed mid-write can
never leave a half-written parquet that a later run would mistake for complete.
Symbols with no vendor coverage get an empty file so they are not retried forever.

Bulk endpoints are not on our plan (see docs/DATA_INVENTORY.md §6), so every
symbol costs one call per dataset. Budget: N_symbols x N_datasets calls, at the
client's ~500 calls/min ceiling.

Usage:
    # everything the registry knows about, for the current price-panel universe
    /opt/anaconda3/envs/quant/bin/python scripts/fetch_fmp_datasets.py --datasets all

    # one dataset, explicit symbols
    /opt/anaconda3/envs/quant/bin/python scripts/fetch_fmp_datasets.py \
        --datasets earnings,key_metrics --symbols AAPL,MSFT

    # universe from a saved universe table (see build_fmp_universe.py)
    /opt/anaconda3/envs/quant/bin/python scripts/fetch_fmp_datasets.py \
        --datasets earnings --universe-file data/raw/fmp/universe/us_equity_universe.parquet

    /opt/anaconda3/envs/quant/bin/python scripts/fetch_fmp_datasets.py --list
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data.fmp.datasets import SYMBOL_DATASETS, describe_datasets, fetch_symbol_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("fetch_fmp_datasets")

RAW_FMP_DIR = ROOT / "data" / "raw" / "fmp"
PRICE_PANEL = ROOT / "data" / "factors" / "prices.parquet"

_PROGRESS_EVERY = 50


def write_atomic(frame: pd.DataFrame, path: Path) -> None:
    """
    Write a parquet so an interrupted run cannot leave a truncated file behind.

    Args:
        frame: Data to write.
        path: Final destination.
    """
    temporary = path.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary)
    os.replace(temporary, path)


def load_universe(universe_file: Path | None, symbols_arg: str | None) -> list[str]:
    """
    Resolve the symbol list from the CLI arguments.

    Args:
        universe_file: Optional parquet with a ``symbol`` column.
        symbols_arg: Optional comma-separated override.

    Returns:
        Sorted unique symbols.
    """
    if symbols_arg:
        return sorted({s.strip().upper() for s in symbols_arg.split(",") if s.strip()})
    if universe_file:
        universe = pd.read_parquet(universe_file)
        if "symbol" not in universe.columns:
            raise SystemExit(f"{universe_file} has no 'symbol' column")
        return sorted(universe["symbol"].dropna().unique().tolist())
    return sorted(pd.read_parquet(PRICE_PANEL).columns)


def fetch_dataset(dataset: str, symbols: list[str], refresh: bool) -> pd.DataFrame:
    """
    Fetch one dataset for every symbol, skipping symbols already on disk.

    Args:
        dataset: Key into the dataset registry.
        symbols: Symbols to fetch.
        refresh: Refetch even when a raw file exists.

    Returns:
        Per-symbol report with ``symbol``, ``status``, ``rows``.
    """
    output_dir = RAW_FMP_DIR / dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    report: list[dict[str, object]] = []
    counts = {"ok": 0, "skipped": 0, "empty": 0, "failed": 0}
    started = time.monotonic()

    for i, symbol in enumerate(symbols, 1):
        # "/" appears in some FMP tickers and would create a subdirectory.
        out_path = output_dir / f"{symbol.replace('/', '-')}.parquet"
        if out_path.exists() and not refresh:
            counts["skipped"] += 1
            continue

        try:
            frame = fetch_symbol_dataset(symbol, dataset)
        except Exception:
            logger.exception("FAILED %s/%s", dataset, symbol)
            report.append({"symbol": symbol, "status": "failed", "rows": 0})
            counts["failed"] += 1
            continue

        write_atomic(frame, out_path)
        if frame.empty:
            report.append({"symbol": symbol, "status": "empty", "rows": 0})
            counts["empty"] += 1
        else:
            report.append({"symbol": symbol, "status": "ok", "rows": len(frame)})
            counts["ok"] += 1

        if i % _PROGRESS_EVERY == 0 or i == len(symbols):
            done = counts["ok"] + counts["empty"] + counts["failed"]
            rate = done / max(time.monotonic() - started, 1e-9)
            remaining = (len(symbols) - i) / rate / 60.0 if rate > 0 else float("nan")
            logger.info(
                "%s [%d/%d] ok=%d skipped=%d empty=%d failed=%d (~%.1f min left)",
                dataset,
                i,
                len(symbols),
                counts["ok"],
                counts["skipped"],
                counts["empty"],
                counts["failed"],
                remaining,
            )

    logger.info("%s DONE %s", dataset, counts)
    return pd.DataFrame(report)


def write_report(dataset: str, report: pd.DataFrame) -> None:
    """Merge this run's per-symbol outcomes into the dataset's cumulative report."""
    if report.empty:
        return
    report_path = RAW_FMP_DIR / dataset / "_fetch_report.csv"
    if report_path.exists():
        report = pd.concat([pd.read_csv(report_path), report]).drop_duplicates(
            "symbol", keep="last"
        )
    report.to_csv(report_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch per-symbol FMP datasets")
    parser.add_argument("--datasets", type=str, default=None, help="Comma-separated, or 'all'")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated subset")
    parser.add_argument("--universe-file", type=Path, default=None, help="Parquet with 'symbol'")
    parser.add_argument("--refresh", action="store_true", help="Refetch existing files")
    parser.add_argument("--list", action="store_true", help="List the dataset registry and exit")
    args = parser.parse_args()

    if args.list:
        registry = describe_datasets()
        for _, row in registry.iterrows():
            logger.info(
                "%-22s %-32s %-15s %s",
                row["dataset"],
                row["endpoint"],
                row["pit_status"],
                row["notes"][:70],
            )
        return

    if not args.datasets:
        raise SystemExit("--datasets is required (use 'all', or --list to see options)")
    datasets = (
        sorted(SYMBOL_DATASETS)
        if args.datasets == "all"
        else [d.strip() for d in args.datasets.split(",")]
    )
    if unknown := set(datasets) - set(SYMBOL_DATASETS):
        raise SystemExit(f"Unknown datasets: {sorted(unknown)}")

    symbols = load_universe(args.universe_file, args.symbols)
    logger.info(
        "Fetching %d dataset(s) x %d symbols = up to %s calls",
        len(datasets),
        len(symbols),
        f"{len(datasets) * len(symbols):,}",
    )

    for dataset in datasets:
        write_report(dataset, fetch_dataset(dataset, symbols, refresh=args.refresh))


if __name__ == "__main__":
    main()
