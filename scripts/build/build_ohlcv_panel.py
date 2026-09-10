#!/usr/bin/env python3
"""
Assemble the OHLCV long panel and microstructure factors from the raw price layer.

The raw layer has stored full adjusted bars (``adj_open``/``adj_high``/``adj_low``/
``adj_close``/``volume``) since the FMP cutover, but ``data/factors/prices.parquet``
keeps only closes. This script surfaces the rest:

    data/factors/ohlcv.parquet                 long panel (date, symbol) x 5 fields
    data/factors/factors_microstructure.parquet range vol, spreads, overnight split

No network access — everything is derived from bars already downloaded.

Usage:
    /opt/anaconda3/envs/quant/bin/python scripts/build/build_ohlcv_panel.py
    /opt/anaconda3/envs/quant/bin/python scripts/build/build_ohlcv_panel.py --symbols AAPL,MSFT
    /opt/anaconda3/envs/quant/bin/python scripts/build/build_ohlcv_panel.py --window 63
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data.store.artifacts import write_artifact
from core.data.factors.microstructure import DEFAULT_WINDOW, compute_microstructure_factors
from core.data.vendors.fmp.prices import PANEL_TIMEZONE
from core.data.vendors.fmp.storage import load_trading_calendar, load_universe_symbols, safe_filename

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("build_ohlcv_panel")

RAW_PRICES_DIR = ROOT / "data" / "raw" / "fmp" / "prices"
CANONICAL_PANEL = ROOT / "data" / "factors" / "prices.parquet"
OHLCV_OUT = ROOT / "data" / "factors" / "ohlcv.parquet"
MICROSTRUCTURE_OUT = ROOT / "data" / "factors" / "factors_microstructure.parquet"
OHLCV_EXPANDED_OUT = ROOT / "data" / "factors" / "ohlcv_expanded.parquet"
MICROSTRUCTURE_EXPANDED_OUT = ROOT / "data" / "factors" / "factors_microstructure_expanded.parquet"

OHLCV_COLUMNS = ("adj_open", "adj_high", "adj_low", "adj_close", "volume")
FACTOR_DTYPE = "float32"


def load_symbol_bars(path: Path, trading_calendar: pd.DatetimeIndex) -> pd.DataFrame | None:
    """
    Read one symbol's raw bars, returning None when unusable.

    Args:
        path: ``{SYMBOL}.parquet`` under the raw price directory.
        trading_calendar: Valid trading days; bars on other dates (vendor bad
            prints on Sundays/holidays) are dropped.

    Returns:
        Bars with a tz-aware ascending index and the five OHLCV columns, or None
        for empty files and pre-cutover files that only carry closes.
    """
    bars = pd.read_parquet(path)
    if bars.empty or not set(OHLCV_COLUMNS).issubset(bars.columns):
        return None
    bars = bars[list(OHLCV_COLUMNS)].sort_index()
    if bars.index.tz is None:
        bars.index = bars.index.tz_localize(PANEL_TIMEZONE)
    bars.index.name = "date"
    bars = bars[~bars.index.duplicated(keep="last")]
    return bars[bars.index.isin(trading_calendar)]


def build_panels(
    symbols: list[str],
    window: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build the OHLCV long panel and the microstructure factor panel together.

    Both are produced in one pass so each raw file is read once.

    Args:
        symbols: Universe to build — always explicit. The raw directory holds the
            expanded universe, so globbing it would silently change what the
            canonical artifacts cover.
        window: Trailing window shared by all microstructure estimators.

    Returns:
        ``(ohlcv_panel, microstructure_panel)``, both MultiIndex (date, symbol).
    """
    wanted = {safe_filename(s) for s in symbols}
    paths = [p for p in sorted(RAW_PRICES_DIR.glob("*.parquet")) if p.stem in wanted]
    if not paths:
        raise SystemExit(f"No raw price files in {RAW_PRICES_DIR}; run fetch_fmp_prices.py first.")

    trading_calendar = load_trading_calendar(CANONICAL_PANEL)
    ohlcv_frames: list[pd.DataFrame] = []
    factor_frames: list[pd.DataFrame] = []
    n_skipped = 0

    for i, path in enumerate(paths, 1):
        bars = load_symbol_bars(path, trading_calendar)
        if bars is None:
            n_skipped += 1
            continue

        symbol = path.stem
        ohlcv_frames.append(bars.assign(symbol=symbol))
        factors = compute_microstructure_factors(bars, window=window)
        factor_frames.append(factors.astype(FACTOR_DTYPE).assign(symbol=symbol))

        if i % 200 == 0 or i == len(paths):
            logger.info("[%d/%d] processed (%d skipped)", i, len(paths), n_skipped)

    def stack(frames: list[pd.DataFrame]) -> pd.DataFrame:
        panel = pd.concat(frames)
        return panel.set_index("symbol", append=True).sort_index()

    ohlcv_panel = stack(ohlcv_frames)
    microstructure_panel = stack(factor_frames).dropna(how="all")
    return ohlcv_panel, microstructure_panel


def main() -> None:
    parser = argparse.ArgumentParser(description="Build OHLCV and microstructure panels")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated subset")
    parser.add_argument(
        "--expanded",
        action="store_true",
        help="Build the full expanded universe into *_expanded staging artifacts "
        "instead of the canonical files",
    )
    parser.add_argument("--universe-file", type=Path, default=None, help="Universe parquet")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="Trailing window")
    args = parser.parse_args()

    if args.symbols:
        symbols = args.symbols.split(",")
        ohlcv_out, micro_out = OHLCV_OUT, MICROSTRUCTURE_OUT
    elif args.expanded:
        universe_file = args.universe_file or (
            ROOT / "data" / "raw" / "fmp" / "universe" / "us_equity_universe.parquet"
        )
        symbols = load_universe_symbols(universe_file)
        ohlcv_out, micro_out = OHLCV_EXPANDED_OUT, MICROSTRUCTURE_EXPANDED_OUT
    else:
        # Canonical build: exactly the canonical panel's symbols, never the raw glob.
        symbols = sorted(pq.ParquetFile(CANONICAL_PANEL).schema_arrow.names)
        symbols = [c for c in symbols if c != "date" and not c.startswith("__")]
        ohlcv_out, micro_out = OHLCV_OUT, MICROSTRUCTURE_OUT

    ohlcv_panel, microstructure_panel = build_panels(symbols, args.window)

    # A --symbols run is a partial build; write_artifact redirects it to scratch/
    # rather than letting a smoke test replace the full panel.
    subset = bool(args.symbols)
    write_artifact(ohlcv_panel, ohlcv_out, subset=subset)
    logger.info(
        "%s: %d symbols", ohlcv_out.name, ohlcv_panel.index.get_level_values("symbol").nunique()
    )

    write_artifact(microstructure_panel, micro_out, subset=subset)
    coverage = microstructure_panel.notna().mean().sort_values()
    logger.info("Factor coverage: %s", coverage.round(3).to_dict())


if __name__ == "__main__":
    main()
