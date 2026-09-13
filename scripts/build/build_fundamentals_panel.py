#!/usr/bin/env python3
"""
Build the point-in-time fundamentals panel and the fundamental factor library.

Reads raw FMP statements (data/raw/fmp/fundamentals/), aligns every metric on its
publication date (acceptedDate; visible the NEXT trading day), and writes:

    data/factors/fundamentals.parquet         PIT metric panel (date, symbol)
    data/factors/factors_fundamental.parquet  ~35 factor columns

Cost ordering matters here. Dailyizing a column across ~10k trading days × ~800
symbols is ~100x more expensive than computing it at quarterly filing frequency,
so every ratio that needs no market price is computed **quarterly first** and
dailyized once. Only numerators that must meet a daily market cap are carried
into the daily panel.

No network access: this reads the raw layer that scripts/ingest/fetch_fmp_fundamentals.py
already downloaded.

Usage:
    /opt/anaconda3/envs/quant/bin/python scripts/build/build_fundamentals_panel.py
    /opt/anaconda3/envs/quant/bin/python scripts/build/build_fundamentals_panel.py --symbols AAPL,MSFT
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data.store.artifacts import write_artifact
from core.data.factors.fundamental_factors import (
    PRICE_DEPENDENT_INPUT_COLUMNS,
    compute_price_dependent_factors,
    compute_statement_factors,
)
from core.data.factors.fundamentals import build_pit_fundamentals_panel
from core.data.factors.quality_scores import compute_altman_z_score, compute_piotroski_f_score
from core.data.factors.statement_metrics import build_statement_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("build_fundamentals_panel")

RAW_DIR = ROOT / "data" / "raw" / "fmp" / "fundamentals"
PRICES_PATH = ROOT / "data" / "factors" / "prices.parquet"
MARKET_CAPS_PATH = ROOT / "data" / "market_caps" / "historical_market_caps.parquet"
PIT_PANEL_OUT = ROOT / "data" / "factors" / "fundamentals.parquet"
FACTORS_OUT = ROOT / "data" / "factors" / "factors_fundamental.parquet"
COMPOSITES_OUT = ROOT / "data" / "factors" / "factors_composites.parquet"

# Above this many symbols the single-pass dailyization does not fit in memory.
BATCH_THRESHOLD_SYMBOLS = 1200

# Kept in the daily panel for backwards compatibility with earlier consumers.
LEGACY_PANEL_COLUMNS = ("shares_diluted", "asset_growth_yoy")

# float32 costs ~7 significant digits and halves a multi-GB panel. Factor ranking
# is scale-invariant, so the precision loss is far below the noise in the inputs.
PANEL_DTYPE = "float32"


def load_quarterly_frames(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """
    Build each symbol's quarterly, publication-dated frame of metrics + factors.

    Args:
        symbols: Universe to build — always explicit. The raw directory holds the
            9,000-symbol expanded universe; globbing it from the canonical build
            both changes what the artifact covers and blows memory.

    Returns:
        ``{symbol: DataFrame}`` indexed by publication_date, carrying the columns
        that need dailyizing (statement factors, Piotroski F, and the numerators
        used by price-dependent factors).
    """
    income_dir = RAW_DIR / "income_statement"
    balance_dir = RAW_DIR / "balance_sheet"
    cash_flow_dir = RAW_DIR / "cash_flow"

    available = {path.stem for path in income_dir.glob("*.parquet")}
    wanted = [s for s in symbols if s in available]
    logger.info("Extracting quarterly metrics for %d symbols", len(wanted))

    per_symbol: dict[str, pd.DataFrame] = {}
    n_empty = 0
    for symbol in wanted:
        balance_path = balance_dir / f"{symbol}.parquet"
        if not balance_path.exists():
            continue
        income_statements = pd.read_parquet(income_dir / f"{symbol}.parquet")
        balance_sheets = pd.read_parquet(balance_path)
        cash_flow_path = cash_flow_dir / f"{symbol}.parquet"
        cash_flows = pd.read_parquet(cash_flow_path) if cash_flow_path.exists() else None

        metrics = build_statement_metrics(income_statements, balance_sheets, cash_flows)
        if metrics.empty:
            n_empty += 1
            continue

        statement_factors = compute_statement_factors(metrics)
        combined = pd.concat(
            [
                statement_factors,
                compute_piotroski_f_score(metrics),
                metrics[list(PRICE_DEPENDENT_INPUT_COLUMNS)],
                metrics[["shares_diluted"]],
            ],
            axis=1,
        )
        combined["asset_growth_yoy"] = statement_factors["asset_growth"]
        per_symbol[symbol] = combined.astype(PANEL_DTYPE)

    logger.info("Extracted %d symbols (%d with no usable statements)", len(per_symbol), n_empty)
    return per_symbol


def load_market_cap(trading_index: pd.DatetimeIndex) -> pd.Series:
    """Load the (date, symbol) market cap series, aligned to the panel timezone."""
    market_caps = pd.read_parquet(MARKET_CAPS_PATH)
    market_cap = market_caps["market_cap"]
    market_cap.index = market_cap.index.set_names(["date", "symbol"])
    dates = market_cap.index.get_level_values("date")
    if dates.tz is None and trading_index.tz is not None:
        market_cap.index = pd.MultiIndex.from_arrays(
            [dates.tz_localize(trading_index.tz), market_cap.index.get_level_values("symbol")],
            names=["date", "symbol"],
        )
    return market_cap


def build_batched(
    symbols: list[str],
    factors_out: Path,
    panel_out: Path,
    batch_size: int = 400,
) -> int:
    """
    Build the expanded-universe fundamentals factors in symbol batches.

    The single-pass build holds every dailyized symbol in memory at once, which
    is ~5+ GB at 9,000 symbols and is exactly what got the 2026-08-08 overnight
    run killed (exit -9). This variant processes ``batch_size`` symbols at a
    time and appends each batch to the parquet file via a ``ParquetWriter``, so
    peak memory stays at one batch.

    Two deliberate differences from the canonical artifact, both documented in
    docs/data/DATA_HEALTH.md:

    - Output rows are sorted (date, symbol) **within each batch**, not globally.
      Sort or group after loading; do not assume a monotonic date index.
    - Sector-neutral and z-scored composites (``value_quality``,
      ``value_quality_sn``, ``roe_sn``) are omitted: they are cross-sectional
      per date, and computing them inside a 400-symbol batch would z-score
      against a fraction of the universe.

    Args:
        symbols: Universe to build.
        factors_out: Destination for the factor panel.
        panel_out: Destination for the PIT metric panel.
        batch_size: Symbols per batch; 400 keeps a batch under ~300 MB.

    Returns:
        Total rows written.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    mcap_raw_dir = ROOT / "data" / "raw" / "fmp" / "market_caps"
    trading_index = pd.read_parquet(PRICES_PATH, columns=[]).index

    def batch_market_caps(batch: list[str]) -> pd.Series:
        """(date, symbol) market-cap series for one batch, read from raw files."""
        frames = []
        for symbol in batch:
            path = mcap_raw_dir / f"{symbol}.parquet"
            if not path.exists():
                continue
            caps = pd.read_parquet(path)
            if caps.empty:
                continue
            if caps.index.tz is None and trading_index.tz is not None:
                caps.index = caps.index.tz_localize(trading_index.tz)
            frames.append(caps["market_cap"].to_frame().assign(symbol=symbol))
        if not frames:
            return pd.Series(dtype="float64")
        stacked = pd.concat(frames)
        stacked.index.name = "date"
        return stacked.set_index("symbol", append=True)["market_cap"]

    factors_writer: pq.ParquetWriter | None = None
    panel_writer: pq.ParquetWriter | None = None
    total_rows = 0
    try:
        for start in range(0, len(symbols), batch_size):
            batch = symbols[start : start + batch_size]
            per_symbol = load_quarterly_frames(batch)
            if not per_symbol:
                continue
            pit_panel = build_pit_fundamentals_panel(per_symbol, trading_index)
            if pit_panel.empty:
                continue

            market_cap = batch_market_caps(list(per_symbol))
            price_dependent = compute_price_dependent_factors(pit_panel, market_cap)
            altman_z = compute_altman_z_score(pit_panel, market_cap)

            statement_columns = [
                c
                for c in pit_panel.columns
                if c not in PRICE_DEPENDENT_INPUT_COLUMNS and c not in LEGACY_PANEL_COLUMNS
            ]
            factors = pd.concat(
                [pit_panel[statement_columns], price_dependent, altman_z], axis=1
            ).astype(PANEL_DTYPE)
            panel_slice = pit_panel[[*PRICE_DEPENDENT_INPUT_COLUMNS, *LEGACY_PANEL_COLUMNS]].astype(
                PANEL_DTYPE
            )

            factors_table = pa.Table.from_pandas(factors)
            panel_table = pa.Table.from_pandas(panel_slice)
            if factors_writer is None:
                factors_writer = pq.ParquetWriter(factors_out, factors_table.schema)
                panel_writer = pq.ParquetWriter(panel_out, panel_table.schema)
            factors_writer.write_table(factors_table)
            panel_writer.write_table(panel_table)
            total_rows += len(factors)
            logger.info(
                "[%d/%d symbols] appended %s rows (total %s)",
                min(start + batch_size, len(symbols)),
                len(symbols),
                f"{len(factors):,}",
                f"{total_rows:,}",
            )
    finally:
        if factors_writer is not None:
            factors_writer.close()
        if panel_writer is not None:
            panel_writer.close()

    logger.info("Wrote %s and %s: %s rows", factors_out.name, panel_out.name, f"{total_rows:,}")
    return total_rows


def write_composites(factors_path: Path, composites_out: Path) -> int:
    """
    Compute the cross-sectional composites in a second pass over two columns.

    ``value_quality`` / ``value_quality_sn`` / ``roe_sn`` are z-scores across the
    cross-section on each date, so a batched build cannot produce them (a batch is
    a fraction of the universe). Computing them afterwards is cheap because only
    ``earnings_yield`` and ``roe`` are needed — two columns instead of forty.

    Args:
        factors_path: The factor panel just written.
        composites_out: Destination for the composite columns.

    Returns:
        Rows written, or 0 when the inputs or the sector file are missing.
    """
    from core.signals.sector_neutral import attach_value_quality_columns

    sectors_path = ROOT / "data" / "sectors" / "sector_classifications.parquet"
    if not sectors_path.exists():
        logger.warning("No sector file at %s; skipping composites", sectors_path)
        return 0

    legs = pd.read_parquet(factors_path, columns=["earnings_yield", "roe"])
    # A batched panel is only sorted within batches; the composites are computed
    # per date, so sort once here rather than assuming order.
    legs = legs.sort_index()
    symbol_to_sector = pd.read_parquet(sectors_path).set_index("symbol")["sector"]
    with_composites = attach_value_quality_columns(legs, symbol_to_sector)

    composite_columns = [c for c in with_composites.columns if c not in ("earnings_yield", "roe")]
    if not composite_columns:
        return 0
    composites = with_composites[composite_columns].astype(PANEL_DTYPE)
    composites.to_parquet(composites_out)
    logger.info(
        "Wrote %s: %s rows, columns %s",
        composites_out.name,
        f"{len(composites):,}",
        composite_columns,
    )
    return len(composites)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PIT fundamentals and factors")
    parser.add_argument("--symbols", type=str, default=None, help="Comma-separated subset")
    parser.add_argument(
        "--expanded",
        action="store_true",
        help="Batched build of the full universe into *_expanded staging artifacts",
    )
    parser.add_argument("--batch-size", type=int, default=400)
    parser.add_argument(
        "--universe-file",
        type=Path,
        default=ROOT / "data" / "raw" / "fmp" / "universe" / "us_equity_universe.parquet",
    )
    args = parser.parse_args()

    if args.expanded:
        from core.data.vendors.fmp.storage import load_universe_symbols, safe_filename

        expanded_symbols = [safe_filename(x) for x in load_universe_symbols(args.universe_file)]
        build_batched(
            expanded_symbols,
            ROOT / "data" / "factors" / "factors_fundamental_expanded.parquet",
            ROOT / "data" / "factors" / "fundamentals_expanded.parquet",
            args.batch_size,
        )
        return

    trading_index = pd.read_parquet(PRICES_PATH, columns=[]).index
    if args.symbols:
        symbols = args.symbols.split(",")
    else:
        # Canonical build: exactly the canonical panel's symbols, never the raw glob.
        import pyarrow.parquet as _pq

        symbols = sorted(
            c
            for c in _pq.ParquetFile(PRICES_PATH).schema_arrow.names
            if c != "date" and not c.startswith("__") and not c.startswith("^")
        )
    # Post-cutover the canonical universe is ~8,900 symbols; dailyizing them in
    # one pass allocates (symbols x dates x columns) ~ 15 GB and gets the process
    # killed. Above the threshold the canonical build uses the same batched writer
    # as --expanded, then computes the cross-sectional composites in a cheap
    # second pass over two columns.
    if not args.symbols and len(symbols) > BATCH_THRESHOLD_SYMBOLS:
        logger.info(
            "%d symbols exceeds the %d single-pass threshold; building in batches",
            len(symbols),
            BATCH_THRESHOLD_SYMBOLS,
        )
        build_batched(symbols, FACTORS_OUT, PIT_PANEL_OUT, args.batch_size)
        write_composites(FACTORS_OUT, COMPOSITES_OUT)
        return

    per_symbol = load_quarterly_frames(symbols)
    if not per_symbol:
        raise SystemExit(f"No usable statements under {RAW_DIR}; run the fetch step first.")

    logger.info("Dailyizing %d symbols onto %d trading days", len(per_symbol), len(trading_index))
    pit_panel = build_pit_fundamentals_panel(per_symbol, trading_index)
    logger.info("Daily panel: %s", pit_panel.shape)

    market_cap = load_market_cap(trading_index)
    price_dependent = compute_price_dependent_factors(pit_panel, market_cap)
    altman_z = compute_altman_z_score(pit_panel, market_cap)

    statement_factor_columns = [
        column
        for column in pit_panel.columns
        if column not in PRICE_DEPENDENT_INPUT_COLUMNS and column not in LEGACY_PANEL_COLUMNS
    ]
    factors = pd.concat(
        [pit_panel[statement_factor_columns], price_dependent, altman_z], axis=1
    ).astype(PANEL_DTYPE)

    sectors_path = ROOT / "data" / "sectors" / "sector_classifications.parquet"
    if sectors_path.exists():
        from core.signals.sector_neutral import attach_value_quality_columns

        symbol_to_sector = pd.read_parquet(sectors_path).set_index("symbol")["sector"]
        factors = attach_value_quality_columns(factors, symbol_to_sector)
    else:
        logger.warning("No sector file at %s; skipping value_quality_sn", sectors_path)

    # A --symbols run is a partial build; write_artifact redirects it to scratch/
    # rather than letting a smoke test replace the full panel.
    subset = bool(args.symbols)
    panel_columns = [*PRICE_DEPENDENT_INPUT_COLUMNS, *LEGACY_PANEL_COLUMNS]
    write_artifact(pit_panel[panel_columns], PIT_PANEL_OUT, subset=subset)

    write_artifact(factors, FACTORS_OUT, subset=subset)
    coverage = factors.notna().mean().sort_values()
    logger.info("Lowest-coverage factors: %s", coverage.head(6).round(3).to_dict())
    logger.info("Highest-coverage factors: %s", coverage.tail(4).round(3).to_dict())


if __name__ == "__main__":
    main()
