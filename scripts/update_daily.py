#!/usr/bin/env python3
"""
Incremental daily update script for quantamental data.

This script:
1. Reads existing Parquet files
2. Fetches only new data since last update
3. Appends new data to Parquet files
4. Rebuilds factors for new dates
5. Refreshes sector classifications (quarterly, >90 days old)
6. Updates DuckDB views

Run daily/weekly to keep data current without full backfill.
Sector classifications are automatically refreshed every 3 months.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.factors.build_factors import build_price_factors
from src.data.factors.io import connect_duckdb, register_parquet
from src.data.factors.macro import compute_macro_zscores, load_default_macro
from src.data.factors.prices import update_prices_panel_incremental
from src.data.sector_classification import (
    add_or_update_sectors,
    get_symbols_needing_refresh,
    load_sector_classifications,
)
from src.utils.io import (
    append_rows_to_parquet,
    get_last_date_from_parquet,
    read_parquet,
    write_parquet,
)


def update_prices(out_root: Path) -> bool:
    """
    Update prices.parquet with new data since last date.
    
    Returns:
        True if new data was added, False otherwise
    """
    prices_path = out_root / 'prices.parquet'
    
    print(f"📈 Updating prices from {prices_path}...")
    
    # Read existing prices
    existing_prices = read_parquet(prices_path)
    
    if existing_prices is None or existing_prices.empty:
        print("⚠️  No existing prices found. Run backfill_all.py first.")
        return False
    
    last_date = existing_prices.index.max()
    print(f"   Last date in prices: {last_date.strftime('%Y-%m-%d')}")
    
    # Check if we need to update (skip if last date is today or in the future)
    # Handle timezone-aware timestamps
    today = pd.Timestamp.now()
    if last_date.tz is not None:
        today = today.tz_localize(last_date.tz)
    
    if last_date >= today.normalize():
        print("   ✅ Prices are already up to date!")
        return False
    
    # Update with new data
    print(f"   Fetching new data since {last_date.strftime('%Y-%m-%d')}...")
    updated_prices = update_prices_panel_incremental(existing_prices)
    
    new_last_date = updated_prices.index.max()
    if new_last_date > last_date:
        new_rows = len(updated_prices) - len(existing_prices)
        print(f"   ✅ Added {new_rows} new dates")
        print(f"   New last date: {new_last_date.strftime('%Y-%m-%d')}")
        write_parquet(updated_prices, prices_path)
        return True
    else:
        print("   ℹ️  No new data available")
        return False


def update_macro(out_root: Path) -> bool:
    """
    Update macro.parquet and macro_z.parquet with new data.
    
    Returns:
        True if new data was added, False otherwise
    """
    macro_path = out_root / 'macro.parquet'
    macro_z_path = out_root / 'macro_z.parquet'
    
    print(f"📊 Updating macro from {macro_path}...")
    
    existing_macro = read_parquet(macro_path)
    
    if existing_macro is None or existing_macro.empty:
        print("⚠️  No existing macro data found. Run backfill_all.py first.")
        return False
    
    last_date = existing_macro.index.max()
    print(f"   Last date in macro: {last_date.strftime('%Y-%m-%d')}")
    
    # Fetch fresh macro data (FRED API typically provides full history)
    print("   Fetching latest macro data...")
    new_macro = load_default_macro()
    
    new_last_date = new_macro.index.max()
    
    if new_last_date > last_date:
        print(f"   ✅ New macro data available through {new_last_date.strftime('%Y-%m-%d')}")
        
        # Compute z-scores
        macro_z = compute_macro_zscores(new_macro)
        
        # Write updated files
        write_parquet(new_macro, macro_path)
        write_parquet(macro_z, macro_z_path)
        return True
    else:
        print("   ℹ️  Macro data is up to date")
        return False


def rebuild_factors(out_root: Path, prices_updated: bool) -> None:
    """
    Rebuild price factors if prices were updated.
    
    Args:
        out_root: Output directory
        prices_updated: Whether prices were updated
    """
    if not prices_updated:
        print("📉 Skipping factor rebuild (no new price data)")
        return
    
    print("📉 Rebuilding price factors...")
    
    prices_path = out_root / 'prices.parquet'
    factors_price_path = out_root / 'factors_price.parquet'
    factors_all_path = out_root / 'factors_all.parquet'
    
    # Read updated prices
    prices = read_parquet(prices_path)
    
    if prices is None or prices.empty:
        print("⚠️  No prices available for factor calculation")
        return
    
    # Build price factors
    market_symbol = '^GSPC'
    factors_price = build_price_factors(prices, market_symbol=market_symbol)
    
    print(f"   ✅ Rebuilt price factors: {factors_price.shape}")
    
    # Write factors
    write_parquet(factors_price, factors_price_path)
    
    # For now, factors_all = factors_price (until fundamentals are added)
    write_parquet(factors_price, factors_all_path)


def update_sectors_if_needed(out_root: Path) -> bool:
    """
    Check if sector classifications need quarterly refresh.
    Only updates symbols that are >90 days old.
    
    Returns:
        True if sectors were updated, False otherwise
    """
    print("📊 Checking sector classifications...")
    
    # Load existing sector data
    sector_df = load_sector_classifications()
    
    if sector_df is None or sector_df.empty:
        print("   ℹ️  No sector data found - run fetch_sectors.py first")
        return False
    
    # Get symbols needing refresh (>90 days old)
    stale_symbols = get_symbols_needing_refresh(sector_df, refresh_days=90)
    
    if not stale_symbols:
        print("   ✅ Sector classifications are up to date")
        return False
    
    print(f"   Found {len(stale_symbols)} symbols needing quarterly refresh")
    print(f"   Updating sectors (this may take a few minutes)...")
    
    # Update stale sectors
    add_or_update_sectors(stale_symbols, force_refresh=True)
    
    print(f"   ✅ Updated {len(stale_symbols)} sector classifications")
    return True


def update_duckdb_views(out_root: Path, db_path: Path) -> None:
    """
    Update DuckDB views to point to updated Parquet files.
    
    Args:
        out_root: Directory containing Parquet files
        db_path: Path to DuckDB database
    """
    print(f"🦆 Updating DuckDB views at {db_path}...")
    
    con = connect_duckdb(db_path)
    
    # Register all views
    views = {
        'prices': out_root / 'prices.parquet',
        'macro': out_root / 'macro.parquet',
        'macro_z': out_root / 'macro_z.parquet',
        'factors_price': out_root / 'factors_price.parquet',
        'factors_all': out_root / 'factors_all.parquet',
    }
    
    for name, path in views.items():
        if path.exists():
            register_parquet(con, name, path)
            print(f"   ✅ Registered view: {name}")
    
    con.close()


def main(out_root: str = 'data/factors', db_path: str = 'data/factors/factors.duckdb'):
    """
    Main incremental update workflow.
    
    Args:
        out_root: Output directory for Parquet files
        db_path: Path to DuckDB database
    """
    out_root_p = Path(out_root)
    db_path_p = Path(db_path)
    
    print("=" * 80)
    print("🔄 INCREMENTAL DATA UPDATE")
    print("=" * 80)
    print()
    
    # Step 1: Update prices
    prices_updated = update_prices(out_root_p)
    print()
    
    # Step 2: Update macro
    macro_updated = update_macro(out_root_p)
    print()
    
    # Step 3: Rebuild factors if prices changed
    rebuild_factors(out_root_p, prices_updated)
    print()
    
    # Step 4: Update sector classifications (quarterly refresh)
    sectors_updated = update_sectors_if_needed(out_root_p)
    print()
    
    # Step 5: Update DuckDB views
    update_duckdb_views(out_root_p, db_path_p)
    print()
    
    print("=" * 80)
    if prices_updated or macro_updated or sectors_updated:
        print("✅ Incremental update completed successfully!")
        if sectors_updated:
            print("   📊 Sector classifications refreshed (quarterly update)")
    else:
        print("✅ Data is already up to date - no changes needed")
    print("=" * 80)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Incremental daily/weekly data update')
    p.add_argument('--out', type=str, default='data/factors', help='Output directory for Parquet files')
    p.add_argument('--db', type=str, default='data/factors/factors.duckdb', help='Path to DuckDB database')
    args = p.parse_args()
    
    main(args.out, args.db)


