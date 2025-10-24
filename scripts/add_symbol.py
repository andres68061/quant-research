#!/usr/bin/env python3
"""
Add new stock symbol(s) to the quantamental database.

This script:
1. Fetches full history for new symbol(s)
2. Adds them to prices.parquet
3. Rebuilds factors including the new symbol(s)
4. Updates DuckDB views

Usage:
    python scripts/add_symbol.py NVDA
    python scripts/add_symbol.py NVDA TSLA AMZN
"""

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.factors.build_factors import build_price_factors
from src.data.factors.io import connect_duckdb, register_parquet
from src.data.factors.prices import add_symbol_to_panel
from src.utils.io import read_parquet, write_parquet


def add_symbols(symbols: List[str], out_root: Path, db_path: Path) -> None:
    """
    Add new symbols to the quantamental database.
    
    Args:
        symbols: List of ticker symbols to add
        out_root: Output directory for Parquet files
        db_path: Path to DuckDB database
    """
    prices_path = out_root / 'prices.parquet'
    factors_price_path = out_root / 'factors_price.parquet'
    factors_all_path = out_root / 'factors_all.parquet'
    
    print("=" * 80)
    print(f"➕ ADDING {len(symbols)} NEW SYMBOL(S)")
    print("=" * 80)
    print(f"Symbols: {', '.join(symbols)}")
    print()
    
    # Step 1: Read existing prices
    print(f"📈 Reading existing prices from {prices_path}...")
    existing_prices = read_parquet(prices_path)
    
    if existing_prices is None or existing_prices.empty:
        print("⚠️  No existing prices found. Run backfill_all.py first.")
        return
    
    existing_symbols = existing_prices.columns.tolist()
    print(f"   Current symbols: {len(existing_symbols)}")
    
    # Filter out symbols that already exist
    new_symbols = [s for s in symbols if s not in existing_symbols]
    already_exist = [s for s in symbols if s in existing_symbols]
    
    if already_exist:
        print(f"   ℹ️  Already have: {', '.join(already_exist)}")
    
    if not new_symbols:
        print("   ✅ All symbols already exist - nothing to add")
        return
    
    print(f"   Adding {len(new_symbols)} new symbols: {', '.join(new_symbols)}")
    print()
    
    # Step 2: Add new symbols to prices
    print("📥 Fetching full history for new symbols...")
    updated_prices = existing_prices.copy()
    
    for symbol in new_symbols:
        print(f"   Fetching {symbol}...")
        updated_prices = add_symbol_to_panel(updated_prices, symbol)
        
        if symbol in updated_prices.columns:
            rows = updated_prices[symbol].notna().sum()
            print(f"   ✅ Added {symbol}: {rows} data points")
        else:
            print(f"   ⚠️  Failed to fetch {symbol}")
    
    # Step 3: Save updated prices
    print()
    print(f"💾 Saving updated prices...")
    write_parquet(updated_prices, prices_path)
    print(f"   ✅ Saved: {updated_prices.shape}")
    print()
    
    # Step 4: Rebuild all factors
    print("📉 Rebuilding all factors (this may take a moment)...")
    market_symbol = '^GSPC'
    factors_price = build_price_factors(updated_prices, market_symbol=market_symbol)
    
    print(f"   ✅ Rebuilt price factors: {factors_price.shape}")
    
    # Write factors
    write_parquet(factors_price, factors_price_path)
    write_parquet(factors_price, factors_all_path)
    print()
    
    # Step 5: Update DuckDB views
    print(f"🦆 Updating DuckDB views at {db_path}...")
    con = connect_duckdb(db_path)
    
    register_parquet(con, 'prices', prices_path)
    register_parquet(con, 'factors_price', factors_price_path)
    register_parquet(con, 'factors_all', factors_all_path)
    
    con.close()
    print("   ✅ DuckDB views updated")
    print()
    
    print("=" * 80)
    print(f"✅ Successfully added {len(new_symbols)} new symbol(s)!")
    print("=" * 80)


def main():
    """Parse arguments and add symbols."""
    parser = argparse.ArgumentParser(
        description='Add new stock symbol(s) to the quantamental database',
        epilog='Example: python scripts/add_symbol.py NVDA TSLA'
    )
    parser.add_argument(
        'symbols',
        nargs='+',
        help='Stock ticker symbol(s) to add'
    )
    parser.add_argument(
        '--out',
        type=str,
        default='data/factors',
        help='Output directory for Parquet files (default: data/factors)'
    )
    parser.add_argument(
        '--db',
        type=str,
        default='data/factors/factors.duckdb',
        help='Path to DuckDB database (default: data/factors/factors.duckdb)'
    )
    
    args = parser.parse_args()
    
    # Normalize symbols to uppercase
    symbols = [s.upper() for s in args.symbols]
    
    out_root = Path(args.out)
    db_path = Path(args.db)
    
    add_symbols(symbols, out_root, db_path)


if __name__ == '__main__':
    main()

