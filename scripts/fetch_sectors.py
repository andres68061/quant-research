#!/usr/bin/env python3
"""
Fetch sector classifications for all stocks in the database.

This script:
1. Reads all symbols from prices.parquet
2. Fetches sector/industry classifications from Yahoo Finance
3. Saves to data/sectors/sector_classifications.parquet
4. Labels stocks without classification as 'Unknown'

Usage:
    python scripts/fetch_sectors.py
    python scripts/fetch_sectors.py --force  # Force refresh all
    python scripts/fetch_sectors.py --symbols AAPL MSFT  # Specific symbols
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.sector_classification import (
    add_or_update_sectors,
    get_sector_summary,
    load_sector_classifications,
)
from src.utils.io import read_parquet


def get_all_symbols_from_prices(prices_path: Path) -> list:
    """
    Extract all symbols from prices.parquet.
    
    Args:
        prices_path: Path to prices.parquet file
        
    Returns:
        List of ticker symbols
    """
    print(f"📈 Reading symbols from {prices_path}...")
    
    df = read_parquet(prices_path)
    
    if df is None or df.empty:
        print("❌ No price data found")
        return []
    
    symbols = df.columns.tolist()
    print(f"   Found {len(symbols)} symbols")
    
    return symbols


def main():
    """Fetch sector classifications for all stocks."""
    parser = argparse.ArgumentParser(
        description='Fetch sector classifications for stocks',
        epilog='Example: python scripts/fetch_sectors.py'
    )
    parser.add_argument(
        '--prices',
        type=str,
        default='data/factors/prices.parquet',
        help='Path to prices.parquet (default: data/factors/prices.parquet)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force refresh all symbols (even if not stale)'
    )
    parser.add_argument(
        '--symbols',
        nargs='+',
        help='Specific symbols to fetch (default: all from prices.parquet)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("📊 FETCH SECTOR CLASSIFICATIONS")
    print("=" * 80)
    print()
    
    # Get symbols to process
    if args.symbols:
        # User provided specific symbols
        symbols = [s.upper() for s in args.symbols]
        print(f"Processing {len(symbols)} specified symbols: {', '.join(symbols)}")
    else:
        # Get all symbols from prices.parquet
        prices_path = Path(args.prices)
        
        if not prices_path.exists():
            print(f"❌ Prices file not found: {prices_path}")
            print("   Run backfill_all.py first to create price data")
            return
        
        symbols = get_all_symbols_from_prices(prices_path)
        
        if not symbols:
            print("❌ No symbols found in prices file")
            return
        
        print(f"Processing all {len(symbols)} symbols from prices.parquet")
    
    print()
    
    # Check existing classifications
    existing_df = load_sector_classifications()
    
    if existing_df is not None and not existing_df.empty:
        print(f"📋 Existing classifications: {len(existing_df)} symbols")
        
        if args.force:
            print("   ⚠️  Force refresh enabled - will update all symbols")
        else:
            print("   ℹ️  Will only fetch new/stale symbols (>90 days old)")
    else:
        print("📋 No existing classifications - will fetch all symbols")
    
    print()
    print("=" * 80)
    print("🚀 STARTING FETCH")
    print("=" * 80)
    print()
    
    # Fetch/update sectors
    updated_df = add_or_update_sectors(symbols, force_refresh=args.force)
    
    print()
    print("=" * 80)
    print("📊 RESULTS")
    print("=" * 80)
    print()
    
    # Show summary
    print(f"✅ Total symbols with classifications: {len(updated_df)}")
    print()
    
    # Count unknowns
    unknown_count = (updated_df['sector'] == 'Unknown').sum()
    if unknown_count > 0:
        print(f"⚠️  Symbols with unknown sector: {unknown_count}")
        unknown_symbols = updated_df[
            updated_df['sector'] == 'Unknown'
        ]['symbol'].tolist()
        print(f"   {', '.join(unknown_symbols[:10])}")
        if len(unknown_symbols) > 10:
            print(f"   ... and {len(unknown_symbols) - 10} more")
        print()
    
    # Show sector breakdown
    print("📈 Sector Breakdown:")
    print()
    summary = get_sector_summary()
    
    for _, row in summary.head(15).iterrows():
        sector = row['sector']
        count = row['count']
        pct = row['percentage']
        
        # Create bar visualization
        bar_length = int(pct / 2)  # Scale to fit terminal
        bar = '█' * bar_length
        
        print(f"   {sector:25s} {count:4d} ({pct:5.1f}%) {bar}")
    
    if len(summary) > 15:
        print(f"   ... and {len(summary) - 15} more sectors")
    
    print()
    print("=" * 80)
    print("✅ SECTOR FETCH COMPLETE")
    print("=" * 80)
    print()
    print(f"📁 Saved to: data/sectors/sector_classifications.parquet")
    print()
    print("Next steps:")
    print("  - Run quarterly: python scripts/update_sectors.py")
    print("  - Add new stock: python scripts/add_symbol.py TICKER")
    print("  - View summary: python -m src.data.sector_classification")
    print()


if __name__ == '__main__':
    main()
