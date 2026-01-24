#!/usr/bin/env python3
"""
Update Commodities Data - Incremental Updates

Updates commodity prices with new data since last fetch.
Should be run daily (can be added to cron).

Usage:
    python scripts/update_commodities.py
    python scripts/update_commodities.py GLD SLV  # Update specific commodities
"""

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.commodities import CommodityDataFetcher, COMMODITIES_CONFIG


def main(symbols=None):
    """
    Update commodities data incrementally.
    
    Args:
        symbols: List of commodity symbols to update. If None, updates all.
    """
    print("=" * 70)
    print("COMMODITIES DATA UPDATER")
    print("=" * 70)
    print()

    fetcher = CommodityDataFetcher()
    
    if not fetcher.prices_file.exists():
        print("✗ No existing data found!")
        print()
        print("Please run the initial fetch first:")
        print("  python scripts/fetch_commodities.py")
        print()
        sys.exit(1)
    
    # Load existing data to show current state
    existing_df = fetcher.load_prices()
    print(f"Current data: {len(existing_df)} rows, {len(existing_df.columns)} commodities")
    print(f"Last date: {existing_df.index[-1].date()}")
    print()
    
    # Determine which symbols to update
    if symbols is None:
        symbols_to_update = list(COMMODITIES_CONFIG.keys())
        print(f"Updating all {len(symbols_to_update)} commodities...")
    else:
        symbols_to_update = symbols
        print(f"Updating {len(symbols_to_update)} commodities: {', '.join(symbols_to_update)}")
    
    print("-" * 70)
    
    # Update each commodity
    updated_df = existing_df.copy()
    update_count = 0
    
    for symbol in symbols_to_update:
        if symbol not in COMMODITIES_CONFIG:
            print(f"⚠️  Unknown commodity: {symbol}")
            continue
        
        # Get last date for this commodity
        if symbol in updated_df.columns:
            last_date = updated_df[symbol].last_valid_index()
            if last_date:
                print(f"\n{symbol}: Last date = {last_date.date()}")
        else:
            print(f"\n{symbol}: New commodity (full fetch)")
        
        # Update
        updated_df = fetcher.update_commodity(symbol)
        
        # Check if we got new data
        if symbol in updated_df.columns:
            new_last_date = updated_df[symbol].last_valid_index()
            if last_date is None or (new_last_date and new_last_date > last_date):
                update_count += 1
    
    print()
    print("-" * 70)
    
    if update_count > 0:
        print(f"✓ Updated {update_count} commodities")
        print(f"  New data range: {updated_df.index[0].date()} to {updated_df.index[-1].date()}")
        print(f"  Total rows: {len(updated_df):,}")
        print()
        
        # Save updated data
        fetcher.save_prices(updated_df)
        
        print()
        print("=" * 70)
        print("✓ DONE - Commodities data updated")
        print("=" * 70)
    else:
        print("✓ All commodities already up to date")
        print()
        print("=" * 70)
        print("✓ DONE - No updates needed")
        print("=" * 70)


if __name__ == "__main__":
    import pandas as pd
    
    # Check if specific symbols were provided as arguments
    if len(sys.argv) > 1:
        symbols = sys.argv[1:]
        main(symbols=symbols)
    else:
        main()

