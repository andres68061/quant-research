#!/usr/bin/env python3
"""
Fetch Commodities Data - Initial Load

Fetches full historical data for all configured commodities and saves to parquet.
Run this once to initialize the commodities database.

Usage:
    python scripts/fetch_commodities.py
"""

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.commodities import CommodityDataFetcher, COMMODITIES_CONFIG


def main():
    """Fetch and save all commodities data."""
    print("=" * 70)
    print("COMMODITIES DATA FETCHER - Initial Load")
    print("=" * 70)
    print()

    fetcher = CommodityDataFetcher()
    
    print(f"Data directory: {fetcher.data_dir}")
    print(f"Output file: {fetcher.prices_file}")
    print()
    
    print(f"Fetching {len(COMMODITIES_CONFIG)} commodities...")
    print("-" * 70)
    
    # Fetch all data
    df = fetcher.fetch_all_commodities()
    
    if df.empty:
        print()
        print("✗ No data fetched. Please check:")
        print("  1. Internet connection")
        print("  2. Alpha Vantage API key in .env file")
        print("  3. API rate limits")
        sys.exit(1)
    
    print()
    print("-" * 70)
    print(f"✓ Successfully fetched {len(df.columns)} commodities")
    print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")
    print(f"  Total rows: {len(df):,}")
    print()
    
    # Display summary
    print("Summary by commodity:")
    print("-" * 70)
    for col in df.columns:
        non_null = df[col].notna().sum()
        first_date = df[col].first_valid_index()
        last_date = df[col].last_valid_index()
        latest_price = df[col].iloc[-1] if pd.notna(df[col].iloc[-1]) else None
        
        if first_date and last_date:
            print(f"  {col:15s}: {non_null:5,} days  "
                  f"({first_date.date()} to {last_date.date()})  "
                  f"Latest: ${latest_price:.2f}" if latest_price else "N/A")
    
    print()
    
    # Save to parquet
    fetcher.save_prices(df)
    
    print()
    print("=" * 70)
    print("✓ DONE - Commodities data initialized")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  • Run 'python scripts/update_commodities.py' to update data")
    print("  • Add to cron for daily updates")
    print("  • View in Streamlit app: Metals Analytics page")
    print()


if __name__ == "__main__":
    import pandas as pd
    main()

