#!/usr/bin/env python3
"""
Fetch Shares Outstanding and Calculate Market Caps

This script:
1. Fetches shares outstanding from Yahoo Finance (FREE!)
2. Calculates historical market caps: market_cap = shares × price
3. Saves results for use in cap-weighted benchmarks

No paid API required - uses existing price data + free yfinance shares data.
"""

import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.data.market_caps import MarketCapCalculator
from src.data.sp500_constituents import SP500Constituents


def main(tickers_source: str = 'sp500', batch_size: int = 0):
    """
    Main execution.
    
    Args:
        tickers_source: 'sp500' (all S&P 500 historical) or 'current' (only current prices)
        batch_size: Number to fetch (0 = all)
    """
    print("=" * 80)
    print("SHARES OUTSTANDING & MARKET CAP CALCULATOR")
    print("=" * 80)
    
    # Initialize calculator
    calc = MarketCapCalculator()
    
    # Get ticker list
    if tickers_source == 'sp500':
        print("\n📋 Using S&P 500 historical constituents")
        sp500 = SP500Constituents()
        sp500.load()
        all_tickers = sorted(sp500.get_ticker_universe())
    else:
        print("\n📋 Using tickers from current price data")
        prices = calc.load_prices()
        all_tickers = sorted(prices.columns.tolist())
    
    print(f"Total tickers: {len(all_tickers)}")
    
    # Check existing shares data
    existing_shares = calc.load_shares_outstanding()
    if not existing_shares.empty:
        already_have = set(existing_shares['ticker'].unique())
        remaining = [t for t in all_tickers if t not in already_have]
        print(f"Already have shares for: {len(already_have)}")
        print(f"Need to fetch: {len(remaining)}")
        tickers_to_fetch = remaining
    else:
        print("No existing shares data - will fetch all")
        tickers_to_fetch = all_tickers
    
    if not tickers_to_fetch:
        print("\n✅ Already have shares for all tickers!")
    else:
        # Limit to batch size if specified
        if batch_size > 0:
            tickers_to_fetch = tickers_to_fetch[:batch_size]
            print(f"\nFetching batch of {len(tickers_to_fetch)} tickers")
        
        # Fetch shares outstanding
        print("\n" + "=" * 80)
        print("STEP 1: FETCH SHARES OUTSTANDING")
        print("=" * 80)
        
        shares_df = calc.fetch_all_shares_outstanding(
            tickers_to_fetch,
            delay=0.5,
            resume=True
        )
    
    # Calculate market caps
    print("\n" + "=" * 80)
    print("STEP 2: CALCULATE HISTORICAL MARKET CAPS")
    print("=" * 80)
    
    market_caps_df = calc.calculate_market_caps(save=True)
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    
    stats = calc.get_summary_stats()
    print(f"\nShares Outstanding:")
    print(f"  Tickers: {stats['tickers_with_shares']}")
    
    print(f"\nMarket Caps:")
    print(f"  Tickers: {stats['tickers_with_market_caps']}")
    if stats['date_range']:
        print(f"  Date Range: {stats['date_range'][0]} to {stats['date_range'][1]}")
    print(f"  Total Records: {stats['total_records']:,}")
    
    # Test example
    if not market_caps_df.empty:
        print(f"\n💡 EXAMPLE USAGE:")
        print("-" * 80)
        
        test_date = market_caps_df.index.get_level_values('date').max()
        print(f"\nMarket caps on {test_date.date()}:")
        
        caps_on_date = calc.get_market_cap_on_date(test_date)
        top_5 = caps_on_date.nlargest(5)
        
        print("\nTop 5 by market cap:")
        for ticker, cap in top_5.items():
            cap_trillion = cap / 1e12
            print(f"  {ticker:10s}: ${cap_trillion:8.2f}T")
        
        # Show weights
        weights = calc.get_weights_on_date(test_date)
        top_5_weights = weights.nlargest(5)
        
        print("\nTop 5 weights in cap-weighted index:")
        for ticker, weight in top_5_weights.items():
            print(f"  {ticker:10s}: {weight*100:6.2f}%")
    
    print("\n" + "=" * 80)
    print("✅ COMPLETE")
    print("=" * 80)
    
    print("\nData saved to:")
    print(f"  • Shares: {calc.shares_file}")
    print(f"  • Market Caps: {calc.market_caps_file}")
    
    print("\nNext steps:")
    print("  1. Use in portfolio simulator for cap-weighted S&P 500 benchmark")
    print("  2. Run daily to keep market caps updated")
    print("  3. Integrate into update_daily.py for automatic updates")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Fetch shares outstanding and calculate market caps"
    )
    parser.add_argument(
        "--source",
        choices=['sp500', 'current'],
        default='current',
        help="Ticker source: 'sp500' (all historical) or 'current' (current prices only)"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=0,
        help="Batch size for shares fetching (0 = all, default: 0)"
    )
    
    args = parser.parse_args()
    
    main(tickers_source=args.source, batch_size=args.batch)

