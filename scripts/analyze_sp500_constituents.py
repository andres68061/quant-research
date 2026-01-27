#!/usr/bin/env python3
"""
Analyze S&P 500 Historical Constituents

This script:
1. Loads the historical S&P 500 constituents data
2. Identifies which tickers need price data
3. Suggests which tickers to fetch
4. Shows summary statistics and changes over time
"""

import sys
from pathlib import Path

import pandas as pd

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.sp500_constituents import SP500Constituents


def get_existing_symbols() -> set:
    """Get list of symbols we have price data for."""
    prices_file = ROOT / "data" / "factors" / "prices.parquet"
    if prices_file.exists():
        df = pd.read_parquet(prices_file)
        # Tickers are columns, not rows
        return set(df.columns.tolist())
    return set()


def check_existing_data(sp500: SP500Constituents) -> pd.DataFrame:
    """Check which S&P 500 constituents we have data for."""
    all_tickers = sp500.get_ticker_universe()
    existing_symbols = get_existing_symbols()
    
    results = []
    for ticker in sorted(all_tickers):
        has_data = ticker in existing_symbols
        results.append({
            'ticker': ticker,
            'has_data': has_data
        })
    
    return pd.DataFrame(results)


def main():
    print("=" * 80)
    print("S&P 500 HISTORICAL CONSTITUENTS ANALYZER")
    print("=" * 80)
    
    # Load S&P 500 constituents
    sp500 = SP500Constituents()
    sp500.load()
    
    # Get summary stats
    stats = sp500.get_summary_stats()
    
    print("\n📊 SUMMARY STATISTICS")
    print("-" * 80)
    print(f"Date Range: {stats['start_date'].date()} to {stats['end_date'].date()}")
    print(f"Number of Dates: {stats['num_dates']:,}")
    print(f"Total Unique Tickers: {stats['total_unique_tickers']:,}")
    print(f"Total Changes: {stats['total_changes']:,}")
    print(f"Total Additions: {stats['total_additions']:.0f}")
    print(f"Total Removals: {stats['total_removals']:.0f}")
    print(f"Avg Additions per Change: {stats['avg_additions_per_change']:.1f}")
    print(f"Avg Removals per Change: {stats['avg_removals_per_change']:.1f}")
    
    # Check existing data
    print("\n💾 CHECKING EXISTING PRICE DATA")
    print("-" * 80)
    
    data_status = check_existing_data(sp500)
    
    have_data = data_status[data_status['has_data']].shape[0]
    missing_data = data_status[~data_status['has_data']].shape[0]
    total = data_status.shape[0]
    pct_complete = (have_data / total) * 100
    
    print(f"Tickers with Data: {have_data:,} ({pct_complete:.1f}%)")
    print(f"Tickers Missing Data: {missing_data:,}")
    
    # Show some missing tickers
    missing_tickers = data_status[~data_status['has_data']]['ticker'].tolist()
    if missing_tickers:
        print(f"\nSample Missing Tickers (first 20):")
        for i, ticker in enumerate(missing_tickers[:20], 1):
            print(f"  {i:2d}. {ticker}")
        
        if len(missing_tickers) > 20:
            print(f"  ... and {len(missing_tickers) - 20} more")
    
    # Show major additions/removals
    print("\n🔄 RECENT INDEX CHANGES (Last 5 Years)")
    print("-" * 80)
    
    recent_date = pd.Timestamp.now() - pd.DateOffset(years=5)
    changes_df = sp500.get_additions_and_removals(start_date=recent_date)
    
    if not changes_df.empty:
        print(f"\nTotal Changes: {len(changes_df)}")
        print(f"\nMost Recent Changes:")
        for _, change in changes_df.tail(10).iterrows():
            date_str = change['date'].strftime('%Y-%m-%d')
            if change['additions']:
                print(f"\n  {date_str} - ADDED {len(change['additions'])}:")
                for ticker in change['additions']:
                    print(f"    + {ticker}")
            if change['removals']:
                print(f"\n  {date_str} - REMOVED {len(change['removals'])}:")
                for ticker in change['removals']:
                    print(f"    - {ticker}")
    
    # Show ticker longevity
    print("\n📅 TICKER LONGEVITY")
    print("-" * 80)
    
    ticker_dates = {}
    for ticker in sp500.get_ticker_universe():
        history = sp500.get_ticker_history(ticker)
        in_sp500 = history[history['in_sp500']]
        if not in_sp500.empty:
            ticker_dates[ticker] = {
                'first_date': in_sp500['date'].min(),
                'last_date': in_sp500['date'].max(),
                'days': (in_sp500['date'].max() - in_sp500['date'].min()).days
            }
    
    # Sort by longevity
    longevity_df = pd.DataFrame.from_dict(ticker_dates, orient='index')
    longevity_df = longevity_df.sort_values('days', ascending=False)
    
    print("\nLongest Running S&P 500 Members:")
    for i, (ticker, row) in enumerate(longevity_df.head(10).iterrows(), 1):
        years = row['days'] / 365.25
        print(f"  {i:2d}. {ticker:8s} - {years:.1f} years "
              f"({row['first_date'].date()} to {row['last_date'].date()})")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS")
    print("-" * 80)
    
    if missing_data > 0:
        print(f"\n1. Fetch missing price data for {missing_data} tickers:")
        print(f"   Run: python scripts/fetch_sp500_prices.py")
        print(f"   (This may take a while - ~{missing_data} API calls)")
        
        # Estimate time
        seconds_per_call = 1.5  # Conservative estimate with rate limits
        total_minutes = (missing_data * seconds_per_call) / 60
        print(f"   Estimated time: ~{total_minutes:.0f} minutes")
    else:
        print("\n✓ All S&P 500 constituents have price data!")
    
    print("\n2. Use point-in-time constituents in backtesting:")
    print("   from src.data.sp500_constituents import SP500Constituents")
    print("   sp500 = SP500Constituents()")
    print("   universe = sp500.get_constituents_on_date(backtest_date)")
    
    print("\n3. Enable 'S&P 500 Historical' benchmark in portfolio simulator")
    
    print("\n" + "=" * 80)
    print("✓ ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

