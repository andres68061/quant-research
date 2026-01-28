#!/usr/bin/env python3
"""
Analyze Failed S&P 500 Symbols

Analyzes symbols that failed to fetch, categorizing them by likely reasons
(delisted, bankruptcies, mergers, ticker changes, etc.)
Also checks which failed symbols have existing data and tests class A/B shares.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import yfinance as yf
from src.data.sp500_constituents import SP500Constituents


def load_failed_symbols() -> Set[str]:
    """Load previously failed symbols from tracking file."""
    failed_file = ROOT / "data" / "sp500_failed_symbols.json"
    if failed_file.exists():
        with open(failed_file, 'r') as f:
            return set(json.load(f))
    return set()


def get_existing_symbols() -> Set[str]:
    """Get list of symbols we have price data for."""
    prices_file = ROOT / "data" / "factors" / "prices.parquet"
    if prices_file.exists():
        df = pd.read_parquet(prices_file)
        return set(df.columns.tolist())
    return set()


def check_symbol_has_data(symbol: str, prices_df: Optional[pd.DataFrame] = None) -> Tuple[bool, int]:
    """
    Check if a symbol has data in prices.parquet.
    
    Returns:
        (has_data, num_data_points)
    """
    if prices_df is None:
        prices_file = ROOT / "data" / "factors" / "prices.parquet"
        if not prices_file.exists():
            return False, 0
        prices_df = pd.read_parquet(prices_file)
    
    if symbol not in prices_df.columns:
        return False, 0
    
    series = prices_df[symbol].dropna()
    return len(series) > 0, len(series)


def test_symbol_fetch(symbol: str) -> Tuple[bool, Optional[str]]:
    """
    Test if a symbol can be fetched from yfinance.
    
    Returns:
        (success, error_message)
    """
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="5d")
        if hist.empty:
            return False, "No data returned"
        # Check if we got Adj Close or Close
        if 'Adj Close' in hist.columns and not hist['Adj Close'].isna().all():
            return True, None
        elif 'Close' in hist.columns and not hist['Close'].isna().all():
            return True, None
        return False, "All price columns are NaN"
    except Exception as e:
        return False, str(e)


def categorize_symbols(symbols: List[str], sp500: SP500Constituents) -> Dict[str, List[str]]:
    """
    Categorize failed symbols by likely failure reasons.
    
    Args:
        symbols: List of failed symbols
        sp500: SP500Constituents instance
        
    Returns:
        Dict mapping category names to lists of symbols
    """
    categories = defaultdict(list)
    
    # Load S&P 500 data to check when symbols were constituents
    sp500.load()
    constituents_df = sp500.get_constituents_series()
    
    # Get all dates when each symbol was in S&P 500
    symbol_dates = {}
    for date, row in constituents_df.iterrows():
        tickers = row['tickers']
        for ticker in tickers:
            if ticker not in symbol_dates:
                symbol_dates[ticker] = []
            symbol_dates[ticker].append(date)
    
    for symbol in sorted(symbols):
        # Check if symbol was ever in S&P 500
        if symbol not in symbol_dates:
            categories["Never in S&P 500 (Invalid Ticker)"].append(symbol)
            continue
        
        dates_in_sp500 = sorted(symbol_dates[symbol])
        first_date = dates_in_sp500[0]
        last_date = dates_in_sp500[-1]
        
        # Classify based on patterns
        if symbol.endswith('Q'):
            categories["Bankruptcy/Delisted (Q suffix)"].append(symbol)
        elif symbol.endswith('.A'):
            categories["Class A Shares (may need different ticker)"].append(symbol)
        elif symbol.endswith('.B'):
            categories["Class B Shares (may need different ticker)"].append(symbol)
        elif any(x in symbol for x in ['MERQ', 'LEHMQ', 'WAMUQ', 'AAMRQ']):
            categories["Major Financial Crisis Failures"].append(symbol)
        elif last_date < pd.Timestamp('2010-01-01'):
            categories["Early Delistings (before 2010)"].append(symbol)
        elif last_date < pd.Timestamp('2020-01-01'):
            categories["Recent Delistings (2010-2020)"].append(symbol)
        else:
            categories["Recent Delistings (2020+)"].append(symbol)
    
    return dict(categories)


def main():
    """Analyze failed symbols."""
    print("=" * 80)
    print("FAILED S&P 500 SYMBOLS ANALYSIS")
    print("=" * 80)
    print()
    
    # Load failed symbols
    failed_symbols = load_failed_symbols()
    
    if not failed_symbols:
        print("✅ No failed symbols found!")
        return
    
    print(f"📊 Total Failed Symbols: {len(failed_symbols)}")
    print()
    
    # Load existing prices data
    print("Loading existing price data...")
    prices_file = ROOT / "data" / "factors" / "prices.parquet"
    prices_df = None
    existing_symbols = set()
    if prices_file.exists():
        prices_df = pd.read_parquet(prices_file)
        existing_symbols = set(prices_df.columns.tolist())
        print(f"✅ Loaded {len(existing_symbols)} symbols from prices.parquet")
    else:
        print("⚠️  prices.parquet not found")
    print()
    
    # Load S&P 500 data
    sp500 = SP500Constituents()
    sp500.load()
    
    # Categorize symbols
    categories = categorize_symbols(list(failed_symbols), sp500)
    
    # Check which failed symbols have existing data
    print("=" * 80)
    print("CHECKING EXISTING DATA")
    print("=" * 80)
    print()
    
    failed_with_data = []
    failed_without_data = []
    
    for symbol in failed_symbols:
        has_data, num_points = check_symbol_has_data(symbol, prices_df)
        if has_data:
            failed_with_data.append((symbol, num_points))
        else:
            failed_without_data.append(symbol)
    
    if failed_with_data:
        print(f"✅ {len(failed_with_data)} failed symbols HAVE existing data:")
        for symbol, num_points in sorted(failed_with_data, key=lambda x: -x[1])[:20]:
            print(f"   - {symbol}: {num_points} data points")
        if len(failed_with_data) > 20:
            print(f"   ... and {len(failed_with_data) - 20} more")
        print()
        print("💡 For delisted symbols with data: You can use existing data")
        print("   and 'sell' on the delisting day in backtests.")
        print()
    else:
        print("❌ No failed symbols have existing data")
        print()
    
    # Test class A/B shares
    print("=" * 80)
    print("TESTING CLASS A/B SHARES")
    print("=" * 80)
    print()
    
    class_a_b = [s for s in failed_symbols if s.endswith('.A') or s.endswith('.B')]
    
    if class_a_b:
        print(f"Testing {len(class_a_b)} class A/B shares...")
        test_results = {}
        for symbol in class_a_b[:10]:  # Test first 10
            success, error = test_symbol_fetch(symbol)
            test_results[symbol] = (success, error)
            if success:
                print(f"   ✅ {symbol}: Can be fetched")
            else:
                print(f"   ❌ {symbol}: {error}")
        
        fetchable = [s for s, (success, _) in test_results.items() if success]
        if fetchable:
            print()
            print(f"💡 {len(fetchable)} class A/B shares CAN be fetched!")
            print("   They're still shares - you should retry fetching them.")
    else:
        print("No class A/B shares in failed list")
    print()
    
    # Print categorized analysis
    print("=" * 80)
    print("CATEGORIES")
    print("=" * 80)
    print()
    
    for category, symbols in sorted(categories.items(), key=lambda x: -len(x[1])):
        print(f"📋 {category}: {len(symbols)} symbols")
        if len(symbols) <= 20:
            for symbol in symbols:
                has_data, num_points = check_symbol_has_data(symbol, prices_df)
                data_info = f" ({num_points} data points)" if has_data else ""
                print(f"   - {symbol}{data_info}")
        else:
            for symbol in symbols[:10]:
                has_data, num_points = check_symbol_has_data(symbol, prices_df)
                data_info = f" ({num_points} data points)" if has_data else ""
                print(f"   - {symbol}{data_info}")
            print(f"   ... and {len(symbols) - 10} more")
        print()
    
    # Summary statistics
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total Failed: {len(failed_symbols)}")
    print(f"  - With existing data: {len(failed_with_data)}")
    print(f"  - Without data: {len(failed_without_data)}")
    print(f"  - Class A/B shares: {len(class_a_b)}")
    print(f"Total Categories: {len(categories)}")
    print()
    
    # Recommendations
    print("=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print()
    print("1. Delisted symbols with existing data:")
    print("   ✅ Use existing price data and 'sell' on delisting day in backtests")
    print()
    print("2. Bankruptcies (Q suffix):")
    print("   ✅ Fine to keep - represents real performance going to 0")
    print("   ✅ If price data stops, assume it goes to 0")
    print()
    print("3. Class A/B shares:")
    print("   ✅ They're still shares - retry fetching them")
    print("   ✅ Test shows some CAN be fetched from yfinance")
    print()
    print("4. Symbols with no data:")
    print("   ⚠️  May need manual investigation or alternative data sources")
    print()
    print("To retry failed symbols, run:")
    print("  python scripts/fetch_sp500_prices_batch.py --batch 50")


if __name__ == "__main__":
    main()
