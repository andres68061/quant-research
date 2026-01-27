#!/usr/bin/env python3
"""
Fetch Missing S&P 500 Historical Constituents

This script fetches price data for all S&P 500 constituents that are missing
from our database. It's designed to be run in batches and can be resumed.
"""

import sys
import time
from pathlib import Path
from typing import List, Set

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.sp500_constituents import SP500Constituents


def get_existing_symbols() -> Set[str]:
    """Get list of symbols we have price data for."""
    import pandas as pd
    
    prices_file = ROOT / "data" / "factors" / "prices.parquet"
    if prices_file.exists():
        df = pd.read_parquet(prices_file)
        return set(df.columns.tolist())
    return set()


def get_missing_symbols(sp500: SP500Constituents) -> List[str]:
    """Get list of S&P 500 symbols we're missing."""
    all_tickers = sp500.get_ticker_universe()
    existing = get_existing_symbols()
    missing = sorted(all_tickers - existing)
    return missing


def fetch_symbol_with_retry(symbol: str, max_retries: int = 3) -> bool:
    """
    Fetch a single symbol using add_symbol.py script.
    
    Args:
        symbol: Ticker to fetch
        max_retries: Number of retry attempts
        
    Returns:
        True if successful, False otherwise
    """
    import subprocess
    
    add_symbol_script = ROOT / "scripts" / "add_symbol.py"
    
    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                [sys.executable, str(add_symbol_script), symbol],
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout per symbol
            )
            
            if result.returncode == 0:
                return True
            else:
                print(f"  ⚠️  Attempt {attempt + 1} failed: {result.stderr.strip()[:100]}")
                
                # If it's a known bad ticker, don't retry
                if "No data found" in result.stderr or "delisted" in result.stderr.lower():
                    return False
                
                time.sleep(2)  # Brief pause before retry
                
        except subprocess.TimeoutExpired:
            print(f"  ⏱️  Attempt {attempt + 1} timed out")
            time.sleep(2)
        except Exception as e:
            print(f"  ❌ Attempt {attempt + 1} error: {str(e)[:100]}")
            time.sleep(2)
    
    return False


def main(batch_size: int = 50, start_index: int = 0):
    """
    Fetch missing S&P 500 constituents.
    
    Args:
        batch_size: Number of symbols to fetch (0 = all)
        start_index: Index to start from (for resuming)
    """
    print("=" * 80)
    print("S&P 500 HISTORICAL CONSTITUENTS FETCHER")
    print("=" * 80)
    
    # Load S&P 500 data
    sp500 = SP500Constituents()
    sp500.load()
    
    # Get missing symbols
    missing = get_missing_symbols(sp500)
    
    print(f"\n📊 STATUS")
    print("-" * 80)
    print(f"Total S&P 500 Tickers (Historical): {len(sp500.get_ticker_universe()):,}")
    print(f"Tickers with Data: {len(get_existing_symbols()):,}")
    print(f"Tickers Missing: {len(missing):,}")
    
    if not missing:
        print("\n✅ All S&P 500 constituents have data!")
        return
    
    # Apply batch size and start index
    if start_index > 0:
        print(f"\nStarting from index {start_index}")
        missing = missing[start_index:]
    
    if batch_size > 0:
        print(f"Fetching batch of {batch_size} symbols")
        to_fetch = missing[:batch_size]
    else:
        print(f"Fetching all {len(missing)} missing symbols")
        to_fetch = missing
    
    print(f"\n⏬ FETCHING {len(to_fetch)} SYMBOLS")
    print("-" * 80)
    
    # Estimate time
    seconds_per_symbol = 2.5  # Conservative estimate
    total_minutes = (len(to_fetch) * seconds_per_symbol) / 60
    print(f"Estimated time: ~{total_minutes:.0f} minutes\n")
    
    # Fetch symbols
    success_count = 0
    failed_symbols = []
    
    start_time = time.time()
    
    for i, symbol in enumerate(to_fetch, 1):
        print(f"\n[{i}/{len(to_fetch)}] Fetching {symbol}...")
        
        success = fetch_symbol_with_retry(symbol)
        
        if success:
            success_count += 1
            print(f"  ✅ Success")
        else:
            failed_symbols.append(symbol)
            print(f"  ❌ Failed (may be delisted/invalid)")
        
        # Show progress every 10 symbols
        if i % 10 == 0:
            elapsed = time.time() - start_time
            avg_time = elapsed / i
            remaining = (len(to_fetch) - i) * avg_time
            print(f"\n  Progress: {success_count}/{i} successful | "
                  f"~{remaining/60:.0f} min remaining")
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("📈 SUMMARY")
    print("=" * 80)
    print(f"Successfully Fetched: {success_count}/{len(to_fetch)}")
    print(f"Failed: {len(failed_symbols)}")
    print(f"Time Elapsed: {elapsed/60:.1f} minutes")
    print(f"Avg Time per Symbol: {elapsed/len(to_fetch):.1f} seconds")
    
    if failed_symbols:
        print(f"\n❌ Failed Symbols ({len(failed_symbols)}):")
        for symbol in failed_symbols[:20]:
            print(f"  - {symbol}")
        if len(failed_symbols) > 20:
            print(f"  ... and {len(failed_symbols) - 20} more")
    
    # Check if we need to continue
    still_missing = get_missing_symbols(sp500)
    if still_missing:
        print(f"\n⏭️  NEXT STEPS")
        print("-" * 80)
        print(f"Still missing {len(still_missing)} symbols")
        next_index = start_index + len(to_fetch)
        print(f"\nTo continue, run:")
        print(f"  python {Path(__file__).name} --batch {batch_size} --start {next_index}")
    else:
        print(f"\n✅ ALL S&P 500 CONSTITUENTS FETCHED!")
        print("\nNext steps:")
        print("  1. Run: python scripts/analyze_sp500_constituents.py")
        print("  2. Use S&P 500 Historical benchmark in portfolio simulator")
    
    print("\n" + "=" * 80)
    print("✓ COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch missing S&P 500 constituent prices")
    parser.add_argument("--batch", type=int, default=50, 
                       help="Number of symbols to fetch (0 = all, default: 50)")
    parser.add_argument("--start", type=int, default=0,
                       help="Index to start from (for resuming, default: 0)")
    parser.add_argument("--all", action="store_true",
                       help="Fetch all missing symbols (overrides --batch)")
    
    args = parser.parse_args()
    
    batch_size = 0 if args.all else args.batch
    
    main(batch_size=batch_size, start_index=args.start)

