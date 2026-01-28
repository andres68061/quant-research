#!/usr/bin/env python3
"""
Fetch Missing S&P 500 Historical Constituents (BATCH OPTIMIZED)

Uses yfinance.download() to fetch multiple symbols in parallel - MUCH faster!
Fetches 50-100 symbols at once instead of one-by-one.

This is 5-10x faster than the sequential version!

Usage:
    python scripts/fetch_sp500_prices_batch.py --batch 50
    python scripts/fetch_sp500_prices_batch.py --all  # Fetch all missing
"""

import json
import sys
import time
from pathlib import Path
from typing import List, Set, Tuple

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from src.data.sp500_constituents import SP500Constituents
from src.data.factors.prices import fetch_symbols_batch, add_symbols_batch
from src.data.factors.build_factors import build_price_factors
from src.data.factors.io import connect_duckdb, register_parquet
from src.utils.io import read_parquet, write_parquet


def get_existing_symbols() -> Set[str]:
    """Get list of symbols we have price data for."""
    prices_file = ROOT / "data" / "factors" / "prices.parquet"
    if prices_file.exists():
        df = pd.read_parquet(prices_file)
        return set(df.columns.tolist())
    return set()


def load_failed_symbols() -> Set[str]:
    """Load previously failed symbols from tracking file."""
    failed_file = ROOT / "data" / "sp500_failed_symbols.json"
    if failed_file.exists():
        with open(failed_file, 'r') as f:
            return set(json.load(f))
    return set()


def save_failed_symbols(failed_symbols: Set[str]) -> None:
    """Save failed symbols to tracking file."""
    failed_file = ROOT / "data" / "sp500_failed_symbols.json"
    failed_file.parent.mkdir(parents=True, exist_ok=True)
    with open(failed_file, 'w') as f:
        json.dump(sorted(list(failed_symbols)), f, indent=2)


def get_missing_symbols(sp500: SP500Constituents, exclude_failed: bool = True) -> List[str]:
    """
    Get list of S&P 500 symbols we're missing.
    
    Args:
        sp500: SP500Constituents instance
        exclude_failed: If True, exclude symbols that previously failed
        
    Returns:
        List of missing symbols
    """
    all_tickers = sp500.get_ticker_universe()
    existing = get_existing_symbols()
    missing = sorted(all_tickers - existing)
    
    if exclude_failed:
        failed = load_failed_symbols()
        missing = [s for s in missing if s not in failed]
    
    return missing


def fetch_batch_symbols(
    symbols: List[str],
    existing_prices: pd.DataFrame
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Fetch a batch of symbols using parallel batch download.
    
    Args:
        symbols: List of symbols to fetch
        existing_prices: Existing prices DataFrame
        
    Returns:
        Tuple of (updated_prices, successful_symbols, failed_symbols)
    """
    if not symbols:
        return existing_prices, [], []
    
    print(f"  📥 Fetching {len(symbols)} symbols in parallel batch...")
    
    # Fetch batch using optimized parallel download
    new_data = fetch_symbols_batch(symbols)
    
    successful = []
    failed = []
    
    if new_data.empty:
        # All failed
        failed = symbols
        return existing_prices, successful, failed
    
    # Check which symbols were successfully fetched
    # A symbol is successful only if it has actual data (not all NaN)
    fetched_symbols = set(new_data.columns)
    for symbol in symbols:
        if symbol in fetched_symbols:
            # Check if symbol has actual data (not all NaN)
            symbol_data = new_data[symbol]
            if symbol_data.notna().sum() > 0:  # Has at least some non-NaN values
                successful.append(symbol)
            else:
                failed.append(symbol)
        else:
            failed.append(symbol)
    
    if not successful:
        return existing_prices, successful, failed
    
    # Add successful symbols to existing prices
    successful_data = new_data[successful]
    successful_data = successful_data.asfreq('B').ffill()
    
    # Ensure timezone consistency before concatenating
    if existing_prices is not None and not existing_prices.empty:
        # Get timezone from existing prices
        existing_tz = existing_prices.index.tz
        
        # Make new data timezone-aware to match existing
        if successful_data.index.tz is None and existing_tz is not None:
            # Convert naive to timezone-aware (match existing)
            successful_data.index = successful_data.index.tz_localize('America/New_York')
        elif successful_data.index.tz is not None and existing_tz is None:
            # Convert timezone-aware to naive (match existing)
            successful_data.index = successful_data.index.tz_localize(None)
        elif successful_data.index.tz is not None and existing_tz is not None:
            # Both have timezones, convert to match
            if successful_data.index.tz != existing_tz:
                successful_data.index = successful_data.index.tz_convert(existing_tz)
    
    if existing_prices is None or existing_prices.empty:
        updated_prices = successful_data
        updated_prices.index.name = 'date'
    else:
        # Merge with existing (both should have matching timezones now)
        updated_prices = pd.concat([existing_prices, successful_data], axis=1)
        updated_prices = updated_prices.sort_index()
    
    return updated_prices, successful, failed


def categorize_failures(failed_symbols: List[str]) -> dict:
    """Categorize failed symbols (for batch, we don't have detailed reasons)."""
    # With batch fetching, we can't get detailed error reasons per symbol
    # But we can note that they failed
    return {
        "No data/Delisted": [(s, "No data returned from batch fetch") for s in failed_symbols]
    }


def main(batch_size: int = 50, start_index: int = 0, exclude_failed: bool = True):
    """
    Fetch missing S&P 500 constituents using batch parallel fetching.
    
    Args:
        batch_size: Number of symbols to fetch per batch (0 = all)
        start_index: Index to start from (for resuming)
        exclude_failed: If True, skip previously failed symbols
    """
    print("=" * 80)
    print("S&P 500 HISTORICAL CONSTITUENTS FETCHER (BATCH OPTIMIZED)")
    print("=" * 80)
    print()
    print("⚡ Using parallel batch fetching - 5-10x faster than sequential!")
    print()
    
    # Load prices file
    prices_path = ROOT / "data" / "factors" / "prices.parquet"
    existing_prices = read_parquet(prices_path)
    
    if existing_prices is None or existing_prices.empty:
        print("⚠️  No existing prices found. Run backfill_all.py first.")
        return
    
    # Load S&P 500 data
    sp500 = SP500Constituents()
    sp500.load()
    
    # Get missing symbols
    missing = get_missing_symbols(sp500, exclude_failed=exclude_failed)
    
    # Load previously failed symbols
    previously_failed = load_failed_symbols()
    
    print(f"📊 STATUS")
    print("-" * 80)
    print(f"Total S&P 500 Tickers (Historical): {len(sp500.get_ticker_universe()):,}")
    print(f"Tickers with Data: {len(get_existing_symbols()):,}")
    print(f"Tickers Missing: {len(missing):,}")
    if previously_failed:
        print(f"Previously Failed (excluded): {len(previously_failed):,}")
    
    if not missing:
        print("\n✅ All fetchable S&P 500 constituents have data!")
        if previously_failed:
            print(f"\n⚠️  Note: {len(previously_failed)} symbols previously failed.")
            print("   Review data/sp500_failed_symbols.json for details.")
            print("   Run with --include-failed to retry them.")
        return
    
    # Apply batch size
    # Note: start_index is deprecated - missing list is recalculated each time
    # so indices become invalid after each batch. Just run the same command again.
    if start_index > 0:
        print(f"\n⚠️  Warning: --start index is deprecated (missing list changes after each batch)")
        print(f"   Ignoring start_index={start_index}, fetching from beginning of current missing list")
    
    if batch_size > 0:
        print(f"Fetching batch of {batch_size} symbols")
        to_fetch = missing[:batch_size]
    else:
        print(f"Fetching all {len(missing)} missing symbols")
        to_fetch = missing
    
    print(f"\n⏬ FETCHING {len(to_fetch)} SYMBOLS (PARALLEL BATCH)")
    print("-" * 80)
    
    # Estimate time (much faster with batch!)
    seconds_per_batch = 15  # Batch of 50 takes ~10-20 seconds
    total_minutes = (len(to_fetch) / batch_size if batch_size > 0 else 1) * (seconds_per_batch / 60)
    print(f"Estimated time: ~{total_minutes:.1f} minutes (vs ~{len(to_fetch) * 2.5 / 60:.1f} min sequential)")
    print()
    
    # Fetch batch
    start_time = time.time()
    
    updated_prices, successful, failed = fetch_batch_symbols(to_fetch, existing_prices)
    
    elapsed = time.time() - start_time
    
    # Save updated prices
    if successful:
        print(f"\n💾 Saving updated prices...")
        write_parquet(updated_prices, prices_path)
        print(f"   ✅ Saved: {updated_prices.shape}")
    
    # Update failed symbols tracking
    if failed:
        all_failed = load_failed_symbols()
        all_failed.update(failed)
        save_failed_symbols(all_failed)
    
    # Summary
    print("\n" + "=" * 80)
    print("📈 SUMMARY")
    print("=" * 80)
    print(f"Successfully Fetched: {len(successful)}/{len(to_fetch)}")
    print(f"Failed: {len(failed)}")
    print(f"Time Elapsed: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    if len(to_fetch) > 0:
        print(f"Avg Time per Symbol: {elapsed/len(to_fetch):.2f} seconds")
        print(f"Speedup: ~{2.5 / (elapsed/len(to_fetch)):.1f}x faster than sequential!")
    
    # Show failures
    if failed:
        print(f"\n❌ FAILED SYMBOLS ({len(failed)}):")
        for symbol in failed[:20]:
            print(f"  - {symbol}")
        if len(failed) > 20:
            print(f"  ... and {len(failed) - 20} more")
        
        print(f"\n💾 Failed symbols saved to: data/sp500_failed_symbols.json")
    
    # Rebuild factors if we got new data
    if successful:
        print(f"\n📉 Rebuilding factors with new symbols...")
        factors_price_path = ROOT / "data" / "factors" / "factors_price.parquet"
        factors_all_path = ROOT / "data" / "factors" / "factors_all.parquet"
        db_path = ROOT / "data" / "factors" / "factors.duckdb"
        
        market_symbol = '^GSPC'
        factors_price = build_price_factors(updated_prices, market_symbol=market_symbol)
        
        write_parquet(factors_price, factors_price_path)
        write_parquet(factors_price, factors_all_path)
        
        # Update DuckDB views
        con = connect_duckdb(db_path)
        register_parquet(con, 'prices', prices_path)
        register_parquet(con, 'factors_price', factors_price_path)
        register_parquet(con, 'factors_all', factors_all_path)
        con.close()
        
        print(f"   ✅ Factors rebuilt")
    
    # Check if we need to continue
    still_missing = get_missing_symbols(sp500, exclude_failed=exclude_failed)
    if still_missing:
        print(f"\n⏭️  NEXT STEPS")
        print("-" * 80)
        print(f"Still missing {len(still_missing)} symbols")
        print(f"\nTo continue, just run the same command again:")
        print(f"  python scripts/{Path(__file__).name} --batch {batch_size}")
        print(f"\nThe missing list is automatically updated, so it will fetch the next batch!")
    else:
        print(f"\n✅ ALL FETCHABLE S&P 500 CONSTITUENTS FETCHED!")
        if previously_failed or failed:
            print(f"\n⚠️  Note: Some symbols failed (likely delisted/invalid).")
            print("   Review data/sp500_failed_symbols.json for details.")
        print("\nNext steps:")
        print("  1. Run: python scripts/analyze_sp500_constituents.py")
        print("  2. Use S&P 500 Historical benchmark in portfolio simulator")
    
    print("\n" + "=" * 80)
    print("✓ COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Fetch missing S&P 500 constituent prices (batch optimized)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This version uses parallel batch fetching - 5-10x faster than sequential!

Examples:
  # Fetch 50 symbols (takes ~15 seconds vs ~2 minutes sequential)
  python scripts/fetch_sp500_prices_batch.py --batch 50
  
  # Just run again to fetch next batch (missing list is auto-updated)
  python scripts/fetch_sp500_prices_batch.py --batch 50
  
  # Fetch all missing (excluding previously failed)
  python scripts/fetch_sp500_prices_batch.py --all
  
  # Retry previously failed symbols
  python scripts/fetch_sp500_prices_batch.py --batch 50 --include-failed
        """
    )
    parser.add_argument("--batch", type=int, default=50, 
                       help="Number of symbols per batch (default: 50)")
    parser.add_argument("--start", type=int, default=0,
                       help="[DEPRECATED] Ignored - missing list is recalculated each time")
    parser.add_argument("--all", action="store_true",
                       help="Fetch all missing symbols (overrides --batch)")
    parser.add_argument("--include-failed", action="store_true",
                       help="Include previously failed symbols (default: exclude them)")
    
    args = parser.parse_args()
    
    batch_size = 0 if args.all else args.batch
    
    main(
        batch_size=batch_size, 
        start_index=args.start,
        exclude_failed=not args.include_failed
    )
