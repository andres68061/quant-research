#!/usr/bin/env python3
"""
FMP Market Cap Fetcher - Batch Mode

Fetches historical market capitalization data from Financial Modeling Prep (FMP)
for S&P 500 historical constituents.

Daily limit: 250 requests (recommended: 240 to leave buffer for other API calls)
Timeline: ~5 days to fetch all 1,194 tickers
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from dotenv import load_dotenv

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load environment variables
load_dotenv(ROOT / ".env")

from src.data.sp500_constituents import SP500Constituents


class FMPMarketCapFetcher:
    """Fetch historical market cap data from FMP API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize FMP market cap fetcher.
        
        Args:
            api_key: FMP API key (defaults to FMP_API_KEY from .env)
        """
        self.api_key = api_key or os.getenv('FMP_API_KEY')
        if not self.api_key:
            raise ValueError("FMP_API_KEY not found in environment or .env file")
        
        self.base_url = "https://financialmodelingprep.com/api/v3"
        self.data_dir = ROOT / "data" / "market_caps"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.historical_file = self.data_dir / "historical.parquet"
        self.progress_file = self.data_dir / "fetch_progress.txt"
        
    def fetch_historical_market_cap(self, ticker: str) -> Optional[pd.DataFrame]:
        """
        Fetch historical market cap for a single ticker.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            DataFrame with columns: date, ticker, market_cap
            None if request fails
        """
        url = f"{self.base_url}/historical-market-capitalization/{ticker}"
        params = {
            'apikey': self.api_key,
            'limit': 5000  # Get all available history
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if not data or isinstance(data, dict) and 'Error Message' in data:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            if df.empty:
                return None
            
            # Add ticker column and standardize
            df['ticker'] = ticker
            df['date'] = pd.to_datetime(df['date'])
            df = df.rename(columns={'marketCap': 'market_cap'})
            df = df[['date', 'ticker', 'market_cap']]
            
            return df
            
        except requests.exceptions.RequestException as e:
            print(f"  ⚠️  Request error: {str(e)[:50]}")
            return None
        except Exception as e:
            print(f"  ❌ Error: {str(e)[:50]}")
            return None
    
    def get_fetched_tickers(self) -> set:
        """Get set of tickers already fetched."""
        if not self.historical_file.exists():
            return set()
        
        try:
            df = pd.read_parquet(self.historical_file)
            return set(df['ticker'].unique())
        except Exception:
            return set()
    
    def save_batch(self, batch_df: pd.DataFrame):
        """Append a batch of market cap data to parquet file."""
        if self.historical_file.exists():
            existing = pd.read_parquet(self.historical_file)
            combined = pd.concat([existing, batch_df], ignore_index=True)
            # Remove duplicates
            combined = combined.drop_duplicates(subset=['date', 'ticker'])
            combined.to_parquet(self.historical_file, index=False)
        else:
            batch_df.to_parquet(self.historical_file, index=False)
    
    def log_progress(self, ticker: str, success: bool):
        """Log progress to file."""
        with open(self.progress_file, 'a') as f:
            status = "SUCCESS" if success else "FAILED"
            f.write(f"{datetime.now().isoformat()} | {ticker:10s} | {status}\n")
    
    def fetch_batch(
        self, 
        tickers: List[str], 
        batch_size: int = 240,
        delay: float = 0.5
    ) -> Dict:
        """
        Fetch market caps for a batch of tickers.
        
        Args:
            tickers: List of tickers to fetch
            batch_size: Max number to fetch (default: 240 to stay under 250/day limit)
            delay: Delay between requests in seconds
            
        Returns:
            Dictionary with stats: success_count, failed_count, failed_tickers
        """
        # Filter out already-fetched tickers
        already_fetched = self.get_fetched_tickers()
        tickers_to_fetch = [t for t in tickers if t not in already_fetched]
        
        if not tickers_to_fetch:
            print("✓ All tickers already fetched!")
            return {'success_count': 0, 'failed_count': 0, 'failed_tickers': []}
        
        # Limit to batch size
        tickers_to_fetch = tickers_to_fetch[:batch_size]
        
        print(f"\n⏬ FETCHING {len(tickers_to_fetch)} TICKERS")
        print("-" * 80)
        
        success_count = 0
        failed_count = 0
        failed_tickers = []
        batch_data = []
        
        start_time = time.time()
        
        for i, ticker in enumerate(tickers_to_fetch, 1):
            print(f"[{i}/{len(tickers_to_fetch)}] Fetching {ticker}...", end=' ')
            
            df = self.fetch_historical_market_cap(ticker)
            
            if df is not None and not df.empty:
                batch_data.append(df)
                success_count += 1
                self.log_progress(ticker, True)
                print(f"✅ {len(df)} records")
            else:
                failed_count += 1
                failed_tickers.append(ticker)
                self.log_progress(ticker, False)
                print("❌ No data")
            
            # Rate limiting
            time.sleep(delay)
            
            # Save every 50 tickers
            if len(batch_data) >= 50:
                print(f"\n  💾 Saving batch of {len(batch_data)} tickers...")
                combined_batch = pd.concat(batch_data, ignore_index=True)
                self.save_batch(combined_batch)
                batch_data = []
        
        # Save remaining data
        if batch_data:
            print(f"\n💾 Saving final batch of {len(batch_data)} tickers...")
            combined_batch = pd.concat(batch_data, ignore_index=True)
            self.save_batch(combined_batch)
        
        elapsed = time.time() - start_time
        
        return {
            'success_count': success_count,
            'failed_count': failed_count,
            'failed_tickers': failed_tickers,
            'elapsed': elapsed
        }


def main(batch_size: int = 240, resume: bool = True):
    """
    Main execution function.
    
    Args:
        batch_size: Number of tickers to fetch (default: 240, max recommended: 240)
        resume: Resume from where last session ended
    """
    print("=" * 80)
    print("FMP MARKET CAP FETCHER - Batch Mode")
    print("=" * 80)
    
    # Load S&P 500 tickers
    sp500 = SP500Constituents()
    sp500.load()
    all_tickers = sorted(sp500.get_ticker_universe())
    
    # Initialize fetcher
    try:
        fetcher = FMPMarketCapFetcher()
    except ValueError as e:
        print(f"\n❌ ERROR: {e}")
        print("\nPlease add FMP_API_KEY to your .env file:")
        print("  FMP_API_KEY=your_api_key_here")
        print("\nGet a free API key at: https://financialmodelingprep.com/developer/docs/")
        return
    
    # Get already-fetched tickers
    already_fetched = fetcher.get_fetched_tickers()
    remaining_tickers = [t for t in all_tickers if t not in already_fetched]
    
    print(f"\n📊 STATUS")
    print("-" * 80)
    print(f"Total S&P 500 Tickers: {len(all_tickers):,}")
    print(f"Already Fetched: {len(already_fetched):,}")
    print(f"Remaining: {len(remaining_tickers):,}")
    print(f"\nFetching up to {batch_size} tickers today...")
    
    if not remaining_tickers:
        print("\n✅ All tickers already fetched!")
        return
    
    # Estimate completion
    days_remaining = (len(remaining_tickers) + batch_size - 1) // batch_size
    print(f"Estimated days to complete: {days_remaining}")
    
    # Fetch batch
    results = fetcher.fetch_batch(remaining_tickers, batch_size=batch_size)
    
    # Summary
    print("\n" + "=" * 80)
    print("📈 SUMMARY")
    print("=" * 80)
    print(f"Successfully Fetched: {results['success_count']}/{batch_size}")
    print(f"Failed: {results['failed_count']}")
    print(f"Time Elapsed: {results['elapsed']/60:.1f} minutes")
    
    if results['failed_tickers']:
        print(f"\n❌ Failed Tickers ({len(results['failed_tickers'])}):")
        for ticker in results['failed_tickers'][:20]:
            print(f"  - {ticker}")
        if len(results['failed_tickers']) > 20:
            print(f"  ... and {len(results['failed_tickers']) - 20} more")
    
    # Check overall progress
    newly_fetched = fetcher.get_fetched_tickers()
    still_remaining = [t for t in all_tickers if t not in newly_fetched]
    
    print(f"\n📊 OVERALL PROGRESS")
    print("-" * 80)
    print(f"Total Fetched: {len(newly_fetched):,} / {len(all_tickers):,}")
    print(f"Remaining: {len(still_remaining):,}")
    
    if still_remaining:
        days_left = (len(still_remaining) + batch_size - 1) // batch_size
        print(f"\n⏭️  Run again tomorrow to continue ({days_left} more days)")
        print(f"   python scripts/fetch_market_caps_fmp.py")
    else:
        print(f"\n✅ ALL TICKERS COMPLETE!")
        print(f"\nData saved to: {fetcher.historical_file}")
        print(f"\nNext step: Update portfolio simulator to use cap-weighted S&P 500")
    
    print("\n" + "=" * 80)
    print("✓ COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fetch market caps from FMP")
    parser.add_argument("--batch", type=int, default=240,
                       help="Number of tickers to fetch (default: 240, recommended max: 240)")
    parser.add_argument("--no-resume", action="store_true",
                       help="Start from scratch (ignore previous progress)")
    
    args = parser.parse_args()
    
    if args.batch > 240:
        print("⚠️  WARNING: Fetching more than 240/day may exceed API limits!")
        print("   FMP free tier allows 250 requests/day")
        print("   Other API calls may push you over the limit")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    main(batch_size=args.batch, resume=not args.no_resume)

