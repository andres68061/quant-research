"""
Market Capitalization Data Management

This module calculates and stores historical market capitalization data using:
    market_cap = shares_outstanding × price

Data sources:
- Shares outstanding: Yahoo Finance (yfinance)
- Prices: Existing data/factors/prices.parquet
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Set

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


class MarketCapCalculator:
    """
    Calculate historical market caps using shares outstanding × price.
    
    This approach is FREE and gives historical data without paid API access.
    """
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize market cap calculator.
        
        Args:
            data_dir: Directory for market cap data (defaults to data/market_caps/)
        """
        if data_dir is None:
            data_dir = Path("data/market_caps")
        
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.shares_file = self.data_dir / "shares_outstanding.parquet"
        self.market_caps_file = self.data_dir / "historical_market_caps.parquet"
        
        # Load existing prices
        self.prices_file = Path("data/factors/prices.parquet")
        self._prices: Optional[pd.DataFrame] = None
    
    def load_prices(self) -> pd.DataFrame:
        """Load price data."""
        if self._prices is None:
            if self.prices_file.exists():
                self._prices = pd.read_parquet(self.prices_file)
                logger.info(f"Loaded prices: {len(self._prices)} dates, {len(self._prices.columns)} symbols")
            else:
                raise FileNotFoundError(f"Prices file not found: {self.prices_file}")
        return self._prices
    
    def fetch_shares_outstanding(self, ticker: str) -> Optional[Dict]:
        """
        Fetch shares outstanding for a ticker from Yahoo Finance.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with: {
                'ticker': str,
                'shares_outstanding': int,
                'fetch_date': datetime,
                'source': str
            }
            None if fetch fails
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            shares = info.get('sharesOutstanding')
            
            if shares is None or shares == 0:
                logger.warning(f"{ticker}: No shares outstanding data")
                return None
            
            return {
                'ticker': ticker,
                'shares_outstanding': shares,
                'fetch_date': datetime.now(),
                'source': 'yfinance'
            }
            
        except Exception as e:
            logger.error(f"{ticker}: Error fetching shares outstanding - {str(e)[:100]}")
            return None
    
    def fetch_all_shares_outstanding(
        self, 
        tickers: list,
        delay: float = 0.5,
        resume: bool = True
    ) -> pd.DataFrame:
        """
        Fetch shares outstanding for multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            delay: Delay between requests (seconds)
            resume: Skip tickers already fetched
            
        Returns:
            DataFrame with columns: ticker, shares_outstanding, fetch_date, source
        """
        # Load existing data if resuming
        existing_df = None
        if resume and self.shares_file.exists():
            existing_df = pd.read_parquet(self.shares_file)
            already_fetched = set(existing_df['ticker'].unique())
            tickers = [t for t in tickers if t not in already_fetched]
            logger.info(f"Resuming: {len(already_fetched)} already fetched, {len(tickers)} remaining")
        
        if not tickers:
            logger.info("All tickers already fetched!")
            return existing_df if existing_df is not None else pd.DataFrame()
        
        print(f"\n⏬ FETCHING SHARES OUTSTANDING FOR {len(tickers)} TICKERS")
        print("-" * 80)
        
        results = []
        success_count = 0
        failed_count = 0
        
        for i, ticker in enumerate(tickers, 1):
            print(f"[{i}/{len(tickers)}] {ticker:10s} ", end='')
            
            result = self.fetch_shares_outstanding(ticker)
            
            if result:
                results.append(result)
                success_count += 1
                print(f"✅ {result['shares_outstanding']:,} shares")
            else:
                failed_count += 1
                print("❌ No data")
            
            time.sleep(delay)
            
            # Progress update every 50
            if i % 50 == 0:
                print(f"\n  Progress: {success_count}/{i} successful")
        
        # Convert to DataFrame
        new_df = pd.DataFrame(results)
        
        # Combine with existing
        if existing_df is not None and not existing_df.empty:
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            combined_df = new_df
        
        # Save
        if not combined_df.empty:
            combined_df.to_parquet(self.shares_file, index=False)
            logger.info(f"Saved {len(combined_df)} tickers to {self.shares_file}")
        
        print("\n" + "-" * 80)
        print(f"✓ Fetched {success_count}/{len(tickers)} successfully")
        print(f"✗ Failed: {failed_count}")
        
        return combined_df
    
    def load_shares_outstanding(self) -> pd.DataFrame:
        """Load shares outstanding data from parquet."""
        if self.shares_file.exists():
            return pd.read_parquet(self.shares_file)
        return pd.DataFrame()
    
    def calculate_market_caps(
        self, 
        tickers: Optional[list] = None,
        save: bool = True
    ) -> pd.DataFrame:
        """
        Calculate historical market caps: market_cap = shares_outstanding × price
        
        Args:
            tickers: List of tickers to calculate (None = all with shares data)
            save: Whether to save results to parquet
            
        Returns:
            DataFrame with MultiIndex (date, ticker) and column: market_cap
        """
        print("\n📊 CALCULATING HISTORICAL MARKET CAPS")
        print("-" * 80)
        
        # Load data
        prices = self.load_prices()
        shares_df = self.load_shares_outstanding()
        
        if shares_df.empty:
            raise ValueError("No shares outstanding data available. Run fetch_all_shares_outstanding first.")
        
        # Filter tickers
        if tickers is None:
            tickers = shares_df['ticker'].unique().tolist()
        
        # Filter to tickers we have both prices and shares for
        available_tickers = [t for t in tickers if t in prices.columns and t in shares_df['ticker'].values]
        
        print(f"Calculating for {len(available_tickers)} tickers with both price & shares data")
        
        # Calculate market caps
        market_caps = []
        
        for ticker in available_tickers:
            # Get shares outstanding (use most recent value)
            shares = shares_df[shares_df['ticker'] == ticker]['shares_outstanding'].iloc[-1]
            
            # Get price series
            price_series = prices[ticker]
            
            # Calculate market cap series
            market_cap_series = price_series * shares
            
            # Create DataFrame
            df = pd.DataFrame({
                'date': price_series.index,
                'ticker': ticker,
                'market_cap': market_cap_series.values
            })
            
            market_caps.append(df)
        
        # Combine all
        result_df = pd.concat(market_caps, ignore_index=True)
        result_df['date'] = pd.to_datetime(result_df['date'])
        
        # Remove NaN market caps
        result_df = result_df.dropna(subset=['market_cap'])
        
        # Set MultiIndex
        result_df = result_df.set_index(['date', 'ticker'])
        
        print(f"✓ Calculated {len(result_df):,} date-ticker market cap records")
        print(f"  Date range: {result_df.index.get_level_values('date').min().date()} to {result_df.index.get_level_values('date').max().date()}")
        
        # Save
        if save:
            result_df.to_parquet(self.market_caps_file)
            print(f"✓ Saved to {self.market_caps_file}")
        
        return result_df
    
    def load_market_caps(self) -> Optional[pd.DataFrame]:
        """Load calculated market caps from parquet."""
        if self.market_caps_file.exists():
            return pd.read_parquet(self.market_caps_file)
        return None
    
    def get_market_cap_on_date(
        self, 
        date: pd.Timestamp,
        tickers: Optional[list] = None
    ) -> pd.Series:
        """
        Get market caps for specific date.
        
        Args:
            date: Date to query
            tickers: Optional list of tickers to filter
            
        Returns:
            Series with ticker as index, market_cap as values
        """
        market_caps = self.load_market_caps()
        
        if market_caps is None:
            raise ValueError("No market cap data available. Run calculate_market_caps first.")
        
        # Get closest date
        available_dates = market_caps.index.get_level_values('date').unique()
        closest_date = available_dates[available_dates <= date].max()
        
        # Filter to date
        caps_on_date = market_caps.xs(closest_date, level='date')
        
        # Filter tickers if specified
        if tickers:
            caps_on_date = caps_on_date[caps_on_date.index.isin(tickers)]
        
        return caps_on_date['market_cap']
    
    def get_weights_on_date(
        self,
        date: pd.Timestamp,
        tickers: Optional[list] = None
    ) -> pd.Series:
        """
        Get market-cap weights for specific date.
        
        Args:
            date: Date to query
            tickers: Optional list of tickers to filter
            
        Returns:
            Series with ticker as index, weight (sums to 1.0) as values
        """
        caps = self.get_market_cap_on_date(date, tickers)
        weights = caps / caps.sum()
        return weights
    
    def get_summary_stats(self) -> Dict:
        """Get summary statistics about market cap data."""
        shares_df = self.load_shares_outstanding()
        market_caps_df = self.load_market_caps()
        
        stats = {
            'tickers_with_shares': len(shares_df) if not shares_df.empty else 0,
            'tickers_with_market_caps': 0,
            'date_range': None,
            'total_records': 0
        }
        
        if market_caps_df is not None and not market_caps_df.empty:
            stats['tickers_with_market_caps'] = len(market_caps_df.index.get_level_values('ticker').unique())
            dates = market_caps_df.index.get_level_values('date')
            stats['date_range'] = (dates.min().date(), dates.max().date())
            stats['total_records'] = len(market_caps_df)
        
        return stats

