"""
S&P 500 Historical Constituents Management

This module provides point-in-time S&P 500 constituent data to eliminate
survivorship bias in backtesting.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

logger = logging.getLogger(__name__)


class SP500Constituents:
    """
    Manages S&P 500 historical constituents data for point-in-time analysis.
    
    This eliminates survivorship bias by tracking which stocks were actually
    in the S&P 500 on any given date.
    """
    
    def __init__(self, csv_path: Optional[Path] = None):
        """
        Initialize S&P 500 constituents manager.
        
        Args:
            csv_path: Path to historical constituents CSV. If None, uses default.
        """
        if csv_path is None:
            csv_path = Path("data/S&P 500 Historical Components & Changes(01-17-2026).csv")
        
        self.csv_path = csv_path
        self._constituents: Optional[pd.DataFrame] = None
        self._ticker_universe: Optional[Set[str]] = None
        
    def load(self) -> pd.DataFrame:
        """Load and parse the historical constituents CSV."""
        if self._constituents is not None:
            return self._constituents
            
        logger.info(f"Loading S&P 500 constituents from {self.csv_path}")
        
        # Read CSV
        df = pd.read_csv(self.csv_path)
        
        # Parse date
        df['date'] = pd.to_datetime(df['date'])
        
        # Split ticker strings into lists
        df['tickers'] = df['tickers'].str.split(',')
        
        # Set date as index
        df = df.set_index('date').sort_index()
        
        self._constituents = df
        
        # Build universe of all unique tickers
        all_tickers = set()
        for tickers_list in df['tickers']:
            all_tickers.update(tickers_list)
        self._ticker_universe = all_tickers
        
        logger.info(f"Loaded {len(df)} dates, {len(all_tickers)} unique tickers")
        
        return df
    
    def get_constituents_on_date(self, date: pd.Timestamp) -> List[str]:
        """
        Get S&P 500 constituents on a specific date.
        
        Args:
            date: Date to query
            
        Returns:
            List of ticker symbols in S&P 500 on that date
        """
        if self._constituents is None:
            self.load()
        
        # Find the most recent date <= query date
        valid_dates = self._constituents.index[self._constituents.index <= date]
        
        if len(valid_dates) == 0:
            logger.warning(f"No constituents data before {date}")
            return []
        
        most_recent_date = valid_dates[-1]
        return self._constituents.loc[most_recent_date, 'tickers']
    
    def get_constituents_series(
        self, 
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None
    ) -> pd.DataFrame:
        """
        Get constituents for a date range.
        
        Args:
            start_date: Start date (inclusive). If None, use first available.
            end_date: End date (inclusive). If None, use last available.
            
        Returns:
            DataFrame with dates and ticker lists
        """
        if self._constituents is None:
            self.load()
        
        df = self._constituents.copy()
        
        if start_date is not None:
            df = df[df.index >= start_date]
        if end_date is not None:
            df = df[df.index <= end_date]
        
        return df
    
    def get_additions_and_removals(
        self,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None
    ) -> pd.DataFrame:
        """
        Identify S&P 500 additions and removals over time.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with columns: date, additions, removals
        """
        df = self.get_constituents_series(start_date, end_date)
        
        changes = []
        prev_tickers = None
        
        for date, row in df.iterrows():
            curr_tickers = set(row['tickers'])
            
            if prev_tickers is not None:
                additions = curr_tickers - prev_tickers
                removals = prev_tickers - curr_tickers
                
                if additions or removals:
                    changes.append({
                        'date': date,
                        'additions': list(additions),
                        'removals': list(removals),
                        'num_additions': len(additions),
                        'num_removals': len(removals)
                    })
            
            prev_tickers = curr_tickers
        
        return pd.DataFrame(changes)
    
    def get_ticker_universe(self) -> Set[str]:
        """
        Get all unique tickers that have ever been in S&P 500.
        
        Returns:
            Set of all ticker symbols
        """
        if self._ticker_universe is None:
            self.load()
        return self._ticker_universe.copy()
    
    def is_in_sp500(self, ticker: str, date: pd.Timestamp) -> bool:
        """
        Check if a ticker was in S&P 500 on a specific date.
        
        Args:
            ticker: Ticker symbol
            date: Date to check
            
        Returns:
            True if ticker was in S&P 500 on that date
        """
        constituents = self.get_constituents_on_date(date)
        return ticker in constituents
    
    def get_ticker_history(self, ticker: str) -> pd.DataFrame:
        """
        Get the full history of when a ticker was in/out of S&P 500.
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            DataFrame with date, in_sp500 columns
        """
        if self._constituents is None:
            self.load()
        
        history = []
        for date, row in self._constituents.iterrows():
            history.append({
                'date': date,
                'in_sp500': ticker in row['tickers']
            })
        
        df = pd.DataFrame(history)
        df['date'] = pd.to_datetime(df['date'])
        return df
    
    def filter_universe_by_date(
        self,
        tickers: List[str],
        date: pd.Timestamp
    ) -> List[str]:
        """
        Filter a list of tickers to only those in S&P 500 on a date.
        
        Args:
            tickers: List of tickers to filter
            date: Date to check
            
        Returns:
            Filtered list of tickers
        """
        sp500_tickers = set(self.get_constituents_on_date(date))
        return [t for t in tickers if t in sp500_tickers]
    
    def get_summary_stats(self) -> Dict:
        """
        Get summary statistics about the constituents data.
        
        Returns:
            Dictionary with statistics
        """
        if self._constituents is None:
            self.load()
        
        changes_df = self.get_additions_and_removals()
        
        return {
            'start_date': self._constituents.index.min(),
            'end_date': self._constituents.index.max(),
            'num_dates': len(self._constituents),
            'total_unique_tickers': len(self._ticker_universe),
            'total_changes': len(changes_df),
            'total_additions': changes_df['num_additions'].sum(),
            'total_removals': changes_df['num_removals'].sum(),
            'avg_additions_per_change': changes_df['num_additions'].mean(),
            'avg_removals_per_change': changes_df['num_removals'].mean(),
        }

