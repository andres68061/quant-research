"""
Enhanced stock data fetching module with database caching and multiple data sources.

This module provides intelligent data fetching that checks the database first,
then fetches only new data from external sources to minimize API calls.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

import pandas as pd

from config.settings import get_database_path

from .database import StockDatabase
from .finnhub_data import FinnhubDataFetcher

# Import our modules
from .stock_data import StockDataFetcher

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedStockDataFetcher:
    """
    Enhanced stock data fetcher with intelligent caching and multiple data sources.
    
    This class provides methods to fetch stock data with the following features:
    - Database caching to avoid redundant API calls
    - Incremental updates (only fetch new data)
    - Multiple data sources (Yahoo Finance, Finnhub)
    - Automatic data validation and cleaning
    """
    
    def __init__(self, finnhub_api_key: Optional[str] = None, db_path: Optional[str] = None):
        """
        Initialize the EnhancedStockDataFetcher.
        
        Args:
            finnhub_api_key (str, optional): Finnhub API key
            db_path (str): Path to the SQLite database. If None, uses config.get_database_path().
        """
        resolved_db_path = str(db_path) if db_path else str(get_database_path())
        self.db = StockDatabase(resolved_db_path)
        self.yf_fetcher = StockDataFetcher()
        
        if finnhub_api_key:
            self.finnhub_fetcher = FinnhubDataFetcher(finnhub_api_key)
            logger.info("Finnhub integration enabled")
        else:
            self.finnhub_fetcher = None
            logger.info("Finnhub integration disabled (no API key provided)")
        
        logger.info("EnhancedStockDataFetcher initialized")
    
    def get_stock_data(self, symbol: str, period: str = "1y", 
                      source: str = "auto", force_refresh: bool = False) -> Optional[pd.DataFrame]:
        """
        Get stock data with intelligent caching.
        
        Args:
            symbol (str): Stock symbol
            period (str): Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            source (str): Data source ('auto', 'yfinance', 'finnhub', 'database')
            force_refresh (bool): Force refresh from external source
            
        Returns:
            pd.DataFrame: Stock data with calculated returns
        """
        try:
            symbol = symbol.upper()
            
            # Determine the best data source
            if source == "auto":
                source = self._select_best_source(symbol)
            
            # Check database first (unless force refresh)
            if not force_refresh:
                db_data = self._get_from_database(symbol, source)
                if db_data is not None:
                    logger.info(f"Retrieved {len(db_data)} records for {symbol} from database")
                    return self._calculate_returns(db_data)
            
            # Fetch from external source
            external_data = self._fetch_from_external(symbol, period, source)
            if external_data is None:
                logger.warning(f"Failed to fetch data for {symbol} from {source}")
                return None
            
            # Store in database
            self._store_in_database(symbol, external_data, source)
            
            # Calculate and return returns
            return self._calculate_returns(external_data)
            
        except Exception as e:
            logger.error(f"Error getting stock data for {symbol}: {e}")
            return None
    
    def get_stock_data_incremental(self, symbol: str, period: str = "1y", 
                                 source: str = "auto") -> Optional[pd.DataFrame]:
        """
        Get stock data with incremental updates (only fetch new data).
        
        Args:
            symbol (str): Stock symbol
            period (str): Data period
            source (str): Data source
            
        Returns:
            pd.DataFrame: Complete stock data (existing + new)
        """
        try:
            symbol = symbol.upper()
            
            # Get existing data from database
            existing_data = self._get_from_database(symbol, source)
            
            # Determine what new data we need
            if existing_data is not None:
                latest_date = existing_data.index.max()
                today = datetime.now().date()
                
                # If we have recent data (within 1 day), return existing data
                if (today - latest_date.date()).days <= 1:
                    logger.info(f"Using cached data for {symbol} (up to date)")
                    return self._calculate_returns(existing_data)
                
                # Fetch only new data
                start_date = (latest_date + timedelta(days=1)).strftime('%Y-%m-%d')
                new_data = self._fetch_from_external(symbol, start_date=start_date, source=source)
                
                if new_data is not None and not new_data.empty:
                    # Combine existing and new data
                    combined_data = pd.concat([existing_data, new_data])
                    combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
                    combined_data.sort_index(inplace=True)
                    
                    # Store updated data
                    self._store_in_database(symbol, combined_data, source, replace=True)
                    
                    logger.info(f"Updated {symbol} with {len(new_data)} new records")
                    return self._calculate_returns(combined_data)
                else:
                    logger.info(f"No new data available for {symbol}")
                    return self._calculate_returns(existing_data)
            else:
                # No existing data, fetch complete dataset
                return self.get_stock_data(symbol, period, source)
                
        except Exception as e:
            logger.error(f"Error in incremental update for {symbol}: {e}")
            return None
    
    def get_multiple_stocks(self, symbols: List[str], period: str = "1y", 
                          source: str = "auto") -> Dict[str, pd.DataFrame]:
        """
        Get data for multiple stocks efficiently.
        
        Args:
            symbols (List[str]): List of stock symbols
            period (str): Data period
            source (str): Data source
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary with symbol as key and data as value
        """
        results = {}
        
        for symbol in symbols:
            try:
                data = self.get_stock_data(symbol, period, source)
                if data is not None:
                    results[symbol] = data
                    logger.info(f"Successfully fetched data for {symbol}")
                else:
                    logger.warning(f"Failed to fetch data for {symbol}")
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {e}")
        
        return results
    
    def get_stock_analysis(self, symbol: str, period: str = "1y", 
                          source: str = "auto") -> Dict[str, Union[pd.DataFrame, Dict]]:
        """
        Get comprehensive stock analysis including data and statistics.
        
        Args:
            symbol (str): Stock symbol
            period (str): Data period
            source (str): Data source
            
        Returns:
            Dict: Contains 'data' (DataFrame) and 'statistics' (Dict)
        """
        data = self.get_stock_data(symbol, period, source)
        
        if data is None:
            return {'data': pd.DataFrame(), 'statistics': {}}
        
        # Calculate statistics
        stats = self._calculate_statistics(data)
        
        return {
            'data': data,
            'statistics': stats
        }
    
    def _select_best_source(self, symbol: str) -> str:
        """Select the best data source for a symbol."""
        # For now, default to yfinance
        # In the future, this could check availability, data quality, etc.
        return "yfinance"
    
    def _get_from_database(self, symbol: str, source: str) -> Optional[pd.DataFrame]:
        """Get data from database."""
        try:
            return self.db.get_stock_data(symbol, source=source)
        except Exception as e:
            logger.error(f"Error getting data from database for {symbol}: {e}")
            return None
    
    def _fetch_from_external(self, symbol: str, period: str = None, 
                           source: str = "yfinance", start_date: str = None) -> Optional[pd.DataFrame]:
        """Fetch data from external source."""
        try:
            if source == "yfinance":
                if start_date:
                    # Fetch from specific date
                    end_date = datetime.now().strftime('%Y-%m-%d')
                    return self.yf_fetcher.fetch_stock_data(symbol, start_date=start_date, end_date=end_date)
                elif period:
                    # Fetch by period
                    return self.yf_fetcher.fetch_stock_data(symbol, period=period)
                else:
                    # Default to 1 year
                    return self.yf_fetcher.fetch_stock_data(symbol, period="1y")
            
            elif source == "finnhub" and self.finnhub_fetcher:
                if start_date:
                    # Fetch from specific date
                    end_date = datetime.now().strftime('%Y-%m-%d')
                    return self.finnhub_fetcher.get_stock_candles(symbol, start_date=start_date, end_date=end_date)
                else:
                    # Convert period to count
                    count = self._period_to_count(period)
                    return self.finnhub_fetcher.get_stock_candles(symbol, count=count)
            
            else:
                logger.warning(f"Unsupported source: {source}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching from external source for {symbol}: {e}")
            return None
    
    def _store_in_database(self, symbol: str, data: pd.DataFrame, source: str, replace: bool = False):
        """Store data in database."""
        try:
            if replace:
                # Delete existing data first
                self.db.delete_symbol_data(symbol, source)
            
            self.db.store_stock_data(symbol, data, source)
        except Exception as e:
            logger.error(f"Error storing data in database for {symbol}: {e}")
    
    def _calculate_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate returns and technical indicators."""
        if data is None or data.empty:
            return pd.DataFrame()
        
        # Use the existing calculation method from StockDataFetcher
        return self.yf_fetcher.calculate_returns(data)
    
    def _calculate_statistics(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate comprehensive statistics."""
        if data is None or data.empty:
            return {}
        
        # Use the existing statistics method from StockDataFetcher
        return self.yf_fetcher.get_basic_statistics(data)
    
    def _period_to_count(self, period: str) -> int:
        """Convert period string to count for Finnhub API."""
        period_map = {
            '1d': 1,
            '5d': 5,
            '1mo': 30,
            '3mo': 90,
            '6mo': 180,
            '1y': 365,
            '2y': 730,
            '5y': 1825,
            '10y': 3650,
            'max': 3650  # Finnhub limit
        }
        return period_map.get(period, 365)
    
    def get_database_stats(self) -> Dict:
        """Get database statistics."""
        return self.db.get_database_stats()
    
    def get_available_symbols(self, source: str = "yfinance") -> List[str]:
        """Get all symbols available in the database."""
        return self.db.get_all_symbols(source)
    
    def cleanup_old_data(self, days_to_keep: int = 3650) -> int:
        """
        Clean up old data from the database.
        
        Args:
            days_to_keep (int): Number of days of data to keep
            
        Returns:
            int: Number of records deleted
        """
        try:
            cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).strftime('%Y-%m-%d')
            
            cursor = self.db.connection.cursor()
            cursor.execute('''
                DELETE FROM stock_data 
                WHERE date < ?
            ''', (cutoff_date,))
            
            deleted_count = cursor.rowcount
            self.db.connection.commit()
            
            logger.info(f"Cleaned up {deleted_count} old records")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old data: {e}")
            return 0
    
    def close(self):
        """Close the database connection."""
        self.db.close()


# Convenience functions
def get_enhanced_stock_data(finnhub_api_key: str = None, symbol: str = None, 
                           period: str = "1y", source: str = "auto") -> Optional[pd.DataFrame]:
    """
    Convenience function to get enhanced stock data.
    
    Args:
        finnhub_api_key (str, optional): Finnhub API key
        symbol (str): Stock symbol
        period (str): Data period
        source (str): Data source
        
    Returns:
        pd.DataFrame: Stock data or None
    """
    fetcher = EnhancedStockDataFetcher(finnhub_api_key)
    try:
        return fetcher.get_stock_data(symbol, period, source)
    finally:
        fetcher.close()


def get_enhanced_stock_analysis(finnhub_api_key: str = None, symbol: str = None, 
                               period: str = "1y", source: str = "auto") -> Dict[str, Union[pd.DataFrame, Dict]]:
    """
    Convenience function to get enhanced stock analysis.
    
    Args:
        finnhub_api_key (str, optional): Finnhub API key
        symbol (str): Stock symbol
        period (str): Data period
        source (str): Data source
        
    Returns:
        Dict: Analysis results
    """
    fetcher = EnhancedStockDataFetcher(finnhub_api_key)
    try:
        return fetcher.get_stock_analysis(symbol, period, source)
    finally:
        fetcher.close()


if __name__ == "__main__":
    # Example usage
    from config.settings import FINNHUB_API_KEY

    print("Testing EnhancedStockDataFetcher...")

    # Create enhanced fetcher
    fetcher = EnhancedStockDataFetcher(FINNHUB_API_KEY)
    
    try:
        # Test getting stock data
        apple_data = fetcher.get_stock_data('AAPL', period='6mo')
        
        if apple_data is not None:
            print(f"✅ Successfully retrieved {len(apple_data)} data points for AAPL")
            print(f"   Date range: {apple_data.index[0].date()} to {apple_data.index[-1].date()}")
            print(f"   Latest close: ${apple_data['Close'].iloc[-1]:.2f}")
            
            # Test incremental update
            print("\n🔄 Testing incremental update...")
            updated_data = fetcher.get_stock_data_incremental('AAPL', period='6mo')
            if updated_data is not None:
                print(f"   Updated data points: {len(updated_data)}")
        else:
            print("❌ Failed to retrieve AAPL data")
        
        # Test multiple stocks
        print("\n📊 Testing multiple stocks...")
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        multiple_data = fetcher.get_multiple_stocks(symbols, period='1mo')
        print(f"   Retrieved data for {len(multiple_data)} stocks")
        
        # Get database stats
        stats = fetcher.get_database_stats()
        print(f"\n📈 Database stats: {stats}")
        
    finally:
        fetcher.close()
        print("✅ Enhanced fetcher test completed")
