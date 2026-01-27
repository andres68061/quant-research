"""
Stock data fetching and processing module.

This module provides functions to fetch, process, and analyze stock market data
using various data sources and APIs.
"""

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import yfinance as yf

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockDataFetcher:
    """
    A class for fetching and processing stock market data.
    
    This class provides methods to download stock data from Yahoo Finance,
    calculate various financial metrics, and perform basic analysis.
    """
    
    def __init__(self):
        """Initialize the StockDataFetcher."""
        self.cache = {}  # Simple in-memory cache
        logger.info("StockDataFetcher initialized")
    
    def fetch_stock_data(
        self, 
        symbol: str, 
        period: str = "1y",
        interval: str = "1d",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """
        Fetch stock data for a given symbol.
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL', 'MSFT')
            period (str): Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval (str): Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            start_date (str, optional): Start date in 'YYYY-MM-DD' format
            end_date (str, optional): End date in 'YYYY-MM-DD' format
            
        Returns:
            pd.DataFrame: Stock data with columns [Open, High, Low, Close, Volume, Dividends, Stock Splits]
        """
        try:
            logger.info(f"Fetching data for {symbol}")
            
            # Create cache key
            cache_key = f"{symbol}_{period}_{interval}_{start_date or 'None'}_{end_date or 'None'}"
            
            # Check cache first
            if cache_key in self.cache:
                logger.info(f"Using cached data for {symbol}")
                return self.cache[cache_key]
            
            # Fetch data from Yahoo Finance
            ticker = yf.Ticker(symbol)
            
            if start_date and end_date:
                data = ticker.history(start=start_date, end=end_date, interval=interval)
            else:
                data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                logger.warning(f"No data found for symbol {symbol}")
                return None
            
            # Cache the data
            self.cache[cache_key] = data
            
            logger.info(f"Successfully fetched {len(data)} data points for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def calculate_returns(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate various return metrics from price data.
        
        Args:
            data (pd.DataFrame): Stock price data
            
        Returns:
            pd.DataFrame: Data with additional return columns
        """
        if data is None or data.empty:
            return pd.DataFrame()
        
        # Create a copy to avoid modifying original data
        result = data.copy()
        
        # Calculate daily returns
        result['Daily_Return'] = result['Close'].pct_change()
        
        # Calculate cumulative returns
        result['Cumulative_Return'] = (1 + result['Daily_Return']).cumprod() - 1
        
        # Calculate log returns
        result['Log_Return'] = np.log(result['Close'] / result['Close'].shift(1))
        
        # Calculate volatility (rolling 30-day)
        result['Volatility_30d'] = result['Daily_Return'].rolling(window=30).std() * np.sqrt(252)
        
        # Calculate moving averages
        result['MA_20'] = result['Close'].rolling(window=20).mean()
        result['MA_50'] = result['Close'].rolling(window=50).mean()
        
        return result
    
    def get_basic_statistics(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate basic statistics for the stock data.
        
        Args:
            data (pd.DataFrame): Stock data with returns calculated
            
        Returns:
            Dict[str, float]: Dictionary containing various statistics
        """
        if data is None or data.empty:
            return {}
        
        stats = {}
        
        # Price statistics
        stats['current_price'] = data['Close'].iloc[-1]
        stats['highest_price'] = data['High'].max() if 'High' in data.columns else data['Close'].max()
        stats['lowest_price'] = data['Low'].min() if 'Low' in data.columns else data['Close'].min()
        stats['price_range'] = stats['highest_price'] - stats['lowest_price']
        
        # Return statistics
        if 'Daily_Return' in data.columns:
            returns = data['Daily_Return'].dropna()
            stats['mean_daily_return'] = returns.mean()
            stats['std_daily_return'] = returns.std()
            stats['annualized_volatility'] = returns.std() * np.sqrt(252)
            stats['sharpe_ratio'] = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
            stats['total_return'] = (data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100
        
        # Volume statistics
        if 'Volume' in data.columns:
            stats['avg_volume'] = data['Volume'].mean()
            stats['max_volume'] = data['Volume'].max()
            stats['min_volume'] = data['Volume'].min()
        else:
            stats['avg_volume'] = 0
            stats['max_volume'] = 0
            stats['min_volume'] = 0
        
        return stats
    
    def fetch_multiple_stocks(
        self, 
        symbols: List[str], 
        period: str = "1y"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple stocks simultaneously.
        
        Args:
            symbols (List[str]): List of stock symbols
            period (str): Data period
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary with symbol as key and data as value
        """
        results = {}
        
        for symbol in symbols:
            data = self.fetch_stock_data(symbol, period=period)
            if data is not None:
                results[symbol] = data
                logger.info(f"Successfully fetched data for {symbol}")
            else:
                logger.warning(f"Failed to fetch data for {symbol}")
        
        return results


# Convenience functions for quick access
def fetch_stock_data(symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
    """
    Quick function to fetch stock data.
    
    Args:
        symbol (str): Stock symbol
        period (str): Data period
        
    Returns:
        pd.DataFrame: Stock data
    """
    fetcher = StockDataFetcher()
    return fetcher.fetch_stock_data(symbol, period=period)


def get_stock_analysis(symbol: str, period: str = "1y") -> Dict[str, Union[pd.DataFrame, Dict[str, float]]]:
    """
    Get comprehensive stock analysis including data and statistics.
    
    Args:
        symbol (str): Stock symbol
        period (str): Data period
        
    Returns:
        Dict: Contains 'data' (DataFrame) and 'statistics' (Dict)
    """
    fetcher = StockDataFetcher()
    
    # Fetch data
    data = fetcher.fetch_stock_data(symbol, period=period)
    if data is None:
        return {'data': pd.DataFrame(), 'statistics': {}}
    
    # Calculate returns
    data_with_returns = fetcher.calculate_returns(data)
    
    # Get statistics
    stats = fetcher.get_basic_statistics(data_with_returns)
    
    return {
        'data': data_with_returns,
        'statistics': stats
    }


if __name__ == "__main__":
    # Example usage
    print("Testing StockDataFetcher...")
    
    # Create fetcher instance
    fetcher = StockDataFetcher()
    
    # Fetch data for Apple
    apple_data = fetcher.fetch_stock_data('AAPL', period='6mo')
    
    if apple_data is not None:
        # Calculate returns
        apple_with_returns = fetcher.calculate_returns(apple_data)
        
        # Get statistics
        stats = fetcher.get_basic_statistics(apple_with_returns)
        
        print(f"✅ Successfully analyzed AAPL data")
        print(f"   Data points: {len(apple_data)}")
        print(f"   Current price: ${stats.get('current_price', 0):.2f}")
        print(f"   Total return: {stats.get('total_return', 0):.2f}%")
        print(f"   Volatility: {stats.get('annualized_volatility', 0):.2f}")
    else:
        print("❌ Failed to fetch AAPL data")
