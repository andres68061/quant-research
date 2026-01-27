"""
Finnhub data fetching module.

This module provides functionality to fetch financial data from Finnhub API,
including stock prices, company information, and market data.
"""

import logging
import os
from datetime import datetime
from typing import Dict, Optional
from urllib.parse import urlencode

import pandas as pd
import requests

from config.settings import FINNHUB_API_KEY as SETTINGS_FINNHUB_API_KEY

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinnhubDataFetcher:
    """
    A class for fetching financial data from Finnhub API.
    
    This class provides methods to download stock data, company information,
    and other financial data from Finnhub's comprehensive API.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize the FinnhubDataFetcher.
        
        Args:
            api_key (str): Finnhub API key
        """
        self.api_key = api_key
        self.base_url = "https://finnhub.io/api/v1"
        self.session = requests.Session()
        self.session.headers.update({
            'X-Finnhub-Token': api_key,
            'User-Agent': 'QuantProject/1.0'
        })
        
        logger.info("FinnhubDataFetcher initialized")
    
    def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """
        Make a request to the Finnhub API.
        
        Args:
            endpoint (str): API endpoint
            params (Dict): Query parameters
            
        Returns:
            Dict: API response or None if failed
        """
        try:
            url = f"{self.base_url}/{endpoint}"
            if params:
                url += f"?{urlencode(params)}"
            
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in API request: {e}")
            return None
    
    def get_stock_candles(self, symbol: str, resolution: str = 'D', 
                         start_date: Optional[str] = None, end_date: Optional[str] = None,
                         count: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        Get stock candle data (OHLCV).
        
        Args:
            symbol (str): Stock symbol
            resolution (str): Data resolution ('1', '5', '15', '30', '60', 'D', 'W', 'M')
            start_date (str, optional): Start date in 'YYYY-MM-DD' format
            end_date (str, optional): End date in 'YYYY-MM-DD' format
            count (int, optional): Number of data points to retrieve
            
        Returns:
            pd.DataFrame: Stock data with OHLCV columns
        """
        try:
            # Convert dates to timestamps
            start_timestamp = None
            end_timestamp = None
            
            if start_date:
                start_timestamp = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
            if end_date:
                end_timestamp = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
            
            params = {
                'symbol': symbol,
                'resolution': resolution
            }
            
            if start_timestamp:
                params['from'] = start_timestamp
            if end_timestamp:
                params['to'] = end_timestamp
            if count:
                params['count'] = count
            
            response = self._make_request('stock/candle', params)
            
            if not response or response.get('s') != 'ok':
                logger.warning(f"No data returned for {symbol}")
                return None
            
            # Convert to DataFrame
            data = response.get('v', [])  # Volume
            timestamps = response.get('t', [])  # Timestamps
            closes = response.get('c', [])  # Close
            
            if not timestamps:
                logger.warning(f"No data points for {symbol}")
                return None
            
            # Create DataFrame (simplified - only close and adjusted close)
            df = pd.DataFrame({
                'Close': closes,
                'Adj Close': closes,  # Finnhub doesn't provide adjusted close, use close
                'Volume': data
            }, index=pd.to_datetime(timestamps, unit='s'))
            
            logger.info(f"Retrieved {len(df)} candle data points for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching candle data for {symbol}: {e}")
            return None
    
    def get_company_profile(self, symbol: str) -> Optional[Dict]:
        """
        Get company profile information.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            Dict: Company profile information
        """
        try:
            response = self._make_request('stock/profile2', {'symbol': symbol})
            
            if response:
                logger.info(f"Retrieved company profile for {symbol}")
                return response
            else:
                logger.warning(f"No company profile found for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching company profile for {symbol}: {e}")
            return None
    
    def get_quote(self, symbol: str) -> Optional[Dict]:
        """
        Get real-time quote for a stock.
        
        Args:
            symbol (str): Stock symbol
            
        Returns:
            Dict: Quote information
        """
        try:
            response = self._make_request('quote', {'symbol': symbol})
            
            if response:
                logger.info(f"Retrieved quote for {symbol}")
                return response
            else:
                logger.warning(f"No quote found for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
            return None
    
    def get_earnings_calendar(self, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """
        Get earnings calendar for a date range.
        
        Args:
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            
        Returns:
            pd.DataFrame: Earnings calendar data
        """
        try:
            params = {
                'from': start_date,
                'to': end_date
            }
            
            response = self._make_request('calendar/earnings', params)
            
            if response and 'earningsCalendar' in response:
                df = pd.DataFrame(response['earningsCalendar'])
                logger.info(f"Retrieved {len(df)} earnings events")
                return df
            else:
                logger.warning("No earnings calendar data found")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching earnings calendar: {e}")
            return None
    
    def get_news(self, symbol: str = None, category: str = 'general', 
                start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        """
        Get news articles.
        
        Args:
            symbol (str, optional): Stock symbol to filter news
            category (str): News category ('general', 'forex', 'crypto', 'merger')
            start_date (str, optional): Start date in 'YYYY-MM-DD' format
            end_date (str, optional): End date in 'YYYY-MM-DD' format
            
        Returns:
            pd.DataFrame: News articles
        """
        try:
            params = {'category': category}
            
            if symbol:
                params['q'] = symbol
            if start_date:
                params['from'] = start_date
            if end_date:
                params['to'] = end_date
            
            response = self._make_request('news', params)
            
            if response:
                df = pd.DataFrame(response)
                logger.info(f"Retrieved {len(df)} news articles")
                return df
            else:
                logger.warning("No news articles found")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return None
    
    def get_market_status(self) -> Optional[Dict]:
        """
        Get market status information.
        
        Returns:
            Dict: Market status information
        """
        try:
            response = self._make_request('stock/market-status')
            
            if response:
                logger.info("Retrieved market status")
                return response
            else:
                logger.warning("No market status data found")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching market status: {e}")
            return None
    
    def get_symbols(self, exchange: str = 'US') -> Optional[pd.DataFrame]:
        """
        Get all symbols for an exchange.
        
        Args:
            exchange (str): Exchange code (e.g., 'US', 'LSE', 'TSE')
            
        Returns:
            pd.DataFrame: Symbols information
        """
        try:
            response = self._make_request('stock/symbol', {'exchange': exchange})
            
            if response:
                df = pd.DataFrame(response)
                logger.info(f"Retrieved {len(df)} symbols for {exchange} exchange")
                return df
            else:
                logger.warning(f"No symbols found for {exchange} exchange")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching symbols for {exchange}: {e}")
            return None
    
    def get_insider_transactions(self, symbol: str, start_date: str = None, 
                               end_date: str = None) -> Optional[pd.DataFrame]:
        """
        Get insider transactions for a stock.
        
        Args:
            symbol (str): Stock symbol
            start_date (str, optional): Start date in 'YYYY-MM-DD' format
            end_date (str, optional): End date in 'YYYY-MM-DD' format
            
        Returns:
            pd.DataFrame: Insider transactions
        """
        try:
            params = {'symbol': symbol}
            
            if start_date:
                params['from'] = start_date
            if end_date:
                params['to'] = end_date
            
            response = self._make_request('stock/insider-transactions', params)
            
            if response and 'data' in response:
                df = pd.DataFrame(response['data'])
                logger.info(f"Retrieved {len(df)} insider transactions for {symbol}")
                return df
            else:
                logger.warning(f"No insider transactions found for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching insider transactions for {symbol}: {e}")
            return None


# Convenience functions
def get_finnhub_stock_data(api_key: str, symbol: str, start_date: str = None, 
                          end_date: str = None, resolution: str = 'D') -> Optional[pd.DataFrame]:
    """
    Convenience function to get stock data from Finnhub.
    
    Args:
        api_key (str): Finnhub API key
        symbol (str): Stock symbol
        start_date (str, optional): Start date
        end_date (str, optional): End date
        resolution (str): Data resolution
        
    Returns:
        pd.DataFrame: Stock data or None
    """
    fetcher = FinnhubDataFetcher(api_key)
    return fetcher.get_stock_candles(symbol, resolution, start_date, end_date)


def get_finnhub_quote(api_key: str, symbol: str) -> Optional[Dict]:
    """
    Convenience function to get real-time quote from Finnhub.
    
    Args:
        api_key (str): Finnhub API key
        symbol (str): Stock symbol
        
    Returns:
        Dict: Quote information or None
    """
    fetcher = FinnhubDataFetcher(api_key)
    return fetcher.get_quote(symbol)


if __name__ == "__main__":
    # Example usage using environment-based API key
    API_KEY = SETTINGS_FINNHUB_API_KEY or os.getenv("FINNHUB_API_KEY")

    if not API_KEY:
        print("❌ FINNHUB_API_KEY not set. Set it in your environment or .env")
        raise SystemExit(1)

    print("Testing FinnhubDataFetcher...")

    # Create fetcher instance
    fetcher = FinnhubDataFetcher(API_KEY)
    
    # Test getting stock data
    apple_data = fetcher.get_stock_candles('AAPL', resolution='D', count=30)
    
    if apple_data is not None:
        print(f"✅ Successfully retrieved {len(apple_data)} data points for AAPL")
        print(f"   Date range: {apple_data.index[0].date()} to {apple_data.index[-1].date()}")
        print(f"   Latest close: ${apple_data['Close'].iloc[-1]:.2f}")
    else:
        print("❌ Failed to retrieve AAPL data")
    
    # Test getting quote
    quote = fetcher.get_quote('AAPL')
    if quote:
        print(f"✅ Current AAPL quote: ${quote.get('c', 0):.2f}")
    else:
        print("❌ Failed to retrieve AAPL quote")
    
    # Test getting company profile
    profile = fetcher.get_company_profile('AAPL')
    if profile:
        print(f"✅ AAPL company: {profile.get('name', 'Unknown')}")
        print(f"   Industry: {profile.get('finnhubIndustry', 'Unknown')}")
    else:
        print("❌ Failed to retrieve AAPL profile")
