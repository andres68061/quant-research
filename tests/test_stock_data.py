"""
Tests for the stock_data module.

This module contains unit tests to verify the functionality of the StockDataFetcher class
and related functions.
"""

import os
import sys
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.stock_data import StockDataFetcher, fetch_stock_data, get_stock_analysis


class TestStockDataFetcher:
    """Test cases for the StockDataFetcher class."""
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.fetcher = StockDataFetcher()
        
        # Create sample data for testing
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        self.sample_data = pd.DataFrame({
            'Open': np.random.uniform(100, 200, 100),
            'High': np.random.uniform(200, 300, 100),
            'Low': np.random.uniform(50, 150, 100),
            'Close': np.random.uniform(100, 200, 100),
            'Volume': np.random.randint(1000000, 10000000, 100)
        }, index=dates)
    
    def test_initialization(self):
        """Test that StockDataFetcher initializes correctly."""
        assert self.fetcher.cache == {}
        assert isinstance(self.fetcher.cache, dict)
    
    def test_calculate_returns(self):
        """Test the calculate_returns method."""
        # Test with valid data
        result = self.fetcher.calculate_returns(self.sample_data)
        
        assert isinstance(result, pd.DataFrame)
        assert 'Daily_Return' in result.columns
        assert 'Cumulative_Return' in result.columns
        assert 'Log_Return' in result.columns
        assert 'Volatility_30d' in result.columns
        assert 'MA_20' in result.columns
        assert 'MA_50' in result.columns
        
        # Test with empty data
        empty_result = self.fetcher.calculate_returns(pd.DataFrame())
        assert empty_result.empty
    
    def test_get_basic_statistics(self):
        """Test the get_basic_statistics method."""
        # Add returns to sample data
        data_with_returns = self.fetcher.calculate_returns(self.sample_data)
        stats = self.fetcher.get_basic_statistics(data_with_returns)
        
        assert isinstance(stats, dict)
        assert 'current_price' in stats
        assert 'highest_price' in stats
        assert 'lowest_price' in stats
        assert 'price_range' in stats
        assert 'mean_daily_return' in stats
        assert 'std_daily_return' in stats
        assert 'annualized_volatility' in stats
        assert 'sharpe_ratio' in stats
        assert 'total_return' in stats
        assert 'avg_volume' in stats
        
        # Test with empty data
        empty_stats = self.fetcher.get_basic_statistics(pd.DataFrame())
        assert empty_stats == {}
    
    def test_fetch_multiple_stocks(self):
        """Test the fetch_multiple_stocks method."""
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        
        # Mock the fetch_stock_data method to avoid actual API calls
        with patch.object(self.fetcher, 'fetch_stock_data') as mock_fetch:
            mock_fetch.return_value = self.sample_data
            
            results = self.fetcher.fetch_multiple_stocks(symbols, period='1y')
            
            assert isinstance(results, dict)
            assert len(results) == 3
            assert all(symbol in results for symbol in symbols)
            assert all(isinstance(data, pd.DataFrame) for data in results.values())


class TestConvenienceFunctions:
    """Test cases for the convenience functions."""
    
    def test_fetch_stock_data_function(self):
        """Test the fetch_stock_data convenience function."""
        # Mock the StockDataFetcher to avoid actual API calls
        with patch('data.stock_data.StockDataFetcher') as mock_fetcher_class:
            mock_fetcher = Mock()
            mock_fetcher.fetch_stock_data.return_value = pd.DataFrame({'Close': [100, 101, 102]})
            mock_fetcher_class.return_value = mock_fetcher
            
            result = fetch_stock_data('AAPL', period='1y')
            
            assert isinstance(result, pd.DataFrame)
            mock_fetcher.fetch_stock_data.assert_called_once_with('AAPL', period='1y')
    
    def test_get_stock_analysis_function(self):
        """Test the get_stock_analysis convenience function."""
        # Mock the StockDataFetcher to avoid actual API calls
        with patch('data.stock_data.StockDataFetcher') as mock_fetcher_class:
            mock_fetcher = Mock()
            sample_data = pd.DataFrame({
                'Close': [100, 101, 102],
                'High': [105, 106, 107],
                'Low': [95, 96, 97],
                'Volume': [1000000, 1100000, 1200000]
            })
            mock_fetcher.fetch_stock_data.return_value = sample_data
            mock_fetcher.calculate_returns.return_value = sample_data
            mock_fetcher.get_basic_statistics.return_value = {'current_price': 102}
            mock_fetcher_class.return_value = mock_fetcher
            
            result = get_stock_analysis('AAPL', period='1y')
            
            assert isinstance(result, dict)
            assert 'data' in result
            assert 'statistics' in result
            assert isinstance(result['data'], pd.DataFrame)
            assert isinstance(result['statistics'], dict)


class TestDataValidation:
    """Test cases for data validation and edge cases."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.fetcher = StockDataFetcher()
    
    def test_calculate_returns_with_none_data(self):
        """Test calculate_returns with None data."""
        result = self.fetcher.calculate_returns(None)
        assert result.empty
    
    def test_calculate_returns_with_single_row(self):
        """Test calculate_returns with data containing only one row."""
        single_row_data = pd.DataFrame({
            'Close': [100],
            'High': [105],
            'Low': [95],
            'Volume': [1000000]
        })
        result = self.fetcher.calculate_returns(single_row_data)
        
        # Should handle single row gracefully
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
    
    def test_statistics_with_missing_columns(self):
        """Test get_basic_statistics with data missing some columns."""
        incomplete_data = pd.DataFrame({
            'Close': [100, 101, 102]
            # Missing High, Low, Volume columns
        })
        
        stats = self.fetcher.get_basic_statistics(incomplete_data)
        
        # Should handle missing columns gracefully
        assert isinstance(stats, dict)
        assert 'current_price' in stats


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v"])
