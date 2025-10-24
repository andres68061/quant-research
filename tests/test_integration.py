"""
Integration tests for the quant project.

This module contains integration tests that test the interaction between
different components of the system.
"""

import os
import sys
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config.settings import validate_api_keys
from data.enhanced_stock_data import EnhancedStockDataFetcher


class TestIntegration:
    """Integration tests for the complete system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a test database under data/
        self.test_db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'stock_data_test.db')
        self.fetcher = EnhancedStockDataFetcher(db_path=self.test_db_path)
        
        # Create sample data
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        self.sample_data = pd.DataFrame({
            'Open': np.random.uniform(100, 200, 30),
            'High': np.random.uniform(200, 300, 30),
            'Low': np.random.uniform(50, 150, 30),
            'Close': np.random.uniform(100, 200, 30),
            'Volume': np.random.randint(1000000, 10000000, 30)
        }, index=dates)
    
    def teardown_method(self):
        """Clean up after each test."""
        if hasattr(self, 'fetcher'):
            self.fetcher.close()
        
        # Remove test database
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)
    
    @patch('data.enhanced_stock_data.StockDataFetcher.fetch_stock_data')
    def test_complete_data_flow(self, mock_fetch):
        """Test the complete data flow from fetch to database to retrieval."""
        # Mock the external API call
        mock_fetch.return_value = self.sample_data
        
        # Test 1: Initial fetch (should call external API)
        data = self.fetcher.get_stock_data('TEST', period='1mo')
        
        assert data is not None
        assert len(data) == 30
        assert 'Daily_Return' in data.columns
        assert 'Cumulative_Return' in data.columns
        
        # Verify external API was called
        mock_fetch.assert_called_once()
        
        # Test 2: Second fetch (should use database cache)
        mock_fetch.reset_mock()
        cached_data = self.fetcher.get_stock_data('TEST', period='1mo')
        
        assert cached_data is not None
        assert len(cached_data) == 30
        # External API should not be called again
        mock_fetch.assert_not_called()
        
        # Test 3: Database statistics
        stats = self.fetcher.get_database_stats()
        assert stats['total_records'] > 0
        assert stats['total_symbols'] > 0
    
    @patch('data.enhanced_stock_data.StockDataFetcher.fetch_stock_data')
    def test_incremental_update_flow(self, mock_fetch):
        """Test incremental update functionality."""
        # Mock initial data
        mock_fetch.return_value = self.sample_data
        
        # Initial fetch
        initial_data = self.fetcher.get_stock_data('TEST', period='1mo')
        assert initial_data is not None
        
        # Mock new data (simulating new data since last fetch)
        new_dates = pd.date_range('2024-01-31', periods=5, freq='D')
        new_data = pd.DataFrame({
            'Open': np.random.uniform(100, 200, 5),
            'High': np.random.uniform(200, 300, 5),
            'Low': np.random.uniform(50, 150, 5),
            'Close': np.random.uniform(100, 200, 5),
            'Volume': np.random.randint(1000000, 10000000, 5)
        }, index=new_dates)
        
        # Mock incremental fetch
        mock_fetch.return_value = new_data
        
        # Test incremental update
        updated_data = self.fetcher.get_stock_data_incremental('TEST', period='1mo')
        
        assert updated_data is not None
        # Should have more data than initial
        assert len(updated_data) >= len(initial_data)
    
    def test_multiple_stocks_integration(self):
        """Test fetching multiple stocks with mocked data."""
        symbols = ['TEST1', 'TEST2', 'TEST3']
        
        with patch('data.enhanced_stock_data.StockDataFetcher.fetch_stock_data') as mock_fetch:
            # Mock data for each symbol
            mock_fetch.return_value = self.sample_data
            
            # Fetch multiple stocks
            results = self.fetcher.get_multiple_stocks(symbols, period='1mo')
            
            assert len(results) == 3
            assert all(symbol in results for symbol in symbols)
            assert all(isinstance(data, pd.DataFrame) for data in results.values())
            assert all(len(data) == 30 for data in results.values())
    
    def test_database_persistence(self):
        """Test that data persists in database across fetcher instances."""
        # Create first fetcher and store data
        with patch('data.enhanced_stock_data.StockDataFetcher.fetch_stock_data') as mock_fetch:
            mock_fetch.return_value = self.sample_data
            
            data1 = self.fetcher.get_stock_data('PERSIST', period='1mo')
            assert data1 is not None
        
        # Close first fetcher
        self.fetcher.close()
        
        # Create new fetcher (should use same database)
        new_fetcher = EnhancedStockDataFetcher(db_path=self.test_db_path)
        
        try:
            # Should retrieve from database without external call
            with patch('data.enhanced_stock_data.StockDataFetcher.fetch_stock_data') as mock_fetch:
                data2 = new_fetcher.get_stock_data('PERSIST', period='1mo')
                
                assert data2 is not None
                assert len(data2) == 30
                # External API should not be called
                mock_fetch.assert_not_called()
        finally:
            new_fetcher.close()
    
    def test_error_handling_integration(self):
        """Test error handling across the system."""
        # Test with invalid symbol
        with patch('src.data.enhanced_stock_data.StockDataFetcher.fetch_stock_data') as mock_fetch:
            mock_fetch.return_value = None  # Simulate API failure
            
            data = self.fetcher.get_stock_data('INVALID', period='1mo')
            assert data is None
        
        # Test database error handling
        with patch('src.data.enhanced_stock_data.StockDatabase.get_stock_data') as mock_db:
            mock_db.side_effect = Exception("Database error")
            
            # Should handle database errors gracefully
            data = self.fetcher.get_stock_data('TEST', period='1mo')
            # Should fall back to external fetch
            assert data is None or isinstance(data, pd.DataFrame)
    
    def test_configuration_integration(self):
        """Test configuration system integration."""
        # Test API key validation
        validation = validate_api_keys()
        assert isinstance(validation, dict)
        assert 'finnhub' in validation
        
        # Test database path configuration
        from config.settings import get_database_path
        db_path = get_database_path()
        assert db_path.exists() or str(db_path).endswith('.db')
    
    def test_ml_data_preparation_integration(self):
        """Test ML data preparation pipeline."""
        # This would test the ml_data_preparation.py script
        # For now, just test that the module can be imported
        try:
            # Import from scripts package after relocation
            import importlib
            spec = importlib.util.find_spec('scripts.ml_data_preparation')
            assert spec is not None
        except Exception as e:
            pytest.fail(f"ML data preparation module not found: {e}")


class TestMockAPIResponses:
    """Test with mocked API responses to avoid external dependencies."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.test_db_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'stock_data_test_mock.db')
    
    def teardown_method(self):
        """Clean up after each test."""
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)
    
    @patch('yfinance.Ticker')
    def test_yfinance_mock_response(self, mock_ticker):
        """Test with mocked yfinance responses."""
        # Create mock ticker
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [103, 104, 105],
            'Volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range('2024-01-01', periods=3))
        mock_ticker.return_value = mock_ticker_instance
        
        # Test with mocked data
        fetcher = EnhancedStockDataFetcher(db_path=self.test_db_path)
        try:
            data = fetcher.get_stock_data('MOCK', period='1mo')
            assert data is not None
            assert len(data) == 3
            assert 'Daily_Return' in data.columns
        finally:
            fetcher.close()
    
    @patch('requests.get')
    def test_finnhub_mock_response(self, mock_get):
        """Test with mocked Finnhub API responses."""
        # Mock Finnhub API response
        mock_response = Mock()
        mock_response.json.return_value = {
            'c': [100, 101, 102],  # Close prices
            'h': [105, 106, 107],  # High prices
            'l': [95, 96, 97],     # Low prices
            'o': [98, 99, 100],    # Open prices
            'v': [1000000, 1100000, 1200000],  # Volume
            't': [1704067200, 1704153600, 1704240000]  # Timestamps
        }
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        # Test Finnhub integration
        fetcher = EnhancedStockDataFetcher(finnhub_api_key="test_key", db_path=self.test_db_path)
        try:
            # This would test Finnhub data fetching
            # For now, just verify the fetcher initializes
            assert fetcher.finnhub_fetcher is not None
        finally:
            fetcher.close()


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "--tb=short"])
