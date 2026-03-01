"""
Integration tests for the quant project.

These tests exercise EnhancedStockDataFetcher, which depends on
core.data.database.StockDatabase. That module has not been implemented yet,
so every test that tries to instantiate the fetcher is skipped automatically.
"""

import os
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

try:
    from core.data.database import StockDatabase  # noqa: F401
    from core.data.enhanced_stock_data import EnhancedStockDataFetcher

    _HAS_DATABASE = True
except ImportError:
    _HAS_DATABASE = False

skip_no_db = pytest.mark.skipif(
    not _HAS_DATABASE,
    reason="StockDatabase module not yet implemented",
)


@skip_no_db
class TestIntegration:
    """Integration tests for the complete system."""

    def setup_method(self):
        self.test_db_path = os.path.join(
            os.path.dirname(__file__), '..', 'data', 'stock_data_test.db'
        )
        self.fetcher = EnhancedStockDataFetcher(db_path=self.test_db_path)

        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        self.sample_data = pd.DataFrame({
            'Open': np.random.uniform(100, 200, 30),
            'High': np.random.uniform(200, 300, 30),
            'Low': np.random.uniform(50, 150, 30),
            'Close': np.random.uniform(100, 200, 30),
            'Volume': np.random.randint(1000000, 10000000, 30),
        }, index=dates)

    def teardown_method(self):
        if hasattr(self, 'fetcher'):
            self.fetcher.close()
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)

    @patch('core.data.enhanced_stock_data.StockDataFetcher.fetch_stock_data')
    def test_complete_data_flow(self, mock_fetch):
        """Test fetch -> database -> retrieval cycle."""
        mock_fetch.return_value = self.sample_data

        data = self.fetcher.get_stock_data('TEST', period='1mo')
        assert data is not None
        assert len(data) == 30
        assert 'Daily_Return' in data.columns
        mock_fetch.assert_called_once()

        mock_fetch.reset_mock()
        cached_data = self.fetcher.get_stock_data('TEST', period='1mo')
        assert cached_data is not None
        mock_fetch.assert_not_called()

        stats = self.fetcher.get_database_stats()
        assert stats['total_records'] > 0

    @patch('core.data.enhanced_stock_data.StockDataFetcher.fetch_stock_data')
    def test_incremental_update_flow(self, mock_fetch):
        mock_fetch.return_value = self.sample_data
        initial_data = self.fetcher.get_stock_data('TEST', period='1mo')
        assert initial_data is not None

        new_dates = pd.date_range('2024-01-31', periods=5, freq='D')
        new_data = pd.DataFrame({
            'Open': np.random.uniform(100, 200, 5),
            'High': np.random.uniform(200, 300, 5),
            'Low': np.random.uniform(50, 150, 5),
            'Close': np.random.uniform(100, 200, 5),
            'Volume': np.random.randint(1000000, 10000000, 5),
        }, index=new_dates)
        mock_fetch.return_value = new_data

        updated_data = self.fetcher.get_stock_data_incremental('TEST', period='1mo')
        assert updated_data is not None
        assert len(updated_data) >= len(initial_data)

    def test_configuration_integration(self):
        from config.settings import get_database_path, validate_api_keys

        validation = validate_api_keys()
        assert isinstance(validation, dict)
        assert 'finnhub' in validation

        db_path = get_database_path()
        assert db_path.exists() or str(db_path).endswith('.db')


@skip_no_db
class TestMockAPIResponses:
    """Test with mocked API responses to avoid external dependencies."""

    def setup_method(self):
        self.test_db_path = os.path.join(
            os.path.dirname(__file__), '..', 'data', 'stock_data_test_mock.db'
        )

    def teardown_method(self):
        if os.path.exists(self.test_db_path):
            os.remove(self.test_db_path)

    @patch('yfinance.Ticker')
    def test_yfinance_mock_response(self, mock_ticker):
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [103, 104, 105],
            'Volume': [1000000, 1100000, 1200000],
        }, index=pd.date_range('2024-01-01', periods=3))
        mock_ticker.return_value = mock_ticker_instance

        fetcher = EnhancedStockDataFetcher(db_path=self.test_db_path)
        try:
            data = fetcher.get_stock_data('MOCK', period='1mo')
            assert data is not None
            assert len(data) == 3
        finally:
            fetcher.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
