#!/usr/bin/env python3
"""
Test script for the database system and enhanced data fetching.

This script demonstrates how to use the database caching system
with incremental updates and multiple data sources.
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from config.settings import FINNHUB_API_KEY
from data.enhanced_stock_data import EnhancedStockDataFetcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger("test_database_system")


def test_database_system():
    """Test the complete database system."""
    logger.info("db_test_start")

    fetcher = EnhancedStockDataFetcher(FINNHUB_API_KEY)

    try:
        # Test 1: Initial data fetch
        logger.info("initial_fetch", extra={"symbol": "AAPL", "period": "6mo"})
        apple_data = fetcher.get_stock_data('AAPL', period='6mo')
        if apple_data is None:
            logger.error("initial_fetch_failed", extra={"symbol": "AAPL"})
            return
        logger.info(
            "initial_fetch_done",
            extra={
                "rows": int(len(apple_data)),
                "start": str(apple_data.index[0].date()),
                "end": str(apple_data.index[-1].date()),
                "last_close": float(apple_data['Close'].iloc[-1]),
                "last_cumret_pct": float(apple_data['Cumulative_Return'].iloc[-1] * 100),
            },
        )

        # Test 2: Cached fetch
        logger.info("cached_fetch", extra={"symbol": "AAPL", "period": "6mo"})
        cached_data = fetcher.get_stock_data('AAPL', period='6mo')
        if cached_data is not None:
            logger.info(
                "cached_fetch_done",
                extra={"rows": int(len(cached_data)), "last_close": float(cached_data['Close'].iloc[-1])},
            )
        else:
            logger.error("cached_fetch_failed", extra={"symbol": "AAPL"})

        # Test 3: Incremental update
        logger.info("incremental_update", extra={"symbol": "AAPL", "period": "6mo"})
        incremental_data = fetcher.get_stock_data_incremental('AAPL', period='6mo')
        if incremental_data is not None:
            logger.info(
                "incremental_update_done",
                extra={"rows": int(len(incremental_data)), "last_close": float(incremental_data['Close'].iloc[-1])},
            )
        else:
            logger.error("incremental_update_failed", extra={"symbol": "AAPL"})

        # Test 4: Multiple stocks
        symbols = ['AAPL', 'MSFT', 'GOOGL']
        logger.info("multi_fetch_start", extra={"symbols": ",".join(symbols), "period": "1mo"})
        multiple_data = fetcher.get_multiple_stocks(symbols, period='1mo')
        for sym, df in multiple_data.items():
            if df is not None:
                logger.info("multi_fetch_symbol", extra={"symbol": sym, "rows": int(len(df)), "last_close": float(df['Close'].iloc[-1])})
            else:
                logger.error("multi_fetch_failed_symbol", extra={"symbol": sym})

        # Test 5: Database statistics
        stats = fetcher.get_database_stats()
        logger.info("db_stats", extra=stats)

        # Test 6: Available symbols
        available_symbols = fetcher.get_available_symbols('yfinance')
        logger.info("db_available_symbols", extra={"count": len(available_symbols), "sample": ",".join(available_symbols[:5]) if available_symbols else ""})

        # Test 7: Comprehensive analysis
        analysis = fetcher.get_stock_analysis('MSFT', period='1y')
        data = analysis.get('data')
        s = analysis.get('statistics', {})
        if data is not None and not data.empty:
            logger.info(
                "analysis_summary",
                extra={
                    "symbol": "MSFT",
                    "rows": int(len(data)),
                    "current_price": float(s.get('current_price', 0)),
                    "total_return_pct": float(s.get('total_return', 0)),
                    "volatility": float(s.get('annualized_volatility', 0)),
                    "sharpe": float(s.get('sharpe_ratio', 0)),
                },
            )
        else:
            logger.error("analysis_failed", extra={"symbol": "MSFT"})

        logger.info("db_test_done")
    except Exception as e:
        logger.exception("db_test_exception", extra={"error": str(e)})
    finally:
        fetcher.close()
        logger.info("db_connection_closed")


def test_configuration():
    """Test the configuration system."""
    from config.settings import get_database_path, get_log_file_path, validate_api_keys

    validation = validate_api_keys()
    logger.info("config_api_validation", extra=validation)

    db_path = get_database_path()
    log_path = get_log_file_path()
    logger.info("config_paths", extra={"db_path": str(db_path), "log_path": str(log_path)})


if __name__ == "__main__":
    test_configuration()
    test_database_system()
