#!/usr/bin/env python3
"""
Debug script to test stock data fetching.
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from data.stock_data import StockDataFetcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger("debug_test")


def test_basic_fetch():
    """Test basic stock data fetching."""
    logger.info("debug_fetch_start")

    fetcher = StockDataFetcher()

    logger.info("fetch_with_period", extra={"symbol": "AAPL", "period": "6mo"})
    data = fetcher.fetch_stock_data('AAPL', period='6mo')

    if data is not None:
        logger.info(
            "fetch_success",
            extra={
                "rows": int(len(data)),
                "start": str(data.index[0].date()),
                "end": str(data.index[-1].date()),
                "last_close": float(data['Close'].iloc[-1]),
            },
        )
    else:
        logger.error("fetch_failed", extra={"symbol": "AAPL"})


if __name__ == "__main__":
    test_basic_fetch()
