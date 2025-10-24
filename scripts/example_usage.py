#!/usr/bin/env python3
"""
Example script demonstrating basic quantitative analysis functionality.
This shows how to use the packages we installed in the quant environment.
"""

import logging
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger("example_usage")


def fetch_stock_data(symbol='AAPL', period='1y'):
    """Fetch stock data using yfinance."""
    logger.info("fetching_data", extra={"symbol": symbol, "period": period})

    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)

        logger.info(
            "fetched_data",
            extra={
                "rows": int(len(data)),
                "start": str(data.index[0].date()) if len(data) > 0 else None,
                "end": str(data.index[-1].date()) if len(data) > 0 else None,
            },
        )
        return data
    except Exception as e:
        logger.exception("fetch_failed", extra={"symbol": symbol, "error": str(e)})
        return None


def calculate_returns(data):
    """Calculate daily returns from price data."""
    if data is None or len(data) == 0:
        return None

    returns = data['Close'].pct_change().dropna()

    logger.info(
        "returns_summary",
        extra={
            "rows": int(len(returns)),
            "mean": float(returns.mean()),
            "std": float(returns.std()),
            "min": float(returns.min()),
            "max": float(returns.max()),
        },
    )

    return returns


def create_visualizations(data, returns):
    """Create basic visualizations of the data."""
    if data is None or returns is None:
        return

    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Stock Analysis Dashboard', fontsize=16, fontweight='bold')

    # 1. Price chart
    axes[0, 0].plot(data.index, data['Close'], linewidth=2, color='blue')
    axes[0, 0].set_title('Stock Price Over Time')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Price ($)')
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Volume chart
    axes[0, 1].bar(data.index, data['Volume'], alpha=0.7, color='green')
    axes[0, 1].set_title('Trading Volume')
    axes[0, 1].set_xlabel('Date')
    axes[0, 1].set_ylabel('Volume')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Returns distribution
    axes[1, 0].hist(returns, bins=50, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 0].set_title('Daily Returns Distribution')
    axes[1, 0].set_xlabel('Daily Return')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Returns over time
    axes[1, 1].plot(returns.index, returns, alpha=0.7, color='red')
    axes[1, 1].set_title('Daily Returns Over Time')
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel('Daily Return')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    filename = f'stock_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    logger.info("visual_saved", extra={"file": filename})

    plt.show()


def basic_statistics(data, returns):
    """Calculate and display basic statistics."""
    if data is None or returns is None:
        return

    stats = {
        "current_price": float(data['Close'].iloc[-1]),
        "highest_price": float(data['High'].max()),
        "lowest_price": float(data['Low'].min()),
        "price_range": float(data['High'].max() - data['Low'].min()),
        "total_return_pct": float((data['Close'].iloc[-1] / data['Close'].iloc[0] - 1) * 100),
        "ann_vol_pct": float(returns.std() * np.sqrt(252) * 100),
        "sharpe": float(returns.mean() / returns.std() * np.sqrt(252)),
        "avg_volume": float(data['Volume'].mean()),
        "max_volume": float(data['Volume'].max()),
        "min_volume": float(data['Volume'].min()),
    }
    logger.info("basic_stats", extra=stats)


def main():
    logger.info("example_start")

    symbol = 'AAPL'
    data = fetch_stock_data(symbol, period='1y')

    if data is not None:
        returns = calculate_returns(data)
        create_visualizations(data, returns)
        basic_statistics(data, returns)
        logger.info("example_done", extra={"symbol": symbol})
    else:
        logger.error("no_data", extra={"symbol": symbol})


if __name__ == "__main__":
    main()
