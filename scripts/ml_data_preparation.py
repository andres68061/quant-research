#!/usr/bin/env python3
"""
Machine Learning Data Preparation Example

This script demonstrates how to use the database system to prepare data
for machine learning models, including feature engineering and dataset creation.
"""

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from config.settings import FINNHUB_API_KEY, TECHNICAL_INDICATORS
from data.enhanced_stock_data import EnhancedStockDataFetcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
logger = logging.getLogger("ml_data_preparation")


def create_ml_dataset(symbols: list, period: str = "2y", features: list = None):
    """
    Create a machine learning dataset with technical indicators.
    
    Args:
        symbols (list): List of stock symbols
        period (str): Data period
        features (list): List of features to include
        
    Returns:
        pd.DataFrame: ML-ready dataset
    """
    logger.info("ml_dataset_start", extra={"num_symbols": len(symbols), "period": period})

    if features is None:
        features = [
            'Close', 'Volume', 'Daily_Return', 'Volatility_30d',
            'MA_20', 'MA_50', 'RSI', 'MACD', 'MACD_Signal',
            'BB_Upper', 'BB_Lower', 'BB_Position'
        ]

    fetcher = EnhancedStockDataFetcher(FINNHUB_API_KEY)

    try:
        all_data = {}
        for symbol in symbols:
            logger.info("fetch_symbol", extra={"symbol": symbol})
            data = fetcher.get_stock_data(symbol, period=period)
            if data is not None:
                data = add_technical_indicators(data)
                all_data[symbol] = data
                logger.info("fetch_symbol_done", extra={"symbol": symbol, "rows": int(len(data))})
            else:
                logger.error("fetch_symbol_fail", extra={"symbol": symbol})

        combined_data = combine_stock_data(all_data, features)
        if combined_data.empty:
            logger.error("dataset_empty")
            return combined_data

        logger.info(
            "ml_dataset_done",
            extra={
                "rows": int(len(combined_data)),
                "cols": int(len(combined_data.columns)),
                "start": str(combined_data.index[0].date()),
                "end": str(combined_data.index[-1].date()),
            },
        )
        return combined_data
    finally:
        fetcher.close()


def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to the stock data.
    
    Args:
        data (pd.DataFrame): Stock data
        
    Returns:
        pd.DataFrame: Data with technical indicators
    """
    df = data.copy()

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']

    bb_period = TECHNICAL_INDICATORS['bollinger_period']
    bb_std = TECHNICAL_INDICATORS['bollinger_std']
    bb_middle = df['Close'].rolling(window=bb_period).mean()
    bb_std_dev = df['Close'].rolling(window=bb_period).std()
    df['BB_Upper'] = bb_middle + (bb_std_dev * bb_std)
    df['BB_Lower'] = bb_middle - (bb_std_dev * bb_std)
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])

    df['Price_Momentum_5'] = df['Close'].pct_change(5)
    df['Price_Momentum_10'] = df['Close'].pct_change(10)
    df['Price_Momentum_20'] = df['Close'].pct_change(20)

    if 'Volume' in df.columns:
        df['Volume_MA_5'] = df['Volume'].rolling(window=5).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA_5']
    else:
        df['Volume_MA_5'] = np.nan
        df['Volume_Ratio'] = np.nan

    if 'High' in df.columns and 'Low' in df.columns:
        df['ATR'] = calculate_atr(df)
    else:
        df['ATR'] = np.nan
    df['Volatility_5d'] = df['Daily_Return'].rolling(window=5).std() * np.sqrt(252)
    df['Volatility_10d'] = df['Daily_Return'].rolling(window=10).std() * np.sqrt(252)

    return df


def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    high = data['High']
    low = data['Low']
    close = data['Close']

    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    return atr


def combine_stock_data(all_data: dict, features: list) -> pd.DataFrame:
    """
    Combine data from multiple stocks into a single dataset.
    
    Args:
        all_data (dict): Dictionary with symbol as key and data as value
        features (list): List of features to include
        
    Returns:
        pd.DataFrame: Combined dataset
    """
    combined_data = []

    for symbol, data in all_data.items():
        feature_data = data[features].copy()
        feature_data['Symbol'] = symbol
        feature_data['Target_Return'] = data['Daily_Return'].shift(-1)
        feature_data['Target_Class'] = (data['Daily_Return'].shift(-1) > 0).astype(int)
        combined_data.append(feature_data)

    if combined_data:
        result = pd.concat(combined_data, axis=0)
        if result.index.tz is not None:
            result.index = result.index.tz_localize(None)
        result.sort_index(inplace=True)
        result.dropna(inplace=True)
        return result
    else:
        return pd.DataFrame()


def create_time_series_features(data: pd.DataFrame, lookback_periods: list = [1, 3, 5, 10]):
    """
    Create time series features with different lookback periods.
    
    Args:
        data (pd.DataFrame): Stock data
        lookback_periods (list): List of lookback periods
        
    Returns:
        pd.DataFrame: Data with time series features
    """
    df = data.copy()

    for period in lookback_periods:
        df[f'Return_Lag_{period}'] = df['Daily_Return'].shift(period)
        df[f'Return_Mean_{period}'] = df['Daily_Return'].rolling(window=period).mean()
        df[f'Return_Std_{period}'] = df['Daily_Return'].rolling(window=period).std()
        df[f'Return_Max_{period}'] = df['Daily_Return'].rolling(window=period).max()
        df[f'Return_Min_{period}'] = df['Daily_Return'].rolling(window=period).min()
        df[f'Price_Lag_{period}'] = df['Close'].shift(period)
        df[f'Price_Change_{period}'] = df['Close'].pct_change(period)
        df[f'Volume_Lag_{period}'] = df['Volume'].shift(period)
        df[f'Volume_Change_{period}'] = df['Volume'].pct_change(period)

    return df


def prepare_ml_features(data: pd.DataFrame, target_col: str = 'Target_Return'):
    """
    Prepare features for machine learning.
    
    Args:
        data (pd.DataFrame): Raw data
        target_col (str): Target column name
        
    Returns:
        tuple: (X, y) features and target
    """
    feature_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in feature_cols if col not in [target_col, 'Target_Class']]

    X = data[feature_cols].copy()
    y = data[target_col].copy()

    valid_idx = X.notna().all(axis=1) & y.notna()
    X = X[valid_idx]
    y = y[valid_idx]

    logger.info(
        "ml_features_prepared",
        extra={
            "features": int(X.shape[1]),
            "samples": int(X.shape[0]),
            "target_min": float(y.min()),
            "target_max": float(y.max()),
        },
    )

    return X, y


def main():
    logger.info("ml_prep_start")

    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    ml_data = create_ml_dataset(symbols, period="2y")

    if ml_data.empty:
        logger.error("ml_dataset_failed")
        return

    logger.info("timeseries_features_start")
    ml_data = create_time_series_features(ml_data)

    logger.info("ml_features_prepare_start")
    X, y = prepare_ml_features(ml_data)

    correlations = X.corrwith(y).abs().sort_values(ascending=False)
    top_features = correlations.head(10)
    logger.info("top_feature_corr", extra={"features": top_features.index.tolist(), "values": [float(v) for v in top_features.values]})

    logger.info("ml_viz_start")
    create_ml_visualizations(ml_data, X, y)

    output_path = Path("data/ml") / "stock_ml_dataset.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ml_data.to_csv(output_path)
    logger.info("ml_dataset_saved", extra={"path": str(output_path), "rows": int(len(ml_data)), "cols": int(len(X.columns))})

    logger.info("ml_prep_done", extra={"symbols": len(symbols)})


def create_ml_visualizations(data: pd.DataFrame, X: pd.DataFrame, y: pd.Series):
    """Create visualizations for ML data."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Machine Learning Data Analysis', fontsize=16, fontweight='bold')

    axes[0, 0].hist(y, bins=50, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Target Return Distribution')
    axes[0, 0].set_xlabel('Daily Return')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)

    corr_matrix = X.corr()
    sns.heatmap(corr_matrix.iloc[:10, :10], annot=True, cmap='coolwarm', center=0, ax=axes[0, 1], cbar_kws={'shrink': 0.8})
    axes[0, 1].set_title('Feature Correlations (Top 10)')

    sample_data = data[data['Symbol'] == 'AAPL'].tail(100)
    axes[1, 0].plot(sample_data.index, sample_data['Daily_Return'], alpha=0.7)
    axes[1, 0].set_title('Sample Time Series (AAPL Returns)')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Daily Return')
    axes[1, 0].grid(True, alpha=0.3)

    correlations = X.corrwith(y).abs().sort_values(ascending=True).tail(10)
    axes[1, 1].barh(range(len(correlations)), correlations.values)
    axes[1, 1].set_yticks(range(len(correlations)))
    axes[1, 1].set_yticklabels(correlations.index, fontsize=8)
    axes[1, 1].set_title('Top 10 Feature Correlations with Target')
    axes[1, 1].set_xlabel('Absolute Correlation')

    plt.tight_layout()

    output_path = Path("results") / "ml_data_analysis.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info("ml_viz_saved", extra={"path": str(output_path)})

    plt.show()


if __name__ == "__main__":
    main()
