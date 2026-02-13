"""
Commodity ML Feature Engineering Module

Creates features for machine learning models predicting commodity price direction.

Design Decisions:
1. Uses LOG returns (time-additive, better for ML)
2. Mix of EXPANDING (long-term baseline) and ROLLING (regime detection) metrics
3. All features are LAGGED (no look-ahead bias)
4. Keeps all outliers (transparent - user decides later on capping)
5. Forward fills missing data (commodities don't gap randomly)

Author: Generated for Quant Analytics Platform
Date: February 2026
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        prices: Price series
        window: RSI window (default 14 days)
        
    Returns:
        RSI series (0-100)
    """
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_downside_deviation_expanding(returns: pd.Series) -> pd.Series:
    """
    Calculate expanding downside deviation (annualized).
    
    Uses only negative returns, expanding window from start.
    Provides long-term baseline for downside risk.
    
    Args:
        returns: Return series (should be log returns)
        
    Returns:
        Annualized expanding downside deviation
    """
    # Get only negative returns
    negative_returns = returns.copy()
    negative_returns[negative_returns > 0] = np.nan
    
    # Calculate expanding std of negative returns
    expanding_dd = negative_returns.expanding().std() * np.sqrt(252)
    
    return expanding_dd


def calculate_downside_deviation_rolling(returns: pd.Series, window: int = 21) -> pd.Series:
    """
    Calculate rolling downside deviation (annualized).
    
    Uses only negative returns, rolling window.
    Captures recent risk regime.
    
    Args:
        returns: Return series (should be log returns)
        window: Rolling window size (default 21 days)
        
    Returns:
        Annualized rolling downside deviation
    """
    # Get only negative returns
    negative_returns = returns.copy()
    negative_returns[negative_returns > 0] = np.nan
    
    # Calculate rolling std of negative returns
    rolling_dd = negative_returns.rolling(window).std() * np.sqrt(252)
    
    return rolling_dd


def create_ml_features(
    prices: pd.Series,
    symbol: str = "COMMODITY",
    include_seasonality: bool = True,
) -> pd.DataFrame:
    """
    Create comprehensive ML features for commodity price prediction.
    
    Features include:
    - Log returns (multiple timeframes)
    - Rolling volatility (captures regime changes)
    - Expanding downside deviation (long-term baseline)
    - Rolling downside deviation (recent regime)
    - RSI (momentum indicator)
    - Distance from moving average (mean reversion)
    - Seasonality (month, quarter)
    
    All features are LAGGED to prevent look-ahead bias.
    
    Args:
        prices: Price series (indexed by date)
        symbol: Commodity symbol (for logging)
        include_seasonality: Include month/quarter features
        
    Returns:
        DataFrame with features (NaN rows NOT dropped - handle in pipeline)
    """
    logger.info(f"Creating ML features for {symbol}, {len(prices)} data points")
    
    # Forward fill missing prices (commodities don't gap randomly)
    prices = prices.fillna(method='ffill')
    
    df = pd.DataFrame(index=prices.index)
    
    # ============================================================================
    # LOG RETURNS (time-additive, better for ML)
    # ============================================================================
    log_returns = np.log(prices / prices.shift(1))
    
    df['log_return_1d'] = log_returns.shift(1)  # Yesterday (LAGGED)
    df['log_return_5d'] = log_returns.rolling(5).sum().shift(1)  # Last week (LAGGED)
    df['log_return_21d'] = log_returns.rolling(21).sum().shift(1)  # Last month (LAGGED)
    df['log_return_63d'] = log_returns.rolling(63).sum().shift(1)  # Last quarter (LAGGED)
    
    # ============================================================================
    # VOLATILITY (ROLLING - captures regime changes)
    # ============================================================================
    df['vol_21d'] = log_returns.rolling(21).std().shift(1) * np.sqrt(252)  # LAGGED
    df['vol_63d'] = log_returns.rolling(63).std().shift(1) * np.sqrt(252)  # LAGGED
    
    # ============================================================================
    # DOWNSIDE DEVIATION (EXPANDING - long-term baseline)
    # ============================================================================
    df['downside_dev_expanding'] = calculate_downside_deviation_expanding(log_returns).shift(1)  # LAGGED
    
    # ============================================================================
    # DOWNSIDE DEVIATION (ROLLING - recent regime)
    # ============================================================================
    df['downside_dev_21d'] = calculate_downside_deviation_rolling(log_returns, 21).shift(1)  # LAGGED
    df['downside_dev_63d'] = calculate_downside_deviation_rolling(log_returns, 63).shift(1)  # LAGGED
    
    # ============================================================================
    # MOMENTUM (ROLLING - recent signals matter)
    # ============================================================================
    df['rsi_14d'] = calculate_rsi(prices, 14).shift(1)  # LAGGED
    
    # ============================================================================
    # MEAN REVERSION (ROLLING - current vs recent average)
    # ============================================================================
    ma_50 = prices.rolling(50).mean()
    df['distance_from_ma_50'] = ((prices - ma_50) / ma_50).shift(1)  # LAGGED
    
    ma_200 = prices.rolling(200).mean()
    df['distance_from_ma_200'] = ((prices - ma_200) / ma_200).shift(1)  # LAGGED
    
    # ============================================================================
    # SEASONALITY (if requested)
    # ============================================================================
    if include_seasonality:
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
    
    # ============================================================================
    # TARGET (NEXT DAY DIRECTION)
    # ============================================================================
    # Predict: Will price go up tomorrow? (1 = yes, 0 = no)
    next_return = log_returns.shift(-1)  # Tomorrow's return (FUTURE)
    df['target'] = (next_return > 0).astype(int)
    
    logger.info(f"Created {len(df.columns)} features for {symbol}")
    
    return df


def prepare_ml_dataset(
    prices_df: pd.DataFrame,
    min_training_days: int = 63,
) -> Dict[str, pd.DataFrame]:
    """
    Prepare ML dataset for all commodities in price DataFrame.
    
    Args:
        prices_df: DataFrame with commodity prices (date × symbols)
        min_training_days: Minimum days required for training (default 63 = ~3 months)
        
    Returns:
        Dictionary {symbol: feature_df}
    """
    datasets = {}
    
    for symbol in prices_df.columns:
        price_series = prices_df[symbol].dropna()
        
        # Check if sufficient data
        if len(price_series) < min_training_days + 5:  # +5 for test week
            logger.warning(f"Insufficient data for {symbol}: {len(price_series)} days (need {min_training_days + 5})")
            continue
        
        # Create features
        features_df = create_ml_features(price_series, symbol=symbol)
        
        # Drop rows where target is NaN (last row, can't predict future)
        features_df = features_df[features_df['target'].notna()]
        
        if len(features_df) >= min_training_days:
            datasets[symbol] = features_df
            logger.info(f"Prepared ML dataset for {symbol}: {len(features_df)} rows, {len(features_df.columns)} features")
    
    return datasets


def get_train_test_split_walk_forward(
    df: pd.DataFrame,
    initial_train_days: int = 63,
    test_days: int = 5,
) -> list:
    """
    Create walk-forward splits with expanding window.
    
    Args:
        df: Feature DataFrame with 'target' column
        initial_train_days: Initial training period (default 63 = ~3 months)
        test_days: Test period length (default 5 = 1 week)
        
    Returns:
        List of (train_indices, test_indices) tuples
    """
    splits = []
    
    total_rows = len(df)
    
    if total_rows < initial_train_days + test_days:
        logger.warning(f"Not enough data for walk-forward: {total_rows} rows (need {initial_train_days + test_days})")
        return splits
    
    # Start with initial training period
    train_end = initial_train_days
    
    while train_end + test_days <= total_rows:
        # Train: from start to train_end (EXPANDING)
        train_indices = df.index[:train_end]
        
        # Test: next test_days
        test_indices = df.index[train_end:train_end + test_days]
        
        splits.append((train_indices, test_indices))
        
        # Move forward by test_days
        train_end += test_days
    
    logger.info(f"Created {len(splits)} walk-forward splits")
    
    return splits


def prepare_features_for_model(
    df: pd.DataFrame,
    feature_columns: Optional[list] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and target for model training.
    
    Args:
        df: Feature DataFrame
        feature_columns: Specific columns to use (None = use all except target)
        
    Returns:
        (X, y) tuple
    """
    if feature_columns is None:
        # Use all columns except target
        feature_columns = [col for col in df.columns if col != 'target']
    
    X = df[feature_columns].copy()
    y = df['target'].copy()
    
    return X, y


def check_class_imbalance(y: pd.Series, threshold: float = 0.65) -> Dict:
    """
    Check for class imbalance in target variable.
    
    Args:
        y: Target series (0 or 1)
        threshold: Threshold for imbalance warning (default 0.65 = 65%)
        
    Returns:
        Dictionary with imbalance statistics
    """
    counts = y.value_counts()
    total = len(y)
    
    if len(counts) < 2:
        return {
            'is_imbalanced': True,
            'class_0_pct': 0.0 if 0 not in counts else 100.0,
            'class_1_pct': 0.0 if 1 not in counts else 100.0,
            'recommendation': 'Only one class present - cannot train classifier',
        }
    
    class_0_pct = counts.get(0, 0) / total
    class_1_pct = counts.get(1, 0) / total
    
    is_imbalanced = max(class_0_pct, class_1_pct) > threshold
    
    result = {
        'is_imbalanced': is_imbalanced,
        'class_0_pct': class_0_pct * 100,
        'class_1_pct': class_1_pct * 100,
        'class_0_count': counts.get(0, 0),
        'class_1_count': counts.get(1, 0),
    }
    
    if is_imbalanced:
        result['recommendation'] = f"Use class_weight='balanced' in model (imbalance > {threshold*100:.0f}%)"
    else:
        result['recommendation'] = "No class weighting needed (balanced)"
    
    return result


def check_outliers(returns: pd.Series, z_threshold: float = 3.0) -> Dict:
    """
    Check for outliers in return distribution.
    
    TRANSPARENT: Reports outliers but does NOT remove them.
    User decides later whether to cap/winsorize.
    
    Args:
        returns: Return series
        z_threshold: Z-score threshold for outlier detection (default 3.0)
        
    Returns:
        Dictionary with outlier statistics
    """
    clean_returns = returns.dropna()
    
    if len(clean_returns) < 10:
        return {'error': 'Insufficient data for outlier analysis'}
    
    mean = clean_returns.mean()
    std = clean_returns.std()
    
    z_scores = (clean_returns - mean) / std
    outliers = clean_returns[abs(z_scores) > z_threshold]
    
    result = {
        'total_returns': len(clean_returns),
        'outlier_count': len(outliers),
        'outlier_pct': len(outliers) / len(clean_returns) * 100,
        'min_return': clean_returns.min() * 100,
        'max_return': clean_returns.max() * 100,
        'mean_return': mean * 100,
        'std_return': std * 100,
        'max_z_score': abs(z_scores).max(),
        'outlier_dates': outliers.index.tolist() if len(outliers) > 0 else [],
    }
    
    # Interpretation
    if result['outlier_pct'] < 1.0:
        result['interpretation'] = "Low outlier count (<1%) - likely real price movements. Keep all data."
    elif result['outlier_pct'] < 3.0:
        result['interpretation'] = "Moderate outliers (1-3%) - review dates for data errors vs real events."
    else:
        result['interpretation'] = "High outlier count (>3%) - may indicate data quality issues. Consider winsorizing."
    
    result['action_taken'] = "NONE - All data kept. User can decide on capping/winsorizing later."
    
    return result


def create_ml_features_with_transparency(
    prices: pd.Series,
    symbol: str = "COMMODITY",
) -> Tuple[pd.DataFrame, Dict]:
    """
    Create ML features with full transparency about data prep decisions.
    
    Args:
        prices: Price series
        symbol: Commodity symbol
        
    Returns:
        (features_df, metadata_dict)
        
    metadata_dict contains:
        - Features created
        - Data cleaning steps
        - Outlier analysis
        - Missing data handling
        - Decisions made vs deferred
    """
    metadata = {
        'symbol': symbol,
        'total_data_points': len(prices),
        'date_range': f"{prices.index[0].date()} to {prices.index[-1].date()}",
    }
    
    # Check missing data
    missing_count = prices.isna().sum()
    metadata['missing_data'] = {
        'count': int(missing_count),
        'pct': float(missing_count / len(prices) * 100),
        'action': 'Forward fill (commodities trade continuously)',
    }
    
    # Forward fill
    prices_filled = prices.fillna(method='ffill')
    
    # Calculate log returns for outlier check
    log_returns = np.log(prices_filled / prices_filled.shift(1))
    
    # Check outliers (but don't remove)
    metadata['outliers'] = check_outliers(log_returns)
    
    # Create features
    features_df = create_ml_features(prices_filled, symbol=symbol)
    
    # Drop rows with NaN target (last row)
    features_df = features_df[features_df['target'].notna()]
    
    # Count NaN in features (before dropping)
    nan_counts = features_df.isna().sum()
    features_with_nan = nan_counts[nan_counts > 0]
    
    if len(features_with_nan) > 0:
        metadata['feature_nan'] = features_with_nan.to_dict()
    else:
        metadata['feature_nan'] = {}
    
    # Drop rows with ANY NaN in features
    initial_rows = len(features_df)
    features_df = features_df.dropna()
    rows_dropped = initial_rows - len(features_df)
    
    metadata['rows_dropped_nan'] = int(rows_dropped)
    metadata['final_rows'] = int(len(features_df))
    
    # Check class imbalance
    metadata['class_distribution'] = check_class_imbalance(features_df['target'])
    
    # Document feature types
    metadata['features'] = {
        'log_returns': ['log_return_1d', 'log_return_5d', 'log_return_21d', 'log_return_63d'],
        'rolling_volatility': ['vol_21d', 'vol_63d'],
        'expanding_metrics': ['downside_dev_expanding'],
        'rolling_risk': ['downside_dev_21d', 'downside_dev_63d'],
        'momentum': ['rsi_14d'],
        'mean_reversion': ['distance_from_ma_50', 'distance_from_ma_200'],
        'seasonality': ['month', 'quarter'],
    }
    
    metadata['total_features'] = len([col for col in features_df.columns if col != 'target'])
    
    # Transparency: What we did and didn't do
    metadata['transparency'] = {
        'data_prep_completed': [
            'Forward filled missing data',
            'Created log returns (not arithmetic)',
            'Mixed expanding and rolling windows',
            'All features lagged (no look-ahead)',
            'Dropped rows with NaN in features',
        ],
        'data_prep_NOT_done': [
            'NO outlier removal/capping (kept all data)',
            'NO scaling/normalization (for XGBoost)',
            'NO PCA or dimensionality reduction',
            'NO Box-Cox transforms',
            'NO synthetic data generation (SMOTE)',
        ],
        'to_be_decided': [
            'Outlier treatment: Cap vs Winsorize vs Keep (user decides)',
            'Class imbalance handling: Use class_weight if needed',
            'Hyperparameter tuning: After baseline evaluation',
            'Additional features: Ratio features, cross-asset features',
        ],
    }
    
    logger.info(f"Feature engineering complete for {symbol}: {metadata['final_rows']} rows, {metadata['total_features']} features")
    
    return features_df, metadata


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("Commodity ML Feature Engineering - Example")
    print("=" * 80)
    
    # Load sample data
    from pathlib import Path
    
    data_file = Path(__file__).parents[1] / "data" / "commodities" / "prices.parquet"
    
    if data_file.exists():
        df = pd.read_parquet(data_file)
        
        # Example: Gold
        if 'GLD' in df.columns:
            print("\n📊 Creating features for Gold (GLD)...\n")
            
            features, metadata = create_ml_features_with_transparency(
                df['GLD'],
                symbol='GLD'
            )
            
            print(f"✅ Features created: {metadata['total_features']}")
            print(f"✅ Final rows: {metadata['final_rows']}")
            print(f"\n📋 Feature Groups:")
            for group, features_list in metadata['features'].items():
                print(f"  {group}: {len(features_list)} features")
            
            print(f"\n⚖️ Class Distribution:")
            dist = metadata['class_distribution']
            print(f"  Down (0): {dist['class_0_pct']:.1f}%")
            print(f"  Up (1): {dist['class_1_pct']:.1f}%")
            print(f"  {dist['recommendation']}")
            
            print(f"\n🔍 Outlier Analysis:")
            outliers = metadata['outliers']
            print(f"  Total returns: {outliers['total_returns']}")
            print(f"  Outliers (>3σ): {outliers['outlier_count']} ({outliers['outlier_pct']:.2f}%)")
            print(f"  Min return: {outliers['min_return']:.2f}%")
            print(f"  Max return: {outliers['max_return']:.2f}%")
            print(f"  Interpretation: {outliers['interpretation']}")
            print(f"  Action taken: {outliers['action_taken']}")
            
            print("\n✅ TRANSPARENCY:")
            print("\n  What we DID:")
            for item in metadata['transparency']['data_prep_completed']:
                print(f"    ✓ {item}")
            
            print("\n  What we DID NOT do:")
            for item in metadata['transparency']['data_prep_NOT_done']:
                print(f"    ✗ {item}")
            
            print("\n  To be DECIDED:")
            for item in metadata['transparency']['to_be_decided']:
                print(f"    ? {item}")
            
            print("\n" + "=" * 80)
            print(f"Sample features (first 5 rows):")
            print("=" * 80)
            print(features.head())
            
        else:
            print("❌ GLD not found in data")
    else:
        print(f"❌ Data file not found: {data_file}")
