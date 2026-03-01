"""
ML feature engineering for commodity/asset price prediction.

Re-exports the full feature pipeline from ``core.data.ml_features`` and adds
Sortino-momentum features used by the signal modules.
"""

from core.data.ml_features import (
    calculate_downside_deviation_expanding,
    calculate_downside_deviation_rolling,
    calculate_rsi,
    check_class_imbalance,
    check_outliers,
    classify_volatility_regime,
    create_ml_features,
    create_ml_features_with_transparency,
    get_regime_trading_recommendation,
    prepare_features_for_model,
    prepare_ml_dataset,
)

__all__ = [
    "calculate_downside_deviation_expanding",
    "calculate_downside_deviation_rolling",
    "calculate_rsi",
    "check_class_imbalance",
    "check_outliers",
    "classify_volatility_regime",
    "create_ml_features",
    "create_ml_features_with_transparency",
    "get_regime_trading_recommendation",
    "prepare_features_for_model",
    "prepare_ml_dataset",
]
