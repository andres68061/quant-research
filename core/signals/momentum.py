"""
Sortino-momentum signal generation, regime detection, and analysis.

Implements three complementary approaches:
1. Grid search for optimal lookback/forecast horizon
2. Bootstrap significance testing
3. ML-based momentum prediction (logistic regression with TSCV)
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from core.backtest.portfolio import calculate_rolling_metrics

logger = logging.getLogger(__name__)


def calculate_sortino_slopes(
    rolling_sortino: pd.Series, x_days: int
) -> pd.Series:
    """Sortino slope over *x_days*."""
    return rolling_sortino.diff(x_days) / x_days


def analyze_momentum_grid_search(
    returns: pd.Series,
    sortino_window: int = 252,
    min_signals: int = 10,
) -> pd.DataFrame:
    """
    Grid search to find optimal (X, K) and compute hit rate Z.

    Args:
        returns: Daily returns series
        sortino_window: Window for rolling Sortino calculation
        min_signals: Minimum signals for a valid result

    Returns:
        DataFrame sorted by hit rate descending, with columns
        X (lookback), K (forecast), Z (hit_rate), CI_lower, CI_upper,
        Total_signals, Successful, Failed
    """
    rolling_metrics = calculate_rolling_metrics(returns, window=sortino_window)
    rolling_sortino = rolling_metrics["sortino_ratio"].dropna()

    lookback_windows = [5, 10, 15, 20, 30, 45, 60, 90]
    forecast_horizons = [5, 10, 15, 20, 30]
    baseline_window = 30

    results: List[Dict] = []

    for x in lookback_windows:
        for k in forecast_horizons:
            recent_slope = calculate_sortino_slopes(rolling_sortino, x)
            baseline_slope = calculate_sortino_slopes(
                rolling_sortino.shift(x), baseline_window
            )

            strong_momentum = (
                (recent_slope > baseline_slope)
                & recent_slope.notna()
                & baseline_slope.notna()
            )

            future_slope = calculate_sortino_slopes(
                rolling_sortino.shift(-k), k
            )
            continued = (future_slope > 0) & future_slope.notna()

            valid_indices = strong_momentum[strong_momentum].index

            if len(valid_indices) < min_signals:
                continue

            outcomes = continued.loc[valid_indices]
            hits = outcomes.sum()
            total = len(outcomes)
            hit_rate = (hits / total * 100) if total > 0 else np.nan

            if total > 0:
                se = np.sqrt(hit_rate / 100 * (1 - hit_rate / 100) / total)
                ci_lower = max(0, hit_rate - 1.96 * se * 100)
                ci_upper = min(100, hit_rate + 1.96 * se * 100)
            else:
                ci_lower = ci_upper = np.nan

            results.append(
                {
                    "X (lookback)": x,
                    "K (forecast)": k,
                    "Z (hit_rate)": hit_rate,
                    "CI_lower": ci_lower,
                    "CI_upper": ci_upper,
                    "Total_signals": total,
                    "Successful": hits,
                    "Failed": total - hits,
                }
            )

    df_results = pd.DataFrame(results)
    return df_results.sort_values("Z (hit_rate)", ascending=False)


def bootstrap_significance_test(
    returns: pd.Series,
    x: int,
    k: int,
    sortino_window: int = 252,
    n_bootstraps: int = 500,
) -> Dict:
    """
    Bootstrap test: is the observed hit rate significantly above random?

    Args:
        returns: Daily returns series
        x: Lookback window
        k: Forecast horizon
        sortino_window: Window for rolling Sortino
        n_bootstraps: Number of bootstrap samples

    Returns:
        Dictionary with actual_hit_rate, random_mean, p_value,
        significant (bool), n_signals, and bootstrap_dist
    """
    rolling_metrics = calculate_rolling_metrics(returns, window=sortino_window)
    rolling_sortino = rolling_metrics["sortino_ratio"].dropna()

    recent_slope = calculate_sortino_slopes(rolling_sortino, x)
    baseline_slope = calculate_sortino_slopes(rolling_sortino.shift(x), 30)
    strong_momentum = (
        (recent_slope > baseline_slope)
        & recent_slope.notna()
        & baseline_slope.notna()
    )
    future_slope = calculate_sortino_slopes(rolling_sortino.shift(-k), k)
    continued = (future_slope > 0) & future_slope.notna()

    valid_indices = strong_momentum[strong_momentum].index

    if len(valid_indices) < 10:
        return {
            "actual_hit_rate": np.nan,
            "random_mean": np.nan,
            "p_value": np.nan,
            "significant": False,
            "n_signals": len(valid_indices),
        }

    outcomes = continued.loc[valid_indices]
    actual_hit_rate = outcomes.mean() * 100

    rng = np.random.default_rng(42)
    bootstrap_hit_rates = [
        outcomes.sample(frac=1, replace=True, random_state=int(rng.integers(1e9))).mean()
        * 100
        for _ in range(n_bootstraps)
    ]

    random_mean = float(np.mean(bootstrap_hit_rates))
    p_value = float(
        np.mean(
            np.abs(np.array(bootstrap_hit_rates) - random_mean)
            >= np.abs(actual_hit_rate - random_mean)
        )
    )

    return {
        "actual_hit_rate": float(actual_hit_rate),
        "random_mean": random_mean,
        "random_std": float(np.std(bootstrap_hit_rates)),
        "p_value": p_value,
        "significant": p_value < 0.05,
        "n_signals": len(valid_indices),
        "bootstrap_dist": bootstrap_hit_rates,
    }


def prepare_ml_features(
    returns: pd.Series,
    sortino_window: int = 252,
    forecast_horizon: int = 10,
) -> pd.DataFrame:
    """
    Prepare features for ML-based momentum prediction.

    Features: Sortino level, Sharpe, volatility, slopes at 5/10/20/30d,
    vs-baseline spread.  Target: will Sortino rise in next K days?

    Args:
        returns: Daily returns series
        sortino_window: Window for rolling Sortino
        forecast_horizon: Forward-looking horizon for target

    Returns:
        Feature DataFrame with 'target' column
    """
    rolling = calculate_rolling_metrics(returns, window=sortino_window)

    features = pd.DataFrame(
        {
            "sortino": rolling["sortino_ratio"],
            "sharpe": rolling["sharpe_ratio"],
            "volatility": rolling["annualized_volatility"],
            "slope_5d": calculate_sortino_slopes(rolling["sortino_ratio"], 5),
            "slope_10d": calculate_sortino_slopes(rolling["sortino_ratio"], 10),
            "slope_20d": calculate_sortino_slopes(rolling["sortino_ratio"], 20),
            "slope_30d": calculate_sortino_slopes(rolling["sortino_ratio"], 30),
        }
    ).dropna()

    features["vs_baseline"] = calculate_sortino_slopes(
        rolling["sortino_ratio"], 20
    ) - calculate_sortino_slopes(rolling["sortino_ratio"].shift(20), 30)

    future_slope = calculate_sortino_slopes(
        rolling["sortino_ratio"].shift(-forecast_horizon), forecast_horizon
    )
    features["target"] = (future_slope > 0).astype(int)

    return features.dropna()


def analyze_ml_prediction(
    returns: pd.Series,
    sortino_window: int = 252,
    forecast_horizon: int = 10,
) -> Dict:
    """
    Logistic regression with time-series cross-validation.

    Args:
        returns: Daily returns series
        sortino_window: Window for rolling Sortino
        forecast_horizon: Forward-looking horizon

    Returns:
        Dictionary with mean_accuracy, std_accuracy, feature_importance, etc.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler

    features = prepare_ml_features(returns, sortino_window, forecast_horizon)

    if len(features) < 100:
        return {
            "error": "insufficient_data",
            "message": f"Only {len(features)} samples available, need at least 100",
        }

    x_df = features.drop("target", axis=1)
    y = features["target"]

    tscv = TimeSeriesSplit(n_splits=5)
    accuracies: List[float] = []
    all_predictions: List[int] = []
    all_actuals: List[int] = []
    feature_importances: List[np.ndarray] = []

    for train_idx, test_idx in tscv.split(x_df):
        x_train, x_test = x_df.iloc[train_idx], x_df.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        scaler = StandardScaler()
        x_train_s = scaler.fit_transform(x_train)
        x_test_s = scaler.transform(x_test)

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(x_train_s, y_train)

        y_pred = clf.predict(x_test_s)
        accuracies.append(accuracy_score(y_test, y_pred))
        all_predictions.extend(y_pred.tolist())
        all_actuals.extend(y_test.tolist())
        feature_importances.append(np.abs(clf.coef_[0]))

    mean_importance = np.mean(feature_importances, axis=0)
    importance_dict = dict(
        sorted(
            zip(x_df.columns, mean_importance),
            key=lambda t: t[1],
            reverse=True,
        )
    )

    return {
        "mean_accuracy": float(np.mean(accuracies)),
        "std_accuracy": float(np.std(accuracies)),
        "accuracies": accuracies,
        "feature_importance": importance_dict,
        "n_samples": len(features),
        "positive_class_pct": float(y.mean() * 100),
        "all_predictions": all_predictions,
        "all_actuals": all_actuals,
    }


def get_current_regime(
    returns: pd.Series,
    x: int,
    k: int,
    sortino_window: int = 252,
) -> Optional[Dict]:
    """
    Determine current Sortino momentum regime.

    Args:
        returns: Daily returns series
        x: Lookback window
        k: Forecast horizon (unused but kept for API symmetry)
        sortino_window: Rolling Sortino window

    Returns:
        Dictionary with current_sortino, slopes, strong_momentum flag,
        or None when insufficient data.
    """
    rolling_metrics = calculate_rolling_metrics(returns, window=sortino_window)
    rolling_sortino = rolling_metrics["sortino_ratio"].dropna()

    if len(rolling_sortino) < x + 30:
        return None

    recent_slope = calculate_sortino_slopes(rolling_sortino, x).iloc[-1]
    baseline_slope = calculate_sortino_slopes(
        rolling_sortino.shift(x), 30
    ).iloc[-1]

    current_sortino = rolling_sortino.iloc[-1]

    if pd.notna(recent_slope) and pd.notna(baseline_slope):
        return {
            "current_sortino": float(current_sortino),
            "recent_slope": float(recent_slope),
            "baseline_slope": float(baseline_slope),
            "strong_momentum": bool(recent_slope > baseline_slope),
            "slope_ratio": (
                float(recent_slope / baseline_slope)
                if baseline_slope != 0
                else np.nan
            ),
        }

    return None
