"""
Commodity Price Direction Prediction Models

Implements XGBoost and LSTM models for predicting commodity price direction.
Uses walk-forward validation with expanding window.

Models:
1. XGBoost Classifier (no scaling needed)
2. LSTM Classifier (StandardScaler applied)

Validation:
- Walk-forward with expanding window
- Initial training: 3 months (63 days)
- Test period: 1 week (5 days)
- Growing training set over time

Author: Generated for Quant Analytics Platform
Date: February 2026
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """
    Walk-forward validation with expanding window for time series.
    
    Design:
    - Expanding window (not rolling) - uses all historical data
    - Fixed test period (5 days default)
    - No data leakage (train on past, test on future)
    """
    
    def __init__(
        self,
        initial_train_days: int = 63,
        test_days: int = 5,
    ):
        """
        Initialize walk-forward validator.
        
        Args:
            initial_train_days: Initial training period (default 63 = ~3 months)
            test_days: Test period length (default 5 = 1 week)
        """
        self.initial_train_days = initial_train_days
        self.test_days = test_days
        
    def create_splits(self, df: pd.DataFrame) -> list:
        """
        Create walk-forward splits.
        
        Args:
            df: Feature DataFrame
            
        Returns:
            List of (train_indices, test_indices) tuples
        """
        splits = []
        total_rows = len(df)
        
        if total_rows < self.initial_train_days + self.test_days:
            logger.warning(
                f"Insufficient data: {total_rows} rows (need {self.initial_train_days + self.test_days})"
            )
            return splits
        
        train_end = self.initial_train_days
        
        while train_end + self.test_days <= total_rows:
            # Expanding window: always from start
            train_indices = df.index[:train_end]
            test_indices = df.index[train_end : train_end + self.test_days]
            
            splits.append((train_indices, test_indices))
            
            # Move forward by test period
            train_end += self.test_days
        
        logger.info(f"Created {len(splits)} walk-forward splits")
        return splits


class XGBoostDirectionModel:
    """
    XGBoost model for predicting commodity price direction.
    
    No scaling required (tree-based model).
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        use_class_weight: bool = False,
        random_state: int = 42,
    ):
        """
        Initialize XGBoost model.
        
        Args:
            n_estimators: Number of boosting rounds (default 100)
            max_depth: Maximum tree depth (default 3 - shallow to prevent overfitting)
            learning_rate: Learning rate (default 0.1)
            use_class_weight: Use balanced class weights if imbalanced
            random_state: Random seed for reproducibility
        """
        try:
            from xgboost import XGBClassifier
            
            params = {
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'learning_rate': learning_rate,
                'random_state': random_state,
                'eval_metric': 'logloss',
                'use_label_encoder': False,
            }
            
            if use_class_weight:
                params['scale_pos_weight'] = 1.0  # Will be set dynamically
            
            self.model = XGBClassifier(**params)
            self.feature_importance_ = None
            
        except ImportError:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")
    
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train the model."""
        # Adjust class weight if needed
        if hasattr(self.model, 'scale_pos_weight'):
            neg_count = (y == 0).sum()
            pos_count = (y == 1).sum()
            if pos_count > 0:
                self.model.scale_pos_weight = neg_count / pos_count
        
        self.model.fit(X, y)
        self.feature_importance_ = pd.Series(
            self.model.feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        return self.model.predict_proba(X)
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance."""
        return self.feature_importance_


class LSTMDirectionModel:
    """
    LSTM model for predicting commodity price direction.
    
    Requires StandardScaler for features.
    """
    
    def __init__(
        self,
        sequence_length: int = 60,
        hidden_units: int = 64,
        dropout_rate: float = 0.3,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32,
        random_state: int = 42,
    ):
        """
        Initialize LSTM model.
        
        Args:
            sequence_length: Number of time steps to look back (default 60)
            hidden_units: LSTM hidden units (default 64)
            dropout_rate: Dropout rate (default 0.3)
            learning_rate: Learning rate (default 0.001)
            epochs: Training epochs (default 50)
            batch_size: Batch size (default 32)
            random_state: Random seed
        """
        try:
            import tensorflow as tf
            from tensorflow import keras
            
            self.sequence_length = sequence_length
            self.hidden_units = hidden_units
            self.dropout_rate = dropout_rate
            self.learning_rate = learning_rate
            self.epochs = epochs
            self.batch_size = batch_size
            
            # Set random seed
            np.random.seed(random_state)
            tf.random.set_seed(random_state)
            
            self.model = None
            self.scaler = StandardScaler()
            self.history = None
            
        except ImportError:
            raise ImportError("TensorFlow not installed. Run: pip install tensorflow")
    
    def _build_model(self, n_features: int):
        """Build LSTM architecture."""
        from tensorflow import keras
        from tensorflow.keras import layers
        
        model = keras.Sequential([
            layers.LSTM(
                self.hidden_units,
                return_sequences=True,
                input_shape=(self.sequence_length, n_features)
            ),
            layers.Dropout(self.dropout_rate),
            layers.LSTM(self.hidden_units, return_sequences=False),
            layers.Dropout(self.dropout_rate),
            layers.Dense(32, activation='relu'),
            layers.Dropout(self.dropout_rate),
            layers.Dense(1, activation='sigmoid'),
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc')],
        )
        
        return model
    
    def _create_sequences(self, X_scaled: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM.
        
        Args:
            X_scaled: Scaled features (n_samples, n_features)
            y: Target labels (n_samples,)
            
        Returns:
            (X_sequences, y_sequences) for LSTM
        """
        X_seq = []
        y_seq = []
        
        for i in range(self.sequence_length, len(X_scaled)):
            X_seq.append(X_scaled[i - self.sequence_length : i])
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: int = 0):
        """
        Train LSTM model.
        
        Args:
            X: Feature DataFrame
            y: Target series
            verbose: Training verbosity (0=silent, 1=progress bar, 2=one line per epoch)
        """
        # Scale features (fit on training data)
        X_scaled = self.scaler.fit_transform(X)
        y_values = y.values
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y_values)
        
        if len(X_seq) < 10:
            raise ValueError(f"Not enough sequences after creating {self.sequence_length}-step sequences: {len(X_seq)}")
        
        # Build model
        n_features = X.shape[1]
        self.model = self._build_model(n_features)
        
        # Train with early stopping
        from tensorflow.keras.callbacks import EarlyStopping
        
        early_stop = EarlyStopping(
            monitor='loss',
            patience=10,
            restore_best_weights=True
        )
        
        self.history = self.model.fit(
            X_seq,
            y_seq,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=verbose,
            callbacks=[early_stop],
            validation_split=0.2,  # Use 20% of training for validation
        )
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        # Scale using fitted scaler
        X_scaled = self.scaler.transform(X)
        
        # Create sequences
        X_seq, _ = self._create_sequences(X_scaled, np.zeros(len(X)))  # Dummy y
        
        if len(X_seq) == 0:
            return np.array([])
        
        # Predict probabilities
        proba = self.model.predict(X_seq, verbose=0)
        predictions = (proba > 0.5).astype(int).flatten()
        
        return predictions
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        X_scaled = self.scaler.transform(X)
        X_seq, _ = self._create_sequences(X_scaled, np.zeros(len(X)))
        
        if len(X_seq) == 0:
            return np.array([])
        
        proba = self.model.predict(X_seq, verbose=0)
        
        # Return in sklearn format: [prob_class_0, prob_class_1]
        return np.column_stack([1 - proba, proba])


def evaluate_model_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional, for AUC)
        
    Returns:
        Dictionary with metrics
    """
    metrics = {}
    
    # Basic accuracy
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    metrics['precision'] = precision
    metrics['recall'] = recall
    metrics['f1_score'] = f1
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['true_positives'] = int(tp)
    
    # ROC AUC (if probabilities provided)
    if y_proba is not None:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
        except:
            metrics['roc_auc'] = None
    
    # Direction accuracy (what traders care about)
    metrics['direction_accuracy'] = metrics['accuracy']
    
    return metrics


def run_walk_forward_validation(
    features_df: pd.DataFrame,
    model_type: str = "xgboost",
    initial_train_days: int = 63,
    test_days: int = 5,
    model_params: Optional[Dict] = None,
    verbose: bool = True,
) -> Dict:
    """
    Run walk-forward validation with expanding window.
    
    Args:
        features_df: DataFrame with features and 'target' column
        model_type: 'xgboost' or 'lstm'
        initial_train_days: Initial training period (default 63 = 3 months)
        test_days: Test period (default 5 = 1 week)
        model_params: Optional model parameters
        verbose: Print progress
        
    Returns:
        Dictionary with results, predictions, and metrics
    """
    logger.info(f"Starting walk-forward validation: {model_type}, {len(features_df)} rows")
    
    if model_params is None:
        model_params = {}
    
    # Separate features and target
    feature_cols = [col for col in features_df.columns if col != 'target']
    X = features_df[feature_cols]
    y = features_df['target']
    
    # Create walk-forward splits
    validator = WalkForwardValidator(initial_train_days, test_days)
    splits = validator.create_splits(features_df)
    
    if len(splits) == 0:
        return {
            'error': 'Insufficient data for walk-forward validation',
            'required_rows': initial_train_days + test_days,
            'available_rows': len(features_df),
        }
    
    # Store results
    all_predictions = []
    all_true = []
    all_probabilities = []
    split_metrics = []
    
    # Run walk-forward
    for i, (train_idx, test_idx) in enumerate(splits):
        if verbose:
            print(f"Split {i+1}/{len(splits)}: Train={len(train_idx)}, Test={len(test_idx)}")
        
        # Get train/test data
        X_train = X.loc[train_idx]
        y_train = y.loc[train_idx]
        X_test = X.loc[test_idx]
        y_test = y.loc[test_idx]
        
        # Train model
        if model_type == "xgboost":
            # Check class imbalance
            class_counts = y_train.value_counts()
            use_class_weight = False
            if len(class_counts) == 2:
                imbalance_ratio = max(class_counts) / len(y_train)
                use_class_weight = imbalance_ratio > 0.65
            
            model = XGBoostDirectionModel(
                use_class_weight=use_class_weight,
                **model_params
            )
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            
        elif model_type == "lstm":
            model = LSTMDirectionModel(**model_params)
            
            # LSTM fit handles scaling internally
            try:
                model.fit(X_train, y_train, verbose=0)
                
                # Predict
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)
                
                # Handle sequence length offset
                # LSTM predictions are shorter due to sequence requirement
                if len(y_pred) < len(y_test):
                    # Align predictions with test indices
                    y_test = y_test.iloc[-len(y_pred):]
            
            except Exception as e:
                logger.warning(f"LSTM training failed on split {i+1}: {str(e)}")
                continue
        
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        # Store predictions
        all_predictions.extend(y_pred)
        all_true.extend(y_test.values)
        if len(y_proba) > 0:
            all_probabilities.extend(y_proba[:, 1])  # Probability of class 1
        
        # Calculate split metrics
        split_acc = accuracy_score(y_test, y_pred)
        split_metrics.append({
            'split': i + 1,
            'train_size': len(train_idx),
            'test_size': len(y_test),
            'accuracy': split_acc,
            'train_start': train_idx[0],
            'train_end': train_idx[-1],
            'test_start': test_idx[0],
            'test_end': test_idx[-1],
        })
    
    # Overall metrics
    all_predictions = np.array(all_predictions)
    all_true = np.array(all_true)
    all_probabilities = np.array(all_probabilities) if all_probabilities else None
    
    overall_metrics = evaluate_model_performance(
        all_true,
        all_predictions,
        all_probabilities.reshape(-1, 1) if all_probabilities is not None else None
    )
    
    # Feature importance (last model)
    feature_importance = None
    if model_type == "xgboost":
        feature_importance = model.get_feature_importance()
    
    results = {
        'model_type': model_type,
        'n_splits': len(splits),
        'predictions': all_predictions,
        'true_labels': all_true,
        'probabilities': all_probabilities,
        'overall_metrics': overall_metrics,
        'split_metrics': split_metrics,
        'feature_importance': feature_importance,
        'model_params': model_params,
        'validation_params': {
            'initial_train_days': initial_train_days,
            'test_days': test_days,
            'window_type': 'expanding',
        }
    }
    
    logger.info(f"Walk-forward complete: {model_type}, accuracy={overall_metrics['accuracy']:.4f}")
    
    return results


def compare_models(
    features_df: pd.DataFrame,
    initial_train_days: int = 63,
    test_days: int = 5,
    xgb_params: Optional[Dict] = None,
    lstm_params: Optional[Dict] = None,
    verbose: bool = True,
) -> Dict:
    """
    Compare XGBoost vs LSTM on the same data.
    
    Args:
        features_df: DataFrame with features and target
        initial_train_days: Initial training period
        test_days: Test period
        xgb_params: XGBoost parameters
        lstm_params: LSTM parameters
        verbose: Print progress
        
    Returns:
        Dictionary with results for both models
    """
    if verbose:
        print("=" * 80)
        print("MODEL COMPARISON: XGBoost vs LSTM")
        print("=" * 80)
    
    # Run XGBoost
    if verbose:
        print("\n🌳 Training XGBoost...\n")
    
    xgb_results = run_walk_forward_validation(
        features_df,
        model_type="xgboost",
        initial_train_days=initial_train_days,
        test_days=test_days,
        model_params=xgb_params or {},
        verbose=verbose,
    )
    
    # Run LSTM
    if verbose:
        print("\n🧠 Training LSTM...\n")
    
    lstm_results = run_walk_forward_validation(
        features_df,
        model_type="lstm",
        initial_train_days=initial_train_days,
        test_days=test_days,
        model_params=lstm_params or {},
        verbose=verbose,
    )
    
    # Comparison
    comparison = {
        'xgboost': xgb_results,
        'lstm': lstm_results,
        'winner': None,
    }
    
    # Determine winner
    xgb_acc = xgb_results.get('overall_metrics', {}).get('accuracy', 0)
    lstm_acc = lstm_results.get('overall_metrics', {}).get('accuracy', 0)
    
    if xgb_acc > lstm_acc:
        comparison['winner'] = 'xgboost'
        comparison['margin'] = (xgb_acc - lstm_acc) * 100
    elif lstm_acc > xgb_acc:
        comparison['winner'] = 'lstm'
        comparison['margin'] = (lstm_acc - xgb_acc) * 100
    else:
        comparison['winner'] = 'tie'
        comparison['margin'] = 0
    
    if verbose:
        print("\n" + "=" * 80)
        print("📊 COMPARISON RESULTS")
        print("=" * 80)
        print(f"XGBoost Accuracy: {xgb_acc:.4f}")
        print(f"LSTM Accuracy:    {lstm_acc:.4f}")
        print(f"Winner: {comparison['winner'].upper()} (margin: {comparison['margin']:.2f}%)")
        print("=" * 80)
    
    return comparison


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 80)
    print("Commodity ML Models - Example")
    print("=" * 80)
    
    from pathlib import Path
    import sys
    
    ROOT = Path(__file__).parents[1]
    sys.path.insert(0, str(ROOT))
    
    from data.ml_features import create_ml_features_with_transparency
    
    # Load sample data
    data_file = ROOT / "data" / "commodities" / "prices.parquet"
    
    if data_file.exists():
        df = pd.read_parquet(data_file)
        
        if 'GLD' in df.columns:
            print("\n📊 Preparing features for Gold (GLD)...\n")
            
            features, metadata = create_ml_features_with_transparency(
                df['GLD'],
                symbol='GLD'
            )
            
            print(f"✅ Dataset ready: {len(features)} rows, {metadata['total_features']} features")
            
            # Run comparison
            print("\n🚀 Running model comparison (this may take a few minutes)...\n")
            
            results = compare_models(
                features,
                initial_train_days=63,
                test_days=5,
                verbose=True,
            )
            
            print("\n✅ Comparison complete!")
        else:
            print("❌ GLD not found in data")
    else:
        print(f"❌ Data file not found: {data_file}")
        print("Run: python scripts/fetch_commodities.py")
