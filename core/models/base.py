"""
Base model interface for quant prediction models.

Defines the protocol that all strategy models should follow,
enabling interchangeable use in walk-forward validation and backtesting.
"""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import pandas as pd


class BaseDirectionModel(ABC):
    """
    Abstract base for binary direction classifiers.

    All concrete models (XGBoost, LSTM, Logistic, etc.) should subclass
    this so that ``WalkForwardValidator`` and the API layer can treat
    them uniformly.
    """

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseDirectionModel":
        """Train the model on feature matrix *X* and labels *y*."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted class labels (0 or 1)."""

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class probabilities, shape (n_samples, 2)."""

    def get_feature_importance(self) -> Optional[pd.Series]:
        """Return feature importance (if available), or None."""
        return None
