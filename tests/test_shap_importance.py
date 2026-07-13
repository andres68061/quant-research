"""Tests for SHAP mean-|value| importance on tree models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.models.commodity_direction import (
    XGBoostDirectionModel,
    compute_mean_abs_shap,
)


@pytest.fixture
def toy_binary_frame() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    n = 80
    X = pd.DataFrame(
        {
            "feat_a": rng.normal(size=n),
            "feat_b": rng.normal(size=n),
            "feat_c": rng.normal(size=n),
        }
    )
    # Label mostly follows feat_a so SHAP should prefer it.
    y = pd.Series((X["feat_a"] > 0).astype(int))
    return X, y


def test_compute_mean_abs_shap_returns_feature_keys(
    toy_binary_frame: tuple[pd.DataFrame, pd.Series],
) -> None:
    X, y = toy_binary_frame
    model = XGBoostDirectionModel(n_estimators=20, max_depth=2)
    model.fit(X, y)
    shap_scores = compute_mean_abs_shap(model, X.tail(20))
    assert set(shap_scores) == set(X.columns)
    assert all(isinstance(v, float) and v >= 0.0 for v in shap_scores.values())
    assert sum(shap_scores.values()) > 0.0
