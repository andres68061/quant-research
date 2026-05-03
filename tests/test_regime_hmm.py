"""Tests for core.signals.regime_hmm — features, HMM, leakage, signal."""

import numpy as np
import pandas as pd
import pytest

from core.signals.regime_hmm import (
    build_regime_features,
    fit_regime_hmm,
    regime_signal,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_DATES = pd.bdate_range("2010-01-04", periods=2000, freq="B")
_SYMBOLS = [f"SYM{i}" for i in range(50)]


@pytest.fixture()
def prices() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    data = 100 + rng.standard_normal((len(_DATES), len(_SYMBOLS))).cumsum(axis=0)
    return pd.DataFrame(data, index=_DATES, columns=_SYMBOLS)


@pytest.fixture()
def factors() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    mi = pd.MultiIndex.from_product([_DATES, _SYMBOLS], names=["date", "symbol"])
    return pd.DataFrame(
        {
            "beta_60d": rng.standard_normal(len(mi)),
            "vol_60d": np.abs(rng.standard_normal(len(mi))) * 0.2,
        },
        index=mi,
    )


@pytest.fixture()
def macro_z() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "macro_z_t10y2y": rng.standard_normal(len(_DATES)),
            "macro_z_fed_funds": rng.standard_normal(len(_DATES)),
            "macro_z_unrate": rng.standard_normal(len(_DATES)),
        },
        index=_DATES,
    )


@pytest.fixture()
def vix_series() -> pd.Series:
    rng = np.random.default_rng(42)
    return pd.Series(
        15 + rng.standard_normal(len(_DATES)) * 5,
        index=_DATES,
        name="VIX",
    )


@pytest.fixture()
def features(prices, factors, macro_z, vix_series) -> pd.DataFrame:
    return build_regime_features(prices, factors, macro_z, vix=vix_series)


# ---------------------------------------------------------------------------
# build_regime_features
# ---------------------------------------------------------------------------


class TestBuildRegimeFeatures:
    def test_returns_expected_columns(self, features):
        assert "return_dispersion" in features.columns
        assert "beta_60d_spread" in features.columns
        assert "vol_60d_median" in features.columns
        assert "vix" in features.columns

    def test_no_nan_rows(self, features):
        assert features.isna().sum().sum() == 0

    def test_without_vix(self, prices, factors, macro_z):
        feat = build_regime_features(prices, factors, macro_z, vix=None)
        assert "vix" not in feat.columns
        assert feat.isna().sum().sum() == 0

    def test_index_is_business_day(self, features):
        assert features.index.freq == "B" or pd.infer_freq(features.index) == "B"


# ---------------------------------------------------------------------------
# fit_regime_hmm
# ---------------------------------------------------------------------------


class TestFitRegimeHMM:
    def test_probabilities_sum_to_one(self, features):
        probs = fit_regime_hmm(features, n_states=3, train_window=500, step=100)
        prob_cols = [c for c in probs.columns if c.startswith("p_")]
        row_sums = probs[prob_cols].sum(axis=1)
        np.testing.assert_allclose(row_sums.values, 1.0, atol=1e-6)

    def test_regime_column_values(self, features):
        probs = fit_regime_hmm(features, n_states=3, train_window=500, step=100)
        assert set(probs["regime"].unique()).issubset({"risk_on", "neutral", "risk_off"})

    def test_two_state_model(self, features):
        probs = fit_regime_hmm(features, n_states=2, train_window=500, step=100)
        assert "p_risk_on" in probs.columns
        assert "p_risk_off" in probs.columns
        assert probs["regime"].isin({"risk_on", "risk_off"}).all()

    def test_too_few_rows_raises(self, features):
        tiny = features.iloc[:10]
        with pytest.raises(ValueError, match="Need at least"):
            fit_regime_hmm(tiny, n_states=3)

    def test_no_data_leakage(self, features):
        """Walk-forward: every prediction date must be strictly after its training window."""
        train_window = 500
        step = 100
        n = len(features)
        cursor = max(train_window, 30)
        while cursor < n:
            end_train_idx = cursor
            first_pred_idx = cursor
            assert first_pred_idx >= end_train_idx
            cursor += step

    def test_filtered_mode_ignores_future_prediction_rows(self, features):
        """Filtered probabilities at date t must not depend on later block rows."""
        train_window = 500
        step = 21
        base = fit_regime_hmm(
            features,
            n_states=3,
            train_window=train_window,
            step=step,
            predict_mode="filtered",
        )

        shocked = features.copy()
        first_future_row = train_window + 1
        shocked.iloc[first_future_row : first_future_row + step - 1] = (
            shocked.iloc[first_future_row : first_future_row + step - 1] * 1000.0
        )
        changed = fit_regime_hmm(
            shocked,
            n_states=3,
            train_window=train_window,
            step=step,
            predict_mode="filtered",
        )

        first_pred_date = features.index[train_window]
        prob_cols = [c for c in base.columns if c.startswith("p_")]
        np.testing.assert_allclose(
            base.loc[first_pred_date, prob_cols].astype(float).to_numpy(),
            changed.loc[first_pred_date, prob_cols].astype(float).to_numpy(),
            atol=1e-8,
        )

    def test_smoothed_mode_runs(self, features):
        probs = fit_regime_hmm(
            features,
            n_states=3,
            train_window=500,
            step=100,
            predict_mode="smoothed",
        )
        assert not probs.empty


# ---------------------------------------------------------------------------
# regime_signal
# ---------------------------------------------------------------------------


class TestRegimeSignal:
    def test_exposure_scale_range(self, features):
        probs = fit_regime_hmm(features, n_states=3, train_window=500, step=100)
        sig = regime_signal(probs, mode="exposure_scale")
        assert sig.min() >= 0.0
        assert sig.max() <= 1.0
        assert sig.name == "regime_signal"

    def test_long_short_values(self, features):
        probs = fit_regime_hmm(features, n_states=3, train_window=500, step=100)
        sig = regime_signal(probs, mode="long_short")
        assert set(sig.unique()).issubset({-1.0, 0.0, 1.0})

    def test_unknown_mode_raises(self, features):
        probs = fit_regime_hmm(features, n_states=3, train_window=500, step=100)
        with pytest.raises(ValueError, match="Unknown mode"):
            regime_signal(probs, mode="invalid")
