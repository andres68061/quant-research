"""Unit tests for core.metrics.factor_regression."""

import numpy as np
import pandas as pd
import pytest

from core.exceptions import DataSchemaError
from core.metrics.factor_regression import default_hac_lags, regress_alpha_on_factors


@pytest.fixture()
def ff5_panel() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2020-01-02", periods=1000)
    return pd.DataFrame(
        {
            "mkt_rf": rng.normal(0.0004, 0.010, 1000),
            "smb": rng.normal(0.0, 0.005, 1000),
            "hml": rng.normal(0.0, 0.005, 1000),
            "rmw": rng.normal(0.0, 0.004, 1000),
            "cma": rng.normal(0.0, 0.004, 1000),
            "rf": np.full(1000, 0.00008),
        },
        index=dates,
    )


class TestDefaultHacLags:
    def test_rule_of_thumb(self):
        # floor(4 * (1000/100)^(2/9)) = floor(4 * 10^0.222) = 6
        assert default_hac_lags(1000) == 6

    def test_minimum_one(self):
        assert default_hac_lags(1) == 1
        assert default_hac_lags(10) >= 1


class TestRegressAlphaOnFactors:
    def test_recovers_known_alpha_and_beta(self, ff5_panel):
        rng = np.random.default_rng(11)
        true_alpha = 0.0005  # 5 bps/day
        true_beta = 0.8
        noise = rng.normal(0.0, 0.002, len(ff5_panel))
        strategy = true_alpha + true_beta * ff5_panel["mkt_rf"] + noise
        strategy = pd.Series(strategy, index=ff5_panel.index)

        result = regress_alpha_on_factors(strategy, ff5_panel)

        assert result["alpha"] == pytest.approx(true_alpha, abs=3e-4)
        assert result["betas"]["mkt_rf"] == pytest.approx(true_beta, abs=0.05)
        assert result["n_obs"] == len(ff5_panel)
        assert result["hac_lags"] == 6
        # A genuine 5 bps/day alpha over 1000 days should be detectable
        assert result["alpha_tstat"] > 2.0

    def test_zero_alpha_not_significant(self, ff5_panel):
        rng = np.random.default_rng(13)
        strategy = pd.Series(
            0.9 * ff5_panel["mkt_rf"] + rng.normal(0.0, 0.003, len(ff5_panel)),
            index=ff5_panel.index,
        )
        result = regress_alpha_on_factors(strategy, ff5_panel)
        assert abs(result["alpha_tstat"]) < 2.0

    def test_rf_subtraction(self, ff5_panel):
        rng = np.random.default_rng(17)
        strategy = pd.Series(
            ff5_panel["rf"] + 0.5 * ff5_panel["mkt_rf"] + rng.normal(0, 0.002, len(ff5_panel)),
            index=ff5_panel.index,
        )
        with_rf = regress_alpha_on_factors(strategy, ff5_panel, rf=ff5_panel["rf"])
        without_rf = regress_alpha_on_factors(strategy, ff5_panel)
        # Subtracting a constant rf shifts alpha down by exactly rf
        assert with_rf["alpha"] == pytest.approx(without_rf["alpha"] - 0.00008, abs=1e-6)

    def test_insufficient_overlap_raises(self, ff5_panel):
        strategy = pd.Series([0.001] * 10, index=pd.bdate_range("2030-01-01", periods=10))
        with pytest.raises(DataSchemaError):
            regress_alpha_on_factors(strategy, ff5_panel)

    def test_missing_columns_raise(self, ff5_panel):
        strategy = pd.Series(0.001, index=ff5_panel.index)
        with pytest.raises(DataSchemaError):
            regress_alpha_on_factors(strategy, ff5_panel, factor_columns=["mkt_rf", "not_a_factor"])

    def test_output_units(self, ff5_panel):
        rng = np.random.default_rng(19)
        strategy = pd.Series(0.0004 + rng.normal(0, 0.002, len(ff5_panel)), index=ff5_panel.index)
        result = regress_alpha_on_factors(strategy, ff5_panel)
        assert result["alpha_bps_per_period"] == pytest.approx(result["alpha"] * 1e4)
        assert result["alpha_ann_pct"] == pytest.approx(result["alpha"] * 252 * 100)
