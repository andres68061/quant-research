"""Tests for the Variance Risk Premium (VRP) proxy signal.

Coverage:
    - compute_realized_variance: scaling, window, NaN behaviour
    - compute_vrp_proxy: sign, index alignment, positive-VIX / zero-return case
    - vrp_exposure: 1-day shift, binary range, continuous range, invalid mode
"""

import numpy as np
import pandas as pd
import pytest

from core.signals.vrp import compute_realized_variance, compute_vrp_proxy, vrp_exposure

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _bdate_series(values: list[float], start: str = "2020-01-01") -> pd.Series:
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)))


def _flat_returns(n: int = 30, daily_ret: float = 0.01) -> pd.Series:
    return _bdate_series([daily_ret] * n)


def _flat_vix(n: int = 30, level: float = 20.0) -> pd.Series:
    return _bdate_series([level] * n)


# ---------------------------------------------------------------------------
# compute_realized_variance
# ---------------------------------------------------------------------------


def test_realized_variance_positive_for_nonzero_returns() -> None:
    rv = compute_realized_variance(_flat_returns(30, 0.01), window=21)
    assert (rv.dropna() > 0).all()


def test_realized_variance_zero_for_zero_returns() -> None:
    rv = compute_realized_variance(_flat_returns(30, 0.0), window=21)
    assert (rv.dropna() == 0.0).all()


def test_realized_variance_annualisation() -> None:
    """252 × r² matches the formula exactly for constant daily returns."""
    daily_r = 0.01
    rv = compute_realized_variance(_flat_returns(30, daily_r), window=21)
    expected = 252 * daily_r**2
    assert abs(rv.iloc[-1] - expected) < 1e-12


def test_realized_variance_nan_before_window_fills() -> None:
    rv = compute_realized_variance(_flat_returns(30, 0.01), window=21)
    assert rv.iloc[:20].isna().all()
    assert rv.iloc[20:].notna().all()


def test_realized_variance_name() -> None:
    rv = compute_realized_variance(_flat_returns(25), window=21)
    assert rv.name == "realized_variance"


def test_realized_variance_rejects_nonpositive_window() -> None:
    with pytest.raises(ValueError, match="window"):
        compute_realized_variance(_flat_returns(25), window=0)


def test_realized_variance_strips_tz() -> None:
    idx = pd.bdate_range("2020-01-01", periods=25, tz="UTC")
    returns = pd.Series([0.01] * 25, index=idx)
    rv = compute_realized_variance(returns, window=21)
    assert rv.index.tz is None


# ---------------------------------------------------------------------------
# compute_vrp_proxy
# ---------------------------------------------------------------------------


def test_vrp_proxy_positive_when_vix_high_rv_zero() -> None:
    """VRP > 0 when VIX is high and returns are zero (RV = 0)."""
    vrp = compute_vrp_proxy(_flat_vix(30, 20.0), _flat_returns(30, 0.0))
    assert (vrp > 0).all()


def test_vrp_proxy_negative_when_vix_low_returns_large() -> None:
    """VRP < 0 when VIX is tiny (≈0) and daily returns are large."""
    vrp = compute_vrp_proxy(_flat_vix(30, 0.1), _flat_returns(30, 0.05))
    assert (vrp < 0).all()


def test_vrp_proxy_name() -> None:
    vrp = compute_vrp_proxy(_flat_vix(30), _flat_returns(30))
    assert vrp.name == "vrp_proxy"


def test_vrp_proxy_index_alignment() -> None:
    """Misaligned inputs are inner-joined correctly."""
    vix = _bdate_series([20.0] * 40, start="2020-01-01")
    ret = _bdate_series([0.01] * 30, start="2020-02-01")
    vrp = compute_vrp_proxy(vix, ret)
    # Overlap starts at 2020-02-01; must have fewer rows than either input
    assert len(vrp) < 30


def test_vrp_proxy_consistent_with_components() -> None:
    """Manual component subtraction must equal vrp_proxy."""
    vix = _flat_vix(30, 20.0)
    ret = _flat_returns(30, 0.01)
    vrp = compute_vrp_proxy(vix, ret)
    implied = (20.0 / 100.0) ** 2
    rv_manual = 252 * 0.01**2
    expected_vrp = implied - rv_manual
    assert abs(vrp.iloc[-1] - expected_vrp) < 1e-10


# ---------------------------------------------------------------------------
# vrp_exposure
# ---------------------------------------------------------------------------


def test_vrp_exposure_binary_is_zero_one() -> None:
    vrp = _bdate_series([0.01, -0.005, 0.02, -0.001, 0.005])
    exp = vrp_exposure(vrp, mode="binary")
    valid = exp.dropna()
    assert set(valid.unique()).issubset({0.0, 1.0})


def test_vrp_exposure_binary_shifted_one_day() -> None:
    """Exposure at t+1 must equal (vrp[t] > 0)."""
    vrp = _bdate_series([0.01, -0.005, 0.02])
    exp = vrp_exposure(vrp, mode="binary")
    # exp[1] should reflect vrp[0] > 0 → 1.0
    assert exp.iloc[1] == pytest.approx(1.0)
    # exp[2] should reflect vrp[1] < 0 → 0.0
    assert exp.iloc[2] == pytest.approx(0.0)


def test_vrp_exposure_binary_first_value_is_nan() -> None:
    vrp = _bdate_series([0.01, 0.02, 0.03])
    exp = vrp_exposure(vrp, mode="binary")
    assert pd.isna(exp.iloc[0])


def test_vrp_exposure_continuous_in_zero_one() -> None:
    n = 300
    vrp = _bdate_series(list(np.random.default_rng(42).normal(0.0, 0.01, n)))
    exp = vrp_exposure(vrp, mode="continuous", quantile_window=60)
    valid = exp.dropna()
    assert valid.between(0.0, 1.0).all()


def test_vrp_exposure_rejects_invalid_mode() -> None:
    vrp = _bdate_series([0.01, 0.02])
    with pytest.raises(ValueError, match="mode"):
        vrp_exposure(vrp, mode="quantile")


def test_vrp_exposure_rejects_nonpositive_quantile_window() -> None:
    vrp = _bdate_series([0.01, 0.02])
    with pytest.raises(ValueError, match="quantile_window"):
        vrp_exposure(vrp, mode="continuous", quantile_window=0)


def test_vrp_exposure_name() -> None:
    vrp = _bdate_series([0.01, 0.02, 0.03])
    exp = vrp_exposure(vrp, mode="binary")
    assert exp.name == "vrp_exposure"
