"""Tests for the clean-returns layer — the guard against vendor bad prints.

These exist because the failure they prevent is silent and catastrophic: a single
non-finite return in a cross-sectional mean turns every symbol's abnormal return
for that date into infinity, which then propagates through any cumulative path.
That is not hypothetical — it produced -inf CAR paths in the first
expanded-universe PEAD run.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.data.factors.returns import (
    DEFAULT_MAX_ABS_RETURN,
    compute_abnormal_returns,
    compute_clean_returns,
)

TZ = "America/New_York"


def _prices(**series: list[float]) -> pd.DataFrame:
    length = len(next(iter(series.values())))
    index = pd.bdate_range("2024-01-02", periods=length, tz=TZ, name="date")
    return pd.DataFrame(series, index=index)


class TestCleanReturns:
    def test_normal_returns_pass_through(self) -> None:
        returns = compute_clean_returns(_prices(AAA=[100.0, 110.0, 99.0]))
        assert abs(returns["AAA"].iloc[1] - 0.10) < 1e-12
        assert abs(returns["AAA"].iloc[2] - (-0.10)) < 1e-12

    def test_zero_price_yields_nan_not_infinity(self) -> None:
        """A zero prior close makes pct_change infinite — the poison case."""
        returns = compute_clean_returns(_prices(AAA=[100.0, 0.0, 50.0]))
        assert not np.isinf(returns["AAA"]).any()
        assert pd.isna(returns["AAA"].iloc[2])

    def test_extreme_return_rejected_as_bad_print(self) -> None:
        # 100 -> 100,000 is +99,900%: an un-adjusted reverse split, not a return.
        returns = compute_clean_returns(_prices(AAA=[100.0, 100_000.0, 100_000.0]))
        assert pd.isna(returns["AAA"].iloc[1])
        # The following day is a legitimate 0% and survives.
        assert returns["AAA"].iloc[2] == 0.0

    def test_bound_is_configurable_and_disableable(self) -> None:
        prices = _prices(AAA=[1.0, 10.0])
        assert pd.isna(compute_clean_returns(prices, max_abs_return=3.0)["AAA"].iloc[1])
        kept = compute_clean_returns(prices, max_abs_return=None)["AAA"].iloc[1]
        assert abs(kept - 9.0) < 1e-12

    def test_legitimate_doubling_survives_the_default_bound(self) -> None:
        """Takeover pops are real; the bound must not eat them."""
        returns = compute_clean_returns(_prices(AAA=[10.0, 20.0]))
        assert abs(returns["AAA"].iloc[1] - 1.0) < 1e-12
        assert DEFAULT_MAX_ABS_RETURN > 1.0

    def test_price_floor_requires_both_ends(self) -> None:
        # 0.50 -> 3.00 clears the floor on one end only; it is exactly the
        # sub-dollar bounce the floor exists to exclude.
        returns = compute_clean_returns(_prices(AAA=[0.50, 3.00, 3.30]), min_price=1.0)
        assert pd.isna(returns["AAA"].iloc[1])
        assert abs(returns["AAA"].iloc[2] - 0.10) < 1e-12

    def test_no_floor_by_default(self) -> None:
        returns = compute_clean_returns(_prices(AAA=[0.50, 0.55]))
        assert abs(returns["AAA"].iloc[1] - 0.10) < 1e-12


class TestAbnormalReturns:
    def _wide(self, n_symbols: int = 30, n_days: int = 5) -> pd.DataFrame:
        rng = np.random.default_rng(1)
        index = pd.bdate_range("2024-01-02", periods=n_days, tz=TZ, name="date")
        data = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n_days, n_symbols)), axis=0))
        return pd.DataFrame(data, index=index, columns=[f"S{i:02d}" for i in range(n_symbols)])

    def test_abnormal_returns_sum_to_about_zero_across_the_cross_section(self) -> None:
        abnormal = compute_abnormal_returns(self._wide(), min_names_per_date=5)
        daily_sum = abnormal.sum(axis=1).dropna()
        assert np.allclose(daily_sum, 0.0, atol=1e-10)

    def test_one_bad_print_does_not_poison_the_whole_cross_section(self) -> None:
        """THE regression test: this is what produced -inf CAR paths."""
        prices = self._wide()
        # A 1,000,000x print on a single name, on a single day.
        prices.iloc[2, 0] = prices.iloc[1, 0] * 1_000_000

        abnormal = compute_abnormal_returns(prices, min_names_per_date=5)
        others = abnormal.iloc[2, 1:]
        assert np.isfinite(others).all(), "one bad print corrupted other symbols"
        assert pd.isna(abnormal.iloc[2, 0]), "the bad print itself should be rejected"

    def test_zero_price_does_not_poison_the_cross_section(self) -> None:
        prices = self._wide()
        prices.iloc[1, 0] = 0.0

        abnormal = compute_abnormal_returns(prices, min_names_per_date=5)
        assert np.isfinite(abnormal.iloc[2, 1:]).all()

    def test_thin_dates_are_blanked(self) -> None:
        prices = self._wide(n_symbols=3)
        abnormal = compute_abnormal_returns(prices, min_names_per_date=20)
        assert abnormal.notna().sum().sum() == 0
