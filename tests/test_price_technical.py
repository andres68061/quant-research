"""Tests for the qlib-Alpha158-inspired price/volume technical factor family."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.data.factors.price_technical import (
    BAR_SHAPE_FACTOR_COLUMNS,
    DEFAULT_WINDOWS,
    PRICE_TECHNICAL_FACTOR_COLUMNS,
    build_price_technical_columns,
    compute_bar_shape_factors,
    compute_extreme_position_factors,
    compute_momentum_factors,
    compute_price_technical_factors,
    compute_trend_quality_factors,
    compute_volume_price_factors,
)
from core.data.factors.price_technical_helpers import (
    compute_days_since_high,
    compute_days_since_low,
    compute_rolling_percentile_rank,
    compute_rolling_trend,
    safe_ratio,
)
from core.exceptions import DataSchemaError

BAR_COLUMNS = ["adj_open", "adj_high", "adj_low", "adj_close", "volume"]


def _index(n_days: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2023-01-02", periods=n_days, tz="America/New_York", name="date")


def _bars(n_days: int = 150, seed: int = 7) -> pd.DataFrame:
    """Synthetic OHLCV bars with a consistent intraday path."""
    rng = np.random.default_rng(seed)
    index = _index(n_days)
    close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n_days))), index=index)
    open_price = close.shift(1).fillna(100.0) * (1.0 + rng.normal(0, 0.002, n_days))
    spread = np.abs(rng.normal(0.01, 0.003, n_days))
    high = pd.concat([open_price, close], axis=1).max(axis=1) * (1.0 + spread)
    low = pd.concat([open_price, close], axis=1).min(axis=1) * (1.0 - spread)
    volume = pd.Series(rng.integers(1_000_000, 5_000_000, n_days).astype("float64"), index=index)
    return pd.DataFrame(
        {
            "adj_open": open_price,
            "adj_high": high,
            "adj_low": low,
            "adj_close": close,
            "volume": volume,
        }
    )


def _flat_bars(n_days: int = 40, price: float = 50.0, volume: float = 1e6) -> pd.DataFrame:
    """Bars where nothing whatsoever moves — the degenerate-denominator case."""
    index = _index(n_days)
    return pd.DataFrame(
        {column: pd.Series(price, index=index) for column in BAR_COLUMNS[:4]}
        | {"volume": pd.Series(volume, index=index)}
    )


class TestSafeRatio:
    def test_masks_nonpositive_denominator_to_nan(self) -> None:
        numerator = pd.Series([1.0, 2.0, 3.0])
        denominator = pd.Series([2.0, 0.0, -4.0])
        result = safe_ratio(numerator, denominator)
        assert result.iloc[0] == 0.5
        assert result.iloc[1:].isna().all()

    def test_signed_mode_keeps_negative_denominator(self) -> None:
        result = safe_ratio(pd.Series([1.0, 1.0]), pd.Series([-2.0, 0.0]), positive_only=False)
        assert result.iloc[0] == -0.5
        assert np.isnan(result.iloc[1])


class TestBarShapeFactors:
    def test_body_and_shadows_partition_the_bar_range(self) -> None:
        factors = compute_bar_shape_factors(_bars())
        partition = (
            factors["bar_body_ratio"].abs()
            + factors["bar_upper_shadow_ratio"]
            + factors["bar_lower_shadow_ratio"]
        )
        assert np.allclose(partition.dropna(), 1.0)

    def test_close_position_is_one_when_close_sits_on_the_high(self) -> None:
        index = _index(3)
        bars = pd.DataFrame(
            {
                "adj_open": pd.Series(100.0, index=index),
                "adj_high": pd.Series(110.0, index=index),
                "adj_low": pd.Series(90.0, index=index),
                "adj_close": pd.Series(110.0, index=index),
                "volume": pd.Series(1e6, index=index),
            }
        )
        factors = compute_bar_shape_factors(bars)
        assert np.allclose(factors["bar_close_position"], 1.0)
        assert np.allclose(factors["bar_upper_shadow_ratio"], 0.0)
        assert np.allclose(factors["bar_lower_shadow_ratio"], 0.5)
        assert np.allclose(factors["bar_body_ratio"], 0.5)
        assert np.allclose(factors["bar_range_pct"], 20.0 / 110.0)

    def test_bounded_ratios_stay_inside_their_ranges(self) -> None:
        factors = compute_bar_shape_factors(_bars()).dropna()
        assert factors["bar_body_ratio"].between(-1.0, 1.0).all()
        assert factors["bar_close_position"].between(-1.0, 1.0).all()
        assert factors["bar_upper_shadow_ratio"].between(0.0, 1.0).all()
        assert factors["bar_lower_shadow_ratio"].between(0.0, 1.0).all()

    def test_zero_range_bar_yields_nan_not_infinity(self) -> None:
        factors = compute_bar_shape_factors(_flat_bars())
        assert (
            factors[list(BAR_SHAPE_FACTOR_COLUMNS)]
            .drop(columns=["bar_range_pct"])
            .isna()
            .all(axis=None)
        )
        assert np.allclose(factors["bar_range_pct"], 0.0)


class TestMomentumFactors:
    def test_roc_matches_the_simple_window_return(self) -> None:
        bars = _bars()
        factors = compute_momentum_factors(bars["adj_close"], 20)
        expected = bars["adj_close"] / bars["adj_close"].shift(20) - 1.0
        assert np.allclose(factors["roc_20d"].dropna(), expected.dropna())

    def test_flat_price_gives_zero_momentum_and_undefined_up_share(self) -> None:
        factors = compute_momentum_factors(_flat_bars()["adj_close"], 10).dropna(how="all")
        assert np.allclose(factors["ma_ratio_10d"].dropna(), 0.0)
        assert np.allclose(factors["price_std_10d"].dropna(), 0.0)
        # No movement at all means there is nothing to apportion between up and down.
        assert factors["sump_10d"].isna().all()

    def test_sump_saturates_on_monotone_paths(self) -> None:
        index = _index(30)
        rising = pd.Series(np.linspace(10.0, 40.0, 30), index=index)
        falling = pd.Series(np.linspace(40.0, 10.0, 30), index=index)
        assert np.allclose(compute_momentum_factors(rising, 10)["sump_10d"].dropna(), 1.0)
        assert np.allclose(compute_momentum_factors(falling, 10)["sump_10d"].dropna(), 0.0)

    def test_sump_is_bounded_on_real_paths(self) -> None:
        sump = compute_momentum_factors(_bars()["adj_close"], 20)["sump_20d"].dropna()
        assert sump.between(0.0, 1.0).all()


class TestTrendQualityFactors:
    def test_perfect_line_gives_unit_rsqr_and_exact_slope(self) -> None:
        index = _index(30)
        close = pd.Series(np.arange(30, dtype="float64") * 2.0 + 100.0, index=index)
        factors = compute_trend_quality_factors(close, 10)
        assert np.isclose(factors["trend_rsqr_10d"].iloc[-1], 1.0)
        assert np.isclose(factors["trend_resid_10d"].iloc[-1], 0.0)
        # Slope is normalised by the close: 2.0 per day on a price of 158.0.
        assert np.isclose(factors["trend_slope_10d"].iloc[-1], 2.0 / close.iloc[-1])

    def test_matches_numpy_polyfit_on_a_noisy_window(self) -> None:
        close = _bars()["adj_close"]
        window = 20
        trend = compute_rolling_trend(close, window)
        tail = close.to_numpy()[-window:]
        positions = np.arange(window, dtype="float64")
        slope, intercept = np.polyfit(positions, tail, 1)
        residuals = tail - (slope * positions + intercept)

        assert np.isclose(trend["trend_slope"].iloc[-1], slope)
        assert np.isclose(trend["trend_rsqr"].iloc[-1], np.corrcoef(positions, tail)[0, 1] ** 2)
        assert np.isclose(
            trend["trend_resid"].iloc[-1], np.sqrt((residuals**2).sum() / (window - 2))
        )

    def test_flat_window_is_a_perfect_zero_slope_fit(self) -> None:
        trend = compute_rolling_trend(_flat_bars()["adj_close"], 5)
        assert np.allclose(trend["trend_slope"].dropna(), 0.0)
        assert np.allclose(trend["trend_rsqr"].dropna(), 1.0)
        assert np.allclose(trend["trend_resid"].dropna(), 0.0)

    def test_window_below_three_raises(self) -> None:
        with pytest.raises(ValueError, match="window >= 3"):
            compute_rolling_trend(_bars(20)["adj_close"], 2)

    def test_series_shorter_than_window_is_all_nan(self) -> None:
        trend = compute_rolling_trend(_bars(5)["adj_close"], 60)
        assert trend.isna().all(axis=None)
        assert len(trend) == 5


class TestExtremePositionFactors:
    def test_max_and_min_bracket_the_current_close(self) -> None:
        factors = compute_extreme_position_factors(_bars()["adj_close"], 20).dropna()
        assert (factors["max_to_close_20d"] >= -1e-12).all()
        assert (factors["min_to_close_20d"] <= 1e-12).all()
        assert (factors["qtlu_to_close_20d"] >= factors["qtld_to_close_20d"]).all()

    def test_extreme_ages_and_rank_on_a_hand_checked_series(self) -> None:
        close = pd.Series([1.0, 5.0, 3.0, 2.0, 4.0, 0.0, 9.0, 1.0], index=_index(8))
        factors = compute_extreme_position_factors(close, 4)

        # Window [1, 5, 3, 2] ending on row 3: the high (5) was 2 days ago, the low
        # (1) was 3 days ago.
        assert factors["days_since_high_4d"].iloc[3] == 2.0
        assert factors["days_since_low_4d"].iloc[3] == 3.0
        assert factors["high_low_age_gap_4d"].iloc[3] == -1.0
        # Row 6 (value 9) is a fresh window high, so its age is 0 and its rank is 1.
        assert factors["days_since_high_4d"].iloc[6] == 0.0
        assert factors["close_rank_4d"].iloc[6] == 1.0
        assert factors["max_to_close_4d"].iloc[6] == 0.0
        assert factors["days_since_high_4d"].iloc[:3].isna().all()

    def test_extreme_age_ties_resolve_to_the_most_recent_occurrence(self) -> None:
        close = pd.Series(3.0, index=_index(6))
        assert np.allclose(compute_days_since_high(close, 3).dropna(), 0.0)
        assert np.allclose(compute_days_since_low(close, 3).dropna(), 0.0)

    def test_ages_stay_within_the_window(self) -> None:
        close = _bars()["adj_close"]
        for window in DEFAULT_WINDOWS:
            ages = compute_days_since_high(close, window).dropna()
            assert ages.between(0.0, window - 1).all()

    def test_percentile_rank_is_bounded_and_one_at_a_new_high(self) -> None:
        close = pd.Series(np.arange(20, dtype="float64"), index=_index(20))
        rank = compute_rolling_percentile_rank(close, 5).dropna()
        assert rank.between(0.0, 1.0).all()
        assert np.allclose(rank, 1.0)


class TestVolumePriceFactors:
    def test_volume_tracking_price_gives_perfect_correlation(self) -> None:
        index = _index(30)
        close = pd.Series(np.linspace(10.0, 40.0, 30), index=index)
        volume = close * 1_000.0
        factors = compute_volume_price_factors(close, volume, 10)
        assert np.allclose(factors["corr_close_volume_10d"].dropna(), 1.0)

    def test_constant_volume_leaves_correlation_undefined(self) -> None:
        bars = _bars()
        volume = pd.Series(2_000_000.0, index=bars.index)
        factors = compute_volume_price_factors(bars["adj_close"], volume, 20)
        assert factors["corr_close_volume_20d"].isna().all()
        assert np.allclose(factors["volume_std_ratio_20d"].dropna(), 0.0)
        assert np.allclose(factors["volume_to_mean_20d"].dropna(), 0.0)

    def test_correlations_are_bounded(self) -> None:
        bars = _bars()
        factors = compute_volume_price_factors(bars["adj_close"], bars["volume"], 20).dropna()
        assert factors["corr_close_volume_20d"].between(-1.0, 1.0).all()
        assert factors["corr_return_volume_change_20d"].between(-1.0, 1.0).all()

    def test_volume_and_dispersion_factors_are_nonnegative(self) -> None:
        bars = _bars()
        factors = compute_volume_price_factors(bars["adj_close"], bars["volume"], 20).dropna()
        assert (factors["volume_std_ratio_20d"] >= 0.0).all()
        assert (factors["wvma_20d"] >= 0.0).all()

    def test_volume_to_mean_flags_a_volume_spike(self) -> None:
        bars = _bars()
        spiked = bars.copy()
        spiked.iloc[100, spiked.columns.get_loc("volume")] *= 20.0
        factors = compute_volume_price_factors(spiked["adj_close"], spiked["volume"], 20)
        assert factors["volume_to_mean_20d"].iloc[100] > 5.0


class TestSchema:
    def test_default_column_tuple_covers_every_stem_and_window(self) -> None:
        assert PRICE_TECHNICAL_FACTOR_COLUMNS == build_price_technical_columns(DEFAULT_WINDOWS)
        assert len(PRICE_TECHNICAL_FACTOR_COLUMNS) == len(set(PRICE_TECHNICAL_FACTOR_COLUMNS))
        assert PRICE_TECHNICAL_FACTOR_COLUMNS[: len(BAR_SHAPE_FACTOR_COLUMNS)] == (
            BAR_SHAPE_FACTOR_COLUMNS
        )

    def test_custom_windows_change_the_emitted_schema(self) -> None:
        factors = compute_price_technical_factors(_bars(80), windows=(5, 21))
        assert list(factors.columns) == list(build_price_technical_columns((5, 21)))
        assert "roc_21d" in factors.columns
        assert "roc_60d" not in factors.columns


class TestComputePriceTechnicalFactors:
    def test_emits_every_declared_column_as_float64(self) -> None:
        factors = compute_price_technical_factors(_bars())
        assert list(factors.columns) == list(PRICE_TECHNICAL_FACTOR_COLUMNS)
        assert (factors.dtypes == "float64").all()
        assert factors.index.equals(_bars().index)

    def test_empty_input_returns_empty_frame_with_schema(self) -> None:
        empty = pd.DataFrame(columns=BAR_COLUMNS)
        factors = compute_price_technical_factors(empty)
        assert factors.empty
        assert list(factors.columns) == list(PRICE_TECHNICAL_FACTOR_COLUMNS)

    def test_missing_bar_column_raises_data_schema_error(self) -> None:
        bars = _bars(30).drop(columns=["adj_high"])
        with pytest.raises(DataSchemaError, match="adj_high"):
            compute_price_technical_factors(bars)

    def test_window_below_minimum_raises_data_schema_error(self) -> None:
        with pytest.raises(DataSchemaError, match=">= 3"):
            compute_price_technical_factors(_bars(30), windows=(2, 20))

    def test_history_shorter_than_the_longest_window_is_nan_not_an_error(self) -> None:
        factors = compute_price_technical_factors(_bars(8))
        assert len(factors) == 8
        assert factors[[c for c in factors.columns if c.endswith("_60d")]].isna().all(axis=None)

    def test_warm_up_rows_are_nan_for_every_windowed_factor(self) -> None:
        factors = compute_price_technical_factors(_bars())
        windowed = [c for c in factors.columns if c.endswith("_60d")]
        assert factors.iloc[:58][windowed].isna().all(axis=None)

    def test_produces_finite_values_once_warmed_up(self) -> None:
        factors = compute_price_technical_factors(_bars())
        warmed = factors.iloc[70:]
        assert not np.isinf(warmed.to_numpy()).any()
        assert warmed.notna().all(axis=None)

    def test_degenerate_flat_bars_never_produce_infinities(self) -> None:
        factors = compute_price_technical_factors(_flat_bars(80))
        assert not np.isinf(factors.to_numpy()).any()


class TestCausality:
    """The whole family must be backward-looking; this is the load-bearing test."""

    def test_no_factor_uses_future_information(self) -> None:
        bars = _bars(150, seed=11)
        mutated = bars.copy()
        # Replace the last 30 rows with something wildly different. A factor that
        # peeked forward — a centred window, a shift with the wrong sign, a fit over
        # the whole sample — would change values on rows that precede the edit.
        tail = slice(120, None)
        for column in ("adj_open", "adj_high", "adj_low", "adj_close"):
            mutated.iloc[tail, mutated.columns.get_loc(column)] *= 4.0
        mutated.iloc[tail, mutated.columns.get_loc("volume")] *= 50.0

        base = compute_price_technical_factors(bars)
        edited = compute_price_technical_factors(mutated)
        pd.testing.assert_frame_equal(base.iloc[:120], edited.iloc[:120])

    def test_truncating_the_future_leaves_the_past_unchanged(self) -> None:
        """Same guarantee from the other direction: rows the model has not seen yet."""
        bars = _bars(150, seed=13)
        full = compute_price_technical_factors(bars)
        truncated = compute_price_technical_factors(bars.iloc[:100])
        pd.testing.assert_frame_equal(full.iloc[:100], truncated)

    def test_helper_extreme_ages_are_causal(self) -> None:
        close = _bars(120, seed=17)["adj_close"]
        mutated = close.copy()
        mutated.iloc[100:] *= 10.0
        pd.testing.assert_series_equal(
            compute_days_since_high(close, 20).iloc[:100],
            compute_days_since_high(mutated, 20).iloc[:100],
        )
