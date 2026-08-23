"""Tests for range-based volatility, spread, and overnight/intraday factors."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.data.factors.microstructure import (
    MICROSTRUCTURE_FACTOR_COLUMNS,
    TRADING_DAYS_PER_YEAR,
    _adjust_for_overnight_gap,
    compute_amihud_illiquidity,
    compute_close_location,
    compute_corwin_schultz_spread,
    compute_garman_klass_volatility,
    compute_microstructure_factors,
    compute_overnight_intraday_split,
    compute_parkinson_volatility,
)


def _bars(n_days: int = 120, seed: int = 7) -> pd.DataFrame:
    """Synthetic OHLCV bars with a consistent intraday path."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2023-01-02", periods=n_days, tz="America/New_York", name="date")
    close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n_days))), index=index)
    open_price = close.shift(1).fillna(100.0) * (1.0 + rng.normal(0, 0.002, n_days))
    spread = np.abs(rng.normal(0.01, 0.003, n_days))
    high = pd.concat([open_price, close], axis=1).max(axis=1) * (1.0 + spread)
    low = pd.concat([open_price, close], axis=1).min(axis=1) * (1.0 - spread)
    volume = pd.Series(rng.integers(1_000_000, 5_000_000, n_days), index=index)
    return pd.DataFrame(
        {
            "adj_open": open_price,
            "adj_high": high,
            "adj_low": low,
            "adj_close": close,
            "volume": volume,
        }
    )


class TestRangeVolatility:
    def test_parkinson_recovers_a_known_constant_range(self) -> None:
        # A constant H/L ratio makes the estimator analytically checkable.
        index = pd.bdate_range("2023-01-02", periods=60, tz="America/New_York")
        high = pd.Series(101.0, index=index)
        low = pd.Series(99.0, index=index)
        result = compute_parkinson_volatility(high, low, window=21).dropna()

        log_range = np.log(101.0 / 99.0)
        expected = np.sqrt(log_range**2 / (4 * np.log(2)) * TRADING_DAYS_PER_YEAR)
        assert np.allclose(result, expected)

    def test_garman_klass_close_to_parkinson_on_symmetric_bars(self) -> None:
        bars = _bars()
        parkinson = compute_parkinson_volatility(bars["adj_high"], bars["adj_low"])
        garman_klass = compute_garman_klass_volatility(
            bars["adj_open"], bars["adj_high"], bars["adj_low"], bars["adj_close"]
        )
        ratio = (garman_klass / parkinson).dropna()
        assert ((ratio > 0.5) & (ratio < 1.6)).all()

    def test_volatility_is_annualized_and_positive(self) -> None:
        bars = _bars()
        vol = compute_parkinson_volatility(bars["adj_high"], bars["adj_low"]).dropna()
        assert (vol > 0).all()
        assert (vol < 5.0).all()

    def test_zero_or_negative_prices_yield_nan_not_error(self) -> None:
        index = pd.bdate_range("2023-01-02", periods=30, tz="America/New_York")
        high = pd.Series(0.0, index=index)
        low = pd.Series(0.0, index=index)
        assert compute_parkinson_volatility(high, low).isna().all()


class TestCorwinSchultz:
    def test_spread_is_nonnegative_and_small(self) -> None:
        bars = _bars()
        spread = compute_corwin_schultz_spread(
            bars["adj_high"], bars["adj_low"], bars["adj_close"]
        ).dropna()
        assert (spread >= 0).all()
        assert (spread < 0.5).all()

    def test_wider_ranges_imply_wider_spreads(self) -> None:
        narrow = _bars(seed=1)
        wide = narrow.copy()
        mid = (wide["adj_high"] + wide["adj_low"]) / 2
        wide["adj_high"] = mid * 1.05
        wide["adj_low"] = mid * 0.95

        narrow_spread = compute_corwin_schultz_spread(
            narrow["adj_high"], narrow["adj_low"], narrow["adj_close"]
        ).dropna()
        wide_spread = compute_corwin_schultz_spread(
            wide["adj_high"], wide["adj_low"], wide["adj_close"]
        ).dropna()
        assert wide_spread.mean() > narrow_spread.mean()

    def test_overnight_gap_adjustment_absorbs_most_of_a_jump(self) -> None:
        """A pure overnight jump is not a spread; the adjustment must absorb it.

        The adjustment is additive on prices while the jump here is proportional,
        so a small residual remains — the test asserts the adjustment removes the
        bulk of the inflation, not that it is exact.
        """
        bars = _bars(seed=3)
        gapped = bars.copy()
        # Shift the whole bar up 10% from day 60 onward: a jump, no range change.
        for column in ("adj_open", "adj_high", "adj_low", "adj_close"):
            gapped.iloc[60:, gapped.columns.get_loc(column)] *= 1.10

        base = compute_corwin_schultz_spread(bars["adj_high"], bars["adj_low"], bars["adj_close"])
        adjusted = compute_corwin_schultz_spread(
            gapped["adj_high"], gapped["adj_low"], gapped["adj_close"]
        )
        # The gap day sits inside this window; the spread must stay in the same
        # neighbourhood rather than spiking on a jump that carries no spread.
        assert abs(adjusted.iloc[65] - base.iloc[65]) < 0.2 * base.iloc[65]

    def test_gap_adjustment_pulls_a_jumped_bar_back_onto_the_prior_close(self) -> None:
        """The adjustment itself: after shifting, the bar no longer sits above the prior close."""
        bars = _bars(seed=3)
        gapped = bars.copy()
        for column in ("adj_open", "adj_high", "adj_low", "adj_close"):
            gapped.iloc[60:, gapped.columns.get_loc(column)] *= 1.10

        adjusted_high, adjusted_low = _adjust_for_overnight_gap(
            gapped["adj_high"], gapped["adj_low"], gapped["adj_close"]
        )
        prior_close = gapped["adj_close"].shift(1)

        # Before: day 60's whole bar gapped above the prior close.
        assert gapped["adj_low"].iloc[60] > prior_close.iloc[60]
        # After: the gap is removed, so the low sits at the prior close.
        assert abs(adjusted_low.iloc[60] - prior_close.iloc[60]) < 1e-9
        # The bar's width is preserved — a shift, not a rescale.
        original_width = gapped["adj_high"].iloc[60] - gapped["adj_low"].iloc[60]
        assert abs((adjusted_high.iloc[60] - adjusted_low.iloc[60]) - original_width) < 1e-9


class TestOvernightIntradaySplit:
    def test_components_sum_to_total_log_return(self) -> None:
        bars = _bars()
        split = compute_overnight_intraday_split(bars["adj_open"], bars["adj_close"], window=21)
        total = np.log(bars["adj_close"] / bars["adj_close"].shift(1))
        expected = total.rolling(21, min_periods=10).sum()
        combined = split["overnight_return_21d"] + split["intraday_return_21d"]
        assert np.allclose(combined.dropna(), expected.reindex(combined.dropna().index), atol=1e-10)

    def test_gap_is_the_difference_of_the_legs(self) -> None:
        bars = _bars()
        split = compute_overnight_intraday_split(bars["adj_open"], bars["adj_close"])
        assert np.allclose(
            split["overnight_intraday_gap"].dropna(),
            (split["overnight_return_21d"] - split["intraday_return_21d"]).dropna(),
        )


class TestAmihudAndCloseLocation:
    def test_illiquidity_falls_as_volume_rises(self) -> None:
        bars = _bars()
        thin = compute_amihud_illiquidity(bars["adj_close"], bars["volume"]).dropna()
        thick = compute_amihud_illiquidity(bars["adj_close"], bars["volume"] * 100).dropna()
        assert thick.mean() < thin.mean()

    def test_close_location_bounded_zero_to_one(self) -> None:
        bars = _bars()
        location = compute_close_location(
            bars["adj_high"], bars["adj_low"], bars["adj_close"]
        ).dropna()
        assert ((location >= 0.0) & (location <= 1.0)).all()

    def test_close_at_high_gives_one(self) -> None:
        index = pd.bdate_range("2023-01-02", periods=30, tz="America/New_York")
        high = pd.Series(110.0, index=index)
        low = pd.Series(100.0, index=index)
        close = pd.Series(110.0, index=index)
        assert np.allclose(compute_close_location(high, low, close).dropna(), 1.0)


class TestComputeMicrostructureFactors:
    def test_emits_every_declared_column(self) -> None:
        factors = compute_microstructure_factors(_bars())
        assert list(factors.columns) == list(MICROSTRUCTURE_FACTOR_COLUMNS)

    def test_empty_input_returns_empty_frame_with_schema(self) -> None:
        empty = pd.DataFrame(columns=["adj_open", "adj_high", "adj_low", "adj_close", "volume"])
        factors = compute_microstructure_factors(empty)
        assert factors.empty
        assert list(factors.columns) == list(MICROSTRUCTURE_FACTOR_COLUMNS)

    def test_neg_amihud_mirrors_amihud(self) -> None:
        factors = compute_microstructure_factors(_bars()).dropna()
        assert np.allclose(factors["neg_amihud_illiquidity"], -factors["amihud_illiquidity"])
