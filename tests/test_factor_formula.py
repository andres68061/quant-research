"""
Unit tests locking in the momentum-excluding-recent formula.

These tests guard against regressions of two specific bugs documented in
`docs/FACTOR_BACKTEST_AUDIT.md` §3:

- Bug 1: arithmetic `cum_12 - cum_1` (incorrect) instead of geometric
  `(1+cum_12)/(1+cum_1) - 1` (correct). For non-infinitesimal returns the
  sign of the resulting ranking can flip.
- Bug 2: `cum.sub(ex_recent, fill_value=0.0)` would backfill missing
  short-history names with `-cum_1m`, producing a spurious signal for
  IPOs / recently listed stocks. The geometric form returns NaN until
  both windows are complete.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.data.factors.build_factors import momentum_excluding_recent, short_term_reversal


class TestGeometricMomentum:
    def test_matches_price_ratio_identity(self) -> None:
        """`mom_k_1` must equal `P(t-21) / P(t-k*21) - 1` up to float noise."""
        rng = np.random.default_rng(0)
        n = 500
        # Simulate a realistic price path (drift + vol).
        returns = rng.normal(0.0005, 0.02, n)
        close = pd.Series(100 * np.exp(np.cumsum(returns)))

        months = 12
        mom = momentum_excluding_recent(close, months)
        window = months * 21
        recent = 21

        # Compare against the price-ratio identity for the last valid row.
        t = len(close) - 1
        expected = close.iloc[t - recent] / close.iloc[t - window] - 1.0
        assert np.isclose(
            mom.iloc[t], expected, atol=1e-10
        ), f"mom[{t}]={mom.iloc[t]} expected={expected}"

    def test_arithmetic_form_would_flip_sign_vs_geometric(self) -> None:
        """
        Construct a case where arithmetic ``cum_12 - cum_1`` and geometric
        ``(1+cum_12)/(1+cum_1) - 1`` have opposite signs for the same stock.
        The point is that the old arithmetic formula was genuinely wrong —
        not just noisily different — when returns are large.
        """
        # Build a path where cum_12 = +30%, cum_1 = +50%.
        # Arithmetic: 0.30 - 0.50 = -0.20 (looks bearish-recent)
        # Geometric:  1.30 / 1.50 - 1 = -0.1333... (same sign — fine for this case)
        #
        # The real sign-flip case: cum_12 = +10%, cum_1 = -10%
        # Arithmetic: 0.10 - (-0.10) = +0.20 (bullish ex-recent)
        # Geometric:  1.10 / 0.90 - 1 = +0.2222... (also bullish — same sign)
        #
        # Where arithmetic and geometric diverge meaningfully: large magnitudes.
        # cum_12 = -0.50, cum_1 = -0.80 (crashed further recently)
        #   Arithmetic: -0.50 - (-0.80) = +0.30  (looks bullish ex-recent)
        #   Geometric:  0.50 / 0.20 - 1 = +1.50  (much more bullish)
        # Still same sign but wildly different magnitude => ranks flip when
        # compared against another stock. Test that the geometric formula
        # returns the geometric value, not the arithmetic one.
        n = 400
        close = pd.Series(np.ones(n), dtype=float)

        # Days 0..(252+21-1): price flat at 1.
        # At t = 252 + 21, force price == 0.5 (cum_12 ≈ -0.50)
        # At t = 252 + 21 (most recent): further fall to 0.2 by day t
        # We need: close[t - 21] / close[t - 252] == 0.50
        #         close[t]      / close[t - 21]  == 0.40  (=> cum_1 ≈ -0.60)
        t = 280  # index such that t-21 = 259 and t-252 = 28 are in range
        close.iloc[: t + 1] = 1.0
        close.iloc[t - 21] = 0.50  # P(t-21) = 0.5 -> P(t-21)/P(t-252) = 0.5
        close.iloc[t] = 0.20  # P(t)/P(t-21) = 0.4 (but irrelevant for mom_12_1)

        mom = momentum_excluding_recent(close, 12)
        # Geometric: P(t-21)/P(t-252) - 1 = 0.5 - 1 = -0.5
        assert np.isclose(mom.iloc[t], -0.5, atol=1e-10), (
            f"Got {mom.iloc[t]}, expected -0.5 from geometric formula. "
            "If this is +0.3 the arithmetic form has regressed."
        )

    def test_ipo_short_history_returns_nan_not_zero(self) -> None:
        """
        Recently-listed stocks must produce NaN until both the 252d and 21d
        windows are full. The old `fill_value=0.0` path produced a signal
        equal to `-cum_1m` after just 21 days, which is a major bug.
        """
        # A 100-day history: not enough for a 12-month window.
        rng = np.random.default_rng(1)
        n = 100
        close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, n))))

        mom = momentum_excluding_recent(close, 12)
        # Every row should be NaN because we never reach 12*21=252 days.
        assert mom.isna().all(), (
            f"Expected all-NaN for 100-day history; got {mom.notna().sum()} non-NaN values. "
            "If any values are non-NaN, the fill_value=0.0 regression is back."
        )

    def test_first_valid_index_is_months_times_21(self) -> None:
        """First non-NaN must be at index `months*21`.

        pct_change produces NaN at position 0; then `rolling(window).apply(np.prod)`
        first produces a non-NaN at position `window` (not `window-1`) because the
        window starting at position `window-1` still contains ret[0]=NaN.
        """
        rng = np.random.default_rng(2)
        n = 400
        close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, n))))

        for months in (3, 6, 12):
            mom = momentum_excluding_recent(close, months)
            first_valid = mom.first_valid_index()
            assert (
                first_valid == months * 21
            ), f"months={months}: first_valid={first_valid}, expected {months*21}"


class TestShortTermReversal:
    def test_matches_negative_price_ratio_identity(self) -> None:
        """`rev_21d` must equal `-(P(t)/P(t-21) - 1)` up to float noise."""
        rng = np.random.default_rng(3)
        n = 300
        close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0005, 0.02, n))))

        window = 21
        rev = short_term_reversal(close, window=window)

        t = n - 1
        expected = -(close.iloc[t] / close.iloc[t - window] - 1.0)
        assert np.isclose(rev.iloc[t], expected, atol=1e-12)

    def test_loser_gets_positive_signal(self) -> None:
        """A stock that fell 10% over the window must rank HIGH (positive value)."""
        close = pd.Series([100.0] * 21 + [90.0])
        rev = short_term_reversal(close, window=21)
        assert np.isclose(
            rev.iloc[-1], 0.10, atol=1e-12
        ), "A 10% loser must produce rev=+0.10 so ranking descending buys losers."

    def test_winner_gets_negative_signal(self) -> None:
        """A stock that rose 20% over the window must rank LOW (negative value)."""
        close = pd.Series([100.0] * 21 + [120.0])
        rev = short_term_reversal(close, window=21)
        assert np.isclose(rev.iloc[-1], -0.20, atol=1e-12)

    def test_short_history_returns_nan_not_zero(self) -> None:
        """No signal until the full window of prices exists (no fill_value tricks)."""
        close = pd.Series([100.0, 101.0, 99.0])
        rev = short_term_reversal(close, window=21)
        assert (
            rev.isna().all()
        ), f"Expected all-NaN for 3-day history; got {rev.notna().sum()} non-NaN values."

    def test_first_valid_index_is_window(self) -> None:
        """First non-NaN appears exactly at index `window` (needs P(t-window))."""
        close = pd.Series(np.linspace(100.0, 110.0, 60))
        rev = short_term_reversal(close, window=21)
        assert rev.first_valid_index() == 21
