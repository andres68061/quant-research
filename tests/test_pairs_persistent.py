"""Tests for the cointegration-persistence pairs index.

Covers the two corrections to pairs_index.py's flawed premise (see
docs/FAILED_STRATEGIES_LOG.md): (1) candidates must actually cross, not
just track tightly; (2) trading stops when cointegration breaks, not on a
fixed calendar schedule.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.signals.pairs import count_cumulative_return_crossings
from core.strategies.pairs_gatev import normalize_price_index
from core.strategies.pairs_persistent import (
    find_crossing_cointegrated_candidates,
    run_pair_until_broken,
    run_pairs_persistent_index,
)


def _tight_pair(n: int = 500, seed: int = 1) -> pd.DataFrame:
    """Near-identical prices sharing one latent random walk, like dual
    share classes -- tightly tracking, minimal deviation, the GOOGL/GOOG
    failure mode. Direction-symmetric by construction (both legs are the
    same latent process plus small idiosyncratic noise)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-01", periods=n, tz="America/New_York")
    common = np.cumsum(rng.normal(0.0003, 0.012, n))
    x = 100 * np.exp(common + rng.normal(0.0, 0.0005, n))
    y = 100 * np.exp(common + rng.normal(0.0, 0.0005, n))
    return pd.DataFrame({"TIGHT_X": x, "TIGHT_Y": y}, index=dates)


def _oscillating_pair(
    n: int = 500, seed: int = 2, amplitude: float = 0.15, period_days: int = 60
) -> pd.DataFrame:
    """Genuinely cointegrated pair sharing one latent random walk plus a
    bounded oscillating wedge between the two legs -- crosses repeatedly,
    the shape we now want to select for. Direction-symmetric by
    construction (the wedge is split +/- around the shared latent
    process, not appended to only one side), so Engle-Granger works
    regardless of which leg is treated as dependent."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-01", periods=n, tz="America/New_York")
    common = np.cumsum(rng.normal(0.0003, 0.012, n))
    wave = amplitude * np.sin(2 * np.pi * np.arange(n) / period_days)
    x = 100 * np.exp(common - wave / 2 + rng.normal(0.0, 0.004, n))
    y = 100 * np.exp(common + wave / 2 + rng.normal(0.0, 0.004, n))
    return pd.DataFrame({"OSC_X": x, "OSC_Y": y}, index=dates)


def _sectors_for(symbols: list[str], sector: str = "Test Sector") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": symbols,
            "sector": [sector] * len(symbols),
            "quoteType": ["EQUITY"] * len(symbols),
        }
    )


class TestCountCrossings:
    def test_tight_pair_has_no_crossings(self) -> None:
        panel = _tight_pair()
        norm = normalize_price_index(panel)
        crossings = count_cumulative_return_crossings(norm["TIGHT_X"], norm["TIGHT_Y"])
        assert crossings == 0

    def test_oscillating_pair_has_many_crossings(self) -> None:
        panel = _oscillating_pair()
        norm = normalize_price_index(panel)
        crossings = count_cumulative_return_crossings(norm["OSC_X"], norm["OSC_Y"])
        assert crossings >= 8

    def test_identical_series_has_zero_crossings(self) -> None:
        idx = pd.bdate_range("2020-01-01", periods=50)
        s = pd.Series(np.linspace(1.0, 1.5, 50), index=idx)
        assert count_cumulative_return_crossings(s, s) == 0

    def test_sub_threshold_noise_does_not_count(self) -> None:
        """A tiny wobble that never clears min_amplitude should not
        register as a crossing (the whole point of the hysteresis band)."""
        idx = pd.bdate_range("2020-01-01", periods=200)
        rng = np.random.default_rng(3)
        a = pd.Series(1.0 + rng.normal(0, 0.005, 200), index=idx)
        b = pd.Series(1.0 + rng.normal(0, 0.005, 200), index=idx)
        assert count_cumulative_return_crossings(a, b, min_amplitude=0.03) == 0


class TestFindCrossingCointegratedCandidates:
    def test_excludes_tight_pair_despite_cointegration(self) -> None:
        panel = _tight_pair()
        out = find_crossing_cointegrated_candidates(
            panel,
            ["TIGHT_X", "TIGHT_Y"],
            min_corr=0.5,
            max_adf_pvalue=0.10,
            min_crossings=3,
            min_obs=100,
        )
        assert out == []

    def test_includes_oscillating_cointegrated_pair(self) -> None:
        panel = _oscillating_pair()
        out = find_crossing_cointegrated_candidates(
            panel,
            ["OSC_X", "OSC_Y"],
            min_corr=0.3,
            max_adf_pvalue=0.10,
            min_crossings=3,
            min_obs=100,
        )
        assert len(out) == 1
        assert out[0]["crossings"] >= 3


class TestRunPairUntilBroken:
    def test_stable_cointegration_does_not_stop_early(self) -> None:
        # Rolling-window ADF significance is genuinely noisy even for a
        # real cointegrated pair (see notebook 17's rolling-window ADF
        # instability for XOM/CVX) -- a couple of consecutive noisy months
        # is expected, not a sign the relationship broke. persistence_checks
        # needs to be high enough to ride that out; 4 (~2-3 months) is.
        panel = _oscillating_pair(n=700, seed=2, period_days=60)
        out = run_pair_until_broken(
            panel,
            symbol_y="OSC_Y",
            symbol_x="OSC_X",
            start=panel.index[260],
            max_end=panel.index[-1],
            hedge_window=60,
            zscore_window=20,
            transaction_cost=0.0,
            monitor_window=252,
            check_every_days=21,
            max_pvalue=0.10,
            persistence_checks=4,
        )
        assert out["stopped_early"] is False
        assert out["stop_date"] is None
        assert len(out["net_returns"]) > 0

    def test_breaking_relationship_stops_early(self) -> None:
        """Spread oscillates for the first ~half on a shared latent random
        walk, then the two legs detach into independent random walks
        (no longer cointegrated). Ample post-break runway is needed: a
        252d monitor window takes a while to fully "forget" the old
        cointegrated data once the break happens, and persistence_checks=4
        needs several consecutive post-break checks on top of that."""
        rng = np.random.default_rng(9)
        n = 900
        dates = pd.bdate_range("2016-01-01", periods=n, tz="America/New_York")
        mid = 400
        common = np.cumsum(rng.normal(0.0003, 0.012, mid))
        wave = 0.15 * np.sin(2 * np.pi * np.arange(mid) / 60)
        x_first = common - wave / 2 + rng.normal(0.0, 0.004, mid)
        y_first = common + wave / 2 + rng.normal(0.0, 0.004, mid)

        # From the midpoint on, each leg gets its OWN independent random walk.
        x_rest = x_first[-1] + np.cumsum(rng.normal(0.0, 0.015, n - mid))
        y_rest = y_first[-1] + np.cumsum(rng.normal(0.0, 0.015, n - mid))

        x = 100 * np.exp(np.concatenate([x_first, x_rest]))
        y = 100 * np.exp(np.concatenate([y_first, y_rest]))
        panel = pd.DataFrame({"BREAK_X": x, "BREAK_Y": y}, index=dates)

        out = run_pair_until_broken(
            panel,
            symbol_y="BREAK_Y",
            symbol_x="BREAK_X",
            start=panel.index[260],
            max_end=panel.index[-1],
            hedge_window=60,
            zscore_window=20,
            transaction_cost=0.0,
            monitor_window=252,
            check_every_days=21,
            max_pvalue=0.10,
            persistence_checks=4,
        )
        assert out["stopped_early"] is True
        assert out["stop_date"] is not None
        # Should stop sometime after the break, not immediately at `start`.
        assert out["stop_date"] > panel.index[mid]


class TestRunPairsPersistentIndex:
    def test_runs_end_to_end_on_synthetic_universe(self) -> None:
        tight = _tight_pair(n=700, seed=11)
        osc = _oscillating_pair(n=700, seed=12, period_days=60)
        panel = pd.concat([tight, osc], axis=1)
        sectors = _sectors_for(list(panel.columns))

        out = run_pairs_persistent_index(
            panel,
            sectors,
            sector_names=["Test Sector"],
            start=panel.index[0],
            end=panel.index[-1],
            formation_months=6,
            top_n_pairs=5,
            max_symbols_per_sector=12,
            min_corr=0.3,
            max_adf_pvalue=0.10,
            min_crossings=3,
            hedge_window=60,
            zscore_window=20,
            transaction_cost=0.0,
            monitor_window=120,
            check_every_days=21,
            max_pvalue=0.10,
            persistence_checks=2,
            min_formation_obs=100,
        )
        assert len(out["formations"]) >= 1
        selected_pairs = {(h["symbol_y"], h["symbol_x"]) for h in out["pair_history"]}
        selected_flat = {s for pair in selected_pairs for s in pair}
        # No TIGHT_* pair (GOOGL/GOOG-like) should ever be selected; some
        # OSC_* pairing should be.
        assert not any(s.startswith("TIGHT") for s in selected_flat)
        assert any(s.startswith("OSC") for s in selected_flat)

    def test_rejects_bad_params(self) -> None:
        tight = _tight_pair(n=200)
        sectors = _sectors_for(list(tight.columns))
        with pytest.raises(ValueError):
            run_pairs_persistent_index(
                tight,
                sectors,
                sector_names=["Test Sector"],
                start=tight.index[0],
                end=tight.index[-1],
                formation_months=1,
            )
        with pytest.raises(ValueError):
            run_pairs_persistent_index(
                tight,
                sectors,
                sector_names=["Test Sector"],
                start=tight.index[-1],
                end=tight.index[0],
            )
