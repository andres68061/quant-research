"""Tests for the rolling multi-pair stat-arb index."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.strategies.pairs_index import build_pairs_universe, run_pairs_stat_arb_index


def _panel_with_one_persistent_pair(n: int = 900, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-01", periods=n, tz="America/New_York")
    a = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.015, n)))
    b = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.015, n)))
    c = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.015, n)))
    x = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n)))
    y = np.exp(1.2 * np.log(x) + rng.normal(0.0, 0.004, n))
    return pd.DataFrame(
        {"NOISE_A": a, "NOISE_B": b, "NOISE_C": c, "PAIR_X": x, "PAIR_Y": y},
        index=dates,
    )


def _sectors_one_group() -> pd.DataFrame:
    symbols = ["NOISE_A", "NOISE_B", "NOISE_C", "PAIR_X", "PAIR_Y"]
    return pd.DataFrame(
        {
            "symbol": symbols,
            "sector": ["Test Sector"] * len(symbols),
            "quoteType": ["EQUITY"] * len(symbols),
        }
    )


def test_build_pairs_universe_pools_and_dedups() -> None:
    sectors = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC"],
            "sector": ["Energy", "Energy", "Tech"],
            "quoteType": ["EQUITY", "EQUITY", "EQUITY"],
        }
    )
    out = build_pairs_universe(
        sectors, ["Energy", "Tech"], price_columns=["AAA", "BBB", "CCC"], max_symbols_per_sector=10
    )
    assert set(out) == {"AAA", "BBB", "CCC"}
    assert len(out) == len(set(out))


def test_run_pairs_stat_arb_index_selects_persistent_pair_and_rolls_forward() -> None:
    prices = _panel_with_one_persistent_pair()
    sectors = _sectors_one_group()

    out = run_pairs_stat_arb_index(
        prices,
        sectors,
        sector_names=["Test Sector"],
        start=prices.index[0],
        end=prices.index[-1],
        formation_months=6,
        trading_months=3,
        top_n_pairs=2,
        hedge_window=60,
        zscore_window=20,
        transaction_cost=0.0,
        min_formation_obs=100,
    )

    assert len(out["periods"]) >= 2
    assert out["net_returns"].index.is_monotonic_increasing
    assert not out["net_returns"].index.has_duplicates

    # The persistent cointegrated pair should be selected in at least one period.
    selected_pairs = {
        (row["symbol_y"], row["symbol_x"]) for p in out["periods"] for row in p["selected_pairs"]
    }
    assert ("PAIR_X", "PAIR_Y") in selected_pairs

    # No leakage: each period's trading window starts exactly at formation end.
    for p in out["periods"]:
        assert p["formation_end"] == p["trading_start"]
        assert pd.Timestamp(p["trading_start"]) <= pd.Timestamp(p["trading_end"])
        assert p["n_pairs_selected"] <= 2

    # The basket re-forms every trading_months (3): consecutive trading
    # windows must be back-to-back with no gap and no overlap.
    for prev, nxt in zip(out["periods"], out["periods"][1:], strict=False):
        assert prev["trading_end"] == nxt["trading_start"]


def test_run_pairs_stat_arb_index_rejects_bad_params() -> None:
    prices = _panel_with_one_persistent_pair(n=50)
    sectors = _sectors_one_group()
    with pytest.raises(ValueError):
        run_pairs_stat_arb_index(
            prices,
            sectors,
            sector_names=["Test Sector"],
            start=prices.index[0],
            end=prices.index[-1],
            formation_months=1,
        )
    with pytest.raises(ValueError):
        run_pairs_stat_arb_index(
            prices,
            sectors,
            sector_names=["Test Sector"],
            start=prices.index[-1],
            end=prices.index[0],
        )


def test_run_pairs_stat_arb_index_handles_no_candidates_gracefully() -> None:
    """A sector with only one symbol can't form a pair; index should be flat, not raise."""
    rng = np.random.default_rng(11)
    n = 400
    dates = pd.bdate_range("2018-01-01", periods=n, tz="America/New_York")
    prices = pd.DataFrame(
        {"AAA": 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))},
        index=dates,
    )
    sectors = pd.DataFrame({"symbol": ["AAA"], "sector": ["Test"], "quoteType": ["EQUITY"]})
    out = run_pairs_stat_arb_index(
        prices,
        sectors,
        sector_names=["Test"],
        start=prices.index[0],
        end=prices.index[-1],
        formation_months=4,
        trading_months=2,
        min_formation_obs=60,
    )
    assert len(out["periods"]) >= 1
    assert (out["net_returns"].fillna(0.0) == 0.0).all()
    assert all(p["n_pairs_selected"] == 0 for p in out["periods"])
