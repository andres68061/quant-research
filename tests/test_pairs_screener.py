"""Tests for walk-forward pairs screener."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.strategies.pairs_screener import resolve_sector_symbols, screen_pairs_walk_forward


def _panel_with_one_cointegrated_pair(n: int = 900, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-01", periods=n, tz="America/New_York")
    # Independent noise names.
    a = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.015, n)))
    b = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.015, n)))
    c = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.015, n)))
    # Cointegrated: Y ≈ 1.2 X + stationary noise.
    x = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n)))
    y = np.exp(1.2 * np.log(x) + rng.normal(0.0, 0.004, n))
    return pd.DataFrame(
        {"NOISE_A": a, "NOISE_B": b, "NOISE_C": c, "PAIR_X": x, "PAIR_Y": y},
        index=dates,
    )


def test_resolve_sector_symbols_caps_and_filters() -> None:
    sectors = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC", "DDD"],
            "sector": ["Energy", "Energy", "Tech", "Energy"],
            "quoteType": ["EQUITY", "EQUITY", "EQUITY", "EQUITY"],
        }
    )
    out = resolve_sector_symbols(
        sectors,
        "Energy",
        price_columns=["AAA", "BBB", "ZZZ"],
        max_symbols=10,
    )
    assert out == ["AAA", "BBB"]


def test_screen_ranks_cointegrated_pair_when_oos_works() -> None:
    prices = _panel_with_one_cointegrated_pair()
    out = screen_pairs_walk_forward(
        prices,
        ["NOISE_A", "NOISE_B", "PAIR_X", "PAIR_Y"],
        train_frac=0.6,
        min_train_corr=0.3,
        max_train_adf_pvalue=0.10,
        max_oos_backtests=10,
        hedge_window=120,
        zscore_window=40,
        transaction_cost=0.0,
    )
    assert out["n_pairs_tested"] == 6
    assert out["n_pairs_passed_train"] >= 1
    assert out["results"], "expected at least one OOS-scored pair"
    # The synthetic cointegrated pair should appear among train passers.
    train_pairs = {(r["symbol_y"], r["symbol_x"]) for r in out["results"]}
    assert ("PAIR_Y", "PAIR_X") in train_pairs or ("PAIR_X", "PAIR_Y") in train_pairs
