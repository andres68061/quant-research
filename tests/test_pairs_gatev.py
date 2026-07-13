"""Tests for Gatev distance pairs formation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.strategies.pairs_gatev import (
    form_pairs_by_distance,
    normalize_price_index,
    resolve_liquid_symbols,
    screen_pairs_gatev,
)


def test_normalize_price_index_starts_at_one() -> None:
    dates = pd.bdate_range("2020-01-01", periods=5, tz="America/New_York")
    prices = pd.DataFrame({"A": [50.0, 55.0, 60.0, 58.0, 62.0]}, index=dates)
    norm = normalize_price_index(prices)
    assert abs(float(norm["A"].iloc[0]) - 1.0) < 1e-12
    assert abs(float(norm["A"].iloc[1]) - 1.1) < 1e-12


def test_form_pairs_ranks_tracking_pair_first() -> None:
    rng = np.random.default_rng(0)
    n = 300
    dates = pd.bdate_range("2018-01-01", periods=n, tz="America/New_York")
    base = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n)))
    twin = base * (1.0 + rng.normal(0.0, 0.002, n))
    noise = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.02, n)))
    prices = pd.DataFrame({"CLOSE": twin, "BASE": base, "NOISE": noise}, index=dates)
    ranked = form_pairs_by_distance(prices, ["CLOSE", "BASE", "NOISE"], top_n=3)
    assert ranked
    top = {ranked[0]["symbol_y"], ranked[0]["symbol_x"]}
    assert top == {"CLOSE", "BASE"}


def test_resolve_liquid_symbols_uses_adv() -> None:
    sectors = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB", "CCC"],
            "sector": ["Energy", "Energy", "Energy"],
            "quoteType": ["EQUITY", "EQUITY", "EQUITY"],
        }
    )
    dates = pd.bdate_range("2020-01-01", periods=3, tz="America/New_York")
    adv = pd.DataFrame(
        {"AAA": [1e6, 1e6, 1e6], "BBB": [9e9, 9e9, 9e9], "CCC": [2e6, 2e6, 2e6]},
        index=dates,
    )
    out = resolve_liquid_symbols(
        sectors,
        "Energy",
        price_columns=["AAA", "BBB", "CCC"],
        dollar_adv=adv,
        max_symbols=2,
    )
    assert out == ["BBB", "CCC"]


def test_screen_pairs_gatev_returns_oos_rows() -> None:
    rng = np.random.default_rng(2)
    n = 800
    dates = pd.bdate_range("2016-01-01", periods=n, tz="America/New_York")
    base = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n)))
    twin = base * (1.0 + rng.normal(0.0, 0.003, n))
    a = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.015, n)))
    b = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.015, n)))
    prices = pd.DataFrame(
        {"BASE": base, "TWIN": twin, "A": a, "B": b},
        index=dates,
    )
    out = screen_pairs_gatev(
        prices,
        ["BASE", "TWIN", "A", "B"],
        formation_frac=0.67,
        top_n=3,
        hedge_window=120,
        zscore_window=40,
        transaction_cost=0.0,
    )
    assert out["method"] == "gatev"
    assert out["n_pairs_passed_train"] >= 1
    assert out["results"]
    formed = {(r["symbol_y"], r["symbol_x"]) for r in out["results"]}
    assert ("BASE", "TWIN") in formed or ("TWIN", "BASE") in formed
