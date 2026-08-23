"""Tests for core.index.cid1_study."""

import numpy as np
import pandas as pd
import pytest

from core.exceptions import DataSchemaError
from core.index.cid1_study import compute_forward_returns, run_cid1_relevance_study
from core.index.top500 import build_quarterly_rebalance_dates

N_SYMBOLS = 60


def _make_universe() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Synthetic panel where higher-numbered symbols have persistently higher
    drift and smoother paths — so trailing Cid-1 should predict forward returns."""
    rng = np.random.default_rng(7)
    days = pd.bdate_range("2020-01-01", "2023-12-31", tz="America/New_York")
    symbols = [f"S{i:02d}" for i in range(N_SYMBOLS)]
    drift = np.linspace(-0.001, 0.002, N_SYMBOLS)
    vol = np.linspace(0.012, 0.004, N_SYMBOLS)
    returns = rng.normal(drift, vol, size=(len(days), N_SYMBOLS))
    prices = pd.DataFrame(100.0 * np.cumprod(1.0 + returns, axis=0), index=days, columns=symbols)
    caps = prices * 1e9
    rebalances = build_quarterly_rebalance_dates(prices.index)
    factors = pd.DataFrame(
        {
            "mom_12_1": rng.normal(size=len(rebalances) * N_SYMBOLS),
            "vol_60d": rng.uniform(0.1, 0.5, size=len(rebalances) * N_SYMBOLS),
            "log_market_cap": rng.normal(23, 1, size=len(rebalances) * N_SYMBOLS),
        },
        index=pd.MultiIndex.from_product([rebalances, symbols], names=["date", "symbol"]),
    )
    return prices, caps, factors


def test_compute_forward_returns_windows_do_not_overlap_signal() -> None:
    prices, _, _ = _make_universe()
    rebalances = build_quarterly_rebalance_dates(prices.index)
    fwd = compute_forward_returns(prices, rebalances, execution_lag_days=1)

    # One row per *executable* rebalance except the last one (which has no
    # forward period). The final quarter-end here is the last trading day,
    # so it cannot execute with a 1-day lag and is dropped too.
    executable = [d for d in rebalances if prices.index.get_loc(d) + 1 < len(prices.index)]
    assert len(fwd) == len(executable) - 1
    # Forward return runs exec(t) -> exec(t+1), strictly after the selection date.
    first, second = rebalances[0], rebalances[1]
    e1 = prices.index[prices.index.get_loc(first) + 1]
    e2 = prices.index[prices.index.get_loc(second) + 1]
    expected = prices.loc[e2, "S10"] / prices.loc[e1, "S10"] - 1.0
    assert fwd.loc[first, "S10"] == pytest.approx(expected)


def test_run_cid1_relevance_study_recovers_planted_signal() -> None:
    prices, caps, factors = _make_universe()
    result = run_cid1_relevance_study(
        prices,
        caps,
        factors,
        start=pd.Timestamp("2020-06-01", tz="America/New_York"),
        top_n=N_SYMBOLS,
        window_days=60,
        n_quantiles=3,
    )
    per_date = result["per_date"]
    assert not per_date.empty
    # Planted structure: persistent drift ordering => positive IC and persistence.
    assert result["ic_summary"]["mean"] > 0.1
    assert result["persistence_summary"]["mean"] > 0.3
    assert result["quantile_spread_summary"]["mean"] > 0.0
    # Fama-MacBeth picks up cid1 even with (noise) controls present.
    assert result["fama_macbeth"]["univariate_cid1"]["mean"] > 0.0
    assert result["fama_macbeth"]["multivariate"]["cid1"]["mean"] > 0.0
    # Quantile table covers every bucket.
    assert list(result["quantile_mean_returns"].columns) == [0, 1, 2]
    # Config echoes what ran.
    assert result["config"]["top_n"] == N_SYMBOLS
    assert result["config"]["controls"] == ["mom_12_1", "vol_60d", "log_market_cap"]


def test_run_cid1_relevance_study_too_few_rebalances_raises() -> None:
    prices, caps, factors = _make_universe()
    with pytest.raises(DataSchemaError):
        run_cid1_relevance_study(
            prices,
            caps,
            factors,
            start=pd.Timestamp("2023-01-01", tz="America/New_York"),
            top_n=N_SYMBOLS,
            window_days=60,
        )
