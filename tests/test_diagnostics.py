"""Tests for backtest diagnostics assembler."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.metrics.diagnostics import build_backtest_diagnostics


def test_build_backtest_diagnostics_shape() -> None:
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2020-01-01", periods=200, tz="America/New_York")
    returns = pd.Series(rng.normal(0.0004, 0.01, size=len(idx)), index=idx)
    payload = build_backtest_diagnostics(returns, rolling_window=63, histogram_bins=20)

    assert payload["rolling_window"] == 63
    assert payload["var_confidence"] == 95
    assert len(payload["rolling"]) > 0
    assert {"date", "sharpe", "sortino", "volatility"} <= set(payload["rolling"][0])
    assert len(payload["drawdown"]) == len(returns)
    assert len(payload["histogram"]["counts"]) == 20
    assert len(payload["histogram"]["bin_edges"]) == 21
    for method in ("historical", "parametric", "monte_carlo"):
        assert "var" in payload["var"][method]
        assert "cvar" in payload["var"][method]


def test_build_backtest_diagnostics_empty() -> None:
    payload = build_backtest_diagnostics(pd.Series(dtype=float))
    assert payload["rolling"] == []
    assert payload["drawdown"] == []
    assert payload["var"] == {}
