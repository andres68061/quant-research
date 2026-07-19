"""HTTP-level tests for /portfolio/walk-forward-optimize."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException

from api.routes.portfolio import WalkForwardOptimizeRequest, walk_forward_optimize


def _two_asset_panel(n: int = 900, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-02", periods=n, tz="America/New_York")
    hi = 100 * np.exp(np.cumsum(rng.normal(0.0006, 0.015, n)))
    lo = 100 * np.exp(np.cumsum(rng.normal(0.0, 0.005, n)))
    return pd.DataFrame({"HI": hi, "LO": lo}, index=idx)


def test_walk_forward_optimize_end_to_end() -> None:
    panel = _two_asset_panel()
    req = WalkForwardOptimizeRequest(
        symbols=["HI", "LO"],
        start_date=str(panel.index[300].date()),
        end_date=str(panel.index[-1].date()),
        lookback_months=12,
        rebalance_months=6,
    )
    with patch("api.routes.portfolio.get_prices", return_value=panel):
        out = walk_forward_optimize(req)

    assert out["total_days"] > 0
    assert len(out["periods"]) >= 3
    for p in out["periods"]:
        assert abs(sum(p["weights"].values()) - 1.0) < 1e-6
    assert "sharpe_ratio" in out["metrics"]
    assert "pain_ratio" in out["metrics"]


def test_walk_forward_optimize_rejects_unknown_symbol() -> None:
    panel = _two_asset_panel()
    req = WalkForwardOptimizeRequest(
        symbols=["HI", "NOPE"],
        start_date=str(panel.index[300].date()),
    )
    with patch("api.routes.portfolio.get_prices", return_value=panel):
        with pytest.raises(HTTPException):
            walk_forward_optimize(req)
