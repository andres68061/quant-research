"""Pairs route must slice tz-aware price panels without naive Timestamp errors.

Regression test for a bug where `/run-pairs-backtest` built tz-naive
`pd.Timestamp` bounds from `start_date`/`end_date` and compared them against
the tz-aware (America/New_York) price panel index, raising a `TypeError` on
every request that supplied a date range (i.e. every request the frontend
actually sends).
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd

from api.routes.pairs import run_pairs_backtest
from api.schemas.pairs import PairsBacktestRequest


def _cointegrated_panel(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2020-01-02", periods=n, tz="America/New_York")
    common = np.cumsum(rng.normal(0, 0.01, n))
    noise = rng.normal(0, 0.02, n)
    log_x = common
    log_y = common + noise
    return pd.DataFrame({"AAA": 100 * np.exp(log_y), "BBB": 100 * np.exp(log_x)}, index=idx)


def test_run_pairs_backtest_accepts_date_range_on_tz_aware_panel() -> None:
    panel = _cointegrated_panel()
    req = PairsBacktestRequest(
        symbol_y="AAA",
        symbol_x="BBB",
        start_date="2020-03-01",
        end_date="2021-06-01",
        hedge_window=60,
        zscore_window=20,
    )
    with patch("api.routes.pairs.get_prices", return_value=panel):
        resp = run_pairs_backtest(req)

    assert resp.total_days > 0
    assert resp.diagnostics.symbol_y == "AAA"
    assert resp.diagnostics.symbol_x == "BBB"
    assert resp.is_held_out is False
    assert resp.train_diagnostics is None


def test_run_pairs_backtest_train_frac_returns_held_out_slice_only() -> None:
    panel = _cointegrated_panel(n=700)
    req = PairsBacktestRequest(
        symbol_y="AAA",
        symbol_x="BBB",
        start_date="2020-01-02",
        end_date=str(panel.index[-1].date()),
        hedge_window=60,
        zscore_window=20,
        train_frac=0.6,
    )
    with patch("api.routes.pairs.get_prices", return_value=panel):
        resp = run_pairs_backtest(req)

    assert resp.is_held_out is True
    assert resp.train_diagnostics is not None
    assert resp.train_start_date is not None
    assert resp.held_out_start_date is not None
    assert resp.train_end_date < resp.held_out_start_date
    assert resp.total_days > 0
