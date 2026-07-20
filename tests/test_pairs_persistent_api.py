"""HTTP-level tests for /run-pairs-persistent-backtest."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException

from api.routes.pairs_persistent import run_pairs_persistent_backtest
from api.schemas.pairs_persistent import PairsPersistentBacktestRequest


def _panel_with_oscillating_pair(n: int = 900, seed: int = 12) -> pd.DataFrame:
    """One genuinely oscillating cointegrated pair plus two noise names.

    Direction-symmetric construction (shared latent walk, wedge split +/-)
    so Engle-Granger passes regardless of which leg is dependent.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2017-01-01", periods=n, tz="America/New_York")
    common = np.cumsum(rng.normal(0.0003, 0.012, n))
    wave = 0.15 * np.sin(2 * np.pi * np.arange(n) / 60)
    x = 100 * np.exp(common - wave / 2 + rng.normal(0.0, 0.004, n))
    y = 100 * np.exp(common + wave / 2 + rng.normal(0.0, 0.004, n))
    a = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.015, n)))
    b = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.015, n)))
    return pd.DataFrame({"OSC_X": x, "OSC_Y": y, "NOISE_A": a, "NOISE_B": b}, index=dates)


def _sectors(symbols: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": symbols,
            "sector": ["Test Sector"] * len(symbols),
            "quoteType": ["EQUITY"] * len(symbols),
        }
    )


def test_run_pairs_persistent_backtest_end_to_end() -> None:
    panel = _panel_with_oscillating_pair()
    req = PairsPersistentBacktestRequest(
        sector_names=["Test Sector"],
        start_date=str(panel.index[0].date()),
        end_date=str(panel.index[-1].date()),
        formation_months=6,
        rescreen_months=6,
        top_n_pairs=3,
        use_adv=False,
        max_adf_pvalue=0.10,
        min_crossings=3,
        hedge_window=60,
        zscore_window=20,
        transaction_cost_bps=0.0,
        monitor_window=120,
        persistence_checks=2,
    )
    with (
        patch("api.routes.pairs_persistent.get_prices", return_value=panel),
        patch(
            "api.routes.pairs_persistent.get_sectors", return_value=_sectors(list(panel.columns))
        ),
    ):
        resp = run_pairs_persistent_backtest(req)

    assert resp.total_days > 0
    assert len(resp.screens) >= 1
    assert len(resp.pair_history) >= 1
    pairs_traded = {(h.symbol_y, h.symbol_x) for h in resp.pair_history}
    assert any("OSC" in y and "OSC" in x for y, x in pairs_traded)
    assert all(h.formation_crossings >= 3 for h in resp.pair_history)


def test_run_pairs_persistent_backtest_freeze_hedge_accepted() -> None:
    panel = _panel_with_oscillating_pair()
    req = PairsPersistentBacktestRequest(
        sector_names=["Test Sector"],
        start_date=str(panel.index[0].date()),
        end_date=str(panel.index[-1].date()),
        formation_months=6,
        rescreen_months=6,
        top_n_pairs=3,
        use_adv=False,
        max_adf_pvalue=0.10,
        hedge_window=60,
        zscore_window=20,
        monitor_window=120,
        persistence_checks=2,
        freeze_hedge_in_trade=True,
    )
    with (
        patch("api.routes.pairs_persistent.get_prices", return_value=panel),
        patch(
            "api.routes.pairs_persistent.get_sectors", return_value=_sectors(list(panel.columns))
        ),
    ):
        resp = run_pairs_persistent_backtest(req)

    assert resp.total_days > 0


def test_run_pairs_persistent_backtest_no_data_returns_400() -> None:
    """A window with no tradeable pairs must surface a 400, not a silent empty."""
    rng = np.random.default_rng(3)
    n = 400
    dates = pd.bdate_range("2020-01-01", periods=n, tz="America/New_York")
    panel = pd.DataFrame(
        {
            "NOISE_A": 100 * np.exp(np.cumsum(rng.normal(0.0, 0.02, n))),
            "NOISE_B": 100 * np.exp(np.cumsum(rng.normal(0.0, 0.02, n))),
        },
        index=dates,
    )
    req = PairsPersistentBacktestRequest(
        sector_names=["Test Sector"],
        start_date=str(panel.index[0].date()),
        end_date=str(panel.index[-1].date()),
        formation_months=6,
        rescreen_months=6,
        use_adv=False,
        hedge_window=60,
        zscore_window=20,
        monitor_window=120,
    )
    with (
        patch("api.routes.pairs_persistent.get_prices", return_value=panel),
        patch(
            "api.routes.pairs_persistent.get_sectors", return_value=_sectors(list(panel.columns))
        ),
    ):
        with pytest.raises(HTTPException) as exc_info:
            run_pairs_persistent_backtest(req)

    assert exc_info.value.status_code == 400
