"""HTTP-level tests for /run-pairs-index-backtest."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd

from api.routes.pairs_index import run_pairs_index_backtest
from api.schemas.pairs_index import PairsIndexBacktestRequest


def _panel_with_one_persistent_pair(n: int = 700, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n, tz="America/New_York")
    a = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.015, n)))
    b = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.015, n)))
    x = 100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n)))
    y = np.exp(1.2 * np.log(x) + rng.normal(0.0, 0.004, n))
    return pd.DataFrame({"NOISE_A": a, "NOISE_B": b, "PAIR_X": x, "PAIR_Y": y}, index=dates)


def _sectors() -> pd.DataFrame:
    symbols = ["NOISE_A", "NOISE_B", "PAIR_X", "PAIR_Y"]
    return pd.DataFrame(
        {
            "symbol": symbols,
            "sector": ["Test Sector"] * len(symbols),
            "quoteType": ["EQUITY"] * len(symbols),
        }
    )


def test_run_pairs_index_backtest_end_to_end() -> None:
    panel = _panel_with_one_persistent_pair()
    sectors = _sectors()
    req = PairsIndexBacktestRequest(
        sector_names=["Test Sector"],
        start_date=str(panel.index[0].date()),
        end_date=str(panel.index[-1].date()),
        formation_months=6,
        trading_months=3,
        top_n_pairs=2,
        hedge_window=40,
        zscore_window=15,
        use_adv=False,
        transaction_cost_bps=0.0,
    )
    with (
        patch("api.routes.pairs_index.get_prices", return_value=panel),
        patch("api.routes.pairs_index.get_sectors", return_value=sectors),
    ):
        resp = run_pairs_index_backtest(req)

    assert resp.total_days > 0
    assert len(resp.periods) >= 2
    assert set(resp.universe) == {"NOISE_A", "NOISE_B", "PAIR_X", "PAIR_Y"}
    selected = {(row.symbol_y, row.symbol_x) for p in resp.periods for row in p.selected_pairs}
    assert ("PAIR_X", "PAIR_Y") in selected
