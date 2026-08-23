"""Tests for lazy per-column factor access and the API universe policy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.data.api_universe import MIN_TRADING_DAYS, resolve_policy, select_api_symbols
from core.data.factor_store import FactorStore, describe_factor_source


def _panel(tmp_path, name: str, columns: dict[str, float], symbols=("AAA", "BBB")) -> None:
    index = pd.MultiIndex.from_product(
        [pd.bdate_range("2024-01-02", periods=5, tz="America/New_York", name="date"), symbols],
        names=["date", "symbol"],
    )
    frame = pd.DataFrame({k: np.full(len(index), v) for k, v in columns.items()}, index=index)
    frame.to_parquet(tmp_path / name)


class TestFactorStoreScan:
    def test_indexes_columns_without_reading_data(self, tmp_path) -> None:
        _panel(tmp_path, "factors_price.parquet", {"mom_12_1": 1.0, "vol_60d": 2.0})
        _panel(tmp_path, "factors_fundamental.parquet", {"roe": 3.0})

        store = FactorStore(tmp_path)
        assert store.available_factors == ["mom_12_1", "roe", "vol_60d"]
        assert store.panel_of("roe") == "factors_fundamental.parquet"

    def test_name_collision_resolved_by_priority_not_silently(self, tmp_path, caplog) -> None:
        # factors_all has priority over factors_fundamental.
        _panel(tmp_path, "factors_all.parquet", {"roe": 1.0})
        _panel(tmp_path, "factors_fundamental.parquet", {"roe": 9.0})

        with caplog.at_level("WARNING"):
            store = FactorStore(tmp_path)
        assert store.panel_of("roe") == "factors_all.parquet"
        assert any("appears in both" in record.message for record in caplog.records)

    def test_unreadable_panel_is_skipped_not_fatal(self, tmp_path) -> None:
        _panel(tmp_path, "factors_price.parquet", {"mom_12_1": 1.0})
        (tmp_path / "factors_fundamental.parquet").write_text("not a parquet file")

        store = FactorStore(tmp_path)
        assert "mom_12_1" in store.available_factors

    def test_missing_directory_yields_no_factors(self, tmp_path) -> None:
        assert FactorStore(tmp_path / "nothing").available_factors == []

    def test_panel_priority_override_scopes_the_store(self, tmp_path) -> None:
        # A caller that must not touch factors_all (e.g. mid-rebuild) scopes the scan.
        _panel(tmp_path, "factors_all.parquet", {"roe": 1.0})
        _panel(tmp_path, "factors_fundamental.parquet", {"roe": 9.0})

        store = FactorStore(tmp_path, panel_priority=("factors_fundamental.parquet",))
        assert store.panel_of("roe") == "factors_fundamental.parquet"
        assert store.available_factors == ["roe"]


class TestFactorStoreLoad:
    def test_loads_single_column_frame(self, tmp_path) -> None:
        _panel(tmp_path, "factors_price.parquet", {"mom_12_1": 1.5, "vol_60d": 2.0})
        store = FactorStore(tmp_path)
        FactorStore.cache_clear()

        frame = store.load_factor("mom_12_1")
        assert list(frame.columns) == ["mom_12_1"]
        assert frame.index.names == ["date", "symbol"]
        assert (frame["mom_12_1"] == 1.5).all()

    def test_universe_restriction_applied(self, tmp_path) -> None:
        _panel(tmp_path, "factors_price.parquet", {"mom_12_1": 1.0}, symbols=("AAA", "BBB", "CCC"))
        store = FactorStore(tmp_path, symbols={"AAA"})
        FactorStore.cache_clear()

        frame = store.load_factor("mom_12_1")
        assert set(frame.index.get_level_values("symbol")) == {"AAA"}

    def test_unknown_factor_raises(self, tmp_path) -> None:
        _panel(tmp_path, "factors_price.parquet", {"mom_12_1": 1.0})
        with pytest.raises(KeyError):
            FactorStore(tmp_path).load_factor("nope")

    def test_load_factors_joins_across_panels(self, tmp_path) -> None:
        _panel(tmp_path, "factors_price.parquet", {"mom_12_1": 1.0})
        _panel(tmp_path, "factors_fundamental.parquet", {"roe": 2.0})
        store = FactorStore(tmp_path)
        FactorStore.cache_clear()

        combined = store.load_factors(["mom_12_1", "roe"])
        assert list(combined.columns) == ["mom_12_1", "roe"]
        assert combined.notna().all().all()

    def test_load_factors_rejects_unknown(self, tmp_path) -> None:
        _panel(tmp_path, "factors_price.parquet", {"mom_12_1": 1.0})
        with pytest.raises(KeyError):
            FactorStore(tmp_path).load_factors(["mom_12_1", "nope"])

    def test_describe_reports_owning_panel(self, tmp_path) -> None:
        _panel(tmp_path, "factors_price.parquet", {"mom_12_1": 1.0})
        described = describe_factor_source(FactorStore(tmp_path))
        assert described == [{"factor": "mom_12_1", "panel": "factors_price.parquet"}]


class TestApiUniversePolicy:
    def _prices(self, n_days: int = 400) -> pd.DataFrame:
        index = pd.bdate_range("2023-01-02", periods=n_days, tz="America/New_York", name="date")
        frame = pd.DataFrame(
            {
                "LONG": np.linspace(10, 20, n_days),
                "SHORTLIVED": [np.nan] * (n_days - 50) + list(np.linspace(5, 6, 50)),
                "^GSPC": np.linspace(100, 200, n_days),
            },
            index=index,
        )
        return frame

    def _sectors(self, tmp_path) -> "pd.Path":  # type: ignore[name-defined]
        frame = pd.DataFrame(
            {
                "symbol": ["LONG", "SHORTLIVED", "SHELL"],
                "sector": ["Technology", "Healthcare", "Financial Services"],
                "industry": ["Software", "Biotechnology", "Shell Companies"],
            }
        )
        path = tmp_path / "sectors.parquet"
        frame.to_parquet(path, index=False)
        return path

    def test_research_policy_drops_short_history_keeps_benchmarks(self, tmp_path) -> None:
        symbols, disclosure = select_api_symbols(
            self._prices(), policy="research", sectors_path=self._sectors(tmp_path)
        )
        assert "LONG" in symbols
        assert "SHORTLIVED" not in symbols  # only 50 days < MIN_TRADING_DAYS
        assert "^GSPC" in symbols  # benchmark always kept — beta needs it
        assert disclosure["policy"] == "research"
        assert disclosure["loaded_symbols"] == len(symbols)

    def test_disclosure_records_every_step(self, tmp_path) -> None:
        _, disclosure = select_api_symbols(
            self._prices(), policy="research", sectors_path=self._sectors(tmp_path)
        )
        steps = [step["step"] for step in disclosure["steps"]]
        assert any("shells" in s for s in steps)
        assert any(str(MIN_TRADING_DAYS) in s for s in steps)
        assert disclosure["reason"]

    def test_full_policy_loads_everything(self, tmp_path) -> None:
        prices = self._prices()
        symbols, disclosure = select_api_symbols(prices, policy="full")
        assert len(symbols) == prices.shape[1]
        assert disclosure["steps"] == []

    def test_unknown_policy_falls_back_without_raising(self) -> None:
        assert resolve_policy("nonsense") == "research"

    def test_env_var_respected(self, monkeypatch) -> None:
        monkeypatch.setenv("API_UNIVERSE", "full")
        assert resolve_policy() == "full"

    def test_sp500_policy_without_table_falls_back(self, tmp_path) -> None:
        """A missing membership table must not silently load the whole market."""
        symbols, disclosure = select_api_symbols(
            self._prices(),
            policy="sp500",
            sectors_path=self._sectors(tmp_path),
            membership_path=tmp_path / "missing.parquet",
        )
        assert disclosure["policy"] == "research"
        assert "LONG" in symbols
