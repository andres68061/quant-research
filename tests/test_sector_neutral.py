"""Tests for sector-neutral factor transforms and FMP commodity config."""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.data.commodities import COMMODITIES_CONFIG
from core.signals.sector_neutral import (
    combine_value_quality,
    demean_factor_within_sector,
    zscore_cross_section,
)


class TestCommodityConfig:
    def test_all_fmp_sourced(self) -> None:
        assert all(c["source"] == "fmp" for c in COMMODITIES_CONFIG.values())
        assert COMMODITIES_CONFIG["WTI"]["fmp_symbol"] == "CLUSD"
        assert COMMODITIES_CONFIG["GLD"]["fmp_symbol"] == "GLD"


class TestSectorNeutral:
    def test_demean_within_sector_zeros_group_mean(self) -> None:
        dates = pd.to_datetime(["2020-01-02", "2020-01-02", "2020-01-02", "2020-01-02"])
        symbols = ["A", "B", "C", "D"]
        idx = pd.MultiIndex.from_arrays([dates, symbols], names=["date", "symbol"])
        factor = pd.Series([10.0, 12.0, 20.0, 30.0], index=idx, name="ey")
        sectors = pd.Series({"A": "Tech", "B": "Tech", "C": "Health", "D": "Health"})
        demeaned = demean_factor_within_sector(factor, sectors)
        assert abs(float(demeaned.loc[(dates[0], "A")]) + float(demeaned.loc[(dates[0], "B")])) < 1e-9
        assert abs(float(demeaned.loc[(dates[0], "C")]) + float(demeaned.loc[(dates[0], "D")])) < 1e-9

    def test_zscore_unit_variance(self) -> None:
        date = pd.Timestamp("2020-01-02")
        symbols = [f"S{i}" for i in range(10)]
        idx = pd.MultiIndex.from_product([[date], symbols], names=["date", "symbol"])
        values = pd.Series(np.linspace(1.0, 10.0, 10), index=idx)
        z = zscore_cross_section(values)
        assert abs(float(z.mean())) < 1e-9
        assert abs(float(z.std(ddof=0)) - 1.0) < 1e-9

    def test_value_quality_composite_shape(self) -> None:
        date = pd.Timestamp("2020-01-02")
        symbols = [f"S{i}" for i in range(8)]
        idx = pd.MultiIndex.from_product([[date], symbols], names=["date", "symbol"])
        ey = pd.Series(np.linspace(0.02, 0.10, 8), index=idx)
        roe = pd.Series(np.linspace(0.05, 0.20, 8), index=idx)
        sectors = pd.Series({s: ("Tech" if i < 4 else "Health") for i, s in enumerate(symbols)})
        composite = combine_value_quality(ey, roe, symbol_to_sector=sectors, sector_neutral=True)
        assert composite.name == "value_quality"
        assert composite.notna().sum() == 8
