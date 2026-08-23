"""Tests for the intraday loader's measured adjustment semantics.

The vendor returns intraday history split-adjusted as of the request date
(measured 2026-08-11 on AAPL/NVDA/TSLA). The loader must therefore NOT blanket-
adjust — it detects which splits are actually missing from the stored snapshot
and repairs only those.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from core.data.fmp.intraday import detect_unapplied_splits, load_intraday_bars

TZ = "America/New_York"


def _bars(adjusted_for_split: bool, split_ratio: float = 4.0) -> pd.DataFrame:
    """Hourly bars around a split on 2020-08-31; price ~flat in adjusted terms."""
    index = pd.date_range("2020-08-25 10:00", periods=60, freq="6h", tz=TZ, name="date")
    index = index[index.dayofweek < 5]
    close = pd.Series(100.0, index=index)
    if not adjusted_for_split:
        close[index < pd.Timestamp("2020-08-31", tz=TZ)] *= split_ratio
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1000,
        }
    )


def _splits(ratio: float = 4.0) -> pd.DataFrame:
    return pd.DataFrame({"date": ["2020-08-31"], "numerator": [ratio], "denominator": [1]})


class TestDetectUnappliedSplits:
    def test_already_adjusted_data_detects_nothing(self) -> None:
        detected = detect_unapplied_splits(_bars(adjusted_for_split=True), _splits())
        assert detected.empty

    def test_stale_snapshot_detects_the_split(self) -> None:
        detected = detect_unapplied_splits(_bars(adjusted_for_split=False), _splits())
        assert len(detected) == 1

    def test_tiny_ratio_is_refused_not_guessed(self) -> None:
        """A 1.1:1 'split' is indistinguishable from a market move; never auto-apply."""
        bars = _bars(adjusted_for_split=False, split_ratio=1.1)
        detected = detect_unapplied_splits(bars, _splits(ratio=1.1))
        assert detected.empty

    def test_empty_inputs(self) -> None:
        assert detect_unapplied_splits(pd.DataFrame(), _splits()).empty


class TestLoadIntradayBars:
    def _write_layout(self, tmp_path, bars: pd.DataFrame, splits: pd.DataFrame) -> tuple:
        intraday_dir = tmp_path / "intraday"
        splits_dir = tmp_path / "splits"
        symbol_dir = intraday_dir / "1hour" / "TEST"
        symbol_dir.mkdir(parents=True)
        splits_dir.mkdir()
        bars.to_parquet(symbol_dir / "2020.parquet")
        splits.to_parquet(splits_dir / "TEST.parquet")
        return intraday_dir, splits_dir

    def test_adjusted_snapshot_is_returned_untouched(self, tmp_path) -> None:
        bars = _bars(adjusted_for_split=True)
        intraday_dir, splits_dir = self._write_layout(tmp_path, bars, _splits())
        loaded = load_intraday_bars(
            "TEST", "1hour", intraday_dir=intraday_dir, splits_dir=splits_dir
        )
        assert np.allclose(loaded["close"], bars["close"])

    def test_stale_snapshot_is_repaired(self, tmp_path) -> None:
        bars = _bars(adjusted_for_split=False)
        intraday_dir, splits_dir = self._write_layout(tmp_path, bars, _splits())
        loaded = load_intraday_bars(
            "TEST", "1hour", intraday_dir=intraday_dir, splits_dir=splits_dir
        )
        # After repair the series is continuous (~100 throughout).
        assert loaded["close"].max() / loaded["close"].min() < 1.1

    def test_missing_symbol_returns_typed_empty(self, tmp_path) -> None:
        loaded = load_intraday_bars(
            "NOPE", "1hour", intraday_dir=tmp_path / "x", splits_dir=tmp_path / "y"
        )
        assert loaded.empty
        assert list(loaded.columns) == ["open", "high", "low", "close", "volume"]
