"""Tests for core.data.vendors.commodities — incremental panel updates."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from core.data.vendors import commodities as mod


def _panel(dates: list[str], **cols: list[float]) -> pd.DataFrame:
    return pd.DataFrame(cols, index=pd.DatetimeIndex(pd.to_datetime(dates), name="date"))


@pytest.fixture
def fetcher(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> mod.CommodityDataFetcher:
    f = mod.CommodityDataFetcher(data_dir=tmp_path / "commodities")
    f.raw_dir = tmp_path / "raw"
    return f


def test_update_commodity_extends_panel_beyond_existing_index(
    fetcher: mod.CommodityDataFetcher, monkeypatch: pytest.MonkeyPatch
) -> None:
    """New dates from the vendor must land in the panel, not be dropped by alignment.

    Regression: assigning a longer Series to an existing column aligned it to the
    frame's old index and silently discarded every new row, so the panel stayed
    frozen while nightly updates reported 'already up to date'.
    """
    existing = _panel(["2026-07-09", "2026-07-10"], GLD=[378.0, 377.0], WTI=[60.0, 61.0])
    fresh = pd.Series(
        [377.0, 380.0, 382.0],
        index=pd.DatetimeIndex(
            pd.to_datetime(["2026-07-10", "2026-07-13", "2026-07-14"]), name="date"
        ),
        name="GLD",
    )
    monkeypatch.setattr(mod, "fetch_commodity_close", lambda symbol, start=None, end=None: fresh)

    updated = fetcher.update_commodity("GLD", existing_df=existing)

    assert updated.index.max() == pd.Timestamp("2026-07-14")
    assert updated.loc["2026-07-14", "GLD"] == 382.0
    # Other columns are preserved and simply NaN on the new dates.
    assert updated.loc["2026-07-10", "WTI"] == 61.0
    assert pd.isna(updated.loc["2026-07-14", "WTI"])
    assert updated.index.is_monotonic_increasing and updated.index.is_unique


def test_update_commodity_overwrites_restated_overlap(
    fetcher: mod.CommodityDataFetcher, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing = _panel(["2026-07-09", "2026-07-10"], GLD=[378.0, 377.0])
    fresh = pd.Series(
        [379.5], index=pd.DatetimeIndex(pd.to_datetime(["2026-07-10"]), name="date"), name="GLD"
    )
    monkeypatch.setattr(mod, "fetch_commodity_close", lambda symbol, start=None, end=None: fresh)

    updated = fetcher.update_commodity("GLD", existing_df=existing)

    assert updated.loc["2026-07-10", "GLD"] == 379.5
    assert len(updated) == 2


def test_update_commodity_empty_fetch_leaves_panel_untouched(
    fetcher: mod.CommodityDataFetcher, monkeypatch: pytest.MonkeyPatch
) -> None:
    existing = _panel(["2026-07-10"], GLD=[377.0])
    monkeypatch.setattr(
        mod,
        "fetch_commodity_close",
        lambda symbol, start=None, end=None: pd.Series(dtype="float64", name="GLD"),
    )
    updated = fetcher.update_commodity("GLD", existing_df=existing)
    pd.testing.assert_frame_equal(updated, existing)
