"""Tests for macro publication-lag handling."""

import pandas as pd

from core.data.factors.macro import apply_macro_publication_lag


def test_apply_macro_publication_lag_delays_reference_dates() -> None:
    """Values should only become observable after their configured lag."""
    raw_macro = pd.DataFrame(
        {
            "cpi_yoy": [1.0, 2.0],
            "unrate": [4.0, 4.1],
            "dgs10": [3.0, 3.1],
        },
        index=pd.to_datetime(["2020-01-01", "2020-02-01"]),
    )

    lagged = apply_macro_publication_lag(raw_macro)

    assert lagged.loc[pd.Timestamp("2020-01-31"), "cpi_yoy"] == 1.0
    assert lagged.loc[pd.Timestamp("2020-01-11"), "unrate"] == 4.0
    assert lagged.loc[pd.Timestamp("2020-01-02"), "dgs10"] == 3.0
    assert pd.Timestamp("2020-01-01") not in lagged.index


def test_apply_macro_publication_lag_honors_custom_lags() -> None:
    """Custom lag maps let callers test release-date assumptions explicitly."""
    raw_macro = pd.DataFrame(
        {"series_a": [10.0]},
        index=pd.to_datetime(["2020-01-01"]),
    )

    lagged = apply_macro_publication_lag(raw_macro, {"series_a": 7})

    assert lagged.index.tolist() == [pd.Timestamp("2020-01-08")]
    assert lagged.iloc[0, 0] == 10.0
