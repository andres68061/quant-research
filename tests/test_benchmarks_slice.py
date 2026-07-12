"""Benchmark route must slice tz-aware price panels without naive Timestamp errors."""

from datetime import date
from unittest.mock import patch

import pandas as pd

from api.routes.benchmarks import get_benchmark_returns


def test_benchmark_slice_tz_aware_prices() -> None:
    idx = pd.date_range("2020-01-01", periods=30, freq="B", tz="UTC")
    df = pd.DataFrame({"^GSPC": [100.0 + i * 0.1 for i in range(len(idx))]}, index=idx)

    with patch("api.routes.benchmarks.get_prices", return_value=df):
        out = get_benchmark_returns(
            benchmark_type="S&P 500 (^GSPC)",
            start_date=date(2020, 1, 6),
            end_date=date(2020, 1, 31),
        )

    assert len(out.dates) > 0
    assert len(out.returns) == len(out.dates)
