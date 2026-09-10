"""Tests for panel structural invariants — the automatic gate on bad data.

Each test corresponds to a defect that actually reached a research result before
this module existed, or to an invariant whose violation would do the same.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from core.data.factors.returns import DEFAULT_MAX_ABS_RETURN
from core.data.quality.validation import (
    EXTREME_RETURN_THRESHOLD,
    assert_panel_valid,
    validate_price_panel,
    violations_as_dicts,
)

TZ = "America/New_York"


def _panel(n_days: int = 30, n_symbols: int = 10) -> pd.DataFrame:
    index = pd.bdate_range("2024-01-02", periods=n_days, tz=TZ, name="date")
    rng = np.random.default_rng(0)
    data = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, (n_days, n_symbols)), axis=0))
    return pd.DataFrame(data, index=index, columns=[f"S{i:02d}" for i in range(n_symbols)])


def _checks(violations) -> set[str]:
    return {v.check for v in violations}


class TestCleanPanelPasses:
    def test_a_sound_panel_has_no_violations(self) -> None:
        assert validate_price_panel(_panel()) == []

    def test_assert_does_not_raise_on_sound_panel(self) -> None:
        assert_panel_valid(_panel())


class TestErrorSeverity:
    def test_zero_price_is_an_error(self) -> None:
        """The exact defect that produced -inf CAR paths in the PEAD study."""
        panel = _panel()
        panel.iloc[5, 0] = 0.0
        violations = validate_price_panel(panel)
        assert "non_positive_price" in _checks(violations)
        assert all(v.severity == "error" for v in violations if v.check == "non_positive_price")

    def test_negative_price_is_an_error(self) -> None:
        panel = _panel()
        panel.iloc[5, 0] = -10.0
        assert "non_positive_price" in _checks(validate_price_panel(panel))

    def test_duplicate_dates_are_an_error(self) -> None:
        panel = _panel()
        panel = pd.concat([panel, panel.iloc[[0]]])
        assert "index_duplicated" in _checks(validate_price_panel(panel))

    def test_unsorted_index_is_an_error(self) -> None:
        panel = _panel().iloc[::-1]
        assert "index_not_sorted" in _checks(validate_price_panel(panel))

    def test_tz_naive_index_is_an_error(self) -> None:
        panel = _panel()
        panel.index = panel.index.tz_localize(None)
        assert "index_tz_naive" in _checks(validate_price_panel(panel))

    def test_assert_raises_on_any_error(self) -> None:
        panel = _panel()
        panel.iloc[3, 1] = 0.0
        with pytest.raises(ValueError, match="non_positive_price"):
            assert_panel_valid(panel)

    def test_error_message_names_every_failing_check(self) -> None:
        panel = _panel()
        panel.iloc[3, 1] = 0.0
        panel.index = panel.index.tz_localize(None)
        with pytest.raises(ValueError) as exc:
            assert_panel_valid(panel)
        assert "non_positive_price" in str(exc.value)
        assert "index_tz_naive" in str(exc.value)


class TestWarningSeverity:
    def test_extreme_return_is_a_warning_not_an_error(self) -> None:
        """Defects stay in the raw panel by design; the compute layer rejects them."""
        panel = _panel()
        panel.iloc[10:, 0] = panel.iloc[9, 0] * 1_000_000
        violations = validate_price_panel(panel)
        extreme = [v for v in violations if v.check == "extreme_return"]
        assert extreme and extreme[0].severity == "warning"
        # Warnings must not block a build.
        assert_panel_valid(panel)

    def test_non_trading_dates_reported_against_a_calendar(self) -> None:
        panel = _panel()
        calendar = panel.index[:-3]
        violations = validate_price_panel(panel, trading_calendar=calendar)
        assert "non_trading_date" in _checks(violations)

    def test_thin_cross_section_flagged(self) -> None:
        panel = _panel(n_symbols=3)
        assert "thin_cross_section" in _checks(validate_price_panel(panel))


class TestOrderingAndSerialization:
    def test_errors_sort_before_warnings(self) -> None:
        panel = _panel()
        panel.iloc[5, 0] = 0.0
        panel.iloc[10:, 1] = panel.iloc[9, 1] * 1_000_000
        severities = [v.severity for v in validate_price_panel(panel)]
        assert severities == sorted(severities, key=lambda s: {"error": 0, "warning": 1}[s])

    def test_serialization_shape(self) -> None:
        panel = _panel()
        panel.iloc[5, 0] = 0.0
        payload = violations_as_dicts(validate_price_panel(panel))
        assert payload
        for entry in payload:
            assert set(entry) == {"check", "severity", "count", "detail", "examples"}

    def test_examples_name_the_offending_symbols(self) -> None:
        panel = _panel()
        panel.iloc[5, 2] = 0.0
        violation = next(v for v in validate_price_panel(panel) if v.check == "non_positive_price")
        assert "S02" in violation.examples


class TestThresholdConsistency:
    def test_validation_and_returns_layer_agree(self) -> None:
        """
        The gate and the compute layer must use the same definition of a defect.

        If they drift, the audit reports clean while the return layer is silently
        discarding data (or the reverse).
        """
        assert EXTREME_RETURN_THRESHOLD == DEFAULT_MAX_ABS_RETURN
