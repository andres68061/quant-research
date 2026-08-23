"""Tests for the single caveat registry — the disclosure contract itself.

These are structural: they enforce that no surface can silently ship without
disclosures and that no caveat can exist without a home.
"""

from __future__ import annotations

import pytest

from core.research.caveats import (
    ALL_SURFACES,
    CAVEAT_REGISTRY,
    as_dicts,
    caveat_by_id,
    caveats_for_surface,
)


class TestRegistryIntegrity:
    def test_ids_are_unique(self) -> None:
        ids = [caveat.id for caveat in CAVEAT_REGISTRY]
        assert len(ids) == len(set(ids))

    def test_every_caveat_targets_at_least_one_known_surface(self) -> None:
        """A caveat with no surfaces is written but never shown — worse than absent."""
        for caveat in CAVEAT_REGISTRY:
            assert caveat.surfaces, f"{caveat.id} declares no surfaces"
            unknown = set(caveat.surfaces) - set(ALL_SURFACES)
            assert not unknown, f"{caveat.id} targets unknown surfaces {unknown}"

    def test_every_surface_has_disclosures(self) -> None:
        """A surface with zero caveats means we forgot, not that it is perfect."""
        for surface in ALL_SURFACES:
            assert caveats_for_surface(surface), f"{surface} has no caveats registered"

    def test_severity_and_kind_are_valid(self) -> None:
        for caveat in CAVEAT_REGISTRY:
            assert caveat.severity in {"high", "medium", "low"}, caveat.id
            assert caveat.kind in {"data", "method"}, caveat.id

    def test_detail_is_substantive(self) -> None:
        """A one-line caveat that restates its title explains nothing."""
        for caveat in CAVEAT_REGISTRY:
            assert len(caveat.detail) > 80, f"{caveat.id} detail is too thin to be useful"
            assert caveat.title.strip()


class TestLookup:
    def test_unknown_surface_raises_rather_than_returning_empty(self) -> None:
        """Silently returning [] for a typo would disclose nothing, invisibly."""
        with pytest.raises(KeyError):
            caveats_for_surface("sector_perfromance")

    def test_sorted_most_severe_first(self) -> None:
        for surface in ALL_SURFACES:
            severities = [c.severity for c in caveats_for_surface(surface)]
            rank = {"high": 0, "medium": 1, "low": 2}
            assert severities == sorted(severities, key=lambda s: rank[s])

    def test_caveat_by_id_roundtrip(self) -> None:
        for caveat in CAVEAT_REGISTRY:
            assert caveat_by_id(caveat.id) is caveat

    def test_unknown_id_raises(self) -> None:
        with pytest.raises(KeyError):
            caveat_by_id("no-such-caveat")


class TestSerialization:
    def test_as_dicts_shape(self) -> None:
        payload = as_dicts(caveats_for_surface("pead_study"))
        assert payload
        for entry in payload:
            assert set(entry) == {"id", "kind", "severity", "title", "detail", "remediation"}


class TestSurfacesDeclareKnownRisks:
    """Spot-checks that the highest-consequence disclosures reach their surfaces."""

    def test_cost_assumptions_disclosed_on_backtest_surfaces(self) -> None:
        for surface in ("factor_screen", "factor_backtest"):
            ids = {c.id for c in caveats_for_surface(surface)}
            assert "flat-transaction-costs" in ids
            assert "short-leg-gross-of-borrow" in ids

    def test_multiple_testing_disclosed_on_the_screen(self) -> None:
        ids = {c.id for c in caveats_for_surface("factor_screen")}
        assert "multiple-testing-correction" in ids

    def test_pead_universe_limitation_disclosed(self) -> None:
        ids = {c.id for c in caveats_for_surface("pead_study")}
        assert "pead-large-cap-universe" in ids

    def test_sector_page_discloses_rebalance_and_costs(self) -> None:
        ids = {c.id for c in caveats_for_surface("sector_performance")}
        assert "sector-index-daily-compounding" in ids
        assert "sector-index-gross-returns" in ids
