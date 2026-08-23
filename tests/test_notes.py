"""Tests for research notes.

The important test is the last class: a note's numbers must match the persisted
experiment output. Notes are hand-written prose around copied figures, and a
hand-copied figure drifts from its source silently — which would make the most
readable artifact in the repo also the least trustworthy.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.research.glossary import lookup
from core.research.notes import NOTES, VERDICT_LABELS, as_dicts, get_note

QUALITY_DIR = Path(__file__).resolve().parents[1] / "data" / "quality"


class TestStructure:
    def test_ids_are_unique(self) -> None:
        ids = [n.id for n in NOTES]
        assert len(ids) == len(set(ids))

    def test_every_note_has_a_known_verdict(self) -> None:
        for note in NOTES:
            assert note.verdict in VERDICT_LABELS

    def test_get_note_round_trips(self) -> None:
        for note in NOTES:
            assert get_note(note.id) is note
        assert get_note("nope") is None

    @pytest.mark.parametrize("note", NOTES, ids=lambda n: n.id)
    def test_required_prose_is_present(self, note) -> None:
        """Every section is required; an empty one means the note is not finished."""
        for field in (
            note.question,
            note.hypothesis,
            note.method,
            note.what_it_means,
            note.control_reading,
            note.reproduce,
            note.one_liner,
        ):
            assert field and len(field.strip()) > 20

    @pytest.mark.parametrize("note", NOTES, ids=lambda n: n.id)
    def test_results_and_caveats_are_non_empty(self, note) -> None:
        assert len(note.results) >= 2, "a single variant is not an experiment"
        assert note.caveats, "a note with no caveats has not been thought through"

    @pytest.mark.parametrize("note", NOTES, ids=lambda n: n.id)
    def test_reproduce_points_at_a_real_script(self, note) -> None:
        script = next(part for part in note.reproduce.split() if part.endswith(".py"))
        assert (Path(__file__).resolve().parents[1] / script).exists(), script

    @pytest.mark.parametrize("note", NOTES, ids=lambda n: n.id)
    def test_glossary_terms_all_exist(self, note) -> None:
        """A note linking to a term that is not defined is worse than not linking."""
        for term in note.glossary_terms:
            assert lookup(term) is not None, f"{note.id} references undefined term {term!r}"


class TestControlDiscipline:
    """The rule the platform keeps getting wrong: results need a like-for-like control."""

    @pytest.mark.parametrize("note", NOTES, ids=lambda n: n.id)
    def test_control_reading_is_written(self, note) -> None:
        assert len(note.control_reading.strip()) > 100

    def test_notes_with_a_headline_also_mark_a_control_row(self) -> None:
        """
        Where a note names a headline variant, a control row must exist to read it
        against — except the factor-screen note, whose control is the other 27
        factors rather than a row in its own table.
        """
        for note in NOTES:
            if note.id == "net-operating-assets":
                continue
            if any(row.is_headline for row in note.results):
                assert any(row.is_control for row in note.results), note.id


class TestNumbersMatchSourceData:
    """Hand-copied figures must agree with the experiment output they came from."""

    def _load(self, filename: str) -> dict:
        path = QUALITY_DIR / filename
        if not path.exists():
            pytest.skip(f"{filename} not present")
        return json.loads(path.read_text())

    def test_pead_numbers_match(self) -> None:
        source = self._load("experiment_pead_tradable_20260814.json")
        by_prefix = {row["variant"][0]: row for row in source["results"]}
        note = get_note("pead-tradability")
        assert note is not None

        for row in note.results:
            key = row.variant[0]
            expected = by_prefix[key]
            assert row.gross_sharpe == pytest.approx(expected["gross_sharpe"], abs=0.001)
            assert row.net_sharpe == pytest.approx(expected["net_sharpe"], abs=0.001)
            assert row.hit_rate == pytest.approx(expected["hit_rate"], abs=0.001)

    def test_pead_control_gap_is_stated_correctly(self) -> None:
        """The +0.18 claim must equal E minus F in the source data."""
        source = self._load("experiment_pead_tradable_20260814.json")
        rows = {row["variant"][0]: row for row in source["results"]}
        gap = rows["E"]["net_sharpe"] - rows["F"]["net_sharpe"]
        assert gap == pytest.approx(0.18, abs=0.01)
        note = get_note("pead-tradability")
        assert note is not None
        assert "+0.18" in note.control_reading

    def test_piotroski_numbers_match(self) -> None:
        source = self._load("experiment_conditional_piotroski_20260813.json")
        by_prefix = {row["variant"][0]: row for row in source["results"]}
        note = get_note("conditional-piotroski")
        assert note is not None

        for row in note.results:
            expected = by_prefix[row.variant[0]]
            assert row.net_sharpe == pytest.approx(expected["sharpe_net"], abs=0.001)
            assert row.t_stat == pytest.approx(expected["t_stat"], abs=0.01)

    def test_piotroski_control_beats_the_signal(self) -> None:
        """The whole conclusion rests on F > B; assert it rather than trusting prose."""
        source = self._load("experiment_conditional_piotroski_20260813.json")
        rows = {row["variant"][0]: row for row in source["results"]}
        assert rows["F"]["sharpe_net"] > rows["B"]["sharpe_net"]
        assert rows["E"]["sharpe_net"] > rows["F"]["sharpe_net"]

    def test_noa_numbers_match_the_screen(self) -> None:
        source = self._load("factor_screen_20260811.json")
        rows = {row["factor"]: row for row in source["results"]}
        noa = rows["neg_net_operating_assets"]
        note = get_note("net-operating-assets")
        assert note is not None

        headline = next(row for row in note.results if row.is_headline)
        assert headline.net_sharpe == pytest.approx(noa["sharpe_net"], abs=0.001)
        assert headline.t_stat == pytest.approx(noa["t_stat"], abs=0.01)

    def test_noa_decade_table_matches_the_screen(self) -> None:
        source = self._load("factor_screen_20260811.json")
        rows = {row["factor"]: row for row in source["results"]}
        note = get_note("net-operating-assets")
        assert note is not None

        for label, *decades in note.decade_table:
            record = rows.get(label)
            if record is None:
                continue
            for value, key in zip(decades, ("2000s", "2010s", "2020s"), strict=True):
                assert float(value) == pytest.approx(record["by_decade"][key], abs=0.01)

    def test_significance_bar_matches_the_screen(self) -> None:
        source = self._load("factor_screen_20260811.json")
        assert source["n_tests"] == 28
        assert source["sidak_t_threshold"] == pytest.approx(3.12, abs=0.01)
        note = get_note("net-operating-assets")
        assert note is not None
        assert "3.12" in note.method
        assert not rows_pass(source)

    def test_noa_did_not_pass_the_bar(self) -> None:
        source = self._load("factor_screen_20260811.json")
        noa = next(r for r in source["results"] if r["factor"] == "neg_net_operating_assets")
        assert noa["sidak_pass"] is False


def rows_pass(source: dict) -> bool:
    """True if any factor in the screen cleared the corrected bar."""
    return any(row.get("sidak_pass") for row in source["results"])


class TestSerialization:
    def test_summary_omits_prose(self) -> None:
        for entry in as_dicts(full=False):
            assert "question" not in entry
            assert "one_liner" in entry

    def test_full_includes_every_section(self) -> None:
        for entry in as_dicts(full=True):
            for key in ("question", "method", "results", "caveats", "reproduce"):
                assert key in entry
