"""Tests for permanent security identifiers.

The properties that matter are all about *stability across rebuilds*. A security
master that re-mints ids on rebuild is worse than none, because every downstream
reference silently repoints to a different company.
"""

from __future__ import annotations

import pandas as pd

from core.data.security_master import (
    QID_PATTERN,
    build_security_master,
    format_qid,
    normalize_symbol,
)


def _universe(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


class TestQidFormat:
    def test_qids_are_padded_and_opaque(self) -> None:
        assert format_qid(42) == "Q0000042"
        assert QID_PATTERN.match(format_qid(1))

    def test_minted_ids_are_sequential_and_unique(self) -> None:
        universe = _universe([{"symbol": s} for s in ("AAPL", "MSFT", "XOM")])
        master = build_security_master(universe)
        qids = list(master.securities["qid"])
        assert qids == ["Q0000001", "Q0000002", "Q0000003"]
        assert len(set(qids)) == 3


class TestSymbolNormalization:
    def test_share_class_separators_unify(self) -> None:
        assert normalize_symbol("BRK.B") == normalize_symbol("BRK-B") == "BRK-B"
        assert normalize_symbol("brk/b") == "BRK-B"

    def test_distinct_tickers_stay_distinct(self) -> None:
        assert normalize_symbol("GOOG") != normalize_symbol("GOOGL")

    def test_preferred_series_never_merges_into_an_unrelated_common(self) -> None:
        """
        Real collisions from the live universe, found by the first build.

        FMP writes preferred series as a "-P" suffix. Deleting punctuation made
        AL-PA (Air Lease preferred A) collide with ALPA (Alpha Healthcare
        Acquisition), and C-PK (Citigroup preferred K) with CPK (Chesapeake
        Utilities) — different companies sharing one permanent id.
        """
        assert normalize_symbol("AL-PA") != normalize_symbol("ALPA")
        assert normalize_symbol("C-PK") != normalize_symbol("CPK")
        assert normalize_symbol("ET-PC") != normalize_symbol("ETPC")

    def test_collision_free_on_the_real_universe_shape(self) -> None:
        """Splitting when in doubt: no two distinct companies share a key."""
        symbols = ["AL-PA", "ALPA", "C-PK", "CPK", "BRK.B", "BRK-B", "GOOG", "GOOGL"]
        keys = [normalize_symbol(s) for s in symbols]
        # BRK.B and BRK-B are the only intended merge.
        assert len(set(keys)) == len(symbols) - 1


class TestRebuildStability:
    def test_rebuild_preserves_every_id(self) -> None:
        universe = _universe([{"symbol": s} for s in ("AAPL", "MSFT")])
        first = build_security_master(universe)
        second = build_security_master(universe, existing=first)
        assert dict(
            zip(second.securities["symbol"], second.securities["qid"], strict=True)
        ) == dict(zip(first.securities["symbol"], first.securities["qid"], strict=True))

    def test_new_entities_continue_the_sequence(self) -> None:
        first = build_security_master(_universe([{"symbol": "AAPL"}]))
        second = build_security_master(
            _universe([{"symbol": "AAPL"}, {"symbol": "NVDA"}]), existing=first
        )
        by_symbol = dict(zip(second.securities["symbol"], second.securities["qid"], strict=True))
        assert by_symbol["AAPL"] == "Q0000001"
        assert by_symbol["NVDA"] == "Q0000002"

    def test_ids_are_never_reused_after_a_delisting(self) -> None:
        """A retired qid must not be handed to the next new company."""
        first = build_security_master(
            _universe([{"symbol": "OLD"}, {"symbol": "AAPL"}]),
        )
        # OLD is gone from the vendor universe entirely.
        second = build_security_master(_universe([{"symbol": "AAPL"}]), existing=first)
        third = build_security_master(
            _universe([{"symbol": "AAPL"}, {"symbol": "NEWCO"}]), existing=second
        )
        by_symbol = dict(zip(third.securities["symbol"], third.securities["qid"], strict=True))
        assert by_symbol["NEWCO"] == "Q0000003"
        assert by_symbol["NEWCO"] != "Q0000001"


class TestIssuerAnchor:
    def test_ticker_change_keeps_the_same_qid(self) -> None:
        """The headline case: FB became META, same security under a new ticker."""
        ciks = {"FB": "0001326801", "META": "0001326801"}
        first = build_security_master(_universe([{"symbol": "FB"}]), cik_by_symbol=ciks)
        second = build_security_master(
            _universe([{"symbol": "META"}]), existing=first, cik_by_symbol=ciks
        )
        assert second.securities.iloc[0]["qid"] == first.securities.iloc[0]["qid"]
        assert second.securities.iloc[0]["id_source"] == "issuer"

    def test_old_ticker_still_resolves_after_the_change(self) -> None:
        """Historical datasets keyed by the old ticker must not become orphans."""
        ciks = {"FB": "0001326801", "META": "0001326801"}
        first = build_security_master(_universe([{"symbol": "FB"}]), cik_by_symbol=ciks)
        second = build_security_master(
            _universe([{"symbol": "META"}]), existing=first, cik_by_symbol=ciks
        )
        assert second.resolve("FB") == second.resolve("META")
        assert set(second.symbols_for(second.resolve("META"))) == {"FB", "META"}


class TestIssuerIsNotIdentity:
    """
    One issuer, many securities.

    An earlier implementation used CIK as the identity and merged 208 real
    securities across 85 issuers — a company's common stock and its bond lines
    became one price series.
    """

    def test_share_classes_of_one_issuer_get_separate_qids(self) -> None:
        ciks = {"GOOG": "0001652044", "GOOGL": "0001652044"}
        master = build_security_master(
            _universe([{"symbol": "GOOG"}, {"symbol": "GOOGL"}]), cik_by_symbol=ciks
        )
        assert master.resolve("GOOG") != master.resolve("GOOGL")

    def test_common_and_bond_lines_get_separate_qids(self) -> None:
        cik = "0001042046"
        ciks = {s: cik for s in ("AFG", "AFGB", "AFGC", "AFGE")}
        master = build_security_master(_universe([{"symbol": s} for s in ciks]), cik_by_symbol=ciks)
        assert master.securities["qid"].nunique() == 4

    def test_securities_of_one_issuer_share_an_issuer_id(self) -> None:
        """Separate identities, but the issuer relationship is still recorded."""
        ciks = {"GOOG": "0001652044", "GOOGL": "0001652044"}
        master = build_security_master(
            _universe([{"symbol": "GOOG"}, {"symbol": "GOOGL"}]), cik_by_symbol=ciks
        )
        assert set(master.securities["issuer_id"]) == {"0001652044"}

    def test_ambiguous_issuer_does_not_carry_a_qid_forward(self) -> None:
        """
        With several securities under one issuer, CIK cannot say which is which,
        so a new ticker must mint a new id rather than guess.
        """
        cik = "0001042046"
        first = build_security_master(_universe([{"symbol": "AFG"}]), cik_by_symbol={"AFG": cik})
        second = build_security_master(
            _universe([{"symbol": "AFGB"}, {"symbol": "AFGC"}]),
            existing=first,
            cik_by_symbol={"AFGB": cik, "AFGC": cik},
        )
        assert second.securities["qid"].nunique() == 2
        assert set(second.securities["id_source"]) == {"new"}


class TestNoSharedIdentities:
    def test_two_securities_can_never_share_a_qid(self) -> None:
        cik = "0000000001"
        ciks = {s: cik for s in ("AAA", "AAB", "AAC")}
        master = build_security_master(_universe([{"symbol": s} for s in ciks]), cik_by_symbol=ciks)
        assert not master.securities["qid"].duplicated().any()


class TestResolution:
    def test_unknown_symbol_resolves_to_none(self) -> None:
        master = build_security_master(_universe([{"symbol": "AAPL"}]))
        assert master.resolve("NOPE") is None

    def test_punctuation_variants_resolve_to_the_same_entity(self) -> None:
        master = build_security_master(_universe([{"symbol": "BRK.B"}]))
        assert master.resolve("BRK-B") == master.resolve("BRK.B") == "Q0000001"

    def test_ticker_reuse_mints_a_second_identity(self) -> None:
        """
        One ticker, two companies, different eras — two securities.

        Inheriting the old id here is how a dead company's history gets spliced
        onto a live one inside a single price column.
        """
        universe = _universe(
            [
                {
                    "symbol": "ZZZ",
                    "company_name": "Old Corp",
                    "ipo_date": pd.Timestamp("1990-01-01"),
                    "delisted_date": pd.Timestamp("2005-06-30"),
                },
                {
                    "symbol": "ZZZ",
                    "company_name": "New Corp",
                    "ipo_date": pd.Timestamp("2015-01-01"),
                    "delisted_date": None,
                },
            ]
        )
        master = build_security_master(universe, vendor="test")
        assert master.securities["qid"].nunique() == 2
        assert "reused_ticker" in set(master.securities["id_source"])

    def test_as_of_routes_a_reused_ticker_to_the_right_era(self) -> None:
        universe = _universe(
            [
                {
                    "symbol": "ZZZ",
                    "company_name": "Old Corp",
                    "ipo_date": pd.Timestamp("1990-01-01"),
                    "delisted_date": pd.Timestamp("2005-06-30"),
                },
                {
                    "symbol": "ZZZ",
                    "company_name": "New Corp",
                    "ipo_date": pd.Timestamp("2015-01-01"),
                    "delisted_date": None,
                },
            ]
        )
        master = build_security_master(universe, vendor="test")
        old = master.resolve("ZZZ", as_of=pd.Timestamp("1998-01-01"))
        new = master.resolve("ZZZ", as_of=pd.Timestamp("2020-01-01"))
        assert old != new
        assert master.describe(old)["company_name"] == "Old Corp"
        assert master.describe(new)["company_name"] == "New Corp"

    def test_a_continuing_ticker_keeps_one_identity(self) -> None:
        """Overlapping spans are the same security, not a reuse."""
        first = build_security_master(
            _universe(
                [{"symbol": "AAPL", "ipo_date": pd.Timestamp("1980-12-12"), "delisted_date": None}]
            )
        )
        second = build_security_master(
            _universe(
                [{"symbol": "AAPL", "ipo_date": pd.Timestamp("1980-12-12"), "delisted_date": None}]
            ),
            existing=first,
        )
        assert second.securities["qid"].tolist() == first.securities["qid"].tolist()

    def test_as_of_outside_every_interval_returns_none(self) -> None:
        universe = _universe(
            [
                {
                    "symbol": "DEAD",
                    "ipo_date": pd.Timestamp("1990-01-01"),
                    "delisted_date": pd.Timestamp("1999-12-31"),
                }
            ]
        )
        master = build_security_master(universe)
        assert master.resolve("DEAD", as_of=pd.Timestamp("1995-01-01")) == "Q0000001"
        assert master.resolve("DEAD", as_of=pd.Timestamp("2020-01-01")) is None

    def test_tz_aware_as_of_is_accepted(self) -> None:
        universe = _universe([{"symbol": "AAPL", "ipo_date": pd.Timestamp("1980-12-12")}])
        master = build_security_master(universe)
        stamp = pd.Timestamp("2020-01-01", tz="America/New_York")
        assert master.resolve("AAPL", as_of=stamp) == "Q0000001"
