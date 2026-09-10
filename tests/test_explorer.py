"""Tests for the data explorer.

Two concerns. First, the SQL builder must produce correct joins and refuse
malformed input. Second — and more important — the read-only boundary must hold:
this endpoint accepts user-authored SQL, so the validator is a security control,
not a convenience.
"""

from __future__ import annotations

import pandas as pd
import pytest

from core.data.store.explorer import (
    QueryError,
    ScreenFilter,
    build_screen_sql,
    validate_read_only,
)


class TestReadOnlyBoundary:
    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT 1",
            "select symbol from universe",
            "  WITH x AS (SELECT 1) SELECT * FROM x",
            "\nSELECT * FROM universe\n",
        ],
    )
    def test_reads_are_allowed(self, sql: str) -> None:
        validate_read_only(sql)

    @pytest.mark.parametrize(
        "sql",
        [
            "DROP TABLE universe",
            "DELETE FROM universe",
            "UPDATE universe SET symbol = 'X'",
            "INSERT INTO universe VALUES (1)",
            "CREATE TABLE evil (x INT)",
            "ATTACH 'other.db'",
            "COPY universe TO 'out.csv'",
            "PRAGMA database_list",
        ],
    )
    def test_writes_are_rejected(self, sql: str) -> None:
        with pytest.raises(QueryError):
            validate_read_only(sql)

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT 1; DROP TABLE universe",
            "SELECT 1;DELETE FROM universe",
            "WITH x AS (SELECT 1) SELECT * FROM x; CREATE TABLE y (z INT)",
        ],
    )
    def test_stacked_statements_are_rejected(self, sql: str) -> None:
        """Appending a second statement is the standard way past a prefix check."""
        with pytest.raises(QueryError, match="one statement"):
            validate_read_only(sql)

    def test_trailing_semicolon_alone_is_fine(self) -> None:
        validate_read_only("SELECT 1;")

    @pytest.mark.parametrize("sql", ["", "   ", "\n"])
    def test_empty_is_rejected(self, sql: str) -> None:
        with pytest.raises(QueryError, match="Empty"):
            validate_read_only(sql)


class TestScreenSqlInjection:
    def test_column_names_must_be_identifiers(self) -> None:
        with pytest.raises(QueryError, match="Invalid column"):
            build_screen_sql(columns=["roe; DROP TABLE universe"], filters=[])

    def test_filter_columns_must_be_identifiers(self) -> None:
        with pytest.raises(QueryError, match="Invalid column"):
            build_screen_sql(
                columns=["roe"],
                filters=[ScreenFilter("x) OR 1=1 --", ">", 1)],
            )

    def test_order_by_must_be_an_identifier(self) -> None:
        with pytest.raises(QueryError, match="Invalid order column"):
            build_screen_sql(columns=["roe"], filters=[], order_by="roe; DROP TABLE x")

    def test_unsupported_operator_is_rejected(self) -> None:
        with pytest.raises(QueryError, match="Unsupported operator"):
            build_screen_sql(columns=["roe"], filters=[ScreenFilter("roe", "LIKE", 1)])  # type: ignore[arg-type]

    def test_string_literals_are_escaped(self) -> None:
        sql = build_screen_sql(
            columns=["roe"],
            filters=[ScreenFilter("sector", "=", "O'Brien")],
            panels=["factors_fundamental"],
        )
        assert "'O''Brien'" in sql

    def test_unsupported_value_type_is_rejected(self) -> None:
        with pytest.raises(QueryError, match="Unsupported filter value"):
            build_screen_sql(
                columns=["roe"],
                filters=[ScreenFilter("roe", ">", {"nested": "dict"})],
                panels=["factors_fundamental"],
            )


class TestScreenSqlShape:
    def test_single_panel_needs_no_join(self) -> None:
        sql = build_screen_sql(
            columns=["roe"],
            filters=[ScreenFilter("roe", ">", 0.2)],
            panels=["factors_fundamental"],
        )
        assert "JOIN" not in sql
        assert "roe > 0.2" in sql

    def test_multiple_panels_join_on_date_and_symbol(self) -> None:
        """A join on symbol alone would produce a cross product across dates."""
        sql = build_screen_sql(
            columns=["roe", "mom_12_1"],
            filters=[],
            panels=["factors_price", "factors_fundamental"],
        )
        assert "JOIN factors_fundamental" in sql
        assert "factors_fundamental.symbol = factors_price.symbol" in sql
        assert "factors_fundamental.date = factors_price.date" in sql

    def test_default_date_is_the_latest_available(self) -> None:
        sql = build_screen_sql(columns=["roe"], filters=[], panels=["factors_fundamental"])
        assert "max(date)" in sql

    def test_explicit_as_of_is_used(self) -> None:
        sql = build_screen_sql(
            columns=["roe"], filters=[], panels=["factors_fundamental"], as_of="2020-06-30"
        )
        assert "DATE '2020-06-30'" in sql
        assert "max(date)" not in sql

    def test_between_renders_both_bounds(self) -> None:
        sql = build_screen_sql(
            columns=["roe"],
            filters=[ScreenFilter("roe", "between", 0.1, 0.5)],
            panels=["factors_fundamental"],
        )
        assert "BETWEEN 0.1 AND 0.5" in sql

    def test_in_renders_a_list(self) -> None:
        sql = build_screen_sql(
            columns=["roe"],
            filters=[ScreenFilter("symbol", "in", ["AAPL", "MSFT"])],
            panels=["factors_fundamental"],
        )
        assert "IN ('AAPL', 'MSFT')" in sql

    def test_empty_in_list_is_rejected(self) -> None:
        with pytest.raises(QueryError, match="non-empty"):
            build_screen_sql(
                columns=["roe"],
                filters=[ScreenFilter("symbol", "in", [])],
                panels=["factors_fundamental"],
            )

    def test_nulls_sort_last(self) -> None:
        """Otherwise a 'top 20 by ROE' screen returns 20 rows of missing data."""
        sql = build_screen_sql(
            columns=["roe"], filters=[], panels=["factors_fundamental"], order_by="roe"
        )
        assert "NULLS LAST" in sql


class TestLiveDatasets:
    """
    Light checks against the real files, skipped when they are absent.

    These catch schema drift — a renamed column that silently breaks the catalog.
    """

    def test_catalog_lists_only_existing_datasets(self) -> None:
        from core.data.store.explorer import DATASETS, describe_datasets

        catalog = describe_datasets()
        names = {d["name"] for d in catalog}
        for dataset in DATASETS:
            if dataset.exists:
                assert dataset.name in names

    def test_every_factor_panel_exposes_columns(self) -> None:
        from core.data.store.explorer import describe_datasets

        for entry in describe_datasets():
            if entry["family"] == "factor":
                assert entry["n_columns"] > 0
                assert "date" not in entry["columns"]
                assert "symbol" not in entry["columns"]

    def test_a_real_screen_returns_rows(self) -> None:
        from core.data.store.explorer import DATASETS_BY_NAME, run_query

        if not DATASETS_BY_NAME["factors_price"].exists:
            pytest.skip("factor panels not built")
        sql = build_screen_sql(
            columns=["mom_12_1"],
            filters=[ScreenFilter("mom_12_1", ">", 0.0)],
            panels=["factors_price"],
            order_by="mom_12_1",
            limit=5,
        )
        result = run_query(sql)
        assert result.row_count > 0
        assert "symbol" in result.columns
        assert all(isinstance(r["mom_12_1"], float) for r in result.rows)

    def test_results_are_json_safe(self) -> None:
        """NaN and Timestamp both break JSON serialisation if they survive."""
        from core.data.store.explorer import DATASETS_BY_NAME, run_query

        if not DATASETS_BY_NAME["factors_price"].exists:
            pytest.skip("factor panels not built")
        result = run_query("SELECT date, symbol, vol_60d FROM factors_price LIMIT 20")
        for row in result.rows:
            assert isinstance(row["date"], str)
            assert row["vol_60d"] is None or isinstance(row["vol_60d"], float)
            assert not (isinstance(row["vol_60d"], float) and pd.isna(row["vol_60d"]))

    def test_limit_is_enforced_and_flagged(self) -> None:
        from core.data.store.explorer import DATASETS_BY_NAME, run_query

        if not DATASETS_BY_NAME["universe"].exists:
            pytest.skip("universe not built")
        result = run_query("SELECT * FROM universe", limit=10)
        assert result.row_count == 10
        assert result.truncated is True
