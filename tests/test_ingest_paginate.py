"""
Unit tests for core.ingest.paginate.

The regression this file exists for: FMP's ``stock-list`` ignores the ``page``
parameter and returns the same 50,138 rows for every page number. Walking it
without repeat detection stored that page 200 times — a 10,027,600-row file that
parsed cleanly, looked plausible, and was wrong. The first test below pins the
detection that stops it, and the rest pin the ordinary walk: advance while pages
come back full, stop on a short page, keep what was already gathered when a page
mid-walk errors, and complain loudly when the safety cap is what ended the walk.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pytest

from core.ingest.paginate import fetch_all_pages
from core.ingest.runner import FetchResult


class PermitCounter:
    """Stand-in for TokenBucket.acquire that counts permits handed out."""

    def __init__(self) -> None:
        self.count = 0

    def __call__(self) -> float:
        self.count += 1
        return 0.0


class RepeatingEndpoint:
    """A vendor endpoint that ignores ``page`` and returns the same rows forever."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self.calls: list[dict[str, Any]] = []

    def __call__(self, path: str, params: dict[str, Any]) -> FetchResult:
        self.calls.append(dict(params))
        return FetchResult(200, rows=list(self._rows))


class PagingEndpoint:
    """A vendor endpoint that genuinely paginates over a fixed dataset."""

    def __init__(self, total_rows: int, fail_on_page: Optional[int] = None) -> None:
        self._dataset = [{"i": index} for index in range(total_rows)]
        self._fail_on_page = fail_on_page
        self.calls: list[dict[str, Any]] = []

    def __call__(self, path: str, params: dict[str, Any]) -> FetchResult:
        self.calls.append(dict(params))
        page = int(params["page"])
        limit = int(params["limit"])
        if self._fail_on_page is not None and page == self._fail_on_page:
            return FetchResult(503, error="HTTP 503")
        return FetchResult(200, rows=self._dataset[page * limit : (page + 1) * limit])


class UnboundedEndpoint:
    """A vendor endpoint that never returns a short page, so only the cap stops it."""

    def __init__(self, page_size: int) -> None:
        self._page_size = page_size
        self.calls: list[dict[str, Any]] = []

    def __call__(self, path: str, params: dict[str, Any]) -> FetchResult:
        self.calls.append(dict(params))
        page = int(params["page"])
        return FetchResult(
            200, rows=[{"i": page * self._page_size + offset} for offset in range(self._page_size)]
        )


def test_fetch_all_pages_stores_a_page_ignoring_endpoint_only_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    rows = [{"symbol": f"SYM{index}"} for index in range(5)]
    endpoint = RepeatingEndpoint(rows)
    acquire = PermitCounter()

    with caplog.at_level(logging.INFO, logger="core.ingest.paginate"):
        result = fetch_all_pages(endpoint, acquire, "stock-list", {}, page_size=2, max_pages=200)

    # Detected on the second page, so the payload is the dataset once — not 200x.
    assert result.status_code == 200
    assert result.rows == rows
    assert len(result.rows or []) == len(rows)
    assert endpoint.calls == [
        {"page": 0, "limit": 2},
        {"page": 1, "limit": 2},
    ]
    assert acquire.count == 2
    assert "does not paginate" in caplog.text


def test_fetch_all_pages_advances_and_concatenates_a_real_paginator() -> None:
    endpoint = PagingEndpoint(total_rows=25)
    acquire = PermitCounter()

    result = fetch_all_pages(endpoint, acquire, "cik-list", {"apikey_free": 1}, page_size=10)

    assert result.status_code == 200
    assert result.rows is not None
    assert [row["i"] for row in result.rows] == list(range(25))
    assert [call["page"] for call in endpoint.calls] == [0, 1, 2]
    assert acquire.count == 3
    # Base params survive alongside the injected page/limit.
    assert all(call["apikey_free"] == 1 for call in endpoint.calls)


def test_fetch_all_pages_stops_on_the_first_short_page() -> None:
    endpoint = PagingEndpoint(total_rows=4)
    acquire = PermitCounter()

    result = fetch_all_pages(endpoint, acquire, "delisted-companies", {}, page_size=10)

    assert len(result.rows or []) == 4
    assert len(endpoint.calls) == 1
    assert acquire.count == 1


def test_fetch_all_pages_with_an_empty_first_page_returns_no_rows() -> None:
    endpoint = PagingEndpoint(total_rows=0)

    result = fetch_all_pages(endpoint, PermitCounter(), "cik-list", {}, page_size=10)

    assert result.status_code == 200
    assert result.rows == []
    assert len(endpoint.calls) == 1


def test_fetch_all_pages_returns_a_non_200_first_page_as_is() -> None:
    endpoint = PagingEndpoint(total_rows=100, fail_on_page=0)

    result = fetch_all_pages(endpoint, PermitCounter(), "cik-list", {}, page_size=10)

    assert result.status_code == 503
    assert result.error == "HTTP 503"
    assert result.rows is None
    assert len(endpoint.calls) == 1


def test_fetch_all_pages_keeps_rows_gathered_before_a_mid_walk_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    endpoint = PagingEndpoint(total_rows=100, fail_on_page=2)

    with caplog.at_level(logging.WARNING, logger="core.ingest.paginate"):
        result = fetch_all_pages(endpoint, PermitCounter(), "cik-list", {}, page_size=10)

    assert result.status_code == 200
    assert [row["i"] for row in result.rows or []] == list(range(20))
    assert "keeping 20 rows already gathered" in caplog.text


def test_fetch_all_pages_warns_when_it_hits_the_page_cap(
    caplog: pytest.LogCaptureFixture,
) -> None:
    endpoint = UnboundedEndpoint(page_size=5)
    acquire = PermitCounter()

    with caplog.at_level(logging.WARNING, logger="core.ingest.paginate"):
        result = fetch_all_pages(endpoint, acquire, "cik-list", {}, page_size=5, max_pages=3)

    assert len(result.rows or []) == 15
    assert acquire.count == 3
    assert "hit the 3-page cap" in caplog.text
    assert "truncated" in caplog.text


def test_fetch_all_pages_does_not_warn_about_the_cap_on_a_normal_walk(
    caplog: pytest.LogCaptureFixture,
) -> None:
    endpoint = PagingEndpoint(total_rows=15)

    with caplog.at_level(logging.WARNING, logger="core.ingest.paginate"):
        fetch_all_pages(endpoint, PermitCounter(), "cik-list", {}, page_size=10, max_pages=50)

    assert "page cap" not in caplog.text


def test_fetch_all_pages_acquires_one_permit_per_page() -> None:
    endpoint = PagingEndpoint(total_rows=35)
    acquire = PermitCounter()

    fetch_all_pages(endpoint, acquire, "cik-list", {}, page_size=10)

    assert acquire.count == len(endpoint.calls) == 4


class ReorderingEndpoint:
    """Returns the same rows every call, shuffled differently each time.

    This is ``all-industry-classification``: comparing a page with the previous
    page never matches, so an equality-based repeat check walked to the page cap
    and stored 25,965 real rows as 25,966,000.
    """

    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows
        self.calls = 0

    def __call__(self, path: str, params: dict[str, object]) -> FetchResult:
        self.calls += 1
        rotated = self._rows[self.calls :] + self._rows[: self.calls]
        return FetchResult(status_code=200, rows=rotated)


def test_fetch_all_pages_detects_repetition_when_row_order_changes() -> None:
    """A reordered repeat of the same rows must end the walk, not accumulate."""
    rows = [{"symbol": f"SYM{index}"} for index in range(6)]
    endpoint = ReorderingEndpoint(rows)
    acquire = PermitCounter()

    result = fetch_all_pages(
        endpoint, acquire, "all-industry-classification", {}, page_size=6, max_pages=200
    )

    assert result.status_code == 200
    assert len(result.rows or []) == len(rows)
    assert {row["symbol"] for row in (result.rows or [])} == {row["symbol"] for row in rows}
    assert endpoint.calls == 2
