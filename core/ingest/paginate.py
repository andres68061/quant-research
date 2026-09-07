"""Page-walking for endpoints that cap their response size.

Many vendor "list everything" endpoints return a fixed page rather than the whole
table — FMP caps ``delisted-companies`` at 100 rows and ``cik-list`` at 1,000 —
and a caller that ignores this silently ingests the first page and calls it a
complete dataset. That failure is invisible downstream: the file exists, parses,
and is wrong.

So paginated specs walk pages until the vendor stops returning full ones, and the
accumulated rows become a single stored file. The page cap is a safety valve
against an endpoint that never returns a short page; hitting it is logged loudly
because it means the stored file is truncated.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from core.ingest.runner import FetchResult

logger = logging.getLogger(__name__)


def fetch_all_pages(
    fetch: Callable[[str, dict[str, Any]], FetchResult],
    acquire: Callable[[], Any],
    path: str,
    params: dict[str, Any],
    page_size: int,
    max_pages: int = 200,
) -> FetchResult:
    """
    Walk pages until exhausted and return one combined result.

    Args:
        fetch: Vendor call.
        acquire: Rate-limit permit, called once per page.
        path: Endpoint path.
        params: Base query parameters; ``page`` and ``limit`` are added.
        page_size: Rows requested per page.
        max_pages: Safety cap on pages walked.

    Returns:
        A :class:`FetchResult` whose ``rows`` are every page concatenated. A
        non-200 on the first page is returned as-is; a failure part-way through
        keeps the rows gathered so far rather than discarding them.
    """
    combined: list[dict[str, Any]] = []
    seen: set[int] = set()
    for page in range(max_pages):
        acquire()
        result = fetch(path, {**params, "page": page, "limit": page_size})
        if result.status_code != 200:
            if page == 0:
                return result
            logger.warning(
                "%s: page %d returned HTTP %s; keeping %d rows already gathered",
                path,
                page,
                result.status_code,
                len(combined),
            )
            return FetchResult(status_code=200, rows=combined, partial=True)

        rows = result.rows or []
        fresh = []
        for row in rows:
            signature = hash(json.dumps(row, sort_keys=True, default=str))
            if signature not in seen:
                seen.add(signature)
                fresh.append(row)

        if rows and not fresh:
            logger.info(
                "%s: page %d added no rows that were not already seen; the endpoint "
                "does not paginate, keeping %d rows",
                path,
                page,
                len(combined),
            )
            break

        combined.extend(fresh)
        if len(rows) < page_size:
            break
    else:
        logger.warning("%s: hit the %d-page cap; stored file is likely truncated", path, max_pages)
        return FetchResult(status_code=200, rows=combined, partial=True)
    return FetchResult(status_code=200, rows=combined)
