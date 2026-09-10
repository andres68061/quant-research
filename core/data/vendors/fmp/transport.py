"""Thread-safe HTTP transport for FMP, shaped for the ingestion runner.

Differs from :mod:`core.data.vendors.fmp.client` in three ways that matter for a
large backfill:

- **No internal throttle.** Pacing belongs to the shared
  :class:`~core.ingest.ratelimit.TokenBucket`, because a per-call sleep does not
  bound a pool of workers.
- **Reports instead of raising.** A 402 on an unentitled endpoint is a fact to
  journal for all 9,011 symbols, not an exception that aborts a run.
- **Connection reuse.** One :class:`requests.Session` per worker thread; TLS
  handshakes on every call would otherwise dominate at these volumes.

The single-call client remains the right tool for interactive and small
scripted use, and is unchanged.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional

import requests

from core.exceptions import ConfigError
from core.ingest.runner import FetchResult

logger = thread_logger = logging.getLogger(__name__)

FMP_BASE_URL = "https://financialmodelingprep.com/stable"
_TIMEOUT_SECONDS = 45

_thread_local = threading.local()


def _session() -> requests.Session:
    """One pooled session per worker thread."""
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=8)
        session.mount("https://", adapter)
        _thread_local.session = session
    return session


def make_fetcher(api_key: Optional[str] = None) -> Any:
    """
    Build the ``fetch(path, params)`` callable the runner expects.

    Args:
        api_key: Override key; defaults to ``FMP_API_KEY`` from settings.

    Returns:
        Thread-safe callable returning :class:`FetchResult`.

    Raises:
        ConfigError: If no API key is configured.

    Example:
        >>> fetch = make_fetcher()
        >>> fetch("profile", {"symbol": "AAPL"}).status_code  # doctest: +SKIP
        200
    """
    if api_key is None:
        from config.settings import FMP_API_KEY

        api_key = FMP_API_KEY
    if not api_key:
        raise ConfigError("FMP_API_KEY is not set; add it to .env")
    key = api_key

    def fetch(path: str, params: dict[str, Any]) -> FetchResult:
        url = f"{FMP_BASE_URL}/{path}"
        try:
            response = _session().get(
                url, params={**params, "apikey": key}, timeout=_TIMEOUT_SECONDS
            )
        except requests.RequestException as exc:
            # Transport failures get a synthetic retryable status so the runner
            # backs off rather than treating them as a vendor verdict.
            return FetchResult(status_code=503, error=f"{type(exc).__name__}: {exc}")

        if response.status_code != 200:
            # Never surface response.url: it carries the API key.
            body = response.text[:300].replace(key, "REDACTED")
            return FetchResult(status_code=response.status_code, error=body)

        content_type = response.headers.get("content-type", "")
        if "json" not in content_type:
            return FetchResult(status_code=200, content=response.content)

        try:
            payload = response.json()
        except ValueError:
            return FetchResult(status_code=200, content=response.content)

        if isinstance(payload, dict):
            payload = [payload]
        if not isinstance(payload, list):
            return FetchResult(status_code=200, rows=[])
        rows = [row for row in payload if isinstance(row, dict)]
        return FetchResult(status_code=200, rows=rows)

    return fetch
