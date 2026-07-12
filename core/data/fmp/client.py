"""Minimal HTTP client for the FMP stable API.

Handles authentication, timeouts, retries with exponential backoff, and a
simple client-side rate limiter that keeps request volume safely under the
Premium plan's 750 calls/minute.

Endpoint catalog: ``docs/vendor/fmp/README.md``.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

import requests

from core.exceptions import ConfigError

logger = logging.getLogger(__name__)

FMP_BASE_URL = "https://financialmodelingprep.com/stable"

_MAX_RETRIES = 5
_BACKOFF_BASE_SECONDS = 2.0
_TIMEOUT_SECONDS = 60
# ~500 calls/min ceiling; Premium allows 750/min but leave headroom for bursts.
_MIN_SECONDS_BETWEEN_CALLS = 0.12

_last_call_monotonic: float = 0.0


def _throttle() -> None:
    """Sleep just enough to stay under the client-side calls-per-minute ceiling."""
    global _last_call_monotonic
    elapsed = time.monotonic() - _last_call_monotonic
    if elapsed < _MIN_SECONDS_BETWEEN_CALLS:
        time.sleep(_MIN_SECONDS_BETWEEN_CALLS - elapsed)
    _last_call_monotonic = time.monotonic()


def fmp_get(
    path: str,
    params: Optional[dict[str, Any]] = None,
    api_key: Optional[str] = None,
) -> Any:
    """
    GET an FMP stable endpoint and return the decoded JSON.

    Args:
        path: Endpoint path relative to the stable base URL, e.g.
            ``"historical-price-eod/dividend-adjusted"``.
        params: Query parameters (the API key is added automatically).
        api_key: Override key; defaults to ``FMP_API_KEY`` from ``config.settings``.

    Returns:
        Decoded JSON (usually a list of dicts).

    Raises:
        ConfigError: If no API key is configured or the key is rejected (HTTP 403).
        requests.HTTPError: If the request still fails after all retries.

    Example:
        >>> rows = fmp_get("quote", {"symbol": "AAPL"})  # doctest: +SKIP
    """
    if api_key is None:
        from config.settings import FMP_API_KEY

        api_key = FMP_API_KEY
    if not api_key:
        raise ConfigError("FMP_API_KEY is not set; add it to .env")

    query = dict(params or {})
    query["apikey"] = api_key
    url = f"{FMP_BASE_URL}/{path}"

    last_error: Optional[Exception] = None
    for attempt in range(_MAX_RETRIES):
        _throttle()
        try:
            response = requests.get(url, params=query, timeout=_TIMEOUT_SECONDS)
        except requests.RequestException as exc:
            last_error = exc
            wait = _BACKOFF_BASE_SECONDS * (2**attempt)
            logger.warning(
                "FMP request error on %s (attempt %d/%d): %s; retrying in %.1fs",
                path,
                attempt + 1,
                _MAX_RETRIES,
                exc,
                wait,
            )
            time.sleep(wait)
            continue

        if response.status_code == 200:
            return response.json()
        if response.status_code == 403:
            raise ConfigError(
                "FMP rejected the API key (HTTP 403). Check FMP_API_KEY in .env "
                "and the plan's endpoint entitlements."
            )
        if response.status_code in (429, 500, 502, 503, 504):
            wait = _BACKOFF_BASE_SECONDS * (2**attempt)
            logger.warning(
                "FMP HTTP %d on %s (attempt %d/%d); backing off %.1fs",
                response.status_code,
                path,
                attempt + 1,
                _MAX_RETRIES,
                wait,
            )
            time.sleep(wait)
            last_error = requests.HTTPError(f"HTTP {response.status_code} on {path}")
            continue

        # Do not use response.raise_for_status(): its message embeds the full
        # request URL including the API key, which would leak into logs.
        raise requests.HTTPError(f"HTTP {response.status_code} on {path}: {response.text[:200]}")

    raise requests.HTTPError(
        f"FMP request failed after {_MAX_RETRIES} retries: {path}"
    ) from last_error
