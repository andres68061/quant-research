"""SEC EDGAR client: company CIK lookup and recent filing accepted dates.

Uses the public ``data.sec.gov`` JSON APIs with SEC fair-access User-Agent.
No HTML scraping.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd
import requests

from core.exceptions import DataSchemaError

logger = logging.getLogger(__name__)

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
DEFAULT_USER_AGENT = "QuantResearchPlatform andraeo@outlook.com"
FORM_TYPES = ("10-K", "10-Q", "10-K/A", "10-Q/A")

__all__ = [
    "DEFAULT_USER_AGENT",
    "fetch_company_tickers",
    "fetch_recent_filings",
    "build_filings_sample",
]


def _sec_headers(user_agent: str = DEFAULT_USER_AGENT) -> dict[str, str]:
    return {
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
        "Host": "www.sec.gov",
    }


def fetch_company_tickers(
    *,
    session: Optional[requests.Session] = None,
    user_agent: str = DEFAULT_USER_AGENT,
    timeout: float = 30.0,
) -> dict[str, str]:
    """
    Map ticker → zero-padded 10-digit CIK string.

    Returns:
        ``{"AAPL": "0000320193", ...}``
    """
    http = session or requests.Session()
    response = http.get(
        SEC_TICKERS_URL,
        headers=_sec_headers(user_agent),
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise DataSchemaError(f"Unexpected SEC tickers payload: {type(payload)}")

    mapping: dict[str, str] = {}
    for row in payload.values():
        if not isinstance(row, dict):
            continue
        ticker = str(row.get("ticker") or "").strip().upper()
        cik_raw = row.get("cik_str")
        if not ticker or cik_raw is None:
            continue
        mapping[ticker] = f"{int(cik_raw):010d}"
    if not mapping:
        raise DataSchemaError("SEC company_tickers.json produced no ticker→CIK rows")
    return mapping


def fetch_recent_filings(
    cik: str,
    *,
    forms: tuple[str, ...] = FORM_TYPES,
    limit: int = 20,
    session: Optional[requests.Session] = None,
    user_agent: str = DEFAULT_USER_AGENT,
    timeout: float = 30.0,
) -> pd.DataFrame:
    """
    Recent SEC filings for one CIK from the submissions JSON.

    Returns:
        DataFrame with columns ``form``, ``filing_date``, ``accepted_date``,
        ``accession_number``.
    """
    cik_padded = f"{int(cik):010d}"
    http = session or requests.Session()
    headers = _sec_headers(user_agent)
    headers["Host"] = "data.sec.gov"
    url = SEC_SUBMISSIONS_URL.format(cik=cik_padded)
    response = http.get(url, headers=headers, timeout=timeout)
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    recent = (payload.get("filings") or {}).get("recent") or {}
    forms_list = recent.get("form") or []
    filing_dates = recent.get("filingDate") or []
    acceptance = recent.get("acceptanceDateTime") or []
    accessions = recent.get("accessionNumber") or []

    rows: list[dict[str, Any]] = []
    for i, form in enumerate(forms_list):
        if form not in forms:
            continue
        accepted_raw = acceptance[i] if i < len(acceptance) else None
        rows.append(
            {
                "form": form,
                "filing_date": filing_dates[i] if i < len(filing_dates) else None,
                "accepted_date": (
                    str(accepted_raw)[:10] if accepted_raw else None
                ),
                "accession_number": accessions[i] if i < len(accessions) else None,
            }
        )
        if len(rows) >= limit:
            break

    return pd.DataFrame(rows)


def build_filings_sample(
    tickers: list[str],
    *,
    ticker_to_cik: Optional[dict[str, str]] = None,
    session: Optional[requests.Session] = None,
    user_agent: str = DEFAULT_USER_AGENT,
    forms: tuple[str, ...] = FORM_TYPES,
    limit_per_symbol: int = 12,
) -> pd.DataFrame:
    """
    Pull recent 10-K/10-Q accepted dates for ``tickers``.

    Returns long panel columns:
    ``symbol``, ``cik``, ``form``, ``filing_date``, ``accepted_date``,
    ``accession_number``.
    """
    mapping = ticker_to_cik or fetch_company_tickers(session=session, user_agent=user_agent)
    frames: list[pd.DataFrame] = []
    for symbol in tickers:
        key = symbol.strip().upper()
        cik = mapping.get(key)
        if not cik:
            logger.warning("No SEC CIK for ticker %s — skipping", key)
            continue
        try:
            filings = fetch_recent_filings(
                cik,
                forms=forms,
                limit=limit_per_symbol,
                session=session,
                user_agent=user_agent,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("SEC filings fetch failed for %s: %s", key, exc)
            continue
        if filings.empty:
            continue
        filings = filings.assign(symbol=key, cik=cik)
        frames.append(filings)

    if not frames:
        return pd.DataFrame(
            columns=[
                "symbol",
                "cik",
                "form",
                "filing_date",
                "accepted_date",
                "accession_number",
            ]
        )
    return pd.concat(frames, ignore_index=True)
