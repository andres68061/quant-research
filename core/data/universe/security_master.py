"""Permanent identifiers for the entities the platform stores data about.

**The problem.** Every table in this repo is keyed by ticker. A ticker is not an
identity — it is a *lease*. It changes when a company renames (FB to META), it is
reassigned to an unrelated company after a delisting (ticker reuse), it is
punctuated differently by every vendor (BRK.B / BRK-B / BRK/B), and it says
nothing about which company still exists after a merger. Every one of those is a
silent data corruption: the panel keeps a single column, the numbers stay
plausible, and two different companies get averaged into one time series.

Switching vendors makes it acute. A new vendor's ticker for the same company may
differ, and nothing in the repo would notice that a 30-year history had been
stitched onto the wrong entity.

**The fix.** A surrogate key we own — ``qid`` — that is minted once and never
re-used, never re-minted, and never derived from anything that can change. Ticker
becomes an *attribute* of a qid over a date interval, not the identity itself.

    qid        Q0000042       permanent, ours, meaningless by design
    issuer_id  0000320193     SEC CIK — the *company*, free external anchor
    symbol     AAPL           an attribute, valid over an interval

**Why meaningless by design.** Any id that encodes information (ticker, exchange,
sector) has to change when that information changes, which defeats the purpose.

**Two levels, and the difference is load-bearing.** A ``qid`` identifies a
*security* — one tradable line, one price series. A CIK identifies an *issuer* —
the company behind it. One issuer routinely has many securities: American
Financial Group files under a single CIK for its common stock and three separate
bond lines, and Alphabet's GOOG and GOOGL share a CIK while having different
prices.

Treating CIK as the identity therefore merges securities that must stay apart. It
is the anchor for *continuity* (the same security under a new ticker), not for
*identity*. So CIK carries a qid forward only when it is unambiguous — exactly
one security under that issuer, before and after. When an issuer has several
securities, each keeps its own qid and they merely share ``issuer_id``.

**Scope.** This covers the entities we key data by: listed securities today, with
the same pattern available for indices, macro series (FRED ``series_id`` is
already a permanent id), and commodities. It deliberately does *not* attempt full
corporate-action lineage — share-class linking, merger successor chains, and
spin-off parentage need vendor data we do not hold. Those are recorded as
unresolved rather than guessed; see the module's ``__all__`` and ADR 0015.

**Rebuild contract.** ``build_security_master`` must be safe to re-run. It loads
the existing master, keeps every qid that resolves to a known entity, and mints
ids only for genuinely new ones. A rebuild that re-minted ids would be worse than
no master at all, because downstream references would silently repoint.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = [
    "MASTER_COLUMNS",
    "ALIAS_COLUMNS",
    "QID_PATTERN",
    "SecurityMaster",
    "build_security_master",
    "format_qid",
    "normalize_symbol",
]

ROOT = Path(__file__).resolve().parents[2]
MASTER_FILE = ROOT / "data" / "universe" / "security_master.parquet"
ALIAS_FILE = ROOT / "data" / "universe" / "symbol_aliases.parquet"

QID_PREFIX = "Q"
QID_DIGITS = 7
QID_PATTERN = re.compile(rf"^{QID_PREFIX}\d{{{QID_DIGITS}}}$")

MASTER_COLUMNS = (
    "qid",
    "symbol",
    "company_name",
    "issuer_id",
    "exchange",
    "is_delisted",
    "first_seen",
    "last_seen",
    "id_source",
)

ALIAS_COLUMNS = ("qid", "vendor", "vendor_symbol", "valid_from", "valid_to")

# Vendors separate share classes differently (BRK.B / BRK-B / BRK/B), so the
# separator is unified — but never deleted.
#
# Deleting it was the first implementation and it was wrong. FMP encodes
# preferred series as a "-P" suffix, so stripping punctuation merged AL-PA (Air
# Lease preferred A) into ALPA (Alpha Healthcare Acquisition, an unrelated
# company), and C-PK (Citigroup preferred K) into CPK (Chesapeake Utilities).
# Two different companies sharing one id is a silent, permanent corruption.
# Failing to merge two ids that belong to one company is recoverable — the CIK
# anchor catches it, and it is visible.
#
# When in doubt, split. The asymmetry of the two errors is the whole design.
_CLASS_SEPARATORS = re.compile(r"[./]")
_CANONICAL_SEPARATOR = "-"


def format_qid(n: int) -> str:
    """Render an integer as a padded qid (``42`` -> ``Q0000042``)."""
    return f"{QID_PREFIX}{n:0{QID_DIGITS}d}"


def normalize_symbol(symbol: str) -> str:
    """
    Canonical form of a vendor ticker, for matching only.

    Upper-cased with share-class separators unified to ``-``, so ``BRK.B``,
    ``BRK-B`` and ``brk/b`` all match — while ``AL-PA`` stays distinct from
    ``ALPA``, which are two unrelated companies. The original vendor string is
    always preserved in the alias table; this form is never stored as the symbol.
    """
    return _CLASS_SEPARATORS.sub(_CANONICAL_SEPARATOR, str(symbol).strip().upper())


@dataclass(frozen=True)
class SecurityMaster:
    """The master table plus its alias index, with resolution helpers."""

    securities: pd.DataFrame
    aliases: pd.DataFrame

    def resolve(self, symbol: str, as_of: Optional[pd.Timestamp] = None) -> Optional[str]:
        """
        Map a ticker to its qid, respecting the date the ticker was in force.

        Args:
            symbol: Vendor ticker, any punctuation style.
            as_of: The date the ticker was observed. **Supplying this is what
                makes ticker reuse safe**: without it, a reused ticker resolves
                to whichever entity holds it most recently, which is exactly the
                bug the master exists to prevent.

        Returns:
            The qid, or None when the symbol is unknown for that date.
        """
        key = normalize_symbol(symbol)
        candidates = self.aliases[self.aliases["normalized"] == key]
        if candidates.empty:
            return None

        if as_of is None:
            if len(candidates) > 1:
                logger.warning(
                    "'%s' maps to %d entities over time; resolving without as_of "
                    "returns the most recent. Pass as_of to disambiguate.",
                    symbol,
                    len(candidates),
                )
            return str(candidates.sort_values("valid_from").iloc[-1]["qid"])

        stamp = pd.Timestamp(as_of)
        if stamp.tzinfo is not None:
            stamp = stamp.tz_localize(None)
        in_force = candidates[
            (candidates["valid_from"] <= stamp)
            & (candidates["valid_to"].isna() | (candidates["valid_to"] >= stamp))
        ]
        if in_force.empty:
            return None
        return str(in_force.sort_values("valid_from").iloc[-1]["qid"])

    def symbols_for(self, qid: str) -> list[str]:
        """Every ticker this entity has traded under, oldest first."""
        rows = self.aliases[self.aliases["qid"] == qid].sort_values("valid_from")
        return [str(s) for s in rows["vendor_symbol"]]

    def describe(self, qid: str) -> Optional[dict[str, object]]:
        """The master row for a qid as a plain dict."""
        rows = self.securities[self.securities["qid"] == qid]
        if rows.empty:
            return None
        return rows.iloc[0].to_dict()

    @property
    def next_id(self) -> int:
        """The next integer to mint, one past the highest id ever issued."""
        if self.securities.empty:
            return 1
        issued = self.securities["qid"].str.removeprefix(QID_PREFIX).astype(int)
        return int(issued.max()) + 1


def _empty_master() -> SecurityMaster:
    return SecurityMaster(
        securities=pd.DataFrame(columns=list(MASTER_COLUMNS)),
        aliases=pd.DataFrame(columns=[*ALIAS_COLUMNS, "normalized"]),
    )


def load_security_master(
    master_file: Path = MASTER_FILE, alias_file: Path = ALIAS_FILE
) -> SecurityMaster:
    """Load the persisted master, or an empty one when it has never been built."""
    if not master_file.exists():
        return _empty_master()
    securities = pd.read_parquet(master_file)
    aliases = pd.read_parquet(alias_file) if alias_file.exists() else pd.DataFrame()
    if aliases.empty:
        aliases = pd.DataFrame(columns=[*ALIAS_COLUMNS, "normalized"])
    elif "normalized" not in aliases.columns:
        aliases["normalized"] = aliases["vendor_symbol"].map(normalize_symbol)
    return SecurityMaster(securities=securities, aliases=aliases)


def build_security_master(
    universe: pd.DataFrame,
    *,
    existing: Optional[SecurityMaster] = None,
    cik_by_symbol: Optional[dict[str, str]] = None,
    vendor: str = "fmp",
) -> SecurityMaster:
    """
    Mint qids for a universe, preserving every id already issued.

    Resolution order for deciding "is this a security we already know?":

    1. **Normalized ticker** — the primary match. Same ticker, same security,
       in the overwhelming majority of cases.
    2. **Issuer (CIK), only when unambiguous** — carries a qid forward when a
       security changed ticker. Applied *only* if that issuer has exactly one
       security both in the existing master and in the incoming universe.
       Otherwise the issuer has multiple securities (common plus bonds, or two
       share classes) and CIK cannot say which one this is, so it is not used
       for identity — the securities keep separate qids and share ``issuer_id``.

    The ordering matters. An earlier version matched on CIK first and merged 208
    distinct securities across 85 issuers into single ids, because it read a
    company id as a security id.

    Args:
        universe: Frame with at least ``symbol``; optionally ``company_name``,
            ``exchange``, ``is_delisted``, ``ipo_date``, ``delisted_date``.
        existing: Previously built master whose ids must be preserved.
        cik_by_symbol: ``{symbol: cik}`` from :mod:`core.data.vendors.sec.client`.
        vendor: Name recorded in the alias table for these tickers.

    Returns:
        A new :class:`SecurityMaster`. Existing qids are unchanged; new
        securities receive ids continuing from the highest ever issued.
    """
    if "symbol" not in universe.columns:
        raise ValueError("universe must have a 'symbol' column")

    current = existing or _empty_master()
    cik_map = {normalize_symbol(k): v for k, v in (cik_by_symbol or {}).items()}

    # An issuer is usable for continuity only where it names exactly one security
    # on each side of the rebuild.
    incoming_per_issuer: dict[str, int] = {}
    for symbol in universe["symbol"]:
        cik = cik_map.get(normalize_symbol(str(symbol)))
        if cik:
            incoming_per_issuer[cik] = incoming_per_issuer.get(cik, 0) + 1

    known_by_issuer: dict[str, list[str]] = {}
    # key -> [(qid, valid_from, valid_to)] so a reused ticker can be told apart
    # from a continuing one by whether the date spans overlap.
    known_by_symbol: dict[str, list[tuple[str, Optional[pd.Timestamp], Optional[pd.Timestamp]]]] = (
        {}
    )
    if not current.securities.empty:
        for _, row in current.securities.iterrows():
            issuer = row.get("issuer_id")
            if pd.notna(issuer) and issuer:
                known_by_issuer.setdefault(str(issuer), []).append(str(row["qid"]))
        for _, row in current.aliases.iterrows():
            known_by_symbol.setdefault(str(row["normalized"]), []).append(
                (str(row["qid"]), _as_naive(row.get("valid_from")), _as_naive(row.get("valid_to")))
            )

    next_id = current.next_id
    records: list[dict[str, object]] = []
    alias_records: list[dict[str, object]] = []

    for _, row in universe.iterrows():
        symbol = str(row["symbol"])
        key = normalize_symbol(symbol)
        issuer = cik_map.get(key)
        first_seen = _as_naive(row.get("ipo_date"))
        last_seen = _as_naive(row.get("delisted_date"))

        unambiguous_issuer = (
            issuer is not None
            and incoming_per_issuer.get(issuer, 0) == 1
            and len(set(known_by_issuer.get(issuer, []))) == 1
        )

        continuation = _find_continuing_qid(known_by_symbol.get(key, []), first_seen, last_seen)

        if continuation is not None:
            qid, id_source = continuation, "symbol"
        elif unambiguous_issuer:
            # Same security, new ticker: the FB -> META case.
            qid, id_source = known_by_issuer[issuer][0], "issuer"
        else:
            # Either genuinely new, or the same ticker in a later era held by a
            # different company (ticker reuse) — which must not inherit an id.
            qid = format_qid(next_id)
            next_id += 1
            id_source = "reused_ticker" if known_by_symbol.get(key) else "new"
            if issuer:
                known_by_issuer.setdefault(issuer, []).append(qid)
        known_by_symbol.setdefault(key, []).append((qid, first_seen, last_seen))

        records.append(
            {
                "qid": qid,
                "symbol": symbol,
                "company_name": row.get("company_name"),
                "issuer_id": issuer,
                "exchange": row.get("exchange"),
                "is_delisted": bool(row.get("is_delisted", False)),
                "first_seen": _as_naive(row.get("ipo_date")),
                "last_seen": _as_naive(row.get("delisted_date")),
                "id_source": id_source,
            }
        )
        alias_records.append(
            {
                "qid": qid,
                "vendor": vendor,
                "vendor_symbol": symbol,
                "valid_from": _as_naive(row.get("ipo_date")) or pd.Timestamp("1900-01-01"),
                "valid_to": _as_naive(row.get("delisted_date")),
                "normalized": key,
            }
        )

    securities = pd.DataFrame(records, columns=list(MASTER_COLUMNS))
    aliases = pd.DataFrame(alias_records, columns=[*ALIAS_COLUMNS, "normalized"])

    # Aliases from previous vendors/eras are history, not noise: they are how a
    # symbol observed in an old dataset still resolves after a vendor switch.
    if not current.aliases.empty:
        aliases = (
            pd.concat([current.aliases, aliases], ignore_index=True)
            .drop_duplicates(subset=["qid", "vendor", "vendor_symbol"], keep="last")
            .reset_index(drop=True)
        )

    duplicated_qids = int(securities["qid"].duplicated().sum())
    if duplicated_qids:
        # Two securities sharing a permanent id is the failure this module
        # exists to prevent, so it is an error rather than a log line.
        offenders = securities[securities["qid"].duplicated(keep=False)]
        raise ValueError(
            f"{duplicated_qids} securities share a qid, e.g. "
            f"{offenders.head(4)[['qid', 'symbol']].to_dict('records')}"
        )

    logger.info(
        "Security master: %d securities (%d new), %d aliases, %d with an issuer id",
        len(securities),
        next_id - current.next_id,
        len(aliases),
        int(securities["issuer_id"].notna().sum()),
    )
    return SecurityMaster(securities=securities, aliases=aliases)


def _find_continuing_qid(
    known: list[tuple[str, Optional[pd.Timestamp], Optional[pd.Timestamp]]],
    first_seen: Optional[pd.Timestamp],
    last_seen: Optional[pd.Timestamp],
) -> Optional[str]:
    """
    Pick the qid this ticker continues, or None when it is a fresh use of it.

    Ticker reuse and ticker continuity look identical from the symbol alone; the
    date spans separate them. A ticker whose new life starts strictly after a
    previous holder's last day is a *different security* wearing the same badge,
    and inheriting the old id there is precisely the corruption this module
    exists to prevent — one price series spliced from two companies.

    Unknown dates resolve to continuity, because most rows have no lifecycle
    dates and treating every one of them as a new entity would mint a new id on
    every rebuild.
    """
    for qid, _known_from, known_to in known:
        if known_to is None or first_seen is None:
            return qid
        if first_seen <= known_to:
            return qid
        if last_seen is not None and last_seen <= known_to:
            return qid
    return None


def _as_naive(value: object) -> Optional[pd.Timestamp]:
    """Coerce a date-ish value to a tz-naive Timestamp, or None."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        return None
    return stamp.tz_localize(None) if stamp.tzinfo is not None else stamp
