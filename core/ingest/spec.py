"""Vendor-agnostic description of one ingestible endpoint.

The point of this module is that adding a vendor should mean writing a list of
:class:`EndpointSpec` values, not another bespoke fetch script. Everything the
runner needs to turn an endpoint into a set of resumable, individually
checkpointed tasks is declared here:

- **How it partitions.** One endpoint becomes one task (a global list) or 9,011
  tasks (one per symbol). :class:`Partition` names the expansion.
- **How it is stored.** Raw payloads land at
  ``data/raw/{vendor}/{name}/{partition_key}.parquet``, one file per task, which
  is what makes an interrupted multi-hour backfill resumable by file existence.
- **Whether a backtest may use it.** ``pit_status`` carries the point-in-time
  classification (ADR-0010) with the data itself, so the question is answered at
  ingestion rather than rediscovered later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

# Point-in-time classifications, re-exported so a vendor catalog imports one
# module. Kept identical to core.data.fmp.datasets for continuity.
POINT_IN_TIME = "point_in_time"
PERIOD_END_ONLY = "period_end_only"
SNAPSHOT = "snapshot"
PIT_STATUSES = frozenset({POINT_IN_TIME, PERIOD_END_ONLY, SNAPSHOT})


class Partition(str, Enum):
    """How one endpoint expands into individually checkpointed tasks."""

    GLOBAL = "global"
    PER_SYMBOL = "per_symbol"
    PER_CIK = "per_cik"
    PER_EXCHANGE = "per_exchange"
    PER_SECTOR = "per_sector"
    PER_INDUSTRY = "per_industry"
    PER_NAME = "per_name"
    BATCH_SYMBOLS = "batch_symbols"


class Subject(str, Enum):
    """Whose data the rows of a response describe.

    ``stock-peers`` is the reason this exists: a request for ``LUV`` returns
    ``CRS, JBHT, JOBY`` — the peers *of* LUV, not LUV. The tie between the file
    and the company it was requested for lives in the request key, so a reader
    that assumes ``symbol`` identifies the subject would attribute one company's
    peer set to another.
    """

    REQUEST_KEY = "request_key"
    """Rows describe the entity that was requested; a ``symbol`` column should
    match the request key, and a mismatch means the vendor returned the wrong
    company's data."""

    RELATED = "related"
    """Rows describe entities related to the requested one. Only the request key
    identifies the subject."""


class Payload(str, Enum):
    """Wire format of a successful response."""

    JSON = "json"
    BINARY = "binary"


@dataclass(frozen=True)
class EndpointSpec:
    """
    One vendor endpoint and everything needed to ingest it.

    Attributes:
        name: Storage directory and CLI identifier. Stable across runs — renaming
            it orphans previously downloaded files.
        endpoint: Path relative to the vendor base URL.
        partition: How the endpoint expands into tasks.
        pit_status: Point-in-time classification (ADR-0010).
        params: Static query parameters, excluding the partition key and API key.
        date_columns: Columns parsed to datetime64 on load.
        primary_date: Column to sort ascending by; None for snapshots.
        payload: Wire format; BINARY payloads are stored verbatim, not as parquet.
        subject: Whether rows describe the requested entity or entities related
            to it. Drives the identity check at ingestion.
        batch_size: Symbols per call when partition is BATCH_SYMBOLS.
        priority: Wave number. Lower runs first; the driver groups by this so the
            highest-value data lands before a long tail that may take hours.
        derivable: True when the repo can compute this from data it already owns
            (vendor technical indicators, for instance). Still ingestible, but
            deprioritised and excluded from --wave defaults.
        cadence: How often a scheduled refresh should re-pull it.
        date_chunk_years: When set, the task is split into fixed-length
            ``from``/``to`` windows. Required wherever the vendor caps a
            response: FMP returns at most 5,000 EOD bars per call, so a
            40-year history silently truncates to ~20 years without chunking.
        history_start: Earliest date to request when chunking.
        paginate: Walk ``page``/``limit`` until exhausted. Required wherever the
            vendor caps a response, or the stored file is a silent first page.
        page_size: Rows per page when paginating.
        max_pages: Safety cap on pages walked.
        notes: Why the dataset is worth having, and its traps.
    """

    name: str
    endpoint: str
    partition: Partition
    pit_status: str
    params: dict[str, Any] = field(default_factory=dict)
    date_columns: tuple[str, ...] = ()
    primary_date: Optional[str] = None
    payload: Payload = Payload.JSON
    subject: Subject = Subject.REQUEST_KEY
    batch_size: int = 100
    priority: int = 3
    derivable: bool = False
    cadence: str = "daily"
    paginate: bool = False
    page_size: int = 100
    max_pages: int = 200
    date_chunk_years: Optional[int] = None
    history_start: str = "1985-01-01"
    notes: str = ""

    def __post_init__(self) -> None:
        if self.pit_status not in PIT_STATUSES:
            raise ValueError(
                f"{self.name}: pit_status {self.pit_status!r} not in {sorted(PIT_STATUSES)}"
            )
        if not self.name or "/" in self.name:
            raise ValueError(f"invalid spec name {self.name!r}: must be a single path segment")

    @property
    def needs_universe(self) -> bool:
        """Whether expanding this spec requires the symbol universe."""
        return self.partition in (Partition.PER_SYMBOL, Partition.BATCH_SYMBOLS)


@dataclass(frozen=True)
class Task:
    """
    One unit of work: a single request whose result is one stored file.

    Attributes:
        spec_name: Owning :class:`EndpointSpec` name.
        key: Partition key — a symbol, a CIK, ``"_all"`` for global endpoints.
        params: Fully resolved query parameters, excluding the API key.
    """

    spec_name: str
    key: str
    params: dict[str, Any]
