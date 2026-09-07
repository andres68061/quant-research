"""Vendor-agnostic, resumable data ingestion.

The pieces fit together as: a declarative manifest (:mod:`core.ingest.catalog`)
describes a vendor's endpoints; :mod:`core.ingest.plan` expands them into tasks;
:mod:`core.ingest.runner` executes one task with retries and atomic storage;
:mod:`core.ingest.pool` drives them concurrently under
:mod:`core.ingest.ratelimit`; and :mod:`core.ingest.journal` records every
outcome for :mod:`core.ingest.report` to render.
"""

from core.ingest.catalog import build_specs, load_manifest, select_specs
from core.ingest.journal import IngestJournal, TaskResult
from core.ingest.plan import plan_run
from core.ingest.pool import execute
from core.ingest.ratelimit import TokenBucket
from core.ingest.runner import IngestRunner, RunSummary
from core.ingest.spec import EndpointSpec, Partition, Payload, Task

__all__ = [
    "EndpointSpec",
    "IngestJournal",
    "IngestRunner",
    "Partition",
    "Payload",
    "RunSummary",
    "Task",
    "TaskResult",
    "TokenBucket",
    "build_specs",
    "execute",
    "load_manifest",
    "plan_run",
    "select_specs",
]
