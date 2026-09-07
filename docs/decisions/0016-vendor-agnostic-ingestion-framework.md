# 0016. Replace per-dataset fetch scripts with a declarative manifest and a shared runner

Date: 2026-09-06
Status: accepted

## Context

Every vendor dataset in the repo had grown its own fetch script —
`fetch_fmp_prices.py`, `fetch_fmp_fundamentals.py`, `fetch_fmp_datasets.py`,
`fetch_fmp_market_caps.py`, `fetch_fmp_intraday.py`. Each re-implemented the same
five mechanisms slightly differently: a throttle, a retry loop, "skip what is
already downloaded", a progress log, and a per-symbol failure report. Adding the
next dataset meant copying the nearest script and editing it, which is how five
subtly different retry policies came to exist.

Three facts made that unsustainable rather than merely untidy.

**The download is large and cannot be made small.** All 18 of Financial Modeling
Prep's bulk endpoints return HTTP 402 on this subscription (probed 2026-08-18),
so every universe-wide pull is per-symbol. Against 9,011 symbols in
`data/universe/security_master.parquet` and 175 entitled endpoints, the full
vendor surface is **1,244,039 requests** measured by planning it. The existing
single-call client (`core/data/fmp/client.py`) sleeps between its own calls and
achieves **~134 calls/min measured**, which puts that sweep at roughly 150 hours.

**Nothing could say what a gap meant.** After a multi-hour run, "symbol XYZ has
no earnings rows" had no recoverable explanation. Vendor genuinely has none, or
an HTTP 502 at 3am? The first is a coverage fact to disclose alongside any
**derived layer** panel built on the data; the second is a bug. A log file
answers this badly and the scripts' CSV reports were per-script and inconsistent.

**Point-in-time status was decided too late.** ADR 0010 requires every vendor
dataset to be classified `point_in_time`, `period_end_only` or `snapshot` before
use. In practice the classification lived in a separate registry consulted
(or not) by whoever later built a factor, long after the data was on disk.

## Decision

Describe each vendor's surface as **data**, and execute it with **one shared
runner**.

1. **Manifest.** `config/vendors/{vendor}.json` lists every endpoint with its
   partition (`global`, `per_symbol`, `per_cik`, `per_exchange`, `per_sector`,
   `per_industry`, `per_name`, `batch_symbols`), its `pit_status`, its pagination
   and date-chunking behaviour, and its wave. Validated on load
   (`core/ingest/catalog.py`): an unknown `pit_status` or partition raises
   `ConfigError` at startup rather than becoming a leaked backtest months later.
   FMP's manifest holds 175 endpoints.

2. **Planning is separate from execution** (`core/ingest/plan.py`). Specs expand
   into tasks — one task per stored file — and `--dry-run` prints the count and
   an estimated wall clock **without making a single HTTP call**. A run is costed
   before it is started, not discovered afterwards.

3. **One shared rate limiter, one thread pool** (`core/ingest/ratelimit.py`,
   `core/ingest/pool.py`). A `TokenBucket` hands out permits to all workers
   against one wall clock, so the ceiling holds regardless of pool size. Sixteen
   threads on that bucket sustain **600–700 calls/min measured**, roughly five
   times the serial client, because the threads overlap each other's network
   latency. A probe of 700 requests paced at 700 calls/min returned **zero HTTP
   429 responses**; the configured ceiling is 600, deliberate headroom against the
   vendor's advertised 750. On a 429 the bucket halves its rate and recovers
   geometrically, so a long backfill tracks the true ceiling without tuning.

4. **The checkpoint is a file on disk** (`core/ingest/runner.py`). Each task
   writes one file at `data/raw/{vendor}/{endpoint}/{partition_key}.parquet`
   (`.json.gz` for nested payloads, `.bin` for binary), via a `.tmp` sibling and
   `os.replace`. Resumption is re-running the same command: existing files are
   skipped, and a process killed mid-write leaves a `.tmp` the existence check
   ignores, so a truncated file can never be mistaken for a complete one.

5. **Every outcome is journalled** (`core/ingest/journal.py`). SQLite at
   `data/quality/ingest_journal.db`, one row per task: endpoint, partition key,
   status, HTTP status code, row count, byte count, duration, attempt count,
   error text. The status vocabulary enforces the distinction that motivated the
   work — `empty` (the vendor genuinely has no rows for this key: information,
   recorded once, never retried) versus `failed` (something went wrong, retry
   next run) — with `ok` and `skipped` alongside.

6. **`pit_status` is a required manifest field**, so the ADR-0010 classification
   is made at ingestion time and travels with the data.

## Alternatives rejected

**Keep writing one fetch script per dataset.** The status quo. It has no upper
bound on duplicated mechanism, and every copy re-derives the retry and resumption
semantics that are easy to get subtly wrong — the two pagination bugs found on the
first production run (an endpoint that ignores `page` and returns its whole row
set every time; another that paginates properly and was truncated by the page cap)
are exactly the class of error that would otherwise have to be found and fixed
independently in each script. One shared page-walker means one fix.

**An off-the-shelf orchestrator — Airflow, Dagster, or Prefect.** Rejected as the
wrong shape of tool for this problem. They solve scheduling, dependency graphs,
distributed execution and a web UI; this is a single-box research repo whose
scheduling need is satisfied by four lines of `cron`. What is actually hard here
is **vendor semantics** — response caps, endpoints that ignore paging, endpoints
that answer a bare call while being symbol-scoped, the difference between "the
vendor has nothing" and "the request failed" — and none of those three would
solve any of it. The cost is real: a scheduler daemon, a metadata database, a
deployment story, and a dependency footprint that would land in the CI tiers of
ADR 0004, all to schedule jobs that `cron` already runs.

**`asyncio` with `aiohttp` instead of a thread pool.** The workload is entirely
I/O-bound, so async is a legitimate fit. Rejected because it buys **no measured
throughput**: the thread pool already saturates the vendor's rate ceiling, which
is the binding constraint — the limit is 600–700 calls/min from the vendor, not
from the client. In exchange it would impose a second concurrency model on a
synchronous codebase (pandas, pyarrow, `requests`, every existing fetcher), and
the vendor adapter surface would become async all the way down. Worth revisiting
only if a future vendor's ceiling is high enough that thread scheduling, rather
than the vendor, becomes the limit.

**Store one file per endpoint rather than one per partition key.** Fewer, larger
files, and a simpler storage layout. Rejected because it destroys the checkpoint:
resumption would become all-or-nothing per endpoint, so an interruption 8,000
symbols into a 9,011-symbol endpoint would discard all 8,000. Per-key files make
the unit of retry, the unit of resumption, and the unit of storage the same
thing.

**Track progress in a state file instead of file existence.** The conventional
approach, and it introduces a second source of truth that can disagree with the
filesystem — a state file saying "done" for a file that was never written, or a
file on disk the state file has forgotten. File existence cannot drift from
itself. The journal records what happened; it is never consulted to decide what
to do next.

## Consequences

**Onboarding a vendor is one JSON manifest plus one transport adapter.** The
adapter is a `fetch(path, params) -> FetchResult` callable — roughly 100 lines,
as in `core/data/fmp/transport.py` — and that is the entire vendor interface. The
runner contains no vendor-specific code. Adding an *endpoint* to an existing
vendor is one JSON object and no Python at all.

**The point-in-time classification travels with the data.** `pit_status` is
declared in the manifest, validated when it loads, and available beside every
stored file. "May a backtest use this?" is answered at ingestion rather than
rediscovered by whoever builds a factor on it six months later.

**The journal is the audit trail for the raw layer.** Coverage questions are now
SQL against `data/quality/ingest_journal.db` rather than grep against a log:
which endpoints failed and with what status, which symbols the vendor has no data
for versus which ones we failed to fetch, what a run cost, and whether an
endpoint's row count changed between runs. A text report per run is written to
`data/quality/ingest_reports/{run_id}.txt`.

**A new class of error becomes possible: a manifest entry that is valid but
wrong.** A misdeclared partition passes validation and produces plausible files —
seven FMP endpoints answer a bare call with a well-formed payload while being
genuinely symbol-scoped, and had to be pinned to `per_symbol` by hand. The
runbook documents the check; nothing enforces it.

**Two storage generations now coexist under `data/raw/fmp/`.** The older
per-dataset fetchers wrote `prices/`, `fundamentals/`, `market_caps/`,
`universe/`, `constituents/`, `intraday/` and `commodities/`; the framework
writes one directory per manifest endpoint name. Both are live, and the
derived-layer builders still read the older paths. Consolidating them is a
separate decision, deliberately not bundled into this one.

**The full sweep is feasible, not free.** 1,244,039 tasks is ~34.6 hours
estimated at 600 calls/min — down from ~150 hours, but still something to plan
rather than launch casually. `--dry-run` exists so that number is seen first.

**Revisit if** the subscription gains bulk endpoints (which would make most
per-symbol waves obsolete rather than merely slow), if a second vendor's
semantics do not fit the `EndpointSpec` fields, or if scheduling grows past what
`cron` can express.

## Related

- [0004](0004-ci-scope-and-dependency-tiers.md) — the dependency-tier policy that
  the orchestrator alternative would have strained.
- [0010](0010-vendor-metric-point-in-time-classification.md) — the point-in-time
  classification that `pit_status` carries.
- [`docs/INGESTION.md`](../INGESTION.md) — the operator runbook.
- [`docs/vendor/fmp/INGESTION.md`](../vendor/fmp/INGESTION.md) — the FMP manifest,
  waves, and entitlements.
