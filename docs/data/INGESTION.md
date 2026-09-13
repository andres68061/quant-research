# Data ingestion runbook

How to download a vendor's data into the **raw layer** (`data/raw/`) with the
framework in [`core/ingest/`](../../core/ingest), and how to tell afterwards what
actually landed.

The framework is deliberately vendor-agnostic. A vendor is described by a JSON
**manifest** (which endpoints exist, how each one partitions, what its
point-in-time status is) plus one **transport adapter** (roughly 100 lines that
turn "path plus query parameters" into an HTTP response). Everything else —
planning, concurrency, rate limiting, retries, atomic storage, resumption, and
the audit trail — is shared. Today there is one vendor, Financial Modeling Prep
(FMP); its specifics live in
[`docs/sources/vendor/fmp/INGESTION.md`](../sources/vendor/fmp/INGESTION.md).

---

## 1. Why this exists

Before the framework, every dataset got its own fetch script. Each one
re-implemented the same five things — a throttle, a retry loop, "skip what I
already downloaded", a progress log, and a per-symbol failure report — slightly
differently, and none of them could answer the question that matters after a
long download:

> Symbol XYZ has no earnings rows. Is that because the vendor has none, or
> because we got an HTTP 502 at 3am and nobody noticed?

That question is the difference between a real coverage gap you can disclose and
a silent hole in a **derived layer** panel built on top of it. The framework
answers it by writing every task outcome to a queryable journal.

---

## 2. Architecture: who owns what

```
config/vendors/{vendor}.json      the manifest: one JSON entry per endpoint
          |
          |  core.ingest.catalog     load + validate + filter by wave
          v
     [EndpointSpec] x N             one per endpoint
          |
          |  core.ingest.plan        expand each spec by its partition keys
          v
        [Task] x M                   one per stored file (M can be ~10^6)
          |
          |  core.ingest.pool        thread pool, progress logging
          |  core.ingest.ratelimit   one shared token bucket for all workers
          |  core.ingest.runner      per task: skip / fetch / retry / store
          |  core.ingest.paginate    walk pages for endpoints that cap a response
          v
   data/raw/{vendor}/{endpoint}/{partition_key}.parquet
          |
          |  core.ingest.journal     one SQLite row per task outcome
          v
   data/quality/ingest_journal.db
          |
          |  core.ingest.report      render a run into text
          v
   data/quality/ingest_reports/{run_id}.txt
```

| Module | Owns |
|---|---|
| [`core/ingest/spec.py`](../../core/ingest/spec.py) | `EndpointSpec` — the declarative description of one endpoint (partition, point-in-time status, pagination, date chunking, wave). `Task` — one request whose result is one stored file. |
| [`core/ingest/catalog.py`](../../core/ingest/catalog.py) | Reading `config/vendors/{vendor}.json`, validating it (an unknown `pit_status` raises `ConfigError` on load rather than becoming a leaked backtest months later), and selecting specs by wave or by name. |
| [`core/ingest/plan.py`](../../core/ingest/plan.py) | Expanding specs into tasks: one global endpoint becomes 1 task, a per-symbol endpoint becomes 9,011, a date-chunked price endpoint becomes 9,011 × (number of windows). Planning is separate from execution so a run can be costed before it is started. |
| [`core/ingest/ratelimit.py`](../../core/ingest/ratelimit.py) | `TokenBucket` — one shared, thread-safe permit source that paces every worker against one wall clock, with adaptive backoff on HTTP 429. |
| [`core/ingest/runner.py`](../../core/ingest/runner.py) | One task end to end: skip if already on disk, acquire a permit, fetch, retry retryable statuses, store atomically, return a `TaskResult`. Also installs the SIGINT/SIGTERM handler that drains rather than kills. |
| [`core/ingest/paginate.py`](../../core/ingest/paginate.py) | Walking `page`/`limit` for endpoints that cap their response, and stopping correctly (see [§11 Traps](#11-traps)). |
| [`core/ingest/pool.py`](../../core/ingest/pool.py) | The thread pool, progress lines every 30 seconds, and turning a crashed worker into a journalled `failed` task instead of a lost one. |
| [`core/ingest/journal.py`](../../core/ingest/journal.py) | The SQLite audit trail: one row per task with status, HTTP status code, row count, byte count, duration, attempt count, and error text. |
| [`core/ingest/report.py`](../../core/ingest/report.py) | Rendering one run's journal rows into the text report a human reads afterwards. |
| [`core/data/vendors/fmp/transport.py`](../../core/data/vendors/fmp/transport.py) | The FMP-specific half: build the URL, attach the API key, redact the key from any error text, normalise the response into `FetchResult(status_code, rows, content, error)`. |
| [`scripts/ingest/ingest_fmp.py`](../../scripts/ingest/ingest_fmp.py) | The command-line driver: resolve partition keys, plan, run, report. |
| [`scripts/ingest/validate_fmp_manifest.py`](../../scripts/ingest/validate_fmp_manifest.py) | Pre-flight check: call each spec once against the live vendor and list the ones that return nothing (see [Trap 4](#trap-4--a-spec-missing-a-required-parameter-returns-empty-not-an-error)). |
| [`scripts/ingest/run_full_ingestion.sh`](../../scripts/ingest/run_full_ingestion.sh) | Unattended sequential driver for several waves in a row. |

**The runner knows nothing about any vendor.** It receives a
`fetch(path, params) -> FetchResult` callable. That is the entire vendor
interface.

---

## 3. Storage layout

```
data/raw/fmp/{endpoint_name}/{partition_key}.parquet
```

- `{endpoint_name}` is the manifest's `name` field (`earnings`, `cik_list`,
  `historical_price_eod_full`). It is the storage identity — renaming it in the
  manifest orphans everything previously downloaded under the old name.
- `{partition_key}` is the symbol (`AAPL.parquet`), the CIK, the sector, the
  literal `_all` for endpoints that take no key, `batch_00007` for
  batched-symbol endpoints, or `AAPL__1985_1994` for a date-chunked window.
- Suffix depends on payload: `.parquet` normally; `.json.gz` when the vendor's
  nested JSON does not survive a DataFrame round-trip (whole financial reports,
  for instance) — the payload is kept rather than the task failed; `.bin` for
  binary payloads such as the spreadsheet report endpoint.

One file per task is not an aesthetic choice. It is the checkpoint (§7), the
unit of retry, and the thing that lets a 12-hour download be stopped and
restarted without bookkeeping.

The journal and reports live outside the raw layer, with the rest of the
data-quality artifacts:

```
data/quality/ingest_journal.db          SQLite, all runs, one row per task
data/quality/ingest_reports/{run_id}.txt   text report, written when a run finishes
```

---

## 4. Cost a run before you spend a call

`--dry-run` plans and prints, and makes zero HTTP requests:

```bash
/opt/anaconda3/envs/quant/bin/python scripts/ingest/ingest_fmp.py --wave 1,2 --dry-run
```

It prints the task count per endpoint, the total, and an estimated wall clock at
the configured rate.

**Read that estimate as a floor, not a forecast.** It is `tasks / rate`, and it
assumes one HTTP call per task with nothing already on disk. Two things push the
real number up:

- A **paginated** endpoint issues one call per page, so one task can be dozens
  of calls. Wave 1 is 66 tasks — arithmetically about 7 seconds at 600
  calls/min — but measured **5.0 minutes** wall clock on the 2026-09-06 run,
  because 27 of its 66 endpoints walk pages. (That run predates the
  repeat-page guard described in [§11 Trap 1](#trap-1--endpoints-that-ignore-the-page-parameter),
  so some of those 5 minutes was spent re-walking pages that will now stop
  early; treat 5.0 minutes as an upper bound for wave 1.)
- A **retried** task issues up to `--max-retries` calls (default 4).

Things that push it down: anything already downloaded is skipped without a call.

---

## 5. Running a wave

```bash
# waves 1 and 2, logging to a file as well as stdout
/opt/anaconda3/envs/quant/bin/python scripts/ingest/ingest_fmp.py \
    --wave 1,2 --log-file runtime/logs/ingest_fmp.log

# one endpoint only, re-fetching files that already exist
/opt/anaconda3/envs/quant/bin/python scripts/ingest/ingest_fmp.py \
    --endpoints earnings --force

# a handful of symbols, for a smoke test
/opt/anaconda3/envs/quant/bin/python scripts/ingest/ingest_fmp.py \
    --endpoints earnings,dividends --symbols AAPL,MSFT,NVDA

# print the most recent run's report and exit
/opt/anaconda3/envs/quant/bin/python scripts/ingest/ingest_fmp.py --report
```

Useful flags:

| Flag | Default | Effect |
|---|---|---|
| `--wave` | `1,2` | Comma-separated wave numbers, or `all`. |
| `--endpoints` | — | Explicit endpoint names; overrides `--wave` and also lifts the `derivable` exclusion. |
| `--symbols` | — | Explicit symbol list instead of the security master's 9,011. |
| `--workers` | `16` | Thread pool size. |
| `--rate` | manifest (`600` for FMP) | Calls per minute ceiling, shared across all workers. |
| `--max-retries` | `4` | Attempts before a task is recorded as `failed`. |
| `--force` | off | Re-fetch tasks whose output file already exists. |
| `--include-derivable` | off | Include endpoints the repo could compute from data it already holds (FMP's technical indicators). |
| `--dry-run` | off | Plan and cost only; no HTTP calls. |
| `--report` | off | Render the last run's report and exit. |

A long run belongs in the background with its log on disk:

```bash
nohup /opt/anaconda3/envs/quant/bin/python scripts/ingest/ingest_fmp.py \
    --wave 2 --log-file runtime/logs/ingest_wave2.log > /dev/null 2>&1 &
```

Progress lines appear every 30 seconds and carry the observed tasks/minute and a
running estimate of the time remaining.

### Pre-flight: validate the manifest against the live vendor

Before committing to a long wave, check that every endpoint spec actually
returns rows. This costs one call per spec — a few hundred, not a few hundred
thousand:

```bash
/opt/anaconda3/envs/quant/bin/python scripts/ingest/validate_fmp_manifest.py
/opt/anaconda3/envs/quant/bin/python scripts/ingest/validate_fmp_manifest.py --symbol MSFT --waves 2,3
```

It calls each spec once with a liquid, long-listed symbol (default `AAPL`) that
should have data for almost everything, and prints two lists: specs that returned
a non-200, and specs that returned HTTP 200 **with no rows**. The second list is
the one that matters — see
[Trap 4](#trap-4--a-spec-missing-a-required-parameter-returns-empty-not-an-error).
A spec returning nothing for `AAPL` is not proof of a bug, but it is the short
list worth reading before spending a day of calls.

### Running several waves unattended

```bash
nohup bash scripts/ingest/run_full_ingestion.sh "1 2 3 4" > runtime/logs/ingestion_all.out 2>&1 &
```

The shell driver runs waves **one at a time on purpose**: the rate limiter lives
inside a single process, so two concurrent `ingest_fmp.py` runs would emit twice
the intended calls per minute and risk the vendor throttling the key. The script
refuses to start when it finds another `ingest_fmp.py` already running. A wave
that fails does not stop the ones after it — partial coverage of a later wave is
more useful than none, and the journal records exactly what was missed. Rate and
pool size are overridable: `RATE=400 WORKERS=8 bash scripts/ingest/run_full_ingestion.sh "2"`.

**Never run two ingestions at once**, by any route — that includes starting a
manual run while a scheduled one is going.

---

## 6. Waves and their measured cost

Waves are a priority ordering, not a dependency graph: lower waves run first so
the highest-value data lands before a long tail that may take hours. A wave-1
run also produces the CIK, sector, industry and exchange lists that later waves
need as partition keys.

| Wave | Contents | Tasks | Wall clock |
|---:|---|---:|---|
| 1 | Global reference row sets and calendars: symbol directories, CIK list, index constituents, screener, calendars (earnings, dividends, splits, IPOs, economic), Commitment of Traders, Treasury rates, latest news feeds | 66 | **5.0 min measured** (2026-09-06 run, before the Trap 1 fix) |
| 2 | Core per-symbol history: all financial statements and their as-reported and growth variants, ratios, key metrics, earnings, dividends, splits, analyst grades, insider trading, employee counts, segments | 441,539 | ~12.3 h estimated |
| 3 | Per-symbol and per-key snapshots: quotes, aftermarket, TTM metrics, consensus, ETF composition, sector and industry performance, SEC filing searches, per-CIK profiles | 162,653 | ~4.5 h estimated |
| 4 | End-of-day price variants (`light`, `full`, `dividend-adjusted`, `non-split-adjusted`), date-chunked into 10-year windows from 1985 | 180,220 | ~5.0 h estimated |
| 5 | Vendor technical indicators (`derivable: true` — the repo can compute these from prices it already owns, so they are excluded from wave defaults) | 81,099 | ~2.3 h estimated |
| 6 | Intraday charts (1min through 4hour), 1-year windows from 2020 | 378,462 | ~10.5 h estimated |
| **all** | Everything above, including intraday | **1,244,039** | **~34.6 h estimated** |

Task counts are **measured**: they come from `--dry-run` against the 9,011-symbol
security master and the 175-endpoint manifest. Wall clocks other than wave 1 are
**estimated** as `tasks / 600 calls per minute`, so they are floors for the
reasons in §4. Wave 1's figure is an observed wall clock from one run.

---

## 7. Resuming an interrupted run

Just run the same command again. Nothing else is required — no `--resume` flag,
no state file, no manual list of what to skip.

```bash
# it died at hour 6; this picks up where it stopped
/opt/anaconda3/envs/quant/bin/python scripts/ingest/ingest_fmp.py --wave 2 --log-file runtime/logs/ingest_wave2.log
```

**Why this is correct**, and not merely usually correct:

1. **Completion is file existence, not a cursor.** Before fetching, the runner
   checks whether `data/raw/{vendor}/{endpoint}/{key}` exists with any of the
   three suffixes. If it does, the task is recorded as `skipped` and costs zero
   calls. There is no in-memory or on-disk progress pointer that can drift out of
   sync with what is actually on disk.
2. **Writes are atomic.** Every payload is written to a `.tmp` sibling and then
   moved into place with `os.replace`, which is atomic within a filesystem. A
   process killed mid-write leaves a `.tmp` file, which the existence check does
   not recognise — so the next run refetches that task. A truncated file can
   never be mistaken for a complete one.
3. **Vendor-empty is a completed state.** When the vendor genuinely returns no
   rows, an empty parquet file is written anyway. That is what stops the next
   run from re-asking 8,000 times for data the vendor does not have (§8).
4. **Interrupts drain rather than kill.** SIGINT (Ctrl-C) and SIGTERM stop
   scheduling new work and let in-flight tasks finish, so stopping a long
   backfill does not create partial files or lose journal rows.

The consequence: a second run over already-downloaded data is cheap and safe.
Re-running wave 2 after it completed costs 441,539 filesystem checks and zero
HTTP calls.

To deliberately refetch — because the vendor restated history, or because a
stored file is suspect — use `--force`, ideally narrowed with `--endpoints`
and/or `--symbols`.

---

## 8. `empty` versus `failed`

This is the distinction the journal exists to preserve. Four statuses:

| Status | Meaning | Retried later? | File on disk? |
|---|---|---|---|
| `ok` | The vendor returned rows and they were stored. | No | Yes, with rows |
| `empty` | HTTP 200 with zero rows: the vendor genuinely has nothing for this partition key. **This is information, not an error.** | No | Yes, empty |
| `skipped` | The output file already existed; no call was made. | n/a | Yes, from an earlier run |
| `failed` | Something went wrong: a non-200 the retries could not clear, a transport error, an unentitled endpoint (HTTP 402), or an interrupt before completion. | Yes, on the next run | No |

Why the split matters in practice: most companies have no executive-compensation
filings, so an endpoint reporting `empty` for 8,000 of 9,011 symbols is working
correctly and its coverage gap is a fact to disclose. An endpoint reporting
`failed` with HTTP 402 for all 9,011 is unentitled on this subscription and its
directory must not be mistaken for data. Both produce "not much data"; only one
is a problem, and only the journal can tell you which.

The rendered report calls out endpoints that failed more often than they
succeeded under a heading that says not to trust them.

---

## 9. Reading the journal with SQL

The journal is plain SQLite at `data/quality/ingest_journal.db`, two tables:

```sql
runs (run_id, vendor, started_at, finished_at, argv, n_tasks)
tasks(run_id, spec_name, partition_key, status, http_code,
      n_rows, n_bytes, duration_s, attempts, error, recorded_at)
```

`started_at` / `finished_at` / `recorded_at` are Unix epoch seconds; wrap them in
`datetime(x, 'unixepoch', 'localtime')` to read them.

**Open it read-only** so a query can never block or be blocked by an ingestion
that is still running:

```bash
sqlite3 -readonly data/quality/ingest_journal.db
```

or from Python:

```python
import sqlite3
connection = sqlite3.connect("file:data/quality/ingest_journal.db?mode=ro", uri=True)
```

### What did the last run cost?

```sql
SELECT r.run_id,
       datetime(r.started_at, 'unixepoch', 'localtime')                       AS started,
       ROUND((COALESCE(r.finished_at, strftime('%s','now')) - r.started_at)/60.0, 1)
                                                                              AS minutes,
       r.n_tasks                                                              AS planned,
       SUM(t.status = 'ok')                                                   AS ok,
       SUM(t.status = 'empty')                                                AS empty,
       SUM(t.status = 'skipped')                                              AS skipped,
       SUM(t.status = 'failed')                                               AS failed,
       SUM(t.attempts)                                                        AS http_calls_min,
       ROUND(SUM(t.n_rows) / 1e6, 2)                                          AS million_rows,
       ROUND(SUM(t.n_bytes) / 1e6, 1)                                         AS megabytes
FROM runs r
JOIN tasks t USING (run_id)
WHERE r.run_id = (SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1)
GROUP BY r.run_id;
```

`http_calls_min` is a **lower bound** on calls spent: `attempts` counts fetch
attempts, and a paginated task makes one call per page inside a single attempt.
A NULL `finished_at` means the run is still going or was killed without draining.

### Which endpoints failed most, and with what?

```sql
SELECT spec_name,
       SUM(status = 'failed') AS failed,
       SUM(status = 'ok')     AS ok,
       SUM(status = 'empty')  AS empty,
       MIN(http_code)         AS a_status_code,
       SUBSTR(MIN(error), 1, 80) AS example_error
FROM tasks
WHERE run_id = (SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1)
GROUP BY spec_name
HAVING failed > 0
ORDER BY failed DESC;
```

An endpoint whose `failed` count equals its task count and whose status code is
402 is **unentitled**, not broken — see the FMP entitlement list in
[`docs/data/DATA_INVENTORY.md` §6](DATA_INVENTORY.md#6-fmp-plan-entitlements-probed).

### Which symbols have no earnings data, and why?

```sql
-- the shape of the answer first
SELECT status, COUNT(*)
FROM tasks
WHERE spec_name = 'earnings'
GROUP BY status;

-- then the symbols themselves, with the reason attached
SELECT partition_key AS symbol, status, http_code, attempts,
       SUBSTR(error, 1, 60) AS error
FROM tasks
WHERE spec_name = 'earnings'
  AND status IN ('empty', 'failed')
ORDER BY status, symbol;
```

`empty` rows are real coverage gaps to disclose when a panel built on this
endpoint is used. `failed` rows are ours to fix by re-running.

### Where did the wall clock actually go?

```sql
SELECT spec_name,
       COUNT(*)                        AS tasks,
       ROUND(SUM(duration_s)/60.0, 1)  AS worker_minutes,
       ROUND(AVG(duration_s), 2)       AS mean_seconds,
       MAX(duration_s)                 AS slowest_seconds
FROM tasks
WHERE run_id = (SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1)
  AND status IN ('ok', 'empty')
GROUP BY spec_name
ORDER BY SUM(duration_s) DESC
LIMIT 15;
```

`worker_minutes` sums across all threads, so it exceeds wall clock by roughly the
pool size when the pool is saturated. A `mean_seconds` far above the rest usually
means an endpoint is paginating hard.

### Has coverage of an endpoint changed between runs?

```sql
SELECT r.run_id,
       datetime(r.started_at, 'unixepoch', 'localtime') AS started,
       SUM(t.status = 'ok')    AS ok,
       SUM(t.status = 'empty') AS empty,
       SUM(t.n_rows)           AS rows_stored
FROM runs r JOIN tasks t USING (run_id)
WHERE t.spec_name = 'insider_trading'
GROUP BY r.run_id
ORDER BY r.started_at;
```

A sharp change in `rows_stored` for an unchanged endpoint is a vendor-side event
worth investigating before the **derived layer** is rebuilt on top of it.

---

## 10. Rate limiting and adaptive backoff

**One shared bucket, not a per-call sleep.** The older single-call client
(`core/data/vendors/fmp/client.py`) sleeps between its own calls to target ~500
calls/min; because the sleep is serial with the network round trip, it actually
achieves about **134 calls/min measured**. It remains the right tool for
interactive and small scripted use, and is unchanged.

The framework instead hands out permits from one `TokenBucket` shared by all
workers, against one wall clock. A pool of 16 threads on that bucket sustains
**600–700 calls/min measured** — roughly a five-fold improvement — because the
threads overlap each other's network latency instead of each waiting alone.

**Chosen ceiling: 600 calls/min** (`rate_limit_per_minute` in
`config/vendors/fmp.json`), against a vendor-advertised 750. A probe of 700
requests paced at 700 calls/min returned **zero HTTP 429 responses**, so 600 is
deliberate headroom rather than a measured cliff.

**Adaptive backoff.** A vendor's published limit and its enforced limit are not
the same number: bursts, per-endpoint sub-limits and noisy neighbours all move
it. So the bucket adapts:

- On HTTP 429, `penalize()` **halves** the rate, with a floor at 5% of the
  configured rate. The new rate is logged at WARNING.
- On every success, `recover()` multiplies the rate by **1.05**, capped at the
  configured rate. Recovery is therefore geometric: about 15 consecutive
  successes undo one penalty.

A long backfill converges near the true ceiling without anyone tuning it. If you
see repeated penalty lines in the log, the ceiling really is lower than
configured — lower `--rate` rather than fighting it.

**Retries** are separate from rate limiting. Statuses `408, 429, 500, 502, 503,
504` are retried with exponential backoff (1.5s, 3s, 6s, …) up to
`--max-retries`. Statuses `402`, `403` and `404` are **verdicts, not hiccups**:
they are not retried, because re-asking an unentitled endpoint 9,011 times four
times over just burns the budget. A transport exception (connection reset,
timeout) is reported as a synthetic 503 so it backs off like a server error
rather than being mistaken for a vendor verdict.

---

## 11. Traps

These are real failures this framework has already hit in production. Each one is
silent by construction: the file exists, parses, and is wrong.

### Trap 1 — endpoints that ignore the `page` parameter

Some vendor "list everything" endpoints return their **entire** row set on every
page and ignore `page` altogether. Naively walking pages until a short page
appears therefore never terminates, and every page is appended again.

On FMP the offenders are `stock-list`, `etf-list`, `actively-trading-list`,
`all-industry-classification` and `symbol-change`. A run that walked them stored
**50,138 genuine rows as 10,027,600** — a 200-fold duplication with no error, no
warning, and a perfectly valid parquet file at the end of it.

**Fix, now in `core/ingest/paginate.py`:** compare each page to the previous page
and stop when a page repeats verbatim, keeping the rows gathered so far and
logging that the endpoint ignores paging.

**How to notice it yourself:** a stored file whose row count is an exact multiple
of a suspiciously round number, or a journal `n_rows` that grows every time you
refetch an endpoint whose underlying data did not change.

### Trap 2 — endpoints that *do* paginate, truncated by the page cap

The mirror image. `cik-list` paginates properly, and the original 200-page safety
cap silently truncated it. The stored file was a valid prefix of the real row
set, which is worse than an error because nothing downstream can detect it.

**Fix:** `max_pages` raised to 2,000 for paginated FMP endpoints, and hitting the
cap now logs at WARNING with the text "stored file is likely truncated". Grep the
run log for `page cap` before trusting a paginated endpoint's output.

The two traps pull in opposite directions, which is exactly why both guards are
needed: stop early when pages repeat, but do not stop early merely because you
have walked a lot of them.

### Trap 3 — the 5,000-row end-of-day price cap

FMP returns at most **5,000 rows per end-of-day price call**. Asking for a
40-year history of daily bars therefore returns roughly the most recent 20 years
and silently drops the rest — again, with HTTP 200 and a well-formed response.

**Fix:** the four `historical-price-eod/*` endpoints carry `date_chunk_years: 10`
in the manifest, so the planner splits each symbol into consecutive 10-year
`from`/`to` windows starting at `history_start` (default 1985-01-01). Ten years
of daily bars is about 2,520 rows, comfortably under the cap. This is why wave 4
is 180,220 tasks rather than 36,044.

The same reasoning applies to intraday bars, which are **bar-capped rather than
range-capped**; those use 1-year windows from 2020.

**If you add a price-like endpoint, assume a response cap until you have
disproved it.** The test is cheap: request a range you know contains more rows
than the cap and count what comes back.

### Trap 4 — a spec missing a required parameter returns *empty*, not an error

The nastiest interaction between two things that are individually correct. When a
vendor endpoint requires a parameter the manifest does not supply, FMP does not
return HTTP 400 — it returns HTTP 200 with an empty list. The runner then does
exactly the right thing with that: it records an honest `empty` and writes an
empty file so the key is never re-asked. Twelve hours later you have a directory
of 9,011 empty files that looks like a genuine, well-documented coverage gap.

Real example: `analyst-estimates` requires a `period` parameter. Without it, it
returns nothing for every symbol.

**Guard:** run `scripts/ingest/validate_fmp_manifest.py` before a long wave. It calls
each spec once for a symbol that should have data for almost everything, and
prints every spec that came back with no rows. A short list of names that a human
can sanity-check in a minute is the whole defence here — nothing in the framework
can distinguish "the vendor has no data" from "we asked wrong", because the
vendor's answer is byte-identical in both cases.

---

## 12. Known gaps and limitations

Stated rather than discovered later:

- **`per_name` endpoints plan to zero tasks.** Seven manifest entries partition
  by a person or company name (`senate_trades_by_name`,
  `house_trades_by_name`, `insider_trading_reporting_name`,
  `mergers_acquisitions_search`, `fundraising_search`,
  `crowdfunding_offerings_search`, `economic_indicators`).
  `scripts/ingest/ingest_fmp.py::resolve_keys` sets that key list to empty, so the
  planner logs "no keys available" and skips them. They are catalogued but not
  yet ingestible; a name source has to be chosen first (for
  `economic_indicators`, the list of series names).
- **Per-CIK endpoints are capped at the first 5,000 CIKs** (`ciks[:5000]` in
  `resolve_keys`), sorted as strings. The FMP CIK list is far larger, so per-CIK
  coverage is a deliberate prefix, not the whole SEC registry.
- **The spreadsheet report endpoint is pinned to one fiscal year.**
  `financial_reports_xlsx` carries `params: {year: 2024, period: "FY"}` in the
  manifest, so it downloads exactly one annual report per symbol, not a history.
- **Partition keys containing a dot would collide.** `safe_key` in
  `core/ingest/plan.py` translates `/`, `\`, `:` and space, but not `.`; a key
  such as `BRK.B` would have its suffix replaced rather than appended and land as
  `BRK.parquet`. This is latent, not active: the security master normalises all
  ticker separators to `-` (ADR 0015), so **zero of the 9,011 symbols contain a
  dot** today. It becomes live the moment a vendor's own punctuation is used as a
  partition key.
- **Row counts in the journal are pre-deduplication.** `n_rows` is what was
  stored, not what is unique. Trap 1 was visible in the journal as an
  implausible row count long before anyone read the file.

---

## 13. Adding an endpoint, or a whole vendor

**A new endpoint on an existing vendor** is one JSON object in
`config/vendors/{vendor}.json` — no Python:

```json
{
  "name": "employee_count",
  "endpoint": "employee-count",
  "partition": "per_symbol",
  "pit_status": "point_in_time",
  "priority": 2,
  "date_columns": ["acceptanceTime", "filingDate", "periodOfReport"],
  "primary_date": "filingDate"
}
```

Fields worth getting right:

| Field | Why it matters |
|---|---|
| `name` | Storage identity. Stable forever; renaming orphans downloaded files. |
| `partition` | `global`, `per_symbol`, `per_cik`, `per_exchange`, `per_sector`, `per_industry`, `per_name`, `batch_symbols`. Decides whether this endpoint costs 1 call or 9,011. |
| `pit_status` | `point_in_time`, `period_end_only`, or `snapshot` — see ADR 0010. Validated on load; an unknown value raises `ConfigError`. This is the field that decides whether a backtest may use the data at all. |
| `primary_date` | Column the stored rows are sorted ascending by. For anything a backtest will read, this should be the **publication_date** column (`filingDate`, `acceptedDate`, `publishedDate`), never the **reference_date** (the fiscal period the value describes). |
| `date_columns` | Columns parsed to `datetime64` before storage. |
| `paginate` / `page_size` / `max_pages` | Set whenever the vendor caps the response, or the stored file is a silent first page (Trap 2). |
| `date_chunk_years` / `history_start` | Set whenever the vendor caps rows per call (Trap 3). |
| `priority` | Wave. Put reference data other endpoints need as partition keys in wave 1. |
| `derivable` | `true` when the repo can compute this itself; excluded from wave defaults. |

Then dry-run it, then run it for two or three symbols, then read the stored file
before running it for 9,011.

**A new vendor** is two things:

1. `config/vendors/{vendor}.json` — `vendor`, `base_url`,
   `rate_limit_per_minute`, `raw_root`, and the `endpoints` list.
2. A transport adapter exposing `fetch(path, params) -> FetchResult`, modelled on
   `core/data/vendors/fmp/transport.py`. It must be thread-safe (one `requests.Session`
   per worker thread), must report non-200 statuses rather than raising, and must
   **never let the API key reach an error string or a log line**.

A driver script like `scripts/ingest/ingest_fmp.py` is a thin wrapper: resolve the
partition keys the vendor's manifest needs, then call `plan_run` and `execute`.

---

## Related documents

- [`docs/sources/vendor/fmp/INGESTION.md`](../sources/vendor/fmp/INGESTION.md) — the FMP-specific
  companion: how the manifest was probed, wave rationale, entitled versus
  unentitled endpoint families, and how to re-probe.
- [`docs/sources/vendor/fmp/ENDPOINT_CATALOG.md`](../sources/vendor/fmp/ENDPOINT_CATALOG.md) —
  every documented FMP path with the HTTP status observed on this key.
- [`docs/data/DATA_INVENTORY.md`](DATA_INVENTORY.md) — what is on disk and who
  produces it.
- [`docs/data/DATA_ARCHITECTURE.md`](DATA_ARCHITECTURE.md) — the raw/derived layer
  split this framework writes into.
- [`docs/decisions/0016-vendor-agnostic-ingestion-framework.md`](../decisions/0016-vendor-agnostic-ingestion-framework.md)
  — why a declarative manifest plus a shared runner, and the alternatives
  rejected.
- [`docs/decisions/0010-vendor-metric-point-in-time-classification.md`](../decisions/0010-vendor-metric-point-in-time-classification.md)
  — the point-in-time classification that `pit_status` carries.
