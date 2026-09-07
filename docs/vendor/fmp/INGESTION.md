# FMP ingestion — manifest, waves, entitlements

Vendor-specific companion to the framework runbook,
[`docs/INGESTION.md`](../../INGESTION.md). That document explains how the shared
machinery works (planning, rate limiting, resumption, the journal, the traps);
this one explains what is specific to **Financial Modeling Prep (FMP)**: where the
endpoint manifest came from, why each endpoint sits in the wave it does, which
endpoint families this subscription can actually reach, the seven endpoints that
lie about being global, and how to re-probe when the vendor changes its surface.

Sibling documents in this folder:

- [`README.md`](README.md) — the vendor overview: base URL, authentication, plan
  limits, and the category-by-category notes.
- [`ENDPOINT_CATALOG.md`](ENDPOINT_CATALOG.md) — every documented path with the
  HTTP status observed on this key.

---

## 1. The manifest

```
config/vendors/fmp.json
```

| Field | Value | Meaning |
|---|---|---|
| `vendor` | `fmp` | Manifest stem and journal label. |
| `base_url` | `https://financialmodelingprep.com/stable` | The **stable** API, not the legacy `v3`/`v4` paths. |
| `rate_limit_per_minute` | `600` | Shared ceiling across all worker threads. Vendor advertises 750 on this plan; 600 is deliberate headroom. |
| `raw_root` | `data/raw/fmp` | Where stored payloads land. |
| `probed_on` | `2026-08-24` | When the endpoint shapes were last established against the live API. |
| `endpoints` | 175 entries | One JSON object per ingestible endpoint. |

Composition of the 175 entries:

| By wave | | By partition | | By point-in-time status | |
|---|---:|---|---:|---|---:|
| 1 — global reference | 66 | `global` | 66 | `point_in_time` | 99 |
| 2 — per-symbol history | 49 | `per_symbol` | 86 | `snapshot` | 59 |
| 3 — per-symbol snapshots | 41 | `per_name` | 7 | `period_end_only` | 17 |
| 4 — end-of-day prices | 4 | `batch_symbols` | 5 | | |
| 5 — technical indicators | 9 | `per_cik` | 5 | | |
| 6 — intraday charts | 6 | `per_exchange` | 2 | | |
| | | `per_sector` | 2 | | |
| | | `per_industry` | 2 | | |

Nine of the 175 carry `derivable: true` (all of wave 5, FMP's technical
indicators). "Derivable" means the repo can compute the same quantity from data
it already owns — a 14-day relative strength index is a function of the price
panel — so they are catalogued and ingestible but excluded from wave defaults.
They exist as a **reconciliation target**: a vendor's own indicator is the
cheapest way to check ours.

---

## 2. The manifest was probed, not hand-written

Two separate live sweeps produced it, and neither one trusted the vendor's
documentation page.

**Sweep 1 — entitlements (2026-08-18).** Every documented example on FMP's
stable-API docs page was called with this repo's live Premium key: 263 examples
covering **230 unique paths**, plus one corrected retest for a documented example
that was missing a required parameter. Result: **176 paths returned HTTP 200 and
54 returned HTTP 402** (Payment Required). This is path-level evidence — a path
is marked as working only when its own documented request succeeded, never
inferred from a sibling in the same family. The per-path results are the `Access`
column of [`ENDPOINT_CATALOG.md`](ENDPOINT_CATALOG.md).

Why the sweep was necessary: **the vendor's plan/tier marketing table does not
describe what the key returns.** It lists 1-minute intraday bars, technical
indicators, Commitment of Traders and Senate trades as a higher tier than ours;
all four return HTTP 200 on this key. Conversely it implies bulk downloads are
available; all 18 bulk paths return 402. Trust the probe, not the tier chart.

**Sweep 2 — shapes (2026-08-24).** The entitlement sweep says *whether* a path
answers; it does not say how the endpoint behaves. FMP's per-endpoint parameter
tables are rendered by JavaScript on their docs site and are not captured in the
local snapshot, so the remaining manifest fields were established by calling each
entitled endpoint and reading the response:

- Which query parameter keys it: `symbol`, `cik`, `exchange`, `sector`,
  `industry`, `name`, or nothing at all.
- Whether a bare call (no key) answers, and if so whether the answer is genuinely
  the whole row set — see [§4](#4-the-seven-endpoints-that-answer-a-bare-call-but-are-not-global).
- Which columns are dates, and which of them is a **publication_date** worth
  sorting by rather than a **reference_date**.
- Whether the response is capped, and by rows or by pages.

The 175 manifest entries are the 176 entitled paths minus one:
**`crowdfunding-offerings`** requires the Central Index Key of a private issuer
raising capital under Regulation Crowdfunding. Those issuers are not in the
security master and not in the SEC CIK list the repo downloads, so there is no
key source to expand the endpoint over. It is catalogued as entitled and left out
of the manifest deliberately, not by oversight. Its two siblings that *do* have
key sources, `crowdfunding-offerings-latest` (global) and
`crowdfunding-offerings-search` (by company name), are in the manifest.

> **Under-documented, flagged honestly:** the manifest's own `notes` field says
> "Regenerate with `scripts/probe_fmp_shapes.py` when the vendor surface
> changes." **That script is not in the repository.** The shape sweep's results
> are checked in — the manifest is the artifact — but the sweep itself is not
> currently reproducible from the repo. Until it is restored, re-probing is the
> manual procedure in [§6](#6-re-probing-when-the-vendor-surface-changes).

---

## 3. Wave assignment rationale

Waves are a **priority ordering**, not a dependency graph. The runner does not
enforce ordering between them; the numbering encodes what should land first if
the run is stopped early, plus one genuine ordering constraint.

**Wave 1 — global reference row sets and calendars (66 endpoints, 66 tasks).**
Everything that takes no partition key: symbol directories, the CIK list, index
constituents (S&P 500, Nasdaq, Dow, current and historical), the screener,
delisted companies, symbol changes, the earnings / dividends / splits / IPO /
economic calendars, Treasury rates, Commitment of Traders, and the "latest" news
and filing feeds.

Three reasons it is first:
1. **It is nearly free** — 66 tasks. There is no scenario where you skip it.
2. **It changes daily and is small**, which makes it the natural nightly refresh
   (see the commented block in [`scripts/crontab.txt`](../../../scripts/crontab.txt)).
3. **It is the only genuine ordering constraint.** The per-CIK, per-sector,
   per-industry and per-exchange endpoints in waves 2 and 3 take their partition
   keys from wave-1 outputs — `scripts/ingest_fmp.py::resolve_keys` reads
   `data/raw/fmp/cik_list/_all.parquet`,
   `data/raw/fmp/available_sectors/_all.parquet`, and so on. Run a later wave
   before wave 1 has landed and those endpoints log "no keys available" and plan
   to zero tasks rather than failing.

**Wave 2 — core per-symbol history (49 endpoints × 9,011 symbols = 441,539
tasks).** The data that research actually runs on: income statement, balance
sheet and cash flow (plus their as-reported and growth variants), ratios, key
metrics, financial growth, enterprise values, financial scores, earnings,
dividends, splits, analyst grades and ratings history, insider trading, employee
counts, revenue segmentation, market capitalisation. Every one of these carries
history, so the marginal value of downloading it now rather than later is high
and it never becomes cheaper.

**Wave 3 — per-symbol and per-key snapshots (41 endpoints, 162,653 tasks).**
Current-state values: quotes, aftermarket quotes and trades, trailing-twelve-month
metrics and ratios, price-target and grade consensus, ETF composition, sector and
industry performance, per-CIK profiles, SEC filing searches. These are `snapshot`
data by point-in-time classification — a single current value with no history — so
re-downloading them tomorrow gives you tomorrow's answer and today's is not
recoverable. That makes them **lower** priority for a backfill (you cannot
backfill a snapshot) even though they are individually useful.

**Wave 4 — end-of-day prices (4 endpoints, 180,220 tasks).** The four
`historical-price-eod/*` variants: `light`, `full`, `dividend-adjusted`, and
`non-split-adjusted`. Separated from wave 2 because they are date-chunked and
therefore expensive per symbol (5 windows each rather than 1 task each), and
because the repo already holds an adjusted-close **panel** at
`data/factors/prices.parquet` — so this wave is a re-derivation and
cross-check rather than a first acquisition.

**Wave 5 — technical indicators (9 endpoints, 81,099 tasks).** `derivable: true`,
so excluded from wave defaults entirely. Run only with `--include-derivable` or
an explicit `--endpoints` list, and only for reconciliation.

**Wave 6 — intraday charts (6 endpoints, 378,462 tasks).** 1-minute through
4-hour bars, chunked into 1-year windows from 2020. By far the largest wave and
the one with the least settled research use, so it is opt-in.

Measured task counts and costs for every wave are tabulated in
[`docs/INGESTION.md` §6](../../INGESTION.md#6-waves-and-their-measured-cost).

---

## 4. The seven endpoints that answer a bare call but are not global

This is the subtlest classification error in the manifest, and it is silent.

An automated shape probe decides whether an endpoint is `global` or `per_symbol`
by calling it with no key and seeing whether it answers. Seven FMP endpoints
answer a bare call with HTTP 200 and a well-formed, plausible payload — and are
nonetheless **genuinely symbol-scoped**. Each is pinned to `per_symbol` in the
manifest by hand, overriding what the bare call implies:

| Manifest name | Path | Wave | Has a truly global sibling? |
|---|---|---:|---|
| `company_notes` | `company-notes` | 2 | No |
| `employee_count` | `employee-count` | 2 | No |
| `historical_employee_count` | `historical-employee-count` | 2 | No |
| `sec_profile` | `sec-profile` | 2 | No |
| `insider_trading` | `insider-trading/search` | 2 | Yes — `insider_trading_latest` |
| `news_press_releases` | `news/press-releases` | 3 | Yes — `news_press_releases_latest` |
| `news_stock` | `news/stock` | 3 | Yes — `news_stock_latest` |

The common shape: a bare call returns *something* — a recent-activity window
across issuers, or an unspecified subset — but **not the endpoint's complete row
set**. Asking the same endpoint with `symbol=AAPL` returns rows the bare call did
not contain, which is the test that settles it.

Why misclassifying them is dangerous rather than merely wrong: the bare call
would be stored once as `data/raw/fmp/{endpoint}/_all.parquet`, a single file
with thousands of rows that **looks like a complete dataset**. Nothing downstream
can tell that it covers a handful of issuers or a few weeks. A panel built on it
would silently describe a population nobody declared, while appearing to cover
the universe.

Each therefore costs 9,011 tasks. Where FMP genuinely provides a global "latest"
feed alongside the symbol-scoped endpoint, that feed is catalogued **separately**
and truthfully in wave 1 — `insider_trading_latest`, `news_stock_latest`,
`news_press_releases_latest`. The two are different datasets and get different
storage directories; neither is a substitute for the other.

**Rule for anyone adding an endpoint:** a bare call returning HTTP 200 is not
evidence that an endpoint is global. Check whether the payload is the *whole*
row set — compare a bare call against the union of two or three symbol-scoped
calls, and look at whether the dates cluster in the recent past.

---

## 5. What this subscription can and cannot reach

Plan: **Premium**. Established by the 2026-08-18 sweep, not by the tier chart.
The full list lives in
[`docs/DATA_INVENTORY.md` §6](../../DATA_INVENTORY.md#6-fmp-plan-entitlements-probed);
the summary that matters for ingestion:

**Unentitled — HTTP 402 for every call (54 paths). Do not build against these:**

| Family | Note |
|---|---|
| **All 18 bulk endpoints** | Including `eod-bulk`, bulk statements, profiles, scores, ratios and metrics. **This is the single most consequential restriction:** every universe-wide download must be per-symbol, which is why wave 2 is 441,539 requests rather than a few hundred. |
| Earnings call transcripts | Directory, dates, latest, and content. |
| `latest-financial-statements`, statement `*-ttm` | Latest cross-company statements and trailing-twelve-month statements. TTM *metrics* and *ratios* still work. |
| Form 13F (`institutional-ownership/*`) | All eight holdings and analytics paths. |
| Exchange and asset-class batch quotes | Mutual funds, ETFs, commodities, crypto, forex, indexes. Symbol-list batch quotes still work. |
| Sector / industry `*-snapshot` paths | The *historical* sector and industry performance and price/earnings paths still work. |
| Selected ETF and fund paths | `etf/holdings`, `etf/asset-exposure`, all `funds/disclosure*`. ETF info, country and sector weightings still work. |
| `esg-*` | Disclosures, ratings, benchmark. |

**Entitled, though commonly assumed otherwise:** intraday
`historical-chart/{1min…4hour}`, `technical-indicators/*`,
`commitment-of-traders-report`, `senate-trades`, `house-trades`,
`economic-indicators`, `treasury-rates`, `company-screener`.

None of the 54 blocked paths are in the manifest, so a normal run should never
see an HTTP 402. If one appears in the journal, the subscription changed:

```sql
SELECT spec_name, COUNT(*) AS tasks, SUBSTR(MIN(error), 1, 80) AS example
FROM tasks WHERE http_code = 402 GROUP BY spec_name ORDER BY tasks DESC;
```

402 is not retried — it is a verdict, not a hiccup — so the cost of a newly
unentitled endpoint is one wasted call per task, not four.

---

## 6. Re-probing when the vendor surface changes

Do this when the subscription changes, when FMP announces new endpoints, or when
a run starts producing HTTP 402 or 404 responses it did not produce before.

**Step 1 — re-check entitlements.** One representative request per endpoint
family, using the smallest possible response, about 50 calls:

```bash
/opt/anaconda3/envs/quant/bin/python scripts/probe_fmp_entitlements.py
/opt/anaconda3/envs/quant/bin/python scripts/probe_fmp_entitlements.py --restricted-only
```

Paste a changed restricted list into
[`docs/DATA_INVENTORY.md` §6](../../DATA_INVENTORY.md#6-fmp-plan-entitlements-probed).

**Step 2 — refresh the path list.** [`ENDPOINT_CATALOG.md`](ENDPOINT_CATALOG.md)
is a snapshot of FMP's documented stable-API paths (230 unique as of 2026-08-18)
with the observed status per path. Re-snapshot it from the live docs page and
update its counts and its `Endpoint list fetched and tested` date.

**Step 3 — diff the catalog against the manifest.** This is the check that
catches an endpoint FMP added and nobody noticed. It reads two files on disk and
makes no HTTP calls:

```bash
/opt/anaconda3/envs/quant/bin/python - <<'PY'
import json, re, pathlib
catalog = pathlib.Path("docs/vendor/fmp/ENDPOINT_CATALOG.md").read_text()
works   = {m.group(1) for m in re.finditer(r"^\| `/([^`]+)` \|.*works — HTTP 200", catalog, re.M)}
blocked = {m.group(1) for m in re.finditer(r"^\| `/([^`]+)` \|.*blocked — HTTP 402", catalog, re.M)}
manifest = {e["endpoint"] for e in json.load(open("config/vendors/fmp.json"))["endpoints"]}
print("catalog: %d entitled, %d blocked | manifest: %d" % (len(works), len(blocked), len(manifest)))
print("entitled but NOT in manifest:", sorted(works - manifest))
print("in manifest but now BLOCKED:", sorted(manifest & blocked))
print("in manifest but not in catalog:", sorted(manifest - works - blocked))
PY
```

Expected output today:

```
catalog: 176 entitled, 54 blocked | manifest: 175
entitled but NOT in manifest: ['crowdfunding-offerings']
in manifest but now BLOCKED: []
in manifest but not in catalog: []
```

Anything else in those three lists is a change to act on. `crowdfunding-offerings`
is the deliberate omission explained in [§2](#2-the-manifest-was-probed-not-hand-written).

**Step 4 — establish the shape of each new endpoint** before adding it to the
manifest. For each new path, answer four questions with real calls:

1. *What keys it?* Call it bare, then with `symbol=AAPL`, then with `cik=`. Bare
   success does not mean global — re-read [§4](#4-the-seven-endpoints-that-answer-a-bare-call-but-are-not-global).
2. *Is the response capped?* Request a range you know exceeds any plausible cap
   and count the rows. End-of-day prices cap at **5,000 rows per call**; intraday
   caps by bar count. A capped endpoint needs `date_chunk_years`; a
   page-capped one needs `paginate` plus a `page_size`.
3. *Does `page` do anything?* Fetch page 0 and page 1 and compare them. If they
   are identical the endpoint ignores paging — the framework now detects this at
   run time, but knowing in advance saves a confusing first run. See
   [Trap 1](../../INGESTION.md#trap-1--endpoints-that-ignore-the-page-parameter).
4. *Which date columns, and what do they mean?* Record every date column in
   `date_columns`, and set `primary_date` to the **publication_date** — the day
   the value became publicly knowable (`filingDate`, `acceptedDate`,
   `publishedDate`) — never the **reference_date**, the fiscal period the value
   describes (`period`, `periodOfReport`, `calendarYear`). Getting this backwards
   is how a backtest reads a quarterly balance sheet weeks before it was filed.

**Step 5 — classify the point-in-time status.** Every entry needs a `pit_status`
of `point_in_time`, `period_end_only` or `snapshot`, per
[ADR 0010](../../decisions/0010-vendor-metric-point-in-time-classification.md).
The value is validated when the manifest loads, so a typo raises `ConfigError`
immediately rather than surfacing months later as a leaked backtest. The
distinction:

| `pit_status` | Meaning | Backtest use |
|---|---|---|
| `point_in_time` | Rows carry the date the information became public | Direct |
| `period_end_only` | Rows are stamped with the fiscal period, **not** the publication date | Must first join `filingDate`/`acceptedDate` from the raw statements on `(symbol, reference_date)` |
| `snapshot` | A single current-value row, no history | Reconciliation and live screening only — **never** backtesting |

**Step 6 — add the entry, then prove it small.**

```bash
# plan only, no calls
/opt/anaconda3/envs/quant/bin/python scripts/ingest_fmp.py --endpoints new_endpoint --dry-run

# three symbols, real calls
/opt/anaconda3/envs/quant/bin/python scripts/ingest_fmp.py \
    --endpoints new_endpoint --symbols AAPL,MSFT,NVDA
```

Then **read the stored file** — row count, column names, date ranges — before
running it across 9,011 symbols.

**Step 7 — validate the whole manifest against the live vendor.** One call per
spec, a few hundred calls total:

```bash
/opt/anaconda3/envs/quant/bin/python scripts/validate_fmp_manifest.py
```

It calls each spec once with `AAPL` (override with `--symbol`) and prints every
spec that returned a non-200 and every spec that returned **HTTP 200 with no
rows**. The second list is the important one: FMP answers a request that is
missing a required parameter with an empty list rather than an error, so the
runner records an honest `empty` for all 9,011 symbols and you end up with a
directory of empty files that looks like a real coverage gap.
`analyst-estimates` was exactly this — it requires a `period` parameter, and
without it returns nothing for every symbol. Run this before any long wave, not
only after a re-probe. Full explanation:
[`docs/INGESTION.md` Trap 4](../../INGESTION.md#trap-4--a-spec-missing-a-required-parameter-returns-empty-not-an-error).

**Step 8 — bump `probed_on`** in `config/vendors/fmp.json`, and update
[`docs/DATA_INVENTORY.md`](../../DATA_INVENTORY.md) per the `data-inventory-sync`
checklist.

---

## 7. FMP-specific operational notes

- **Base URL is `stable`.** Legacy `v3` and `v4` paths still resolve for some
  endpoints but are not what the catalog or manifest describe. Do not mix them.
- **The API key must never reach a log line.** `core/data/fmp/transport.py`
  redacts the key from error bodies and never logs `response.url`, because the
  key travels as a query parameter. Preserve that property in any change.
- **Transport errors become a synthetic HTTP 503** so they back off like a server
  error rather than being recorded as a vendor verdict.
- **`financial_reports_xlsx` is a binary payload** — stored as `.bin`, not
  parquet — and is pinned to one fiscal year (`year: 2024, period: "FY"`) in the
  manifest, so it downloads one annual report per symbol rather than a history.
- **Run at most one ingestion process at a time.** The rate limiter is
  per-process, so two concurrent runs emit twice the configured calls per minute.
  `scripts/run_full_ingestion.sh` runs waves sequentially and refuses to start
  when it finds another `ingest_fmp.py` already running; nothing enforces this if
  you launch `ingest_fmp.py` directly.
- **Seven `per_name` endpoints currently plan to zero tasks** because the driver
  supplies no name keys. They are catalogued, not yet ingestible. See
  [`docs/INGESTION.md` §12](../../INGESTION.md#12-known-gaps-and-limitations).
- **Per-CIK endpoints are capped at the first 5,000 CIKs**, a deliberate prefix
  of the SEC registry rather than full coverage.
- **The single-call client `core/data/fmp/client.py` is unchanged and still the
  right tool** for interactive and small scripted use. It throttles per call and
  achieves about 134 calls/min measured; the ingestion framework's shared token
  bucket with 16 workers sustains 600–700 calls/min measured. Use the client for
  one-off questions, the framework for anything that loops over the universe.
