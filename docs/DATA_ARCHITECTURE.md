# Data Architecture

Decision record for how this platform stores research data, and the FMP
migration plan. Written 2026-07-11, when FMP Premium became the primary market
data vendor (replacing yfinance).

## The decision: layered Parquet + DuckDB, not one massive table

**Storage** is Parquet files organized in two layers. **Querying** is pandas for
pipelines and DuckDB for ad-hoc SQL (`data/factors/factors.duckdb` registers
views over the Parquet files — DuckDB scans Parquet directly, no import step).

We deliberately do NOT use a database server (Postgres/Timescale) or one giant
table:

- **Columnar Parquet is the right shape for cross-sectional research.** A factor
  backtest reads a few columns across all symbols and dates — exactly what
  columnar scans are fast at. A row-store server is optimized for the opposite
  (point lookups, concurrent writers) and we have one writer (the update
  scripts) and a handful of readers.
- **One massive table forces a sparse, mixed-frequency schema.** Daily prices,
  quarterly fundamentals, and event-dated constituent changes have different
  keys and frequencies. Cramming them into one table means NULL oceans and
  accidental joins across frequencies (a classic leakage source). Instead: one
  dataset = one folder with one consistent schema.
- **Files are auditable and cheap to version.** A corrupted vendor payload can
  be re-fetched per symbol; a bad derivation can be rebuilt from the raw layer;
  everything diffs and backs up trivially. No server to run, upgrade, or lose.
- **Scale headroom is years away.** The full price panel is ~36 MB, the factor
  panel ~300 MB. Parquet + DuckDB handles hundreds of GB on a laptop. If we ever
  outgrow it, the escape hatch is partitioned Parquet on object storage with the
  same layout — not a rewrite.

## The two layers

```
data/raw/        IMMUTABLE vendor payloads, exactly as fetched (append/refetch only)
data/factors/    DERIVED panels: deterministic functions of the raw layer
data/sectors/    DERIVED classifications
```

Rule: **anything in the derived layer must be rebuildable from the raw layer by
running a script.** Never hand-edit derived files; never compute "raw" data.

### Raw layer (per-vendor, per-dataset, per-entity)

```
data/raw/fmp/prices/{SYMBOL}.parquet    dividend-adjusted daily OHLCV per symbol
data/raw/macro_fred.parquet             long-format FRED panel (existing)
```

Per-symbol files make downloads chunked, resumable, and parallelizable: a
failed run refetches only missing symbols (`scripts/fetch_fmp_prices.py` skips
existing files). Empty files are written for symbols FMP does not cover, so
reruns don't re-probe known gaps. `_fetch_report.csv` records status per symbol.

Future FMP datasets follow the same pattern:

```
data/raw/fmp/fundamentals/income_statement/{SYMBOL}.parquet
data/raw/fmp/market_caps/{SYMBOL}.parquet
data/raw/fmp/constituents/sp500_historical.parquet
data/raw/fmp/delisted_companies.parquet
```

### Derived layer (what the platform actually loads)

```
data/factors/prices.parquet          wide date x symbol adjusted closes  <- THE interface
data/factors/factors_price.parquet   MultiIndex (date, symbol) factor panel
data/factors/factors_all.parquet     + market cap columns
data/factors/macro*.parquet          business-day macro panels
```

The key property of the migration: **`prices.parquet` keeps its exact schema**
(wide, tz-aware America/New_York index, adjusted closes). `api/dependencies.py`,
`core/backtest/`, every notebook — none of them change. Only the script that
produces the file changes vendor.

## Point-in-time rules (hedge-fund-grade, non-negotiable)

- **Prices**: use dividend-adjusted closes for return math. Never forward-fill
  across delistings; a vanished price is a delisting event, not missing data.
- **Universe membership**: point-in-time only (`sp500_universe_filter()`).
  Rebuild/verify from FMP `historical-sp500-constituent`.
- **Fundamentals (when added)**: store BOTH `period_end` and
  `filedDate`/`acceptedDate`, and align features on the FILING date, never the
  period end — a Q4 balance sheet is not knowable on Dec 31. Enforce with tests.
- **Vendor revisions**: FMP restates data (splits, corrections). The raw layer
  is refreshed by refetch (`--refresh`), not by patching values in place.

## FMP migration status & plan

1. ✅ `core/data/fmp/` client (retry/backoff/timeout, ~500 calls/min throttle)
   and price parsing; tests in `tests/test_fmp_prices.py`.
2. ✅ `scripts/fetch_fmp_prices.py` — resumable per-symbol download. Full run
   2026-07-11: 768 symbols fetched OK (incl. `^GSPC` via the `/full` endpoint —
   the dividend-adjusted endpoint rejects index symbols with HTTP 402), 62 with
   no FMP coverage (mostly long-delisted recycled tickers), ~55 min wall time.
3. ✅ Validation: median daily-return correlation vs yfinance **0.9993** across
   754 symbols with >1 year of overlap; corrupted-cell count (|daily return| >
   75%) dropped from 563 to 452, and the worst yfinance offenders (TNB, CBE,
   KRI...) are absent from FMP rather than corrupted. Low-correlation names
   (MCIC, COL, DIGI...) are ticker-reuse entity mismatches where the vendors
   disagree about which company the symbol refers to — quarantine candidates.
4. ✅ Cutover 2026-07-11: yfinance panel backed up to
   `prices_backup_20260711_*_pre_fmp_cutover.parquet`; `prices.parquet` is now
   FMP-built (10,460 dates × 773 symbols, 1985-01-02 → 2026-07-10); factor
   panels rebuilt from it.
5. ✅ Daily updates: `scripts/update_daily.py` now fetches from FMP via
   `core.data.fmp.panel.update_panel_from_fmp`. The fetch window overlaps the
   last 7 calendar days so recent vendor restatements overwrite stale rows
   (vendor-latest wins), and every fetch also lands in the raw layer.
6. ✅ Quarantine system (`core/data/quality.py` + `scripts/scan_data_quality.py`):
   automatic scan for bad-print signatures with a persistent review list.

## Quarantine system

Four checks run over the price panel (`scripts/scan_data_quality.py`, also
invoked automatically by `update_daily.py` after new data lands):

| Check | Signature | Status |
|---|---|---|
| `spike_reversal` | extreme move that reverses within 3 days (a real crash does not un-crash) | **quarantined** |
| `entity_mismatch` | daily-return correlation < 0.90 vs the yfinance reference panel | flagged |
| `extreme_returns` | 3+ days with \|return\| > 75% | flagged |
| `stale_prices` | 15+ identical consecutive closes | flagged |

Only *internal* corruption evidence auto-quarantines. Cross-vendor
disagreement proves one vendor is wrong but not which one — validated
2026-07-11: COST/NKE/HUBB disagreements are yfinance's bad 1980s data, and
CRWD's is a 2026 split that yfinance missed. FMP was right every time, so
mismatches are flags, not exclusions.

The list lives at `data/quality/quarantine.parquet` (+ CSV for humans) with
statuses `quarantined` (dropped from prices/factors at API load —
`api/dependencies.py`), `flagged` (in the data, awaiting review), `cleared`
(manually reviewed and kept; rescans never re-quarantine it). Review UI: the
frontend "Data Coverage" page; API: `GET /data-coverage`.

Full review completed 2026-07-11: every flagged finding adjudicated with
evidence (FMP split records, worst-divergence dates, vendor A/B around those
dates). Final state: **22 symbols quarantined** (FMP-side spinoff/split
adjustment failures inside the modern window — EXPE, FMC, FIS, BCO, MDLZ, JCI,
IDXX, AIV, SVU, SW, CVG — plus delisting-tail junk), **47 cleared** with notes
(yfinance was the wrong side, pre-2000 micro-price artifacts, benign stale
runs, real events like GME 2021 and the HealthSouth 2003 crash), 16 findings
still flagged with documentation (ticker-reuse tails pending lifecycle
truncation, two ambiguous 2008-era divergences).

### Bad-print repair (derived-layer cleaning)

`core.data.quality.repair_isolated_bad_prints` removes single-day quotes that
jump >75% and snap back to within 25% of the pre-spike level on the next bar
(LEN 1990 doubled quotes, AET 2005/2006 split-day prints). The raw layer keeps
the vendor values; only `prices.parquet` is cleaned, and every removal is
logged to `data/quality/bad_print_repairs.csv` (first run: 79 prints across 17
symbols). Factor panels are rebuilt from the repaired panel.

### Known coverage caveats (validated 2026-07-11)

- Live large caps match yfinance closely: daily return correlation ≥ 0.997 on
  AAPL/MSFT/GE/XOM/JPM over 40 years of shared dates.
- FMP Premium serves full history back to at least 1985 via date-range chunks
  (the "30 years" marketing limit did not bind in testing; responses cap at
  5000 rows per call, hence chunking).
- Symbols delisted long ago under recycled tickers (TNB, CBE, KRI, NCC, MEE,
  PBG...) have **no FMP EOD data**. These are precisely the symbols with
  corrupted yfinance quotes. Consequence: FMP fixes the corruption by removing
  the garbage, but pre-2018 survivorship-free backtests lose those names'
  histories entirely. The cutover validation must quantify how much of the
  point-in-time universe this affects per year.
