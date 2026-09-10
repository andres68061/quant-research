---
name: fmp-api
description: Reference for the Financial Modeling Prep (FMP) API used for market data in this repo. Use when fetching stock prices, fundamentals, index constituents, market caps, delistings, intraday bars, earnings, analyst data, or any financial data from FMP; when adding a new FMP dataset; or when writing/editing scripts that call financialmodelingprep.com.
---

# FMP API Usage

## Basics

- Base URL: `https://financialmodelingprep.com/stable/`
- Auth: `?apikey=` query param. The key is `FMP_API_KEY` in `.env`, loaded via
  `config/settings.py`. **Never** hardcode it, and never log a response URL —
  `requests`' `raise_for_status()` embeds the full URL including the key, which
  is why `core/data/vendors/fmp/client.py` raises its own error instead.
- Always go through `core.data.vendors.fmp.client.fmp_get`: it handles auth, retries,
  exponential backoff, and a client-side ~500 calls/min throttle.

## Entitlements — probe, never assume

The vendor's plan/tier table **does not match** what our key returns. On
2026-08-18, all 230 unique documented paths were tested: 176 returned HTTP 200
and 54 returned HTTP 402. The exact path-level result is in
`docs/vendor/fmp/ENDPOINT_CATALOG.md`.

Restricted groups include transcript directory/content, exchange and
asset-class batch quotes, latest/TTM statements, point-in-time market snapshots,
selected ETF/fund data, all Form 13F and ESG paths, and **every bulk endpoint**.
The Premium key does serve 1-minute intraday, technical indicators, COT and
Senate trades despite the marketing page listing them as Ultimate-only.

**Because bulk is unavailable, every universe-wide download is per-symbol** and
rate-limit bound. Budget calls before starting anything wide.

Re-probe after any subscription change and update `docs/DATA_INVENTORY.md` §6:

```bash
/opt/anaconda3/envs/quant/bin/python scripts/ingest/probe_fmp_entitlements.py --restricted-only
```

## Adding a new per-symbol dataset

Do **not** write a new script. Add one entry to `SYMBOL_DATASETS` in
`core/data/vendors/fmp/datasets.py`:

```python
"my_dataset": SymbolDataset(
    name="my_dataset",
    endpoint="some-endpoint",
    pit_status=POINT_IN_TIME,        # required — see below
    params={"limit": 1000},
    date_columns=("date",),
    primary_date="date",
    notes="Why this is worth having, and its traps.",
),
```

Then it is fetchable and resumable for free:

```bash
/opt/anaconda3/envs/quant/bin/python scripts/ingest/fetch_fmp_datasets.py --list
/opt/anaconda3/envs/quant/bin/python scripts/ingest/fetch_fmp_datasets.py --datasets my_dataset
```

### pit_status is the field that matters

Vendor date columns are all called `date` and mean three different things.
Getting this wrong leaks future information and makes backtests look *better*,
so nothing about the result will prompt a second look. See ADR 0010.

- `POINT_IN_TIME` — the date the information became public. Usable directly.
- `PERIOD_END_ONLY` — the fiscal period described, **not** when it was published.
  Vendor ratios (`key-metrics`, `ratios`, `financial-growth`) are all this. Must
  be joined to `filingDate`/`acceptedDate` from the raw statements before use.
- `SNAPSHOT` — one current-value row, no history. Never backtestable.

## Endpoint reference

Local snapshot: `docs/vendor/fmp/` (`README.md` is the category index). Per-endpoint
parameter tables were not captured (JS-rendered) — probe the endpoint when exact
parameters matter.

## Traps that have already bitten us

- **Intraday is bar-capped, not range-capped.** Requesting seven months of 1-min
  bars returns the most recent ~1,170 and silently drops the rest. The call looks
  successful. Chunk sizes live in `core/data/vendors/fmp/intraday.py`.
- **Intraday bars are NOT split-adjusted** (unlike `historical-price-eod/dividend-adjusted`).
  A 4:1 split reads as a 75% overnight crash. Use `apply_split_adjustment` with
  the `splits` dataset.
- **`/historical-price-eod/dividend-adjusted` rejects index symbols** (`^GSPC`)
  with a 402; use `/full` for those.
- **A single symbol can 402** even when the endpoint is entitled. Catch per
  symbol and record it — do not let one ticker kill a multi-hour run.
- **The 5000-row response cap** on EOD history; `generate_date_chunks` keeps
  chunks at 5 years (~1260 rows).

## Repo conventions for FMP fetchers

- Fetch scripts in `scripts/`, pure transforms in `core/data/`.
- Write vendor payloads verbatim to `data/raw/fmp/...` before deriving anything.
- **Resumability is mandatory** for anything over a few minutes: skip symbols
  whose file exists, and write via `core.data.vendors.fmp.storage.write_atomic` so a
  killed process cannot leave a truncated parquet that a later run treats as
  complete. There is no separate state file.
- Narrow fetch windows with `load_fetch_windows` when a universe table is
  available — a company that lived 1998–2003 costs one chunk, not nine.
- Use `safe_filename` for tickers: `BRK/B` would otherwise create a directory.
- After adding a dataset or artifact, run the `data-inventory-sync` checklist.
