# S&P 500 Point-in-Time Membership

Canonical log for **what we trust**, **where it lives**, and **how to update**
the survivorship-free universe used by backtests.

_Last reviewed: 2026-07-12._

## Decision (trust this)

**Live source of truth for backtests:** the newest file matching

```text
data/S&P 500 Historical Components & Changes*.csv
```

Resolved by `resolve_sp500_historical_csv()` in
`core/data/sp500_constituents.py` (highest modification time wins). As of
2026-07-12 that is:

```text
data/S&P 500 Historical Components & Changes (Updated).csv
```

Wired into `sp500_universe_filter()` → factor backtests /
`survivorship_free=True` API runs.

**FMP** (`data/raw/fmp/constituents/sp500_membership.parquet`) is a
**cross-check only**. Reconciliation uses
`normalize_equity_ticker()` (`.` → `-`, uppercase) so `BRK.B` / `BRK-B`
do not inflate Jaccard gaps. Still do not `--promote` FMP over the CSV
until rename/recycle diffs are reviewed. See
`data/quality/sp500_reconciliation.txt`.

| Window | FMP vs Updated CSV (Jaccard) | Takeaway |
|---|---|---|
| Full history | ~0.85 | Mostly notation + denser CSV snapshots |
| 2020+ | ~0.95 | Close enough for research windows |
| Live roster | ~0.98 | Remaining diffs = notation + few recent adds |

## Upstream public repo

Source project: **fja05680 / sp500** (GitHub). Maintainer rebuilds membership
from Andreas Clenow’s original 1996–2019 list plus Wikipedia-driven changes.

| Upstream file | Role |
|---|---|
| `S&P 500 Historical Components & Changes (Updated).csv` | **What we copy into `quant/data/`** — historical membership since 1996 |
| `S&P 500 Historical Components & Changes.csv` | Original Clenow list (1996–2019); input to upstream notebook |
| `sp500_changes_since_2019.csv` | Manual change log since 2019; input to upstream notebook |
| `sp500.csv` | Current Wikipedia roster; output of `sp500.ipynb` |
| `sp500.ipynb` | Pull current members from Wikipedia |
| `sp500_historical.ipynb` | Merge original + changes → `(Updated).csv` |
| `sp500_by_date.ipynb` | Example: roster on a date; adds/removes since then |

### Upstream maintainer notes (paraphrased)

1. Clenow original is trusted through ~2019. Every couple of months they
   compare latest row to Wikipedia, Google exact change dates (Wikipedia
   “Selected Changes” is incomplete), update `sp500_changes_since_2019.csv`,
   and rerun `sp500_historical.ipynb`.
2. Tickers on any CSV date are the correct index members that day. If a
   symbol disappears later, treat as exit (removal *or* rename). No need to
   stitch rename chains for membership — sell the old ticker; the new one
   appears if it is in the index.
3. Price history for delisted/renamed names still needs a proper vendor
   (Norgate / EOD / FMP in this repo). Membership CSV alone is not enough.
4. Final current row is forced to match Wikipedia.
5. Count is often ≠ 500 (typically ~487–507). Early years lean light; prefer
   **2001+** (or our platform default **2015+**) if exact 500 matters.

## How we use it in this repo

```text
Upstream (Updated).csv
        │  (manual copy / snp500 updater)
        ▼
data/S&P 500 Historical Components & Changes (Updated).csv
        │  resolve_sp500_historical_csv()  → newest mtime
        ▼
SP500Constituents / sp500_universe_filter()
        │
        ▼
create_signals_from_factor(..., universe_filter=...)
```

Optional FMP refresh (does **not** change the live CSV unless `--promote`):

```bash
/opt/anaconda3/envs/quant/bin/python scripts/refresh_sp500_constituents.py
# review data/quality/sp500_reconciliation.txt
# only if mean Jaccard ≥ 0.95 AND notation reviewed:
# /opt/anaconda3/envs/quant/bin/python scripts/refresh_sp500_constituents.py --promote
```

## Update checklist (every few months)

1. In the **sp500** repo: update changes vs Wikipedia, rerun
   `sp500_historical.ipynb`, confirm final row matches Wikipedia.
2. Copy the new `(Updated).csv` into this repo’s `data/` (keep the
   `S&P 500 Historical Components & Changes*.csv` name pattern).
3. Optionally archive the previous copy under `data/archive/` so only one
   live file sits in `data/` (resolver already prefers newest mtime).
4. Run `scripts/refresh_sp500_constituents.py` and skim the reconciliation
   report — expect high agreement in recent years; investigate big drops.
5. Restart the API if it was already running (membership is loaded via the
   CSV path at filter construction time).
6. Note the update date and CSV last row date here or in
   `data/quality/sp500_reconciliation.txt`.

### Last ingest

| Field | Value |
|---|---|
| Copied file | `S&P 500 Historical Components & Changes (Updated).csv` |
| CSV last snapshot | 2026-06-02 |
| Prior file | archived `data/archive/...(01-17-2026).csv` (ended 2026-01-14) |
| Decision | Trust Updated CSV; FMP = cross-check only |

## Known limitations

- Pre-~2001 symbol counts are thinner (~487); disclosed, not fatal if you
  start later.
- Symbol renames: membership exits the old ticker; continuation prices must
  exist under the new ticker in our FMP panel (e.g. BK → BNY).
- FMP historical rebuild has fewer snapshot dates and different ticker
  punctuation — do not treat disagreement alone as “CSV wrong.”
