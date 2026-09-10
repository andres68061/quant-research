# Data model — how the tables are organised, and why

Answers the architectural questions: is this one giant table or families? Is raw
separated from derived? How is reference data (sectors, membership) stored, and
does it repeat descriptive values over time or record only changes?

## The three layers

```
data/raw/          immutable vendor payloads      ~105,000 files, 2.8 GB
   └── fmp/{dataset}/{SYMBOL}.parquet             refetched, never edited

data/factors/      derived panels                 25 files, 4.6 GB
   └── one file per FAMILY, not one giant table   rebuilt from raw, disposable

data/{universe,sectors,market_caps,quality}/      reference + metadata
   └── small tables that label or describe        rebuilt or appended
```

**Raw is separated from derived, and it matters.** Raw holds what the vendor
said, per symbol, one file each — so a defect is traceable to its source, a
single symbol can be refetched without touching anything else, and any derived
artifact can be deleted and rebuilt. Derived files are all reproducible from raw
plus code; none is a source of truth.

## Factor panels are families, not one table

Seven panel files, split by **what produces them and how often they change** —
not arbitrarily:

| Panel | Rows | Cols | Source | Rebuild cost |
|---|---:|---:|---|---|
| `factors_price` / `factors_all` | 26.7M | 7 / 8 | price panel | ~5 min |
| `factors_fundamental` | 29.3M | 34 | quarterly statements | ~7 min |
| `factors_composites` | 29.3M | 3 | cross-sectional z-scores | ~1 min |
| `factors_microstructure` | 27.0M | 10 | OHLCV bars | ~4 min |
| `factors_earnings_surprise` | 23.5M | 6 | announcements | ~1 min |
| `factors_vendor_metrics` | 5.4M | 42 | vendor ratios + filing dates | ~1 min |

**Why families rather than one wide table.** A single 100+ column table would be
~5 GB, would have to be rewritten in full whenever any one input changed, and
would force every reader to pay for columns they do not want. Splitting by source
means a price-factor change does not touch fundamentals, and
`core.data.store.factor_store` reads one column (~43 MB) instead of everything (~7 GB).

**Why not one file per factor.** 100+ tiny files multiplies metadata overhead and
makes the multi-column reads (`load_factors`) that the Cid-1 study needs slow.
Family-level granularity is the point where rebuild cost and read cost are both
acceptable.

**The seam that keeps this honest:** `FactorStore` maps *factor name → owning
panel* by reading parquet metadata at startup. Consumers ask for a factor by name
and never learn which file it lives in, so panels can be split or merged without
touching a single caller.

## Fact tables vs reference tables

**Fact tables** (the panels above) are `(date, symbol) → values`. One row per
observation, values change every row. That is correct for measurements.

**Reference tables** describe entities rather than measurements, and they follow
two different patterns depending on whether history matters:

### Current-state snapshot — `data/sectors/sector_classifications.parquet`

```
symbol, sector, industry, industryKey, sectorKey, last_updated, quoteType
```

One row per symbol, overwritten on rebuild. **This repeats no values over time
because it stores no time** — it is a snapshot of *today's* labels.

That is a known limitation, not a design win: the vendor sells no historical
classification, so a company reclassified in 2020 carries that label back through
all of its history. Registered as the `sector-labels-current-only` caveat
(medium severity) and disclosed on every surface that groups by sector.

### Interval table — `data/universe/index_membership.parquet`

```
symbol, index_name, valid_from, valid_to      (valid_to = NaT while still current)
```

This is the pattern the question is really about: **record what changed and when,
not the value on every date.** S&P 500 membership for 1,202 symbols over 30 years
is 1,255 rows — one per continuous membership spell, with re-entries as separate
rows (COV has three). Storing a daily flag instead would be 1,202 × 7,600 ≈ 9.1M
rows to express the same 1,255 facts.

Reading it is an interval lookup (`valid_from <= date <= valid_to`), which
`core.data.universe.filters.load_membership_filter` does.

### Which pattern to use

- **Changes rarely, history matters** → interval table (membership, lifecycle,
  ticker changes). Compact, and the change dates are themselves the information.
- **Changes rarely, history unavailable** → current snapshot + a registered
  caveat saying so (sectors today).
- **Changes every period** → fact panel (prices, factors, market caps).

The mistake to avoid is a fact-shaped table for slowly-changing descriptive data
— repeating "Technology" 7,600 times per symbol — which is how a 30 KB table
becomes a 500 MB one that is also harder to query for "when did this change?".

## Is the factor expansion sustainable?

Adding a factor family today means: a pure computation module in
`core/data/factors/`, a build script that writes one panel, and one entry in
`FactorStore.PANEL_PRIORITY`. No consumer changes, no schema migration. The
recent additions (microstructure, earnings surprise, vendor metrics, composites,
price-technical) each followed exactly that shape.

The controls that keep it from sprawling:

- **Every panel is reproducible** from raw + code, so a bad build is deleted, not
  repaired.
- **`data-inventory-sync`** requires the inventory and health audit to be updated
  when an artifact appears.
- **`core.data.quality.validation`** gates panel builds on structural invariants.
- **`FactorStore`** decouples factor names from file layout.

The real limit is not file count but **research discipline**: 100+ factor columns
make it trivially easy to find a spurious winner, which is why the
`strategy-evaluation` skill mandates the Šidák-corrected bar and reporting the
whole cross-section.

## Known debt

- `factors_price` is a strict subset of `factors_all` (same rows, one fewer
  column). Kept because `factors_all` depends on market caps and can fail
  independently; `FactorStore` resolves the overlap by priority.
- `*_expanded.parquet` staging files predate the ADR-0013 cutover and are now
  redundant with the canonical panels.
- Raw layer is ~105,000 small files. Fine for per-symbol refetch, slow to walk;
  the audit reads parquet metadata rather than data to stay fast.
