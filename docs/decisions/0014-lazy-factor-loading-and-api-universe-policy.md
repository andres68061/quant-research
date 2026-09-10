# 0014. Load factor columns lazily; give the API an explicit universe policy

Date: 2026-08-13
Status: accepted

## Context

ADR 0013 replaced the canonical price panel's 774 S&P names with the full US
universe (~8,900 symbols). It predicted a memory cost and said to measure before
optimizing. Measured:

| Artifact | In-process memory |
|---|---|
| `prices.parquet` (float64) | 0.75 GB |
| `factors_fundamental.parquet` (~21M rows x 37 cols) | ~5.2 GB |
| `factors_all.parquet` (~25M rows) | ~1.0 GB |
| **Total at API startup** | **~7 GB** |

The old loader read every panel eagerly into module-level globals. At 774 symbols
that was ~400 MB and nobody noticed; at 8,900 it makes the API unstartable on a
normal machine.

The eager load was also unnecessary. Every consumer needs one of exactly two
things: the **list** of factor names (schema only), or **one factor column** for a
backtest. Nothing reads the full 37-column frame.

## Decision

Two changes, addressing two different problems.

**1. Factor columns load lazily (`core/data/store/factor_store.py`).**
`FactorStore` scans parquet *metadata* at startup to map factor name -> owning
panel (free), and reads a single column on demand — 43 MB measured, 0.4 s, with
an 8-entry LRU for repeat requests. Routes that ranked on `factors[factor_col]`
now call `get_factor_frame(factor_col)`; `index_top500`, which needs three control
columns, calls `load_factors([...])`. The cache is a module-level function, not a
cached method, because an `lru_cache` on a method keys on `self` and would pin
every store instance for the process lifetime.

**2. The API loads a policy-selected universe (`core/data/universe/api_universe.py`),
and says which one.** `API_UNIVERSE` selects:

- `research` (default) — operating companies with >= 300 trading days of history.
  Excludes shells/SPACs and fund vehicles. Measured: 8,910 -> 6,382 symbols.
- `sp500` — symbols that were ever S&P 500 members (884). Reproduces pre-cutover
  results.
- `full` — everything; for batch jobs and machines with headroom.

The policy, its per-step counts, and the reason are exposed via
`get_universe_disclosure()` and served on `GET /data/factors`, because a silently
narrowed universe makes every number on the page describe a population nobody
declared.

## Alternatives rejected

- **Keep eager loading, shrink the universe until it fits.** Solves the symptom
  by discarding the small-cap tail — the entire reason for the cutover. Also
  picks a universe for a memory reason and then reports research results as if
  the universe were a research choice.

- **float32 everywhere.** Halves the panel but changes numerics platform-wide
  against the CLAUDE.md `float64` default, and 3.5 GB is still too much. Worth
  doing for stored artifacts (already done); not a fix for the load pattern.

- **Serve factors from DuckDB instead of pandas.** The right long-term answer and
  the repo already registers DuckDB views. Rejected *for now* because every
  consumer expects a `(date, symbol)` DataFrame, so it is a rewrite of the runner
  and every route rather than a loader change. Lazy column reads capture most of
  the benefit at a fraction of the risk. Revisit if per-request latency becomes
  the constraint rather than memory.

- **Memory-map the panels.** Parquet is columnar and compressed, so mapping does
  not give random row access without decompression; the format already gives us
  column pruning, which is the win we actually needed.

## Consequences

API startup drops from ~7 GB to ~0.5 GB (research-policy prices) plus ~43 MB per
factor actually requested. A backtest pays a one-off 0.4 s column read, cached
thereafter.

`get_factors()` still exists and still returns the eager frame when one was
loaded — but nothing populates it now, so callers must migrate to
`get_factor_frame`/`get_factor_store`. It is kept rather than deleted so an
out-of-tree caller fails loudly on `None` instead of silently importing a name
that vanished.

Results computed through the API under the default policy are **not** identical
to results computed by scripts over the full panel: the research policy excludes
~2,500 symbols. Any comparison between a page number and a script number must
state the policy. This is why the disclosure is served rather than logged.
