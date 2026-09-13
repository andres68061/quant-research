# Documentation index — start here

This page exists so you never have to guess which document answers your
question, and so nothing gets written in two places.

## The four records that matter

Every fact about the platform's integrity lives in exactly one of these. If you
are about to write a caveat, a flaw, a decision, or a result — this table tells
you where it goes, and the other three are wrong by definition.

| Question | Record | Source of truth |
|---|---|---|
| **"What must I know before trusting this number?"** | [DATA_HEALTH.md](data/DATA_HEALTH.md) | `core/research/caveats.py` (caveats, all surfaces) + `core/data/quality/health.py::known_flaws` (measured flaws) |
| **"Why is it implemented this way?"** | [decisions/](decisions) | One ADR per decision, with alternatives rejected |
| **"Did we already try this and fail?"** | [FAILED_STRATEGIES_LOG.md](FAILED_STRATEGIES_LOG.md) | Real numbers from real runs; entries are never deleted |
| **"What are we building next?"** | [ROADMAP.md](ROADMAP.md) | Forward-looking only; no results |
| **"What did that experiment actually show?"** | `/research-notes` | `core/research/notes.py` — every variant plus its control; figures asserted against the experiment JSON by `tests/test_notes.py` |
| **"What does this word mean?"** | `/glossary` | `core/research/glossary.py` — the single definition registry, read by every surface |

### Why there are two research records

The failure log is the **ledger**: one line per outcome, never deleted, so a dead
end stays dead. A research note is the **argument**: the full variant table, the
control it must be read against, what it means, and what would change it. The log
answers "have we tried this?"; the note answers "and what did we learn?". A note
always names the log or roadmap entry it corresponds to.

### Why caveats are in code, not prose

Caveats used to be written inline wherever a number was computed — four
locations, each correct, none complete. They now live in a single registry
(`core/research/caveats.py`) tagged with the **surfaces** they apply to, so:

- every page/endpoint asks `caveats_for_surface(...)` and cannot forget one,
- the same caveat text appears identically on the sector page, the PEAD page,
  and in DATA_HEALTH.md,
- adding a research surface without disclosures fails the
  `research-disclosure` skill's checklist.

**Measured** flaws (coverage gaps, vendor defects, calendar anomalies) are
different: they are recomputed from disk by `scripts/ops/audit_data_health.py` on
every run, because asserting them in prose is how docs go stale. Both streams
render into DATA_HEALTH.md and the `/data-health` page.

## Live surfaces

| Page | What it answers |
|---|---|
| `/explorer` | **"What do we have on this company?"**, "screen the universe", "run a query". Company profile (identity, permanent id, index history, every factor, price chart), structured screener that shows its generated SQL, and a read-only SQL console |
| `/research-notes` | What each experiment found — every variant, its control, the verdict, and how to re-run it |
| `/glossary` | Every term the platform uses, defined to stand alone, cross-linked |
| `/data-health` | Universe funnel with definitions, survivorship, dataset coverage, flaw + caveat registry, per-symbol drilldown, glossary |
| `/methodology` | Every formula, each with a plain-language reading beside it |
| `/sectors` → Performance | Point-in-time sector indices with full methodology disclosure |
| `/pead` | Post-earnings drift event study |
| `/data-coverage` | Dataset inventory and quarantine review |

A **watchdog banner** appears on every page whenever the scheduled data checks
find a problem, so a failure does not depend on anyone reading a log.

## The tree — where every document lives

Documents are placed by where their subject sits in the data flow, the same
shape `core/`, `scripts/` and `data/` use: roots feed a trunk, the trunk
grows branches, branches carry leaves. Living records and decisions sit at
the top because they are about the whole tree.

```
docs/
  README.md  ROADMAP.md  FAILED_STRATEGIES_LOG.md  PLATFORM_STATUS.md  BACKLOG.txt
  decisions/     ADRs — why things are the way they are
  sources/   roots      the vendors and their quirks
  data/      trunk      the tables built from them, and how they are kept honest
  research/  branches   what we do with subsets of the data
  platform/  leaves     the pages and features a user touches
  archive/              implementation reports and one-off fix notes; history, not reference
```

### sources/ — roots

| Document | Scope |
|---|---|
| [vendor/fmp/](sources/vendor/fmp) | FMP endpoint catalog, probed entitlements, ingestion specifics. **Frozen at 2026-09-09** (ADR 0019) |
| [MACRO_VINTAGES.md](sources/MACRO_VINTAGES.md) | FRED: publication lags per series, and why they are not true vintages |
| [DATA_SOURCES_AND_MARKET_CAP.md](sources/DATA_SOURCES_AND_MARKET_CAP.md) | Where shares outstanding and market caps come from |
| [COMMODITY_DATA_AVAILABILITY.md](sources/COMMODITY_DATA_AVAILABILITY.md) | Which commodity series exist, from which vendor, from when |

### data/ — trunk

| Document | Scope |
|---|---|
| [DATA_ARCHITECTURE.md](data/DATA_ARCHITECTURE.md) | Raw vs derived layering — the one rule everything else follows |
| [DATA_MODEL.md](data/DATA_MODEL.md) | How the tables are organised — panel families, fact vs interval tables |
| [DATA_INVENTORY.md](data/DATA_INVENTORY.md) | **What artifacts exist**, their shape, and which script produces each |
| [DATA_HEALTH.md](data/DATA_HEALTH.md) | **How good they are** — coverage, bias, measured flaws (generated) |
| [DATA_QUALITY_AND_FILTERING.md](data/DATA_QUALITY_AND_FILTERING.md) | Quarantine, bad prints, eligibility filters |
| [INGESTION.md](data/INGESTION.md) | Operator runbook for the vendor-agnostic ingestion engine |
| [MONITORING.md](data/MONITORING.md) | How a data problem reaches you — guard layers, watchdog, alerting, the Data Monitor |
| [SP500_MEMBERSHIP.md](data/SP500_MEMBERSHIP.md) | Point-in-time index membership |
| [SECTOR_CLASSIFICATION.md](data/SECTOR_CLASSIFICATION.md) | Sector labels: source, storage, refresh |
| [RECONSTRUCTED_SNP_EXPLAINED.md](data/RECONSTRUCTED_SNP_EXPLAINED.md) | How the reconstructed S&P benchmark table is built |

### research/ — branches

| Document | Scope |
|---|---|
| [FACTOR_BACKTEST_AUDIT.md](research/FACTOR_BACKTEST_AUDIT.md) | Methodology audit of the factor pipeline |
| [EVENT_DRIVEN_BACKTEST.md](research/EVENT_DRIVEN_BACKTEST.md) | Event-study backtest design |
| [BENCHMARK_OPTIONS.md](research/BENCHMARK_OPTIONS.md) | Which benchmark to compare against, and when |
| [factor_screen_*.md](research) | Generated, dated experiment reports |
| [ml/](research/ml) | ML price prediction: design, quick start, terminology |

### platform/ — leaves

| Document | Scope |
|---|---|
| [RESUME_PLATFORM_MAP.md](platform/RESUME_PLATFORM_MAP.md) | Which page demonstrates which claim |
| [COMMODITIES_QUICK_REFERENCE.md](platform/COMMODITIES_QUICK_REFERENCE.md) | The commodities page and its API |
| [RATIO_ANALYSIS_FEATURE.md](platform/RATIO_ANALYSIS_FEATURE.md) | The ratio-analysis feature |

**Where does a new document go?** About a vendor → `sources/`. About a table
we build or how we keep it honest → `data/`. About a method, experiment or
model → `research/`. About a page → `platform/`. A report of work done, once
the work is merged → `archive/` (or, better, a commit message). Something the
whole platform must obey → an ADR in `decisions/`.

## Skills that enforce all of this

| Skill | Fires when |
|---|---|
| `research-disclosure` | Any surface that shows a research number |
| `strategy-evaluation` | Judging whether a signal/factor/strategy is any good |
| `explain-in-context` | Any user-facing explanation (self-contained, no orphan jargon) |
| `data-inventory-sync` | Any data artifact, fetcher, or entitlement change |
| `decision-log` | Any non-trivial engineering/methodology decision |
| `strategy-experiment-log` | Before and after any strategy experiment |
