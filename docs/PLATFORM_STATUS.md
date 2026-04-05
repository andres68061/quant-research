# Platform status and audit snapshot

This document captures the **quant platform health audit** conclusions (maturity vs. a “quant analytics and strategies” product), **operational** notes, **gaps**, and **strategy model** so agents and humans have context without re-reading old plan files.

- **Prioritized backlog**: see [roadmap.txt](../roadmap.txt) at the repo root.
- **Data & factor inventory** (artifacts, sources, academic gap map): [DATA_INVENTORY.md](DATA_INVENTORY.md).
- **Architecture and layers**: [ARCHITECTURE.md](../ARCHITECTURE.md).
- **Migration history**: [migration.log](../migration.log) (when present).
- **Non-negotiable strategy boundaries**: [.cursor/rules/quant-strategies.mdc](../.cursor/rules/quant-strategies.mdc).

## Scorecard (what exists today)

| Area | Status | Notes |
|------|--------|--------|
| Factor-based backtesting | In place | `core/backtest/`, `core/signals/factor_signals.py` |
| ML alpha (walk-forward, classifiers) | In place | `core/models/`, API `run-ml-strategy` |
| Sortino / momentum analysis | In place | `core/signals/momentum.py` |
| Portfolio optimization (mean-variance, frontier) | In place | `core/optimization/` |
| Data pipeline | In place | Parquet + DuckDB under `data/`; update scripts in `scripts/`; FF5 daily from Kenneth French library ([DATA_INVENTORY.md](DATA_INVENTORY.md)) |
| REST API | In place | FastAPI, `api/routes/` |
| UI | In place | React + TypeScript + Vite + Tailwind (`frontend/`) |
| Replay / time scrubber | In place | `core/replay/`, replay API |
| Strategy registry (v1) | In place | `core/strategies/` metadata + `run_factor_cross_section_backtest`; `GET /strategies` catalog; ML still via `POST /run-ml-strategy` only |
| Event-driven backtest (v0 + HTTP) | In place | `core/backtest/events/` + `POST /backtest/events/simulate` ([EVENT_DRIVEN_BACKTEST.md](EVENT_DRIVEN_BACKTEST.md)); intraday/UI still open |
| Options / implied vol (partial) | In place (core) | `core/surfaces/` — Black–Scholes European price, `implied_volatility` (Brent), `implied_vol_surface_grid`; Dupire/SABR/SVI and API/UI not built |

## Gaps (not yet product-complete)

These are **intentional backlog** items, not bugs:

- **Event-driven backtesting** — v0 core and **simulate REST** are in; **intraday bars and UI** still open. Cross-sectional factor path remains primary for equity research. [EVENT_DRIVEN_BACKTEST.md](EVENT_DRIVEN_BACKTEST.md).
- **Data / factor transparency** — see [DATA_INVENTORY.md](DATA_INVENTORY.md) for Parquet artifacts, ingestion sources, and gap vs common systematic factor families.
- **Options pricing / implied vol surface** — **partial:** Black–Scholes + IV + small IV grid in `core/surfaces/`; full surface parameterizations (Dupire, SABR/SVI), chains ingestion, and API/UI still open.
- **Live or paper trading** integration.
- **Authentication / multi-user** and saved workspaces.
- **Persistent backtest runs** (versioned results, compare runs).
- **Alerting** (signals, data freshness, job failures).

## Strategy model (how we build strategies)

1. **Logic lives in Python** under `core/` (pure functions / pipelines; testable; no leakage).
2. **Configurable parameters** are expressed as **Pydantic** models in `api/schemas/` and exposed via REST.
3. **Frontend** calls the API only: **no user-supplied Python** execution, no quant math in React.
4. **Registry (v1 shipped)**: [`core/strategies/`](../core/strategies/) provides named `StrategyMetadata`, a shared factor pipeline for backtest/replay, and `GET /strategies`. **v2 (backlog)**: unify ML execution behind the same registry pattern and optional shared config schema.

For step-by-step extension workflow, see **How to Add a New Strategy** in [ARCHITECTURE.md](../ARCHITECTURE.md).

## Operations: data updates

Daily data refresh runs via **crontab** on the host (daily at 6 PM after market close):

- `scripts/update_daily.py` — prices, macro, FF5, price factors, sectors, DuckDB views.
- `scripts/update_commodities.py` — 14 commodity series.
- `scripts/fetch_shares_and_market_caps.py` — shares outstanding + historical market caps.

Cron reference file: [`scripts/crontab.txt`](../scripts/crontab.txt). Restore with `crontab scripts/crontab.txt`. Full schedule details in [DATA_INVENTORY.md §4](DATA_INVENTORY.md).

## Migration status (UI stack)

- **Current runtime**: React + FastAPI + Python `core/`.
- **Legacy**: Streamlit / Dash code was **removed** during migration. No `_archive/` folder exists on disk.

## Related documentation

- Data inventory and factor coverage: [DATA_INVENTORY.md](DATA_INVENTORY.md)
- Event-driven backtest (design): [EVENT_DRIVEN_BACKTEST.md](EVENT_DRIVEN_BACKTEST.md)
- Feature-specific notes: `docs/*.md`
- Notebooks: `notebooks/` (prototyping only; production logic belongs in `core/`)
