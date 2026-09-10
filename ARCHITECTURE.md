# Architecture Reference

## Project Goal

A production-style quantitative analytics platform that replays strategies through time via walk-forward validation. The platform supports factor-based portfolio strategies, ML alpha strategies (directional classifiers, cross-sectional ranking), and Sortino momentum analysis. The UI is a professional research terminal: dark theme, dense layout, precise typography, purposeful animations.

## Platform status and agent rules

- **Maturity snapshot, gaps, operations, migration notes**: [docs/PLATFORM_STATUS.md](docs/PLATFORM_STATUS.md).
- **Prioritized backlog**: [docs/BACKLOG.txt](docs/BACKLOG.txt).
- **Strategy boundaries** (logic in `core/`, parameters via API schemas, no user code execution in the frontend): enforced by Cursor project rules under [.cursor/rules/](.cursor/rules/) — see `quant-strategies.mdc`.
- **Strategy registry (v1)**: [`core/strategies/`](core/strategies/) holds named strategy metadata, [`GET /strategies`](api/routes/strategies.py) exposes a read-only catalog, and [`run_factor_cross_section_backtest`](core/strategies/factor_runner.py) centralizes the factor pipeline used by `POST /run-backtest` and `GET /replay/frames`. ML execution remains on `POST /run-ml-strategy` until a v2 unifies runners.

## 3-Layer Design

```
┌───────────────────────────────────────────────┐
│  Layer 3: React Frontend                      │
│  React + TypeScript + Vite + Tailwind + Plotly│
│  Calls the API, renders charts/controls/KPIs  │
└──────────────────────┬────────────────────────┘
                       │ HTTP / JSON
┌──────────────────────▼────────────────────────┐
│  Layer 2: FastAPI Backend                     │
│  REST endpoints, Pydantic schemas             │
│  Swagger docs at /docs                        │
└──────────────────────┬────────────────────────┘
                       │ Python imports
┌──────────────────────▼────────────────────────┐
│  Layer 1: Core Quant Engine                   │
│  Data, features, signals, models, backtest,   │
│  metrics, replay — all pure Python            │
└──────────────────────┬────────────────────────┘
                       │
                   data/ (Parquet, DuckDB, SQLite)
```

**Rules:**

- The frontend never imports Python. It only calls the API.
- API route handlers call `core.*` functions and return Pydantic models. No business logic in routes.
- All quant math lives in `core/`. No computations in the UI or in API routes.

## Directory Map

Eleven top-level folders. Each answers "what kind of thing lives here" in one word;
if a new file does not fit one of these, that is a design question, not a reason
for a new folder.

```
quant/
  core/                  Layer 1 — Quant Engine (pure functions, DataFrames in/out, no I/O in math)
    data/                  Everything about getting and shaping data
      vendors/               Clients for external providers: fmp/, sec/, fred, banxico, yfinance, vix, commodities
      factors/               Derived quantities: price/fundamental factors, returns, liquidity, market caps, macro, FF5
      universe/              Eligibility and identity: membership, lifecycle, filters, security master, sectors
      quality/               Validation invariants, quarantine, data-health audit, watchdog
      store/                 Panel I/O: guarded artifact writes, lazy factor-panel reads, DuckDB access
    ingest/                Vendor-agnostic ingestion engine: spec → plan → rate-limited pool → journal → report
    features/              Feature engineering and labels for ML (incl. commodity features)
    models/                ML models (XGBoost, RF, Logistic, LSTM) and the ML results cache
    signals/               Sortino momentum, factor signals, regime detection
    backtest/              Portfolio simulation, walk-forward, benchmarks, replay frames, mean-variance
    backtest/events/       Event-study backtests
    metrics/               Sharpe, Sortino, drawdown, VaR, factor regression, option pricing / IV surface
    strategies/            Strategy registry, factor cross-section runner, pairs, PEAD, top-N and sector indices
    research/              Caveat registry, glossary, research notes, Cid-1 study
  api/                   Layer 2 — FastAPI Backend
    main.py                App factory, CORS, lifespan (data loading)
    dependencies.py        Shared data loaders (factors, prices, sectors)
    schemas/               Pydantic request/response models
    routes/                Thin endpoint modules — validate, call core.*, serialize
  frontend/              Layer 3 — React Frontend
    src/pages/             One file per page
    src/components/        Reusable UI (charts, cards, controls, layout, tables)
    src/lib/               API client (api.ts), TypeScript types, formatters
    src/stores/            Zustand stores
  scripts/               Command-line entry points, by kind
    ingest/                Vendor fetchers + the ingestion supervisor (ingest_fmp.py, ingest_daemon.sh)
    build/                 Derived-layer builders (build_*_panel.py, build_price_factors.py, ...)
    experiments/           One-off research scripts; their findings live in docs/, not here
    ops/                   crontab.txt, install_crontab.sh, manage_ingest_daemon.sh, watchdog, audits
  config/                settings.py (.env loading), launchd plist template, vendor manifests
  tests/                 pytest suite; file names mirror the core/ module under test
  notebooks/             Jupyter research notebooks (quant kernel)
  docs/                  Documentation, ADRs (decisions/), roadmap, failure log, backlog
  data/                  Parquet, DuckDB, SQLite — gitignored; raw/ is the source of truth, the rest is derived
  runtime/               Written by running code, gitignored: logs/, outputs/ml_results/
  docker/                Dockerfiles (API + frontend); docker-compose.yml at the root
  Makefile               Dev commands (api, frontend, test, lint, up, down)
```

**Where does a new file go?** Talks to a vendor → `core/data/vendors/`. Computes a
factor → `core/data/factors/`. Decides eligibility → `core/data/universe/`. Checks
data → `core/data/quality/`. Reads/writes panels → `core/data/store/`. Trades on
something → `core/strategies/` or `core/signals/`. Measures a result → `core/metrics/`.
A command you run → `scripts/<kind>/`. A number you want to remember → `docs/`.

## Running the Project

```bash
# Terminal 1 — API
conda activate quant
make api                 # uvicorn on :8000, Swagger at /docs

# Terminal 2 — Frontend
conda activate quant
make frontend            # Vite dev server on :5173, proxies /api to :8000

# Docker (alternative)
make up                  # docker compose up --build
make down                # docker compose down
```

## Pages

### Strategies

| Route       | Page                | Description                                                |
| ----------- | ------------------- | ---------------------------------------------------------- |
| `/`         | Portfolio Simulator | Factor-based backtest with equity curve, KPIs, and VaR     |
| `/ml-alpha` | ML Alpha            | ML direction prediction, walk-forward, confusion matrix    |
| `/momentum` | Sortino Momentum    | Grid search heatmap, bootstrap test, regime detection      |
| `/replay`   | Strategy Replay     | Frame-by-frame replay with timeline scrubber and live KPIs |
| `/manual-portfolio` | Manual Portfolio | Pick stocks, set weights, compare against benchmark  |

### Analytics

| Route              | Page                | Description                                              |
| ------------------ | ------------------- | -------------------------------------------------------- |
| `/etf-optimizer`   | ETF Optimizer       | Efficient frontier, tangency portfolio, CAL, rebalancing |
| `/metals`          | Metals Analytics    | Commodity prices, returns, correlation, seasonality      |
| `/economic`        | Economic Indicators | FRED data with recession bands, multi-panel charts       |
| `/sectors`         | Sector Breakdown    | Treemap, distribution charts, classification table       |
| `/excluded-stocks` | Excluded Stocks     | Price-filtered exclusion analysis, stock detail viewer   |

### Reference

| Route                 | Page                | Description                                             |
| --------------------- | ------------------- | ------------------------------------------------------- |
| `/methodology`        | Methodology         | KaTeX-rendered equations and strategy definitions       |
| `/sharpe-limitations` | Sharpe Ratio Limits | Monte Carlo simulation showing Sharpe ratio blind spots |
| `/linear-algebra`     | Linear Algebra Viz  | Interactive matrix ops, transforms, portfolio variance  |

## API Endpoints

| Method | Path                            | Description                          |
| ------ | ------------------------------- | ------------------------------------ |
| GET    | `/health`                       | Health check                         |
| GET    | `/data/assets`                  | Available assets                     |
| GET    | `/data/factors`                 | Available factor columns             |
| GET    | `/data/prices`                  | Price series for one symbol          |
| GET    | `/strategies`                   | Registered strategy catalog (metadata) |
| POST   | `/run-backtest`                 | Factor-based backtest                |
| GET    | `/equity-curve`                 | Equity curve from last backtest      |
| POST   | `/run-ml-strategy`              | ML walk-forward strategy             |
| GET    | `/metrics/performance`          | Performance metrics for a symbol     |
| GET    | `/metrics/var`                  | VaR (Historical, Parametric, MC)     |
| GET    | `/walkforward/results`          | Walk-forward fold results            |
| GET    | `/replay/frames`                | Frame-by-frame replay data           |
| GET    | `/momentum/grid-search`         | Sortino momentum grid search         |
| GET    | `/momentum/bootstrap`           | Bootstrap significance test          |
| GET    | `/momentum/regime`              | Current momentum regime              |
| POST   | `/portfolio/optimize`           | Efficient frontier + tangency        |
| POST   | `/portfolio/simulate`           | Portfolio NAV with rebalancing       |
| GET    | `/banxico/cetes28`              | CETES 28 risk-free rate              |
| GET    | `/commodities/list`             | Available commodities                |
| GET    | `/commodities/prices`           | Commodity price series               |
| GET    | `/commodities/returns`          | Commodity returns + stats            |
| GET    | `/commodities/correlation`      | Commodity correlation matrix         |
| GET    | `/commodities/seasonality`      | Monthly seasonality analysis         |
| GET    | `/fred/catalog`                 | FRED indicator catalog               |
| GET    | `/fred/series`                  | FRED time series data                |
| GET    | `/fred/recessions`              | NBER recession periods               |
| GET    | `/sectors/summary`              | Sector distribution summary          |
| GET    | `/sectors/breakdown`            | Symbols by sector/industry           |
| POST   | `/simulation/sharpe-comparison` | Monte Carlo Sharpe-comparison sim    |
| GET    | `/exclusions/summary`           | Price-filtered exclusion summary     |
| GET    | `/exclusions/detail/{sym}`      | Excluded stock detail + price series |
| GET    | `/benchmarks/returns`           | Benchmark return series + metrics    |

## How to Add a New Strategy

1. **Core logic** — Create a module in `core/` (e.g. `core/signals/mean_reversion.py`). Write pure functions that take DataFrames and return results. Add tests in `tests/`.

2. **Registry (recommended)** — Add a `StrategyMetadata` entry in [`core/strategies/registry.py`](core/strategies/registry.py) so `GET /strategies` stays accurate. For a new HTTP surface, set `post_path` to the new route (e.g. `/run-my-strategy`).

3. **API endpoint** — Add a Pydantic schema in `api/schemas/`. Create a route module in `api/routes/`. Register the router in `api/main.py`. The route handler calls `core.*` functions and returns the schema.

4. **Frontend page** — Add TypeScript types in `frontend/src/lib/types.ts`. Add API client methods in `frontend/src/lib/api.ts`. Create a page component in `frontend/src/pages/`. Add the route in `App.tsx` and the nav link in `TopBar.tsx`.

## How to Add a New Frontend Page

1. Create `frontend/src/pages/MyPage.tsx` — use `AppLayout` with `LeftSidebar`, `RightSidebar`, `BottomPanel` slots.
2. Add types and API methods in `lib/types.ts` and `lib/api.ts`.
3. Import the page in `App.tsx` and add a `<Route>`.
4. Add a nav link in `components/layout/TopBar.tsx`.

## Tech Stack

**Backend:** Python 3.11, FastAPI, Pydantic, XGBoost, scikit-learn, pandas, numpy, DuckDB, Parquet

**Frontend:** React 19, TypeScript, Vite, TailwindCSS, Plotly.js, TanStack Query, Zustand, Framer Motion, KaTeX

## Visual Design Rules

These rules ensure a consistent hedge-fund research terminal aesthetic:

- **Background:** `bg-zinc-950` (near-black). Panels: `bg-zinc-900`.
- **Borders:** `border-zinc-800` (subtle 1px). No shadows. No gradients.
- **Text:** `text-zinc-100` (primary), `text-zinc-400` (secondary), `text-zinc-500` (labels).
- **Color semantics:** `text-emerald-400` = positive/long. `text-red-400` = negative/short. `text-blue-400` = neutral.
- **Numbers:** Always `font-mono tabular-nums`. Use JetBrains Mono.
- **Labels:** `text-[10px] uppercase tracking-wider text-zinc-500`.
- **Metrics:** `text-xl font-mono` for big numbers, `text-[10px]` for labels.
- **Headings:** Small and crisp. No large hero text.
- **Charts:** Transparent backgrounds (`paper_bgcolor: "transparent"`), zinc-toned axes, 2-3 semantic colors max.
- **Motion:** 150-200ms transitions for panels. Replay updates smooth, no flicker. Staggered card entrances.
