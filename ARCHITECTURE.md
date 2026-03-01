# Architecture Reference

## Project Goal

A production-style quantitative analytics platform that replays strategies through time via walk-forward validation. The platform supports factor-based portfolio strategies, ML alpha strategies (directional classifiers, cross-sectional ranking), and Sortino momentum analysis. The UI is a professional research terminal: dark theme, dense layout, precise typography, purposeful animations.

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

```
quant/
  core/                  Layer 1 — Quant Engine
    data/                  Data loading, SP500 constituents, commodities, factors
    features/              Feature engineering, target/label construction
    models/                ML models (XGBoost, RF, Logistic, LSTM)
    backtest/              Portfolio simulation, walk-forward splits, benchmarks
    metrics/               Sharpe, Sortino, drawdown, VaR, cumulative returns
    signals/               Sortino momentum, factor-based signal generation
    surfaces/              Vol surface modeling (placeholder)
    replay/                Frame-by-frame replay precomputation
    utils/                 I/O helpers, ML result caching
  api/                   Layer 2 — FastAPI Backend
    main.py                App factory, CORS, lifespan (data loading)
    config.py              Host, port, allowed origins
    dependencies.py        Shared data loaders (factors, prices, sectors)
    schemas/               Pydantic request/response models
    routes/                Endpoint modules (health, data, strategy, metrics, etc.)
  frontend/              Layer 3 — React Frontend
    src/
      pages/               One file per page (PortfolioSimulator, MLAlphaReplay, etc.)
      components/          Reusable UI (charts, cards, controls, layout, tables)
      lib/                 API client (api.ts), TypeScript types, formatters
      stores/              Zustand stores (replay state)
    index.html             Entry HTML with font imports
    vite.config.ts         Dev server, Tailwind plugin, API proxy
  _archive/              Archived code (old Streamlit app, old Dash frontend, old src/)
  config/                Settings and environment
  scripts/               CLI utilities (backfill, data prep)
  tests/                 pytest suite
  notebooks/             Jupyter notebooks
  data/                  Parquet, DuckDB, SQLite (gitignored)
  docs/                  Feature-specific documentation
  docker/                Dockerfiles (API + frontend)
  docker-compose.yml     Full-stack orchestration
  Makefile               Dev commands (api, frontend, test, lint, up, down)
```

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
| Route          | Page                | Description                                              |
|----------------|---------------------|----------------------------------------------------------|
| `/`            | Portfolio Simulator | Factor-based backtest with equity curve, KPIs, and VaR   |
| `/ml-alpha`    | ML Alpha            | ML direction prediction, walk-forward, confusion matrix  |
| `/momentum`    | Sortino Momentum    | Grid search heatmap, bootstrap test, regime detection    |
| `/replay`      | Strategy Replay     | Frame-by-frame replay with timeline scrubber and live KPIs |

### Analytics
| Route            | Page                | Description                                              |
|------------------|---------------------|----------------------------------------------------------|
| `/etf-optimizer` | ETF Optimizer       | Efficient frontier, tangency portfolio, CAL, rebalancing |
| `/metals`        | Metals Analytics    | Commodity prices, returns, correlation, seasonality      |
| `/economic`      | Economic Indicators | FRED data with recession bands, multi-panel charts       |
| `/sectors`       | Sector Breakdown    | Treemap, distribution charts, classification table       |

### Reference
| Route          | Page                | Description                                              |
|----------------|---------------------|----------------------------------------------------------|
| `/methodology` | Methodology         | KaTeX-rendered equations and strategy definitions        |

## API Endpoints

| Method | Path                       | Description                            |
|--------|----------------------------|----------------------------------------|
| GET    | `/health`                  | Health check                           |
| GET    | `/data/assets`             | Available assets                       |
| GET    | `/data/factors`            | Available factor columns               |
| GET    | `/data/prices`             | Price series for one symbol            |
| POST   | `/run-backtest`            | Factor-based backtest                  |
| GET    | `/equity-curve`            | Equity curve from last backtest        |
| POST   | `/run-ml-strategy`         | ML walk-forward strategy               |
| GET    | `/metrics/performance`     | Performance metrics for a symbol       |
| GET    | `/metrics/var`             | VaR (Historical, Parametric, MC)       |
| GET    | `/walkforward/results`     | Walk-forward fold results              |
| GET    | `/replay/frames`           | Frame-by-frame replay data             |
| GET    | `/momentum/grid-search`    | Sortino momentum grid search           |
| GET    | `/momentum/bootstrap`      | Bootstrap significance test            |
| GET    | `/momentum/regime`         | Current momentum regime                |
| POST   | `/portfolio/optimize`      | Efficient frontier + tangency          |
| POST   | `/portfolio/simulate`      | Portfolio NAV with rebalancing         |
| GET    | `/banxico/cetes28`         | CETES 28 risk-free rate                |
| GET    | `/commodities/list`        | Available commodities                  |
| GET    | `/commodities/prices`      | Commodity price series                 |
| GET    | `/commodities/returns`     | Commodity returns + stats              |
| GET    | `/commodities/correlation` | Commodity correlation matrix           |
| GET    | `/commodities/seasonality` | Monthly seasonality analysis           |
| GET    | `/fred/catalog`            | FRED indicator catalog                 |
| GET    | `/fred/series`             | FRED time series data                  |
| GET    | `/fred/recessions`         | NBER recession periods                 |
| GET    | `/sectors/summary`         | Sector distribution summary            |
| GET    | `/sectors/breakdown`       | Symbols by sector/industry             |

## How to Add a New Strategy

1. **Core logic** — Create a module in `core/` (e.g. `core/signals/mean_reversion.py`). Write pure functions that take DataFrames and return results. Add tests in `tests/`.

2. **API endpoint** — Add a Pydantic schema in `api/schemas/`. Create a route module in `api/routes/`. Register the router in `api/main.py`. The route handler calls `core.*` functions and returns the schema.

3. **Frontend page** — Add TypeScript types in `frontend/src/lib/types.ts`. Add API client methods in `frontend/src/lib/api.ts`. Create a page component in `frontend/src/pages/`. Add the route in `App.tsx` and the nav link in `TopBar.tsx`.

## How to Add a New Frontend Page

1. Create `frontend/src/pages/MyPage.tsx` — use `AppLayout` with `LeftSidebar`, `RightSidebar`, `BottomPanel` slots.
2. Add types and API methods in `lib/types.ts` and `lib/api.ts`.
3. Import the page in `App.tsx` and add a `<Route>`.
4. Add a nav link in `components/layout/TopBar.tsx`.

## Tech Stack

**Backend:** Python 3.11, FastAPI, Pydantic, XGBoost, scikit-learn, pandas, numpy, DuckDB, Parquet

**Frontend:** React 18, TypeScript, Vite, TailwindCSS, Plotly.js, TanStack Query, Zustand, Framer Motion, KaTeX

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
