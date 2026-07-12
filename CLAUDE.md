# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

This project runs in the **`quant`** conda environment. Always use explicit absolute paths — never rely on the active shell environment.

| Tool | Command |
|---|---|
| Python | `/opt/anaconda3/envs/quant/bin/python` |
| pip | `/opt/anaconda3/envs/quant/bin/pip` |
| pytest | `/opt/anaconda3/envs/quant/bin/python -m pytest` |
| jupyter | `/opt/anaconda3/envs/quant/bin/jupyter` |

Pre-flight check before any Python command:
```bash
/opt/anaconda3/envs/quant/bin/python -c "import sys; print(sys.prefix)"
# Must print: /opt/anaconda3/envs/quant
```

When a dependency is missing: add it to `requirements.txt` with a version floor, then install via the env's pip.

## Commands

```bash
# Start API (Terminal 1)
make api                  # uvicorn on :8000, Swagger at /docs

# Start frontend (Terminal 2)
make frontend             # Vite dev server on :5173, proxies /api to :8000

# Tests
make test                 # pytest tests/ -v
/opt/anaconda3/envs/quant/bin/python -m pytest tests/test_backtest.py -v   # single file

# Lint (must pass before commit)
make lint                 # ruff check . && black --check . && isort --check .

# Docker
make up                   # docker compose up --build
make down
```

Line length: 100. Formatter: Black. Import sorter: isort. Linter: ruff (`E`, `F`, `I`, `B`; E501/E402 relaxed).

## Architecture

Three strict layers — no logic crosses downward:

```
frontend/   → HTTP + presentation only (React 19, TypeScript, Vite)
api/        → Thin FastAPI handlers: validate (Pydantic), call core.*, serialize
core/       → All quant math: data, signals, backtest, metrics, models, replay
data/       → Parquet + DuckDB (gitignored; loaded at API startup)
```

**Key constraint:** No business logic in `api/routes/`. No quant math in `frontend/src/`. No I/O inside `core/` computation functions (accept DataFrames, return DataFrames).

### Core modules

| Module | Responsibility |
|---|---|
| `core/data/` | Data loading, SP500 constituents, commodities, factor construction |
| `core/backtest/portfolio.py` | Portfolio simulation, `create_signals_from_factor`, `calculate_portfolio_returns` |
| `core/strategies/factor_runner.py` | `run_factor_cross_section_backtest` — single entry point for factor backtests |
| `core/strategies/registry.py` | Named strategy catalog (`StrategyMetadata`); exposed via `GET /strategies` |
| `core/signals/` | Sortino momentum, factor signals, regime detection (HMM + baselines) |
| `core/metrics/` | Sharpe, Sortino, drawdown, VaR, cumulative returns |
| `core/replay/precompute.py` | Frame-by-frame replay data for `GET /replay/frames` |
| `core/features/` | Feature engineering and label construction for ML |
| `core/models/` | XGBoost, RF, Logistic, LSTM for ML walk-forward |
| `api/dependencies.py` | Loads `factors`, `prices`, `sectors` Parquet files once at startup |

### Data files (loaded at startup)

**Raw layer** (canonical sources of truth):

- `data/raw/macro_fred.parquet` — Long-format raw FRED panel `(reference_date, series_id, value)`. Native frequency, no publication lag, no business-day fill, no standardisation. Built by `scripts/fetch_raw_macro.py`.
- `data/factors/prices.parquet` — Adjusted-close stock panel (yfinance-adjusted; treated as the raw stock layer under the light option). Built by `scripts/backfill_all.py` / `scripts/update_daily.py`.
- `data/factors/vix.parquet` — VIX close history. Already raw (no transformation applied upstream).

**Derived layer** (deterministic functions of the raw layer):

- `data/factors/macro.parquet` — Wide business-day macro panel with publication lag applied. Derived from `data/raw/macro_fred.parquet` via `core.data.factors.macro.derive_macro_panel_from_raw`.
- `data/factors/macro_z.parquet` — 5-year rolling z-scores. Derived from `data/factors/macro.parquet` via `core.data.factors.macro.compute_macro_zscores`.
- `data/factors/factors_price.parquet` — MultiIndex (date, symbol) factor panel (vol_60d, beta_60d, mom_*).
- `data/sectors/sector_classifications.parquet` — Sector/industry classifications.

### Strategy registry pattern
Every user-facing strategy gets a `StrategyMetadata` entry in `core/strategies/registry.py` with: `id`, `title`, `kind` (`FACTOR_CROSS_SECTION` or `ML_DIRECTION`), `post_path`, `hypothesis`, `reference`, `expected_sharpe_range`, and `known_limitations`. The registry drives `GET /strategies`.

For factor backtests, always delegate to `run_factor_cross_section_backtest` in `core/strategies/factor_runner.py` — do not reimplement the slice + signal + portfolio pipeline in routes.

## Adding a New Strategy

1. **Core logic** — new module under `core/` (pure functions, DataFrames in/out). Add tests in `tests/`.
2. **Registry** — add `StrategyMetadata` entry in `core/strategies/registry.py`.
3. **API** — Pydantic schema in `api/schemas/`, route module in `api/routes/`, router registration in `api/main.py`.
4. **Frontend** — types in `frontend/src/lib/types.ts`, API methods in `lib/api.ts`, page in `frontend/src/pages/`, route in `App.tsx`, nav link in `TopBar.tsx`. Use `AppLayout` with `LeftSidebar`, `RightSidebar`, `BottomPanel` slots.

## Python Standards

- **No `print()`** — use `logging` with structured context.
- **Type-hint every public function.** Run `mypy --strict` on `core/` and `api/`.
- Raise domain exceptions (`DataSchemaError`, `LeakageError`, `ConfigError`), not bare `Exception`.
- Modules: one concern per file, ≤ 300 lines. Use `__all__` in `__init__.py`.
- Naming: `calculate_sharpe_ratio` (verb+noun, snake_case), DataFrames named by content (`daily_returns`), constants `UPPER_SNAKE_CASE`.

## Numerics & Data Rules

- Default `float64`. Guard divisions: `eps = 1e-10`. Guard `log()`: `np.maximum(x, eps)`.
- Time-series indexes must be **tz-aware**, **monotonic**, **unique**. No forward-fill over weekends/holidays.

### Point-in-time vocabulary (use these exact terms everywhere)

Every dated observation has up to three dates. Name them consistently in code,
schemas, parquet columns, and docs:

- **`reference_date`** — the period the value describes (June CPI → June 30; a Q2 balance sheet → the quarter end, aka `period_end` for fundamentals).
- **`publication_date`** — when the value became publicly knowable (FRED release day; SEC `filing_date`/`accepted_date` for fundamentals).
- **`as_of_date`** — the backtest clock. A value is usable iff `publication_date <= as_of_date`.

Rules: raw layer stores `reference_date` (and `publication_date` when the vendor provides it); derived/signal layers must align on **publication_date, never reference_date** (macro panels shift by per-series lags in `MACRO_PUBLICATION_LAGS_DAYS`; fundamentals must join on filing date). EOD prices are the special case where publication ≈ reference (same evening). Never name an ambiguous column `date` in new datasets — say which date it is.

### Project vocabulary (use these exact terms in code, schemas, and docs)

| Term | Meaning | Never call it |
|---|---|---|
| **panel** | wide DataFrame: date index × symbol columns (`prices.parquet`) | "table", "matrix" |
| **long panel** | MultiIndex (date, symbol) DataFrame (`factors_*.parquet`) | "stacked df" |
| **raw layer** | immutable vendor payloads in `data/raw/`; refetched, never edited | "cache", "backup" |
| **derived layer** | deterministic rebuilds from raw (`data/factors/`, `data/sectors/`) | — |
| **universe** | the set of symbols eligible on a date (point-in-time membership) | "watchlist", "symbols" |
| **adj_close** | split+dividend adjusted close; the only price used for return math | "price" (ambiguous) |
| **gross / net returns** | before / after transaction costs; always label which | plain "returns" in results |
| **signal lag** | trading days between factor observation and execution (`signal_lag_days`, default 1) | — |
| **quarantined / flagged / cleared** | excluded at load / needs review / reviewed-and-kept (`data/quality/`) | "blacklist", "bad" |
| **bad print** | isolated vendor quote error that snaps back; repaired in derived layer, kept in raw | "outlier" (outliers can be real) |
| **restatement** | vendor revising history (splits, corrections); handled by overlapping refetch | — |
| **ticker reuse** | one symbol, two companies over time; guard with membership filter + lifecycle truncation | — |
| **staleness cap** | max trading days a fundamental forward-fills before dying (273) | — |

### Timezone conventions (current state — reconcile at the join, not ad hoc)

- Equity layer (`prices.parquet`, factor panels): **tz-aware `America/New_York`**.
- Macro / FF5 / calendar-like data: **tz-naive calendar dates**.
- When joining the two, localize the naive side explicitly: `naive_df.index.tz_localize("America/New_York")`. Never strip tz from the equity layer, never compare naive vs aware (pandas raises — that's a feature).
- Use `.loc` with `pd.Timestamp` for date slicing — never `.iloc` for date-based work.
- ML: always time-based train/test splits. Fit transforms on train only. Walk-forward folds: retrain each fold.
- Validate shape, dtypes, NaN counts on every load — do not silently drop NaN rows in returns.
- Parquet naming: `{source}_{frequency}_{content}.parquet`. Columns: lowercase snake_case with unit suffix (`return_pct`, `vol_ann`).

## Testing

Test files mirror source: `core/metrics/risk.py` → `tests/test_metrics_risk.py`. Function names: `test_{function_name}_{scenario}`.

Coverage expectations:
- Every new `core/` function: happy path + edge case (empty input, NaN) + type check.
- ML code: verify walk-forward splits have no leakage (`train.max() < test.min()`).
- Do not mock core computation functions — test with real (small) DataFrames.

## Frontend Standards

- Functional components only. One component per file, name matches export (`KPICard.tsx` → `KPICard`).
- TanStack Query for all API calls — no raw `fetch` in components. API methods in `lib/api.ts`, types in `lib/types.ts`.
- No quant math in TypeScript. No `any` types. No external UI libraries (Tailwind only — no MUI, Chakra).

**Visual theme (hedge-fund terminal):**
- Background: `bg-zinc-950` (page), `bg-zinc-900` (panels). No shadows, no gradients, no `rounded-xl`.
- Text: `text-zinc-100` primary, `text-zinc-400` secondary, `text-zinc-500` labels. Borders: `border-zinc-800`.
- Semantics: `text-emerald-400` positive/long, `text-red-400` negative/short, `text-blue-400` neutral.
- Numbers: `font-mono tabular-nums` (JetBrains Mono). Charts: transparent backgrounds, zinc axes, 2-3 colors max.

## Notebooks

All notebooks in `notebooks/` must use the `quant` kernel. Verify with `jupyter kernelspec list`. If missing, re-register:
```bash
/opt/anaconda3/envs/quant/bin/python -m ipykernel install --user --name quant --display-name "Python (quant)"
```

Notebook research flow: validate one variable fully (load → inspect → chart → transform → assert) before moving to the next. Keep reusable computation in `core/` — do not leave business logic in notebooks.

## Environment Variables

API keys and secrets are loaded from `.env` via `config/settings.py`. Required keys for full functionality: `FINNHUB_API_KEY`, `FRED_API_KEY`, `FMP_API_KEY`. See `.env.example` for the full list.
