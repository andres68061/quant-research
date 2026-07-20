# Quant Research Platform

[![CI](https://github.com/andres68061/quant-research/actions/workflows/ci.yml/badge.svg)](https://github.com/andres68061/quant-research/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A quantitative research platform: point-in-time data layer, cost-aware backtest
engine, and a catalog of tested strategies — including the ones that failed,
with the real numbers. A React terminal UI and FastAPI layer sit on top of the
Python engine.

## Research findings

The headline results, positive and negative. Full methodology in
[docs/FACTOR_BACKTEST_AUDIT.md](docs/FACTOR_BACKTEST_AUDIT.md), negative
results preserved in [docs/FAILED_STRATEGIES_LOG.md](docs/FAILED_STRATEGIES_LOG.md).

1. **XOM/CVX cointegration pair — the one validated signal.** Engle-Granger
   pair discovered on a formation window and scored on a genuinely held-out
   window (notebook 17): net Sharpe **+0.27**, held-out Pain Ratio **4.9**.
   Modest, but it survives costs and out-of-sample evaluation.

2. **Multi-pair stat-arb baskets — six attempts, all failed, family retired.**
   Ranking pairs by price-path proximity (Gatev SSD: net Sharpe **−0.27**;
   ADF p-value ranking: **−0.48**) selects pairs with too little deviation to
   profit from after costs — SSD picked GOOGL/GOOG (Alphabet's two share
   classes) in 28 of 28 rolling periods. Every attempt is logged with numbers
   in [the failed strategies log](docs/FAILED_STRATEGIES_LOG.md).

3. **A "+0.744 Sharpe" pairs result was retracted after validation.** The
   cointegration-persistence basket looked good once; it did not reproduce,
   flipped sign under ±6–12-month start shifts, and counting all 10 basket
   configurations tried, the Deflated Sharpe Ratio (Bailey & López de Prado)
   verdict is **0.12–0.43** — less likely than not to be real skill. It ships
   as a fully-disclosed research product, not a claimed edge
   ([docs/ROADMAP.md](docs/ROADMAP.md) has the corrected robustness grid).

4. **In-sample optimization roughly doubles apparent Sharpe.** The same
   5-asset portfolio optimized naively shows Sharpe **1.05**; the honest
   walk-forward version shows **0.49**. The platform ships both so the gap is
   visible, not hidden.

5. **Factor cross-sections (momentum, low-vol, value, quality) are run with
   point-in-time S&P 500 membership, 1-day signal lag, ADV-bucketed
   transaction costs, and fundamentals joined on filing date** — expected
   net Sharpe ranges per strategy are in `core/strategies/registry.py`, kept
   deliberately sober (0.2–0.8).

### Methodological guardrails

- **Purged walk-forward validation**: training rows whose forward-looking
  labels overlap the test window are purged (plus optional embargo) —
  `core/backtest/walkforward.py`.
- **Multiple-testing corrections**: grid-search results carry trial counts;
  bootstrap p-values are Šidák-adjusted for the size of the searched grid
  (`core/signals/momentum.py`); Deflated Sharpe Ratio for best-of-N selection
  (`core/metrics/deflated_sharpe.py`).
- **Factor-model inference**: alpha t-stats vs FF5 with Newey-West (HAC)
  standard errors — `core/metrics/factor_regression.py`.
- **Point-in-time discipline**: every dated value distinguishes
  `reference_date` / `publication_date` / `as_of_date`; macro panels apply
  per-series publication lags; regime HMMs are re-fit walk-forward on rolling
  windows with filtered (not smoothed) probabilities.
- **334-test suite** covering leakage, survivorship, timezone joins, and cost
  accounting, run in CI on every push.

## Architecture

```
core/       -> Layer 1: Quant Engine (Python) — all math lives here
api/        -> Layer 2: FastAPI REST API — thin validation/serialization
frontend/   -> Layer 3: React + TypeScript + Vite — presentation only
```

The frontend talks to the API over HTTP/JSON. The API calls `core/` modules. No business logic lives in the UI.

## Quick Start

### Prerequisites

- Python 3.11+ (conda recommended)
- Node.js 20+ (`conda install nodejs`)

### 1. Start the API

```bash
conda activate quant
make api
```

The API runs at `http://localhost:8000`. Swagger docs at `http://localhost:8000/docs`.

### 2. Start the Frontend

In a second terminal:

```bash
conda activate quant
make frontend
```

The frontend runs at `http://localhost:5173`. It proxies `/api/*` requests to the FastAPI backend.

### 3. Open the App

Go to `http://localhost:5173` in your browser.

Note: price/factor Parquet files under `data/` are built locally (gitignored)
via `scripts/backfill_all.py` and need `FMP_API_KEY` / `FRED_API_KEY` in
`.env` — see `.env.example`. The test suite runs entirely on synthetic
fixtures and needs no data or keys.

## Pages

| Page                | Route                 | Description                                                                                |
| ------------------- | --------------------- | ------------------------------------------------------------------------------------------ |
| Portfolio Simulator | `/`                   | Factor-based backtesting with equity curve, performance metrics, and VaR                   |
| ML Alpha            | `/ml-alpha`           | ML direction prediction with purged walk-forward validation, replay, feature importance    |
| Sortino Momentum    | `/momentum`           | Grid search heatmap, bootstrap test with multiple-testing correction, regime detection     |
| Pairs Trading       | `/pairs`              | Engle-Granger pair screening, spread charts, held-out validation                           |
| Pairs Index         | `/pairs-index`        | Multi-pair basket backtests (research tool; see failed-strategies log)                     |
| Pairs Persistent    | `/pairs-persistent`   | Cointegration-persistence basket with event-driven stops and DSR disclosure               |
| Portfolio           | `/portfolio`          | Efficient frontier, tangency portfolio, manual weights, walk-forward vs naive comparison   |
| Metals Analytics    | `/metals`             | Commodity prices, returns, correlation, seasonality analysis                               |
| Economic Indicators | `/economic`           | FRED data with recession bands, multi-panel indicator charts                               |
| Fama-French         | `/fama-french`        | FF5 factor panel, cumulative premia, factor stats                                          |
| Sector Breakdown    | `/sectors`            | Treemap, sector distribution, full classification table                                   |
| Excluded Stocks     | `/excluded-stocks`    | Price-filtered exclusion analysis, stock detail viewer                                     |
| Methodology         | `/methodology`        | KaTeX-rendered equations and strategy definitions                                          |
| Sharpe Ratio Limits | `/sharpe-limitations` | Monte Carlo simulation showing Sharpe ratio blind spots                                    |
| Linear Algebra Viz  | `/linear-algebra`     | Interactive matrix ops, 3D transforms, portfolio variance                                  |

## Tech Stack

**Backend (Python):**

- FastAPI + Pydantic
- XGBoost, scikit-learn, statsmodels (TensorFlow optional, LSTM only)
- pandas, numpy, scipy
- DuckDB, Parquet

**Frontend (TypeScript):**

- React 19 + Vite
- TailwindCSS (dark theme)
- Plotly.js (charts)
- TanStack Query (data fetching)
- Zustand (state management)

## Docker

```bash
make up      # docker compose up --build
make down    # docker compose down
```

## Development

```bash
make test    # pytest (334 tests, no data files or API keys required)
make lint    # ruff + black + isort (enforced in CI)
make clean   # remove caches
```

Reproducibility: `requirements.txt` carries version floors;
`requirements.lock.txt` is a full `pip freeze` of the working environment;
CI installs `requirements-ci.txt`. Notebook outputs are stripped via
pre-commit (`pip install pre-commit && pre-commit install`).

## Project Structure

```
quant/
  core/           # Quant engine (data, features, models, backtest, metrics, signals)
  api/            # FastAPI (routes, schemas, dependencies)
  frontend/       # React + TypeScript + Vite
  config/         # Settings and environment
  scripts/        # CLI utilities
  tests/          # pytest suite
  notebooks/      # Jupyter notebooks (outputs stripped)
  data/           # Parquet, DuckDB (gitignored)
  docs/           # Research docs: backtest audit, failed strategies log, roadmap
  docs/decisions/ # ADRs: why implementations are shaped the way they are
  docker/         # Dockerfiles
```

## License

MIT — see [LICENSE](LICENSE).
