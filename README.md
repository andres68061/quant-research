# Quant Analytics Platform

A 3-layer quantitative analytics platform with a Python quant engine, FastAPI backend, and React frontend.

## Architecture

```
core/       -> Layer 1: Quant Engine (Python)
api/        -> Layer 2: FastAPI REST API
frontend/   -> Layer 3: React + TypeScript + Vite
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

## Pages

| Page | Route | Description |
|------|-------|-------------|
| Portfolio Simulator | `/` | Factor-based backtesting with equity curve, performance metrics, and VaR |
| ML Alpha | `/ml-alpha` | ML direction prediction with walk-forward validation, feature importance, confusion matrix |
| Sortino Momentum | `/momentum` | Grid search heatmap, bootstrap significance test, regime detection |
| Strategy Replay | `/replay` | Frame-by-frame strategy replay with timeline scrubber and live KPIs |
| ETF Optimizer | `/etf-optimizer` | Efficient frontier, tangency portfolio, CAL, rebalancing simulation |
| Metals Analytics | `/metals` | Commodity prices, returns, correlation, seasonality analysis |
| Economic Indicators | `/economic` | FRED data with recession bands, multi-panel indicator charts |
| Sector Breakdown | `/sectors` | Treemap, sector distribution, full classification table |
| Excluded Stocks | `/excluded-stocks` | Price-filtered exclusion analysis, stock detail viewer |
| Methodology | `/methodology` | KaTeX-rendered equations and strategy definitions |
| Sharpe Ratio Limits | `/sharpe-limitations` | Monte Carlo simulation showing Sharpe ratio blind spots |
| Linear Algebra Viz | `/linear-algebra` | Interactive matrix ops, 3D transforms, portfolio variance |

## Tech Stack

**Backend (Python):**
- FastAPI + Pydantic
- XGBoost, scikit-learn, TensorFlow
- pandas, numpy, scipy
- DuckDB, Parquet

**Frontend (TypeScript):**
- React 18 + Vite
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
make test    # pytest
make lint    # ruff + black + isort
make clean   # remove caches
```

## Project Structure

```
quant/
  core/           # Quant engine (data, features, models, backtest, metrics, signals)
  api/            # FastAPI (routes, schemas, dependencies)
  frontend/       # React + TypeScript + Vite
  _archive/       # Archived legacy code (Streamlit, Dash, old src/)
  config/         # Settings and environment
  scripts/        # CLI utilities
  tests/          # pytest suite
  notebooks/      # Jupyter notebooks
  data/           # Parquet, DuckDB (gitignored)
  docker/         # Dockerfiles
```
