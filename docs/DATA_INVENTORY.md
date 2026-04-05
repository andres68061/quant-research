# Data inventory and factor coverage

This document answers: **what variables and files exist in the repo today**, **which external sources the code can pull from**, and **how that compares to common systematic equity factor families**. Update it when you add Parquet artifacts, new ingestion scripts, or core factor builders.

- **Prioritized backlog**: [roadmap.txt](../roadmap.txt)
- **Platform gaps**: [PLATFORM_STATUS.md](PLATFORM_STATUS.md)

## 1. Artifacts on disk (ground truth)

Paths are relative to the repository root unless noted.

### Loaded by the FastAPI app at startup

Defined in [`api/dependencies.py`](../api/dependencies.py):

| Path | Role |
|------|------|
| `data/factors/factors_price.parquet` | Factor panel (API cache); **MultiIndex** `(date, symbol)` expected by downstream code |
| `data/factors/prices.parquet` | Wide **daily** price levels; `DatetimeIndex`, columns = symbols |
| `data/sectors/sector_classifications.parquet` | Sector labels (when present) |

If a file is missing, the loader logs a warning and exposes `None` from getters.

### Produced by `scripts/backfill_all.py`

See [`scripts/backfill_all.py`](../scripts/backfill_all.py). Output directory defaults to `data/factors/`.

| File | Contents (high level) |
|------|------------------------|
| `prices.parquet` | Wide close panel from `build_prices_panel` (yfinance-backed in `core/data/factors/prices.py`) |
| `macro.parquet` | Macro series from `load_default_macro` |
| `macro_z.parquet` | Z-scored macro (`compute_macro_zscores`) |
| `factors_price.parquet` | Price-derived factors from `build_price_factors` (see below) |
| `fundamentals_daily.parquet` | Dailyized fundamentals (FMP bulk), when fundamentals pipeline runs |
| `factors_vq.parquet` | Value/quality composites from `compute_value_quality_factors`, when fundamentals present |
| `factors_all.parquet` | `factors_price` joined with `factors_vq` on index (or copy of `factors_price` if no VQ) |
| `fama_french_5.parquet` | Fama-French 5 daily factor returns (decimal): `mkt_rf`, `smb`, `hml`, `rmw`, `cma`, `rf`; from Kenneth French data library via `pandas_datareader` |
| `factors.duckdb` (default) | DuckDB views registered over the same Parquet files (including `fama_french_5`) |

**Note:** The API currently loads **`factors_price.parquet`** from `data/factors/` (not `factors_all.parquet`). Backfill may write both; align names if you need fundamentals in the API. `fama_french_5.parquet` is a **market-level** time series (not per-symbol); join on `date` when needed.

### Price-derived factor columns (`build_price_factors`)

Implemented in [`core/data/factors/build_factors.py`](../core/data/factors/build_factors.py). For each symbol in the price panel:

| Column | Description |
|--------|-------------|
| `mom_12_1` | 12–1 style momentum (months as ~21 trading days per month, excludes recent window) |
| `mom_6_1` | 6–1 style momentum |
| `mom_3_1` | 3–1 style momentum |
| `vol_60d` | Annualized trailing vol from daily returns (60-day window, √252 scaling) |
| `beta_60d` | Rolling beta vs market column (default market symbol from backfill: `^GSPC`) |
| `log_market_cap` | Natural log of historical market cap; joined from `data/market_caps/historical_market_caps.parquet` via `merge_market_cap` (NaN where market cap data is missing for a given date/symbol) |

### Value / quality composites (`compute_value_quality_factors`)

From [`core/data/factors/fundamentals_fmp.py`](../core/data/factors/fundamentals_fmp.py), when quarterly FMP ratios are dailyized:

| Column | Description |
|--------|-------------|
| `value_composite` | Cross-sectional z-mix of earnings yield, PB, PS (signs oriented toward “cheap”) |
| `quality_composite` | Cross-sectional z-mix of ROE, ROA, gross margin, leverage |

Underlying dailyized fields include `pe`, `pb`, `ps`, `roe`, `roa`, `gross_margin`, `debt_to_equity`, `earnings_yield`, etc.

### Other on-disk artifacts (not in `data/factors/`)

These files exist on disk and are produced by standalone scripts. They are **not** loaded by the FastAPI app at startup and are **not** consumed by `build_price_factors`.

| Path | Shape / contents | Producer | Notes |
|------|------------------|----------|-------|
| `data/market_caps/historical_market_caps.parquet` | ~6M rows, MultiIndex `(date, ticker)`, 725 stocks, 1962–present | [`scripts/fetch_shares_and_market_caps.py`](../scripts/fetch_shares_and_market_caps.py) (yfinance shares × prices) | **Not yet wired into factor pipeline.** Could provide per-stock size signal (`log_market_cap`). |
| `data/market_caps/shares_outstanding.parquet` | ~725 rows; columns: `ticker`, `shares_outstanding`, `fetch_date`, `source` | Same script | Point-in-time snapshot of latest shares outstanding. |
| `data/commodities/prices.parquet` | ~5,300 dates × 14 columns (GLD, SLV, WTI, BRENT, etc.) | [`scripts/update_commodities.py`](../scripts/update_commodities.py) / [`scripts/fetch_commodities.py`](../scripts/fetch_commodities.py) | Used by commodity API routes; not merged into the equity factor table. |
| `data/cetes28_daily.parquet` | Mexican CETES 28-day rates (Banxico) | Banxico API route / script | MX risk-free rate proxy. |
| `data/ml/stock_ml_dataset.csv` | ML training dataset (~916 KB) | Legacy (Aug 2025) | Pre-built feature set; may be stale. |
| `data/S&P 500 Historical Components & Changes*.csv` | Historical S&P 500 membership changes (newest matching file in `data/` by mtime) | fja05680/sp500 dataset, copied in by local updater | Used by [`core/data/sp500_constituents.py`](../core/data/sp500_constituents.py) (`resolve_sp500_historical_csv`) for survivorship-bias-free universe construction. |
| `data/sp500_failed_symbols.json` | Symbols that failed yfinance fetch | Backfill scripts | Diagnostic; excluded from price panel. |

### Survivorship-bias-free universe

`scripts/backfill_all.py` (default `--universe auto`) loads **all unique historical S&P 500 tickers** from the newest `S&P 500 Historical Components & Changes*.csv` in `data/` via [`core/data/sp500_constituents.py`](../core/data/sp500_constituents.py). This includes stocks that have since been delisted or removed from the index.

At **backtest time**, `create_signals_from_factor` accepts an optional `universe_filter` callable. The default in the API (`survivorship_free=True`) passes `sp500_universe_filter()`, which restricts the tradable universe at each date to stocks that were in the S&P 500 on that date. This eliminates survivorship bias.

### How strategies consume factors

- **Factor cross-section** ([`core/strategies/factor_runner.py`](../core/strategies/factor_runner.py)): ranks on a **single user-chosen factor column** from the in-memory factor `DataFrame`; accepts `universe_filter` for point-in-time membership.
- **Signals** ([`core/signals/factor_signals.py`](../core/signals/factor_signals.py)): thin wrappers over [`create_signals_from_factor`](../core/backtest/portfolio.py).

## 2. Ingestion sources referenced in code

| Source | Where used | Notes |
|--------|------------|--------|
| **yfinance** | Price panels, batch scripts, `scripts/fetch_shares_and_market_caps.py` (shares + market caps) | No API key; subject to Yahoo rate limits and symbol coverage |
| **FRED / fredapi** | Macro defaults, metals tests, [`api/routes/fred.py`](../api/routes/fred.py) | Needs `FRED_API_KEY` where applicable |
| **Financial Modeling Prep (FMP)** | `core/data/factors/fundamentals_fmp.py`, `scripts/backfill_all.py` | Needs `FMP_API_KEY` for ratios bulk / universe helpers |
| **Banxico** | [`api/routes/banxico.py`](../api/routes/banxico.py) | MX macro series |
| **Commodity feeds** | [`core/data/commodities.py`](../core/data/commodities.py) | Fetch/cache helpers for commodity analytics |
| **Kenneth French data library** | [`core/data/factors/fama_french.py`](../core/data/factors/fama_french.py), `scripts/backfill_all.py`, `scripts/update_daily.py` | FF5 daily via `pandas_datareader`; no API key; public data |
| **pandas-datareader** | FF5 pull (above), environment check in `scripts/test_environment.py` | In `requirements.txt`; used by the Kenneth French reader |

This is not an exhaustive list of every `requests.get` in the repo; search `scripts/` and `api/routes/` when adding a new row.

## 3. Systematic / academic factor families vs this repo

Conservative mapping: **“in data / core today”** means we either store proxies or compute them in `core/data/factors/` or `core/signals/`. **Not present** means no first-class Fama–French style portfolios or vendor factors unless you add them.

| Family | In data / derived in core today | Not present (typical gap) | Proxy / notes |
|--------|----------------------------------|---------------------------|---------------|
| **Market (Mkt-RF)** | `mkt_rf` in `fama_french_5.parquet`; benchmark index in prices (`^GSPC`) | — | FF market excess return now available; per-stock `beta_60d` is a rolling estimate |
| **Size (SMB)** | `smb` in `fama_french_5.parquet`; **`log_market_cap`** per-stock column in `factors_all` (from `data/market_caps/`) | — | FF market-level return + per-stock size signal both available |
| **Value (HML)** | `hml` in `fama_french_5.parquet`; `value_composite` from FMP fundamentals | — | FF official return series; repo also has custom cross-sectional value signals |
| **Momentum** | `mom_12_1`, `mom_6_1`, `mom_3_1`; Sortino path in `core/signals/momentum.py` | FF momentum factor (UMD) | 12–1 style is implemented; FF UMD is **not** in the 5-factor download (separate dataset) |
| **Quality** | `quality_composite`, ROE/ROA/margin/leverage | — | Cross-sectional z-scores; not exactly **q**-factor definitions |
| **Low volatility** | `vol_60d` | ACWI-min-vol style optimized portfolios | Vol is a feature, not a managed min-vol portfolio |
| **Investment (CMA)** | `cma` in `fama_french_5.parquet`; partial via fundamentals growth/margins | — | FF official return series now in data |
| **Profitability (RMW)** | `rmw` in `fama_french_5.parquet`; partial via quality composite | — | FF official return series now in data |
| **Risk-free rate** | `rf` in `fama_french_5.parquet` | — | Daily T-bill proxy from Kenneth French |
| **Carry (rates/FX/commodities)** | Commodity and macro modules | Full carry book | Asset-class specific; not unified in equity factor table |

**Fama–French five factors (Mkt-RF, SMB, HML, RMW, CMA):** Now **downloaded** daily from the Kenneth French data library into `fama_french_5.parquet` via [`core/data/factors/fama_french.py`](../core/data/factors/fama_french.py). These are **market-level long–short portfolio returns** (not per-stock signals). The repo also computes **custom per-stock cross-sectional features** (momentum, value/quality composites, vol, beta) from price and FMP fundamentals — these are complementary, not redundant.

**Takeaway:** The platform now ships both **FF5 market-level factor returns** (for benchmarking and attribution) and **custom per-stock factor signals** (for cross-sectional research). For **additional** academic factors (e.g. FF momentum UMD, liquidity, short-term reversal), add pulls using the same `pandas_datareader` pattern in `core/data/factors/fama_french.py`.

## 4. Scheduling (cron)

All data updates run **daily at 6 PM** (after US market close) via `crontab`. The authoritative reference copy is [`scripts/crontab.txt`](../scripts/crontab.txt) — restore with `crontab scripts/crontab.txt` if lost.

| Time | Script | What it updates | Log |
|------|--------|-----------------|-----|
| 18:00 | `scripts/update_daily.py` | Prices (yfinance), macro (FRED), FF5 (Kenneth French), price factors, sectors (quarterly auto-refresh), DuckDB views | `logs/update.log` |
| 18:05 | `scripts/update_commodities.py` | 14 commodity series | `logs/commodities_update.log` |
| 18:10 | `scripts/fetch_shares_and_market_caps.py` | Shares outstanding + historical market caps (yfinance) | `logs/market_caps_update.log` |

Python interpreter for all jobs: `/opt/anaconda3/envs/quant/bin/python`.

**Not scheduled (manual / one-off):**

- `scripts/backfill_all.py` — full rebuild from scratch (universe + prices + macro + factors + fundamentals). Run once to bootstrap, then rely on daily incremental updates.
- FMP fundamentals — requires paid `FMP_API_KEY`; skip if no subscription.

## 5. Maintenance

When you:

- add a Parquet output in `scripts/backfill_all.py` or a new loader in `api/dependencies.py`, or  
- add columns in `build_price_factors` / fundamentals, or
- add a new cron job,

update **§1** (artifacts), **§4** (scheduling), and if relevant **§3** (gap map). Keep ingestion rows in **§2** accurate (script name + module path). Update `scripts/crontab.txt` and reinstall with `crontab scripts/crontab.txt`.
