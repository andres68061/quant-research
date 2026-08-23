# Data inventory and factor coverage

This document answers: **what variables and files exist in the repo today**, **which external sources the code can pull from**, and **how that compares to common systematic equity factor families**. Update it when you add Parquet artifacts, new ingestion scripts, or core factor builders.

- **Prioritized backlog**: [roadmap.txt](../roadmap.txt)
- **Platform gaps**: [PLATFORM_STATUS.md](PLATFORM_STATUS.md)

## 1. Artifacts on disk (ground truth)

Paths are relative to the repository root unless noted.

### Raw vs derived data layers

The repo separates **raw** from **derived** data:

| Layer | Definition | Files |
|-------|------------|-------|
| Raw | Source of truth. No publication lag, no business-day fill, no standardisation, no per-symbol cleaning beyond yfinance's own adjustment. | `data/raw/macro_fred.parquet`, `data/factors/prices.parquet` (light option: yfinance-adjusted close treated as raw), `data/factors/vix.parquet` |
| Derived | Deterministic functions of the raw layer. Rebuilt by scripts; safe to delete and regenerate. | `data/factors/macro.parquet`, `data/factors/macro_z.parquet`, `data/factors/factors_price.parquet`, `data/factors/factors_all.parquet`, `data/factors/fama_french_5.parquet` |

The macro raw layer is long-format (`reference_date`, `series_id`, `value`) at the FRED native frequency. `core/data/factors/macro.py::derive_macro_panel_from_raw` is the canonical pivot + publication-lag + business-day fill helper. Notebooks load from the raw layer and apply transformations inline so each step is visible.

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
| `prices.parquet` | Wide close panel from `build_prices_panel` (yfinance-backed in `core/data/factors/prices.py`); treated as the raw stock layer under the light option |
| `../raw/macro_fred.parquet` | **Raw** long-format FRED panel `(reference_date, series_id, value)` at native frequency. Source of truth for all macro derivations. Built by `scripts/fetch_raw_macro.py`. |
| `macro.parquet` | **Derived** publication-lagged business-day macro panel from `derive_macro_panel_from_raw(raw_long)` |
| `macro_z.parquet` | **Derived** 5-year rolling z-scores from `compute_macro_zscores(macro)` |
| `factors_price.parquet` | Price-derived factors from `build_price_factors` (see below) |
| `fundamentals_daily.parquet` | Dailyized fundamentals (per-symbol FMP statement calls), when fundamentals pipeline runs |
| `factors_vq.parquet` | Value/quality composites from `compute_value_quality_factors`, when fundamentals present |
| `factors_all.parquet` | `factors_price` joined with `factors_vq` on index (or copy of `factors_price` if no VQ) |
| `ohlcv.parquet` | **Long panel** (date, symbol) of `adj_open/high/low/close/volume`. Built by `scripts/build_ohlcv_panel.py` from the raw price layer — the closes-only `prices.parquet` discards four of five fields |
| `factors_microstructure.parquet` | 10 range/volume factors from the OHLCV panel (see §3b). Same producer |
| `fundamentals.parquet` | **Derived** PIT metric panel: the statement items that must meet a daily market cap, plus legacy columns |
| `factors_fundamental.parquet` | 37 fundamental factor columns (see §3a). Built by `scripts/build_fundamentals_panel.py` |
| `factors_earnings_surprise.parquet` | SUE / PEAD factors, announcement-dated, held 60 trading days (see §3c). Built by `scripts/build_event_factors.py` |
| `factors_vendor_metrics.parquet` | 42 vendor ratios re-dated onto real filing dates (see §3c). Same producer |
| `../universe/index_membership.parquet` | S&P membership intervals — the labeling table that replaced "membership = which panel you loaded" (ADR 0013) |
| `archive/prices_sp500_774_*.parquet` | The pre-cutover 774-name S&P union panel, kept verbatim so any earlier result stays reproducible |
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

### 3a. Fundamental factor columns (`factors_fundamental.parquet`)

Built by [`scripts/build_fundamentals_panel.py`](../scripts/build_fundamentals_panel.py) from
the raw statements — **no additional API calls**. The three statements carry ~147
vendor columns per symbol; these are the factor definitions derived from them.

Statement-only factors ([`core/data/factors/fundamental_factors.py`](../core/data/factors/fundamental_factors.py)),
computed at quarterly publication frequency then dailyized once:

| Column | Definition | Reference |
|---|---|---|
| `gross_profitability` | gross profit TTM / total assets | Novy-Marx (2013) |
| `operating_profitability` | (gross profit − SG&A) TTM / book equity | Fama-French RMW |
| `roa`, `roe`, `cfo_to_assets` | TTM net income or CFO over assets / equity | — |
| `gross_margin`, `asset_turnover` | margin and efficiency | — |
| `leverage`, `debt_to_equity`, `current_ratio` | balance-sheet structure | — |
| `rd_intensity` | R&D TTM / revenue TTM | — |
| `accruals` / `neg_accruals` | (net income − CFO) TTM / average assets | Sloan (1996) |
| `asset_growth` / `neg_asset_growth` | YoY total asset growth | Cooper, Gulen & Schill (2008) |
| `net_share_issuance` / `neg_` | log growth in diluted share count | Pontiff & Woodgate (2008) |
| `net_operating_assets` / `neg_` | (operating assets − operating liabilities) / lagged assets | Hirshleifer et al. (2004) |
| `capex_intensity` / `neg_` | capex TTM / total assets | Titman, Wei & Xie (2004) |
| `inventory_growth` / `neg_`, `revenue_growth` | YoY growth | — |
| `piotroski_f` | 0–9 financial-strength score | Piotroski (2000) |

Price-dependent factors (need a **daily** market cap, so computed after dailyization):
`book_to_market`, `earnings_yield`, `cash_flow_to_price`, `sales_to_price`, `fcf_yield`,
`ebitda_to_ev`, `rd_to_market`, `net_payout_yield`, `altman_z`.

Plus the legacy composites `value_quality`, `value_quality_sn`, `roe_sn`.

**Sign convention:** higher = the side the published anomaly says to go long. A
`neg_` prefix marks a factor whose raw form predicts negatively, so the shared
descending ranker never needs to know the sign.

**Caveats:** inventory, capex and turnover ratios are meaningless for financials —
filter by sector, do not special-case the formulas. Piotroski follows the paper's
*beginning-of-year* asset scaling, which differs from vendor implementations that
use contemporaneous assets (this reconciles our score to FMP's `financial_scores`
for AAPL and MSFT).

### 3b. Microstructure factor columns (`factors_microstructure.parquet`)

From [`core/data/factors/microstructure.py`](../core/data/factors/microstructure.py),
using the OHLC fields that were already on disk but unused:

| Column | Definition | Reference |
|---|---|---|
| `parkinson_vol` | annualized high-low range volatility | Parkinson (1980) |
| `garman_klass_vol` | annualized OHLC volatility | Garman & Klass (1980) |
| `range_to_close_vol` | Parkinson / close-to-close vol; >1 = wilder intraday path | — |
| `corwin_schultz_spread` | effective bid-ask spread, overnight-gap adjusted | Corwin & Schultz (2012) |
| `amihud_illiquidity` / `neg_` | mean \|return\| per $1M volume | Amihud (2002) |
| `overnight_return_21d`, `intraday_return_21d`, `overnight_intraday_gap` | return decomposition | Lou, Polk & Skouras (2019) |
| `close_location_21d` | mean position of close within the day's range | — |

**Caveat:** Corwin-Schultz was validated on 1990s–2000s data and **overstates the
spread for modern mega-caps** (~42 bps for AAPL vs. a true ~1 bp). Use it as a
cross-sectional liquidity ranking; the cost schedule lives in `core/data/liquidity.py`.

### 3c. Event and vendor-metric factor columns

Built by [`scripts/build_event_factors.py`](../scripts/build_event_factors.py),
also with no additional API calls.

**`factors_earnings_surprise.parquet`** — from
[`core/data/factors/earnings_surprise.py`](../core/data/factors/earnings_surprise.py).
The `earnings` dataset is the only fundamental input that is **announcement
dated**, so it needs no filing-date join.

| Column | Definition |
|---|---|
| `sue_price_scaled` | (actual EPS − estimate) / pre-announcement price. **The default** — scale-free and stable through zero earnings (Livnat & Mendenhall 2006) |
| `sue_std_scaled` | surprise / std. dev. of the firm's own past 8 surprises (classical SUE) |
| `eps_surprise_pct` | surprise / \|estimate\|. Useless near zero; for vendor reconciliation only |
| `revenue_surprise_pct`, `eps_growth_yoy` | revenue surprise and YoY EPS growth |
| `days_since_earnings` | calendar days since the announcement — slices the event window |

Unlike the fundamentals panel (273-day staleness cap), this one is held for **60
trading days** and then expires. PEAD is an event effect with a defined decay, not
a standing characteristic — forward-filling it to the next announcement would
invent signal that the literature does not claim.

**`factors_vendor_metrics.parquet`** — from
[`core/data/factors/vendor_metrics.py`](../core/data/factors/vendor_metrics.py).
42 vendor ratios (ROIC, cash conversion cycle, turnover, coverage, growth) from
`key_metrics` / `ratios` / `financial_growth`, re-indexed from fiscal period end
onto the **real filing date** joined from the raw statements. Without that join
these leak ~34 days (measured on AAPL); see ADR 0010.

Price-embedding vendor ratios (`priceToEarningsRatio`, `priceToBookRatio`,
`marketCap`, `dividendYield`, …) are deliberately **excluded** — they use the
vendor's period-end price and so are stale on every day but one. We compute their
equivalents against a daily market cap in §3a.

### Publication-date caveat (affects everything in §3a and §3c)

FMP's `acceptedDate` is a placeholder equal to the period end for **21.5% of
statement rows** — 100% of the 1980s, ~50% of the 1990s, ~3% of the 2020s (the
pre-EDGAR era). Taken literally those rows make a quarter knowable on the day it
closed. `resolve_publication_date` substitutes a conservative 45-day lag and flags
the row via `publication_date_imputed`, carried into the daily panel (ADR 0012).

**Any result leaning on pre-2000 history should be re-run with imputed rows
excluded before it is believed.**

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
| `data/S&P 500 Historical Components & Changes*.csv` | Historical S&P 500 membership (newest file by mtime; usually `(Updated).csv` from fja05680/sp500) | Manual copy from upstream sp500 repo after `sp500_historical.ipynb` | **Canonical PIT universe.** Procedure: [`docs/SP500_MEMBERSHIP.md`](SP500_MEMBERSHIP.md). Loaded by [`core/data/sp500_constituents.py`](../core/data/sp500_constituents.py). |
| `data/sp500_failed_symbols.json` | Symbols that failed yfinance fetch | Backfill scripts | Diagnostic; excluded from price panel. |

### Per-symbol FMP datasets (`data/raw/fmp/{dataset}/{SYMBOL}.parquet`)

Registry: [`core/data/fmp/datasets.py`](../core/data/fmp/datasets.py). Fetcher:
[`scripts/fetch_fmp_datasets.py`](../scripts/fetch_fmp_datasets.py) (`--list` prints
this table). Adding a dataset means adding one registry entry, not a new script.

**Read the `pit_status` column before backtesting anything.**

| pit_status | Meaning | Backtest use |
|---|---|---|
| `point_in_time` | Rows carry the date the information became public | Direct |
| `period_end_only` | Stamped with the fiscal period, **not** the publication date | Must join `filingDate`/`acceptedDate` from the raw statements on `(symbol, date)` first |
| `snapshot` | Single current-value row, no history | Reconciliation and live screening only — never backtesting |

| Dataset | Endpoint | PIT status | Why |
|---|---|---|---|
| `key_metrics` | `key-metrics` | period_end_only | ROIC, EV multiples, cash conversion cycle |
| `ratios` | `ratios` | period_end_only | ~60 vendor ratios |
| `financial_growth` | `financial-growth` | period_end_only | YoY and multi-year growth rates |
| `enterprise_values` | `enterprise-values` | period_end_only | Vendor EV build-up; cross-check for ours |
| `financial_scores` | `financial-scores` | snapshot | Vendor Altman Z / Piotroski F — our reconciliation target |
| `earnings` | `earnings` | **point_in_time** | Announcement-dated actual vs. estimated EPS/revenue → PEAD |
| `dividends` | `dividends` | point_in_time | Audits the price adjustment factors |
| `splits` | `splits` | point_in_time | Audits adjusted-price continuity; required to adjust intraday bars |
| `analyst_estimates` | `analyst-estimates` | snapshot | Consensus for *future* periods as of today — not a revision history |
| `grades_historical` | `grades-historical` | point_in_time | Monthly analyst rating counts |
| `ratings_historical` | `ratings-historical` | point_in_time | Vendor composite rating over time |
| `price_target_summary` | `price-target-summary` | snapshot | Rolling average price targets |
| `shares_float` | `shares-float` | snapshot | Free float vs. shares outstanding |
| `employee_count` | `employee-count` | point_in_time | Headcount with filing date → sales per employee |
| `insider_trading` | `insider-trading/search` | point_in_time | Form 4 transactions; use `filingDate`, not `transactionDate` |
| `stock_peers` | `stock-peers` | snapshot | Vendor peer group for relative-value screens |

Cost: one call per symbol per dataset (no bulk on our plan). The 16 datasets across
774 symbols is ~12,400 calls, ~70 minutes.

### Index membership labels (`data/universe/index_membership.parquet`)

Built by [`scripts/build_index_membership.py`](../scripts/build_index_membership.py) from
the S&P historical CSV. Columns: `symbol`, `index_name`, `valid_from`, `valid_to`
(NaT = still a member). One row per continuous membership interval, so a name that
left and rejoined has several (1,255 intervals over 1,202 symbols, 503 current).

**Why this exists (ADR 0013):** before the cutover, "is an S&P 500 member" was
implied by *which file you loaded* — the canonical panel held only S&P names. The
canonical panel is now the whole US market, so membership became an explicit
label. Restrict a backtest to index members by filtering these intervals, never
by loading a different panel.

### Universe table (`data/raw/fmp/universe/us_equity_universe.parquet`)

Built by [`scripts/build_fmp_universe.py`](../scripts/build_fmp_universe.py) from the
screener (live names above a market-cap floor) **plus** the delisted-companies feed.
Columns: `symbol`, `company_name`, `exchange`, `sector`, `industry`, `market_cap`,
`is_delisted`, `ipo_date`, `delisted_date`.

The delisted leg is not optional — a universe built from the screener alone silently
asks "how would this have done, restricted to companies that survived to today?".

The market-cap floor applies to **today's** market cap, so this table decides what to
**download**, not what is tradable on a given date. Point-in-time eligibility still
comes from the market-cap panel at backtest time.

`ipo_date`/`delisted_date` also narrow each symbol's fetch window: a company that
lived 1998–2003 costs one 5-year chunk instead of nine, cutting the price backfill
by ~37%.

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
| **Financial Modeling Prep (FMP)** | `core/data/fmp/` (client + per-dataset fetchers), `scripts/fetch_fmp_*.py` | Needs `FMP_API_KEY`. **Premium plan — bulk endpoints are NOT entitled**; every download is per-symbol. See [§6 FMP entitlements](#6-fmp-plan-entitlements-probed) |
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
- `scripts/backfill_expanded_universe.py` — the multi-hour expanded-universe backfill (below).
- `scripts/fetch_fmp_intraday.py` — intraday bars; cost-gated, see below.

### Long-running backfills (resumable)

Every fetcher skips symbols whose raw file already exists and writes atomically
(temp file + rename), so an interrupted run resumes on rerun with no lost work and
no truncated parquet mistaken for complete. This is the checkpointing mechanism —
there is no separate state file to corrupt.

```bash
# See the call budget before committing to it
/opt/anaconda3/envs/quant/bin/python scripts/backfill_expanded_universe.py --estimate

# Run it (steps: universe -> prices -> fundamentals -> market_caps -> panels)
/opt/anaconda3/envs/quant/bin/python scripts/backfill_expanded_universe.py \
    > logs/expanded_backfill.log 2>&1 &
```

At 9,011 symbols the budget is ~129,000 calls ≈ **4.3 hours** (prices 1.7h,
fundamentals 0.9h, market caps 1.7h) plus minutes of CPU for the panel rebuilds.

**Intraday is the expensive one.** The endpoint is *bar-capped, not range-capped*:
asking for seven months of 1-minute bars silently returns the most recent ~1,170
and drops the rest. Chunk sizes in `core/data/fmp/intraday.py` are set below each
interval's cap. Budget per symbol-year: 1min ~85 calls, 5min ~45, 1hour ~5, 4hour ~3.
The script refuses runs over 2 hours without `--yes`.

Intraday bars are stored as fetched: **split-adjusted as of the fetch date,
never dividend-adjusted** (measured 2026-08-11 on AAPL/NVDA/TSLA splits — the
earlier "unadjusted" claim was wrong). A split occurring after the fetch leaves
the stored snapshot stale; read via `core.data.fmp.intraday.load_intraday_bars`,
which detects and repairs exactly those splits. Never apply a blanket
adjustment — it double-adjusts.

## 5. Maintenance

When you:

- add a Parquet output in `scripts/backfill_all.py` or a new loader in `api/dependencies.py`, or  
- add columns in `build_price_factors` / fundamentals, or
- add a new cron job,

update **§1** (artifacts), **§4** (scheduling), and if relevant **§3** (gap map). Keep ingestion rows in **§2** accurate (script name + module path). Update `scripts/crontab.txt` and reinstall with `crontab scripts/crontab.txt`.

The `data-inventory-sync` skill (`.claude/skills/data-inventory-sync/`) carries the
full checklist and its trigger conditions — including the **mandatory**
`scripts/audit_data_health.py` rerun after any data change.

**Companion documents** (different jobs, don't merge them):

- **This file** — what artifacts exist and who produces them (inventory).
- **[`DATA_MODEL.md`](DATA_MODEL.md)** — how the tables are *organised*: panel
  families vs one wide table, raw/derived separation, and the fact-table vs
  interval-table (slowly-changing reference data) patterns.
- **[`DATA_HEALTH.md`](DATA_HEALTH.md)** — how good the data is: coverage,
  survivorship, leakage flags, and the known-flaw registry. Generated section is
  recomputed from disk; the same snapshot drives `GET /data-health` and the
  frontend **Data Health** page (`/data-health`). Flaw registry source of truth:
  `core/data/health.py::known_flaws`.

Expanded-universe staging artifacts (`prices_fmp.parquet`, `ohlcv_expanded.parquet`,
`factors_microstructure_expanded.parquet`, `factors_fundamental_expanded.parquet`,
`fundamentals_expanded.parquet`) sit beside the canonical 774-name files until the
cutover decision; the expanded fundamentals files are batch-sorted (date-sorted
within 400-symbol groups, not globally) and omit the cross-sectional composites —
see `build_fundamentals_panel.py --expanded`.

## 6. FMP plan entitlements (probed)

**Do not infer entitlements from the vendor's marketing tiers** — they do not match
what the key actually returns. On **2026-08-18**, all 263 documented examples
covering 230 unique paths were called with the live key. **176 paths returned
HTTP 200 and 54 returned HTTP 402.** The path-level results are in
[`docs/vendor/fmp/ENDPOINT_CATALOG.md`](vendor/fmp/ENDPOINT_CATALOG.md).

Plan: **Premium**, 750 calls/min (client throttles to ~500/min in
[`core/data/fmp/client.py`](../core/data/fmp/client.py)).

**Restricted (HTTP 402 — do not build against these):**

| Paths | Notes |
|---|---|
| `earnings-transcript-list`, `earning-call-transcript*` | Transcript directory, dates, latest, and content |
| `batch-exchange-quote`, asset-class batch quotes | Mutual funds, ETFs, commodities, crypto, forex, and indexes; symbol-list batches still work |
| `latest-financial-statements`, statement `*-ttm` | Latest cross-company statements and TTM income/balance/cash-flow statements; TTM metrics and ratios still work |
| sector/industry performance and P/E `*-snapshot` | Historical sector/industry performance and P/E paths still work |
| `institutional-ownership/*` | All eight Form 13F holdings and analytics paths |
| selected ETF/fund paths | `etf/holdings`, `etf/asset-exposure`, and all `funds/disclosure*`; ETF info/country/sector paths still work |
| `esg-*` | Disclosures, ratings, and benchmark |
| **all 18 bulk endpoints** | Includes `eod-bulk`, statements, profiles, scores, ratios, and metrics — **every universe download must be per-symbol** |

**Entitled but commonly assumed otherwise:** intraday `historical-chart/{1min…4hour}`,
`technical-indicators/*`, `commitment-of-traders-report`, `senate-trades`,
`economic-indicators`, `treasury-rates`, `company-screener`.

**Universe breadth available** (screener, 2026-08-07): 49,939 symbols total; 32,805
with financial statements; 9,543 US common stocks on NYSE/NASDAQ/AMEX; 4,568 above
$300M market cap; 2,590 above $2B.

Re-probe when the subscription changes:

```bash
/opt/anaconda3/envs/quant/bin/python scripts/probe_fmp_entitlements.py
```

## 7. Stale and superseded code (audit 2026-08-14)

Kept here rather than silently deleted: each entry is a decision the next session
should be able to make with context.

| Item | Status | Why | Recommended action |
|---|---|---|---|
| `scripts/backfill_all.py` | **Broken** | Calls `load_bulk_ratios_range` (FMP *bulk* endpoints, which return **402** on our Premium plan — see §6) and `build_prices_panel` (yfinance). Both paths were superseded by the FMP per-symbol fetchers. | Retire. Its jobs are now `fetch_fmp_prices.py` → `build_price_factors.py` → `build_fundamentals_panel.py`. Was already removed from the cutover rebuild chain. |
| `core/data/factors/fundamentals_fmp.py` | Partly dead | `load_bulk_ratios_range` / `fetch_bulk_ratios_year` hit unentitled bulk endpoints. `compute_value_quality_factors` is still used. | Delete the bulk functions; keep the composites. |
| `scripts/find_metals_series.py` | Orphan | Referenced nowhere. One-off FRED exploration. | Delete or move to `notebooks/`. |
| `scripts/test_fred_metals.py` | Orphan | Referenced nowhere; a script named `test_*` that pytest does not collect. | Delete. |
| `scripts/ml_data_preparation.py` | Orphan | Referenced nowhere; superseded by `core/features/`. | Delete after confirming no notebook imports it. |
| `data/factors/*_expanded.parquet` | Redundant | Staging artifacts from before the ADR-0013 cutover; the canonical panels now cover the same universe. | Delete once one full rebuild has been verified. |
| yfinance code paths | Legacy | `core/data/factors/prices.py`, `scripts/fetch_shares_and_market_caps.py` predate the FMP cutover. | Leave until a session needs to touch them; they are not on any live path. |

**Why this accumulated:** the FMP migration replaced the data source but the old
bootstrap script was never retired, and nothing failed loudly because nobody ran
it — until the cutover rebuild chain called it and got an argument error. Broken
code that is never executed looks identical to working code.
