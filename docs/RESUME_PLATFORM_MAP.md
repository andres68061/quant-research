# Resume ↔ Platform Map

Interview cheat sheet: what the resume claims, where it lives in this repo.

_Last updated: 2026-07-12._

| Resume theme | Route / UI | API / core |
|---|---|---|
| Factor long–short (mom / vol / beta / 52w high) | `/` Factor Backtest | `POST /run-backtest`, factors `mom_*`, `vol_60d`, `beta_60d`, `near_52w_high` |
| Pairs (Engle–Granger cointegration) | `/pairs` | `POST /run-pairs-backtest`, `POST /screen-pairs`, `core/signals/pairs.py` |
| → validated signal: **XOM/CVX** | — | `notebooks/17_strategy_pairs_cointegration.ipynb` — walk-forward screen (2012–2022) then held-out test (2023–2026, never seen during selection): net Sharpe 0.89, ADF p=0.009. Rejected DAL/UAL (higher raw Sharpe, but not cointegrated OOS). |
| Risk diagnostics (rolling Sharpe/Sortino/vol, DD, hist, VaR×3) | `/` → **Risk** tab after backtest | `diagnostics` on `/run-backtest`; `core/metrics/diagnostics.py` |
| Efficient frontier / tangency / CAL | `/portfolio` | `POST /portfolio/optimize` |
| Commodities dashboards | `/metals` (nav label **Commodities**) | `/commodities/*` |
| Macro + recession shading | `/economic` | `/fred/*`, `/fred/recessions` |
| Sharpe limitations simulator | `/sharpe-limitations` | `POST /simulation/sharpe-comparison` |
| ML walk-forward + SHAP | `/ml-alpha` | `POST /run-ml-strategy` → `shap_importance` |
| Parquet + DuckDB | ETL / notebooks | `scripts/update_daily.py`, `core/data/factors/io.py` |
| Quarantine + manual overrides | `/data-coverage` Quarantine tab | `POST /data-coverage/quarantine/review` |
| Excluded-stock inspection | `/excluded-stocks` | `/exclusions/*` |
| PIT S&P membership | Data Coverage + backtest checkbox | `sp500_universe_filter`, `docs/SP500_MEMBERSHIP.md` |
| SEC EDGAR | Data Coverage `edgar_note` + sample parquet | `core/data/sec/`, `scripts/fetch_sec_filings_sample.py` |

## Stack wording

- **UI:** React + Vite (not Streamlit)
- **API:** FastAPI
- **Market data:** FMP + FRED; SEC EDGAR for filing-date sample / cross-check
- **Survivorship:** PIT membership CSV; disclosed pre-2015 price coverage gap
- **Macro:** fixed publication lags (`docs/MACRO_VINTAGES.md`), not ALFRED vintages
