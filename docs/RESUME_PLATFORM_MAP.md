# Resume ↔ Platform Map

Interview cheat sheet: what the resume claims, where it lives in this repo.

_Last updated: 2026-07-18._

| Resume theme | Route / UI | API / core |
|---|---|---|
| Factor long–short (mom / vol / beta / 52w high) | `/` Factor Backtest | `POST /run-backtest`, factors `mom_*`, `vol_60d`, `beta_60d`, `near_52w_high` |
| Pairs (Engle–Granger cointegration) | `/pairs` | `POST /run-pairs-backtest`, `POST /screen-pairs`, `core/signals/pairs.py` |
| → **selection-bias fix shipped: held-out mode** | — | The rolling hedge/z-score signal itself was always causal (no future data within its own window) — the risk was picking *which* symbols to test after already knowing how they perform. `POST /run-pairs-backtest` now takes an optional `train_frac`: splits the range into a train slice (diagnostic only, never traded) and a held-out slice (every reported metric). The `/pairs` UI's "Validate out-of-sample" toggle is on by default. `core/strategies/pairs_runner.py:run_pairs_holdout_backtest`. |
| → validated signal: **XOM/CVX** | — | `notebooks/17_strategy_pairs_cointegration.ipynb` — walk-forward screen (2012–2022) then held-out test (2023–2026, never seen during selection): net Sharpe 0.89, ADF p=0.009. Rejected DAL/UAL (higher raw Sharpe, but not cointegrated OOS). |
| Long/short pairs stat-arb index (rolling multi-pair basket) | `/pairs-index` | `POST /run-pairs-index-backtest`, `core/strategies/pairs_index.py` |
| → honest negative result | — | `notebooks/18_strategy_pairs_stat_arb_index.ipynb` — walk-forward Gatev-SSD basket, no lookahead, 2012–2026: net Sharpe **-0.27** (SSD) / **-0.48** (Engle-Granger ranking), both underperforming the single XOM/CVX pair. Root cause: proximity ranking alone keeps selecting degenerate matches (GOOGL/GOOG, same company's two share classes, selected 28/28 periods) — diversification did not fix single-pair fragility on this data. Good interview answer for "tell me about a strategy that didn't work and why." |
| Risk diagnostics (rolling Sharpe/Sortino/vol, DD, hist, VaR×3) | `/` → **Risk** tab after backtest | `diagnostics` on `/run-backtest`; `core/metrics/diagnostics.py` |
| Efficient frontier / tangency / CAL | `/portfolio` | `POST /portfolio/optimize` |
| → **in-sample bias fix shipped: walk-forward mode** | — | `/portfolio/optimize` + "Simulate" still fit and evaluate weights on the identical `[start_date, end_date]` (textbook Markowitz estimation-risk look-ahead) — that combo is left as-is for the "what does a naive backtest look like" comparison. The fix is `POST /portfolio/walk-forward-optimize` (`core/optimization/portfolio.py:run_walk_forward_tangency`): re-fits on a trailing lookback window every rebalance period, reports only realized returns after each fit. The `/portfolio` page's "Walk-forward validation" panel runs it on the same selected symbols/dates. Verified on real data (AAPL/MSFT/XOM/JNJ/KO, 2015–2026): naive Sharpe **1.05** vs honest walk-forward Sharpe **0.49** — roughly half. |
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
