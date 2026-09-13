# Notebooks

Research notebooks, arranged by what stage of the work they belong to. All
use the `quant` kernel (see CLAUDE.md). Outputs are stripped on commit.

```
notebooks/
  explore/     look at the data before doing anything with it
  ideas/       a signal idea taken far enough to decide whether it deserves a strategy
  strategies/  a strategy evaluated with the standard toolkit (see the strategy-evaluation skill)
  pipelines/   notebooks that produce an artifact other code reads
```

| Notebook | Question it answers |
|---|---|
| `explore/browse_databases` | What tables exist and what is in them (DuckDB) |
| `explore/sp500_historical_analysis` | What the point-in-time S&P membership looks like over time |
| `ideas/regime_hmm` | Can a hidden-Markov regime on macro/vol features be estimated stably? |
| `ideas/cross_sectional_quality_of_momentum` | Does the *quality* of a momentum path matter beyond its size? |
| `ideas/vol_risk_premium` | Is there a tradable gap between implied and realised volatility? |
| `ideas/markov_credit_macro_transitions` | Do credit/macro state transitions carry information? |
| `ideas/weather_commodity_demand` | Does weather predict commodity demand shocks? |
| `strategies/momentum_backtest` | Cross-sectional momentum, the baseline everything is compared to |
| `strategies/sortino_momentum` | Momentum ranked by Sortino instead of raw return |
| `strategies/factor_cross_section` | Generic factor long/short via `run_factor_cross_section_backtest` |
| `strategies/short_term_reversal` | 1-month reversal |
| `strategies/earnings_yield` | Value via earnings yield |
| `strategies/value_quality_sector_neutral` | Value + quality composite, sector-neutral |
| `strategies/near_52w_high` | Proximity to the 52-week high |
| `strategies/pairs_cointegration` | Cointegration pairs |
| `strategies/pairs_stat_arb_index` | The cointegration-persistence pairs index |
| `pipelines/precompute_ml_results` | Precomputes ML results for the Quick View cache |

The outcome of every strategy notebook is recorded in
`docs/FAILED_STRATEGIES_LOG.md` or `docs/ROADMAP.md`, never only in the notebook.
Reusable computation belongs in `core/`; a notebook shows the steps, it does
not own the logic.
