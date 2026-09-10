# Roadmap / Next-Steps Log

Working log for the platform build-out. Each entry has enough context to
resume cold ("continue with the roadmap" should be sufficient instruction).
Update this file whenever an item ships or a decision changes.

_Last updated: 2026-07-19 (bundle A regime overlay: answered, negative — see failure log)._

## Resume skills coverage — only via strategy or analytics (not vanity)

Goal: every keyword under resume **ML**, **Time-Series**, and **Portfolio & Risk**
has at least one live platform example. Do **not** ship a page whose only job is
to demo a library. Bundle gaps into features that answer a research question.

### Already covered (no new build required for the claim)

| Resume claim | Where it already lives |
|---|---|
| Gradient boosting / trees / classification + walk-forward + SHAP | `/ml-alpha`, `run_walk_forward_validation` |
| Feature engineering | `core/features/`, ML pipeline |
| Cointegration + stationarity (ADF) | `/pairs`, `/pairs-index`, Engle–Granger |
| Factor strategies (mom / vol / beta) | `/` Factor Backtest |
| Efficient frontier / tangency / CAL + walk-forward weights | `/portfolio` (+ walk-forward panel) |
| Sharpe / Sortino / drawdowns / distributions / VaR·CVaR (incl. MC VaR) | Risk tab, metrics, Pain/Martin |
| Monte Carlo (risk) | VaR Monte Carlo in diagnostics |
| Supervised learning | ML direction models |
| Regime code exists (not yet a first-class product) | `core/signals/regime_hmm.py` + notebook |

### Gaps → justified bundles (add these; order = preference)

**A. Regime overlay on factor / pairs books** *(covers: regime-aware evaluation, unsupervised HMM)*  
Wire existing HMM (or a simple vol/macro baseline) as an **exposure scaler** on an
already-registered strategy (e.g. momentum or XOM/CVX pairs): risk-off → cut gross.
Research question: does regime gating improve Pain Ratio / max DD without killing
Sharpe? If no → `FAILED_STRATEGIES_LOG.md`. If yes → registry overlay + UI toggle.

**B. Classic ML baselines on the same walk-forward task** *(covers: KNN, SVM, regression)*  
On `/ml-alpha`, add KNN / linear-SVM (and optionally logistic) as **baselines beside
XGB/RF** under the identical walk-forward splits + metrics. Research question: does
boosting beat dumb baselines OOS, or are we overfitting capacity? No new product
surface — same page, honest horse-race.

**C. Residual / missing-factor diagnostics** *(covers: ARIMA/SARIMA, residual analysis, stationarity)*  
After any strategy NAV, regress (or difference) strategy returns on FF5 / market and
inspect residuals: ACF, ADF, optional ARIMA(p,d,q) on residuals. Research question:
is “alpha” just leftover autocorrelation or an omitted factor? Analytics tab on
Factor Backtest / pairs results — not a standalone ARIMA toy.

**D. Vol forecast → position sizing** *(covers: GARCH)*  
Fit a simple GARCH (vs EWMA baseline) on trailing returns of a traded book; size
next-day gross to a target vol. Attach to one strategy (pairs or factor L/S).
Research question: does vol-targeting improve Pain Ratio vs fixed notional after
costs? Negative result is fine (log it).

**E. PCA in the multi-factor blend** *(covers: PCA, unsupervised compression)*  
When building the already-planned **composite multi-factor index** (Open #6), form
the composite via PCA on the cross-section of factor z-scores (or equal-weight as
control). Research question: does the first PC beat naive equal blend OOS?
Do not add a bare “PCA explorer” page.

**F. Isolation Forest on the research data layer** *(covers: Isolation Forest on this repo)*  
Job experience already owns Isolation Forest; platform should not invent a fake
“IF dashboard.” Optional, only if useful: run IF on **return / ADV / factor
extremes** as an extra quarantine/analytics check on Data Coverage (why: catch
bad prints / regime breaks the rule-based scanner misses). Skip if quarantine
rules already suffice.

### Explicit non-goals (do not build just to match the resume)

- Standalone ARIMA/SARIMA/GARCH forecasting pages with no trading or residual question
- KNN/SVM demos outside the ML walk-forward comparison
- A second Monte Carlo product beyond VaR / path stress already planned for risk
- Claiming PCA while docs still say “no PCA” in ML price prediction — only use PCA
  in the factor-blend path above, and keep ML direction interpretable

### Coverage checklist (tick when a bundle ships)

- [x] A — regime overlay on a real strategy — **answered 2026-07-19, negative**:
      HMM / VIX / MA-200 exposure gating all fail to improve the mom_12_1 book
      (see `FAILED_STRATEGIES_LOG.md` "Regime overlay" for numbers and
      mechanism). Reusable `core/backtest/overlay.py` shipped; no UI overlay
      toggle warranted by the evidence.  
- [ ] B — KNN/SVM baselines on `/ml-alpha`  
- [ ] C — residual / FF5 / ARIMA residual diagnostics  
- [ ] D — GARCH (or EWMA+) vol targeting on a real book  
- [ ] E — PCA composite factor blend (with equal-weight control)  
- [ ] F — Isolation Forest quarantine assist (optional)

## Up next — universe/factor breakdown analytics with walk-forward replay

Confirmed build order: **(1) general/by-sector breakdown → (2) factor
cross-section → (3) pairs candidates.** Spec, as described: pick a universe
+ filters, see aggregate return/risk metrics update live; distribution
views (boxplot, histogram, sunburst by sector/pair) of those metrics across
the universe, not just the aggregate; a time-scrubber (extending the
existing `/replay/frames` infra used by `/ml-alpha`) so the distributions
animate through history; a "phantom" line/marker on each chart showing the
running average of the metric up to the scrubbed point, so today's
cross-section can be seen against its own history. Multi-day feature, not
started.

## Cointegration-persistence pairs index — shipped as research product; NOT a validated edge

_Updated 2026-07-18 after full validation. The previously-logged
"+0.744 Sharpe positive result" **did not reproduce and is retracted** —
see `docs/FAILED_STRATEGIES_LOG.md` attempt 6 for the full post-mortem
(non-reproduction, ±6-12-month start-shift sign flips, Deflated Sharpe
verdict). What follows is the corrected state._

**Why this approach:** every SSD/significance-ranked attempt at the
multi-pair basket failed — see `docs/FAILED_STRATEGIES_LOG.md`. The shared
flaw: ranking by "how tightly prices track" selects pairs with too little
deviation to profit from after costs.

**Shipped (core + API + UI):** `core/strategies/pairs_persistent.py`,
`POST /run-pairs-persistent-backtest`, `/pairs-persistent` page, registry
`pairs_persistent_index` (with the negative evidence in
`known_limitations`), tests in `tests/test_pairs_persistent.py` +
`tests/test_pairs_persistent_api.py`.

1. **Candidate filter**: Engle-Granger cointegration **and** a minimum
   number of hysteresis-band crossings of the normalized price paths over
   the formation lookback (real oscillation with tradeable amplitude, not
   tight tracking).
2. **Event-driven stops** (`run_pair_until_broken`): rolling EG monitor
   (252d window, checked every 21d) stops a pair after `persistence_checks`
   (4) consecutive failures. Finding that still holds: rolling-window ADF
   is noisy even for genuinely cointegrated pairs; 2 checks false-stops,
   4 rides it out.
3. **`rescreen_months` decoupled from `formation_months`** (the fix for
   attempt 6's fragility): screening cadence and lookback length are
   independent; free slots re-fill annually regardless of how long the
   lookback is. With the old coupled design a 60-month lookback meant the
   basket sat empty for years once its pairs died.
4. **`freeze_hedge_in_trade`** (execution redesign, off by default):
   freezes execution weights at entry instead of re-hedging daily beta
   drift, which was charged at 10 bps per unit turnover every day in a
   trade. Clean paired comparison (identical pairs/days, only execution
   differs): improves Sharpe ~+0.2 and cuts max DD ~5pp in every cell
   tested.

**Validated result (2026-07-18, lookback 60mo / rescreen 12mo / 10 sectors
/ top 10 / 10 bps / frozen hedge; start-shift robustness grid, end
2026-07):**

| start | Sharpe | Cid-1 | Cid-2 | total ret | max DD | beta vs eq-w mkt |
|---|---|---|---|---|---|---|
| 2011-01 | 0.652 | 2.672 | 0.228 | +54.0% | −12.6% | −0.000 |
| 2012-01 | 0.330 | 0.010 | 0.001 | +19.8% | −10.7% | −0.000 |
| 2013-01 | 0.664 | 0.065 | 0.007 | +41.8% | −12.6% | −0.000 |

Sign-stable across starts (unlike the coupled design), no dead zones
(2,100-2,600 trading days; 75-85 pairs), and genuinely market-orthogonal
(|beta| < 0.005 — the "alpha is orthogonal to market returns" box is
structurally ticked). **But the Deflated Sharpe Ratio verdict is honest
and negative**: counting all 10 pairs-basket configurations this repo has
tried, the expected max Sharpe under pure selection luck is ≈0.70
annualized; the observed 0.33-0.66 gives DSR 0.12-0.43 — *less likely than
not* that there is real skill here. Evaluation cadence used: once per
year (as-of year-end truncations 2018-2026), per CLAUDE.md convention.

**Disposition:** keep as a fully-disclosed research product. Do not deploy
or claim edge. The one thing that can raise the DSR without new
researcher degrees of freedom is genuinely new out-of-sample data — re-run
the same frozen config at the end of each calendar year and append the
result here. No further parameter/formation-criterion iterations on this
family without a pre-registered hypothesis (every extra trial raises the
luck bar for all of them).

**New core metrics shipped alongside:** `core/metrics/deflated_sharpe.py`
— `calculate_probabilistic_sharpe_ratio` (PSR), `expected_max_sharpe_under_null`,
`calculate_deflated_sharpe_ratio` (DSR), Bailey & López de Prado (2012,
2014). Evaluation-time diagnostics (need trial-family context), not part
of the generic per-backtest metrics dict.

## New findings this session (acted on)

- **Pain Ratio / Martin Ratio shipped** as diagnostic-only additions to
  every strategy's metrics (`core/metrics/performance.py`:
  `calculate_pain_index`/`calculate_pain_ratio`/`calculate_ulcer_index`/
  `calculate_martin_ratio`), exposed through `PerformanceMetrics` in every
  API response and surfaced as a KPI card on `/pairs` and `/portfolio`.
  Pain Ratio = annualized return / mean(|drawdown|) — integrates both
  depth and duration of every drawdown, unlike Calmar (single worst
  drawdown only). Recomputing the pairs findings on Pain Ratio instead of
  Sharpe did not change the conclusion (XOM/CVX held-out Pain Ratio 4.9 vs
  the SSD basket's -0.09) — it reinforces it.
- **`/pairs` selection-bias fix shipped**: `run_pairs_holdout_backtest`
  (`core/strategies/pairs_runner.py`) + `train_frac` on
  `POST /run-pairs-backtest` + a "Validate out-of-sample" toggle on the
  `/pairs` UI (on by default). Splits the date range into a train slice
  (cointegration diagnostic only, never traded) and a held-out slice (every
  reported metric); makes it impossible to accidentally blend a
  self-selected pair's full-history performance into what looks like a
  clean backtest.
- **`/portfolio` in-sample look-ahead fix shipped**:
  `run_walk_forward_tangency` (`core/backtest/mean_variance.py`) +
  `POST /portfolio/walk-forward-optimize` + a "Walk-forward validation"
  panel on `/portfolio` (collapsed by default, reuses the page's selected
  symbols/dates). Re-fits weights on a trailing lookback window every
  rebalance period and only reports realized returns from after each fit.
  Verified on real data: AAPL/MSFT/XOM/JNJ/KO 2015–2026 — naive
  `/optimize` + `/simulate` on the identical window shows Sharpe **1.05**;
  the honest walk-forward version shows Sharpe **0.49**, roughly half.
  This was previously undisclosed; distinct from the already-fixed
  factor-lookahead issue in `docs/PORTFOLIO_SIMULATION_FIXES_APPLIED.md`.

## PEAD validated on the expanded universe (2026-08-13) — the first positive result

`scripts/experiments/experiment_pead_by_size.py`, 267,780 announcements, 1992-2026, event-time
(day 0 excluded), quintiles by price-scaled SUE within each calendar quarter,
non-operating vehicles excluded, returns require price >= $1 and |ret| <= 300%.

| Bucket | Events | Spread @20d | Spread @60d | t |
|---|---:|---:|---:|---:|
| all | 267,780 | +0.96% | **+1.70%** | **10.05** |
| small | 23,243 | +0.95% | **+1.53%** | **3.60** |
| mid | 23,197 | +0.39% | +0.51% | 1.69 |
| large | 23,289 | +0.57% | +1.14% | 3.94 |

Small > mid and small > large, matching the literature's size prediction, and
the magnitudes (1-2% per quarter) are consistent with modern post-decay estimates
rather than the 2-4% of the 1980s papers.

**Two changes made this visible**, and the second matters more than the first:
4x more events, and **bad-print rejection**. Before it, the same study returned
`-inf` CAR paths — the panel contains daily "returns" up to +101,599,900% from
un-adjusted reverse splits, and one of those in a cross-sectional mean destroys
every symbol's abnormal return for that date. See `core/data/factors/returns.py`.

**Answered 2026-08-14: the effect is real, the obvious strategy is not.**
Steps 1 and 2 below are done — `core/strategies/pead.py` builds the overlapping
book and charges liquidity-scaled costs. Result: long/short goes from Sharpe
+0.70 under flat 10 bps to **−0.11** under real costs; restricting to liquid
names halves the gross edge. Full table in `FAILED_STRATEGIES_LOG.md`.

Still open:
1. ~~Cost model~~ — done (ADV-scaled).
2. ~~Tradable form~~ — done (overlapping book).
3. **Robustness.** Vary the $1 floor and the 300% bad-print bound; the result
   must not depend on either.
4. **Higher-conviction construction.** Every rescue tried so far reduces cost by
   reducing edge proportionally. Untried: trade only the most extreme surprises
   (a fixed small book rather than every qualifying announcement), or enter on
   the open instead of the close.

## Cutover landed (2026-08-13) — what it broke and what it enables

The canonical panel is now **8,910 symbols** (ADR 0013). Membership became a
label (`data/universe/index_membership.parquet`, 1,255 intervals). Four things
broke and were fixed, all of which are worth knowing about before the next
widening:

1. **Single-pass panel builds do not fit in memory** at 8,900 symbols. Price
   factors, fundamentals and event factors all now batch and append via
   `ParquetWriter`; the fundamentals canonical path switches to batching above
   1,200 symbols and computes cross-sectional composites in a cheap second pass
   (`factors_composites.parquet`).
2. **The API needed ~7 GB.** Fixed by lazy per-column factor loading
   (`core/data/store/factor_store.py`) plus an explicit, disclosed universe policy
   (`API_UNIVERSE`, default `research` = 6,369 symbols). ADR 0014. Startup is
   now ~3 s at ~1.5 GB.
3. **The universe filter was quadratic-ish**: a per-date `.loc` + MultiIndex
   append made one backtest take ~100 s at the new width. Rewritten as a single
   positional pass (~35% faster end to end, bit-identical output, pinned by
   `tests/test_universe_filter_perf.py`).
4. **20% of the universe is shells/SPACs** — see the flaw registry. Every
   cross-sectional screen must exclude them
   (`core.data.universe.filters.build_universe_filter`).

## Screen results are in (2026-08-11) — what survived and what's next

The systematic screen ran: **28 factors, uniform pipeline, zero Šidák survivors**
(`docs/research/factor_screen_20260811.md`; negatives logged in the failure log).
The PEAD event study also ran: **t=1.5 on large caps, not significant**. Both are
now permanent surfaces: `scripts/experiments/screen_factor_library.py`, `/pead` page,
`GET /backtest/events/pead-study`.

Forward-looking items that survive contact with the data:

1. **`neg_net_operating_assets`** — the one factor with stable decade Sharpes
   (0.27 / 0.53 / 0.65, t=2.16). Below the corrected threshold, but the only
   candidate whose profile isn't decay-shaped. Worth: sector-neutral variant,
   costs sensitivity, and the imputed-dates exclusion re-run.
2. **PEAD on the expanded universe** — the raw earnings data now covers ~6,800
   symbols; build the surprise panel against `prices_fmp.parquet` and re-run the
   event study on the small-cap tail where the literature puts the effect.
3. **Piotroski as specified** — standalone it is negative (−0.33); the paper
   applies it WITHIN the high book-to-market quintile. Test the conditional
   version before declaring it dead.
4. **The panel cutover decision** (774 → 8,908) — prerequisite for 2.

## Superseded — validate the newly available factors (2026-08-07)

The FMP footprint went from 10 endpoints to 26, the fundamental factor library
from 8 columns to 37, and the universe from 774 to 9,011 symbols. **None of the
new factors has been backtested.** They are computed, coverage-checked, and
registered — that is all. Treat every `expected_sharpe_range` in the registry as
a prior from the literature, not as evidence from this repo.

Work in this order:

1. **Post-earnings announcement drift (PEAD)** — the genuinely new strategy
   family, not a variation on an existing factor. The **signal now exists**:
   `factors_earnings_surprise.parquet` carries three SUE definitions plus
   `days_since_earnings`, announcement-dated, held 60 trading days (82% coverage).
   What is still missing is the **event-time backtest**. The shared
   cross-sectional runner rebalances on a calendar, so it would hold a stale mix
   of fresh and 59-day-old surprises and dilute exactly the effect being
   measured. Build the event-study harness first (align on announcement, measure
   cumulative abnormal return over the drift window), then decide whether a
   tradable calendar-rebalanced version survives costs.

2. **Screen the 29 new fundamental factors properly.** Running 29 factors and
   reporting the best one is exactly the multiple-testing trap ADR 0003 exists
   for — apply the Šidák correction, and report the whole cross-section of
   results, not the winner. Per CLAUDE.md, walk-forward at least annually.

3. **Re-run the existing factors on the expanded universe.** 774 → ~9,000 names
   is the first time there is real cross-sectional dispersion. Several factors
   (accruals, illiquidity, Piotroski) are documented as living in the small-cap
   tail and may show up for the first time. Note that the quarantine rules in
   `data/quality/` and the dollar-ADV filters will be doing much heavier lifting
   — small-cap bad prints are more common than large-cap ones.

4. **Microstructure factors** — `overnight_intraday_gap` is untested and depends
   on opening prices, the noisiest field in the vendor bar. Verify the split
   adjustment before trusting any result.

5. **Screen the 42 vendor metrics** in `factors_vendor_metrics.parquet`. The
   filing-date join is done (ADR 0010) so they are now point-in-time and safe to
   use, but none has been tested. Several — ROIC, cash conversion cycle, income
   quality — are genuinely distinct from anything in §3a rather than
   re-parameterisations, so they are worth a look. Same multiple-testing
   discipline as item 2.

6. **Re-check pre-2000 results with `publication_date_imputed` excluded.** 21.5%
   of statement rows had a placeholder filing date and now carry a conservative
   45-day lag instead (ADR 0012). A factor that only works on imputed history is
   an artifact of that choice, not a finding.

Whatever comes out of 1–4 goes to `FAILED_STRATEGIES_LOG.md` or ships — see the
`strategy-experiment-log` skill.

## Recently shipped

- **Data Monitor + 38-series macro layer (2026-09-10)** — the first EDA
  surface: distribution (moments, percentile-of-today, ADF), histogram with
  the latest value marked, rolling ±2σ level profile, year × month
  seasonality with hit-rate and n, years overlaid, Treasury curve snapshots
  and 2s10s / 3m10y / butterfly history, and a per-series freshness board
  wired into the watchdog. Found and fixed on the way: the commodity panel
  frozen for two months by an index-alignment bug, and monthly FRED lags
  that leaked values a month early (ADR 0018). Core in
  `core/research/eda.py` + `monitor.py`; page at `/data-monitor`.
  **Next on this surface, in order of value:** (a) cross-series view —
  rolling correlation / lead-lag between any two monitored series (e.g.
  breakevens vs WTI, 2s10s vs HY OAS) so the monitor answers "what moves
  with what", (b) regime shading — NBER recessions and the HMM regime as
  bands on every level chart, (c) commodity term structure once a
  futures-chain source exists (front-month only today), (d) ALFRED vintages
  (item 1 below) so the monitor can show first-print vs revised.

- **FMP footprint expansion + fundamental factor library (2026-08-07)** —
  probed entitlements empirically (`scripts/ingest/probe_fmp_entitlements.py`; the
  complete 2026-08-18 sweep found 176/230 paths working and records all 54
  HTTP-402 paths in `docs/vendor/fmp/ENDPOINT_CATALOG.md`); dataset registry
  with mandatory point-in-time classification (`core/data/vendors/fmp/datasets.py`, 16
  per-symbol datasets, ADR 0010); mined the raw statements from 6 derived fields
  to 37 factor columns (`statement_metrics.py`, `fundamental_factors.py`,
  `quality_scores.py`) with Piotroski and Altman reconciled against the vendor's
  own scores (ADR 0011 fixed a beginning-of-year scaling bug the reconciliation
  exposed); surfaced the OHLC fields that were on disk but unused
  (`ohlcv.parquet`, 10 microstructure factors); survivorship-free 9,011-symbol
  universe table (4,568 live + 4,371 delisted); resumable/atomic backfill
  tooling with per-symbol fetch windows (−37% calls) and an intraday fetcher
  that respects the endpoint's bar cap. 8 new registry entries.

  Also: earnings-surprise (SUE/PEAD) factors from announcement-dated data, and a
  filing-date join that makes the vendor ratio datasets point-in-time — measured
  at a 34-day leak on AAPL before the fix (ADR 0010). Reconciliation against the
  vendor's own scores validated Altman Z (n=684, corr 0.98) and showed Piotroski
  agrees in distribution and rank but not per-name, because we use TTM quarterly
  where the paper uses annual (ADR 0011). Measuring the publication lag exposed
  that 21.5% of statement rows carry a placeholder `acceptedDate` equal to the
  period end — a ~35 day lookahead now replaced by a conservative 45-day lag and
  flagged per row (ADR 0012). **No new factor has been validated — see "Up next"
  above.**

- **Simple Top-500 index + Cid-1 relevance study (2026-07-30)** — quarterly
  top-500-by-market-cap cap-weighted index ("S&P without the committee"):
  `core/strategies/top500_index.py`, `core/metrics/cross_section.py` (per-stock
  trailing Cid-1, ADR-0009), `core/research/cid1_study.py` (persistence / IC /
  Fama-MacBeth / quintile sort, Newey-West), `GET /index/top500/*`,
  `/index-top500` page. Index result 2005→2026 (gross, dividend-adjusted
  prices vs ^GSPC price index): +13.3% ann, Sharpe 0.70, corr 0.998,
  TE 1.18%, quarterly one-sided turnover 1.4%. Cid-1 experiment outcome
  is **negative** — logged in `FAILED_STRATEGIES_LOG.md`, never a
  selection criterion.
- **Market-cap panel integrity fix (2026-07-30)** — found
  `data/market_caps/historical_market_caps.parquet` was never rebuilt from
  the raw FMP layer (legacy values: 3Com $4.5T in 2008, MCI $1.6e16;
  Citigroup 2004 2× overstated). Rebuilt via
  `scripts/ingest/fetch_fmp_market_caps.py --build-only` (backup:
  `data/backups/historical_market_caps_backup_20260730_pre_raw_rebuild.parquet`).
  Downstream `log_market_cap` factor values will silently improve on next
  `factors_all` rebuild.
- **Long/short pairs stat-arb index (rolling multi-pair basket)** — walk-forward
  Gatev-SSD basket formation/re-formation, no lookahead;
  `core/strategies/pairs_index.py`, `POST /run-pairs-index-backtest`,
  `/pairs-index` UI, registry `pairs_stat_arb_index`. **Honest result**
  (`notebooks/18_strategy_pairs_stat_arb_index.ipynb`): the systematic basket
  lost money net of costs under every formation criterion tried (SSD,
  Engle-Granger significance, min-dispersion filter, formation-internal
  walk-forward Sharpe) and underperformed the single hand-vetted XOM/CVX
  pair — diversification did not fix single-pair fragility on this data.
  Root cause: proximity/significance ranking alone (no future data to
  validate against inside a live formation window) keeps selecting
  degenerate matches (GOOGL/GOOG, the same company's two share classes,
  selected in 28/28 periods).
- **Gatev distance formation** — normalize-to-1 SSD ranking on formation
  window, ADV-ranked sector universe, OOS z-score backtest;
  `core/strategies/pairs_gatev.py`, `/pairs` method toggle.
- **Pairs walk-forward screen** — same-sector Engle–Granger train filter +
  OOS PnL; `POST /screen-pairs`.
- **Pairs trading (Engle–Granger)** — `core/signals/pairs.py`,
  `POST /run-pairs-backtest`, `/pairs` UI, registry
  `pairs_cointegration`, `notebooks/17_strategy_pairs_cointegration.ipynb`.
- **`near_52w_high` (George & Hwang 2004)** — proximity to 252-day high
  factor + registry + panel patch + research notebook.
- Prior: resume-alignment (SHAP, Risk tab, quarantine UI, EDGAR, `beta_60d`);
  FMP stack, lifecycle, value_quality / roe_sn.


## 0. Survivorship gap — leave disclosed

Prefer 2015+ windows. Needs Norgate/CRSP/Tiingo to close.

## Open / optional

1. True ALFRED macro vintages (ingest) when a nowcast / revision strategy needs them
2. FMP membership `--promote` only after rename/recycle review (notation already fixed)
3. PIT sector source (replace today's FMP sector labels used in `*_sn` factors)
4. Persist `neg_vol_60d` / `neg_beta_60d` columns if we want one-click BAB/low-vol longs
  (ranker is descending — high factor = long)
5. **Cross-sectional factor long/short index** — package the existing
   `run_factor_cross_section_backtest` long/short mechanics (already used
   per-factor in `/` Factor Backtest) as its own named "index" product with
   a fixed rule set and its own registry entry/NAV, rather than an
   ad-hoc user-adjustable backtest.
6. **Composite multi-factor blend long/short index** — combine several
   existing factors (e.g. value + quality + momentum) into one composite
   score, then long/short on the composite; new signal-construction work on
   top of the existing per-factor infrastructure.

## From external review (2026-07-19) — remaining items

Shipped from the same review: purged/embargoed walk-forward
(`core/backtest/walkforward.py`), Šidák trial-count correction on the
momentum grid search, Newey-West FF5 alpha regression
(`core/metrics/factor_regression.py`), GitHub Actions CI, LICENSE,
stripped notebook outputs + nbstripout pre-commit, `requirements.lock.txt`,
research-first README. Still open:

7. **Borrow cost on the short leg** — long-short backtests currently charge
   ADV-bucketed transaction costs but no borrow/financing on the short book.
   Add a flat general-collateral assumption (25–50 bps/yr) as a
   `borrow_rate_annual_bps` parameter in `calculate_portfolio_returns`,
   plus a flag for hard-to-borrow names; matters most where the short leg
   concentrates in small/junky names (low momentum, high vol).
8. **Capacity / participation-rate analysis** — the dollar-ADV panel exists;
   answer "at what AUM does this strategy stop working?" by charging
   market impact as a function of order size / ADV participation.
9. **Bundled sample fixture** — a tiny committed dataset (≈10 symbols,
   2 years) powering one end-to-end demo backtest without an FMP key, so
   the repo is self-verifying for outside readers.
10. **Reframe ML page around cross-sectional panel prediction** — rank
    stocks cross-sectionally instead of predicting one series' next-day
    direction (near-zero signal-to-noise, discounted as a toy); converges
    the ML story with the factor story. Wire the Newey-West alpha
    regression into `/run-backtest` output as an FF5 attribution card
    while in there (needs FF5 parquet loaded at API startup).
