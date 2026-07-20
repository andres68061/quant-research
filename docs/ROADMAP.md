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
  `run_walk_forward_tangency` (`core/optimization/portfolio.py`) +
  `POST /portfolio/walk-forward-optimize` + a "Walk-forward validation"
  panel on `/portfolio` (collapsed by default, reuses the page's selected
  symbols/dates). Re-fits weights on a trailing lookback window every
  rebalance period and only reports realized returns from after each fit.
  Verified on real data: AAPL/MSFT/XOM/JNJ/KO 2015–2026 — naive
  `/optimize` + `/simulate` on the identical window shows Sharpe **1.05**;
  the honest walk-forward version shows Sharpe **0.49**, roughly half.
  This was previously undisclosed; distinct from the already-fixed
  factor-lookahead issue in `docs/PORTFOLIO_SIMULATION_FIXES_APPLIED.md`.

## Recently shipped

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
