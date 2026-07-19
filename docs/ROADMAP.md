# Roadmap / Next-Steps Log

Working log for the platform build-out. Each entry has enough context to
resume cold ("continue with the roadmap" should be sufficient instruction).
Update this file whenever an item ships or a decision changes.

_Last updated: 2026-07-18 (resume skills coverage section: justified ML/TS/portfolio gaps)._

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

- [ ] A — regime overlay on a real strategy  
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

## Cointegration-persistence pairs index — shipped, positive result

**Why:** every SSD/significance-ranked attempt at the multi-pair basket
failed — see `docs/FAILED_STRATEGIES_LOG.md` for the full history. The
shared flaw: ranking by "how tightly prices track" selects pairs with too
little deviation to profit from after costs (the opposite of what a
mean-reversion strategy needs).

**Replacement approach, shipped**: `core/strategies/pairs_persistent.py`
(+ `core.signals.pairs.count_cumulative_return_crossings`), tests in
`tests/test_pairs_persistent.py`. Not yet wired to an API route or UI page
— currently a validated core module only.

1. Candidate filter (`find_crossing_cointegrated_candidates`): requires
   genuine Engle-Granger cointegration **and** a minimum number of
   hysteresis-band sign-changes in the normalized price-path difference
   over the formation window (the pair's paths must visibly cross
   repeatedly with real amplitude, not just sit close together). A naive
   zero-crossing count is noise-dominated and backwards for this purpose —
   two near-identical prices jitter across zero constantly from pure
   noise, which would rank them as the *most* active pair. The hysteresis
   band (`min_amplitude`, default 3% of normalized price) filters that out.
2. Trading duration is event-driven (`run_pair_until_broken`), not a fixed
   calendar window: a rolling Engle-Granger monitor (default: 252-day
   trailing window, checked every 21 days) stops a pair once its ADF
   p-value exceeds 0.10 for `persistence_checks` (default 4, ≈3 months)
   consecutive checks. New formation rounds only top up free slots left by
   pairs that already stopped — a pair that's still working keeps trading
   past its formation round.
3. Formation lookback sweep on real 2012-2026 data, 10 sectors, top 10
   pairs, same cost/entry/exit params as the platform default:

   | formation_months | Sharpe | Pain Ratio | pairs_ever | formations |
   |---|---|---|---|---|
   | 12 | 0.01 | 0.01 | 74 | 15 |
   | 36 | 0.22 | 0.24 | 40 | 5 |
   | **60** | **0.74** | **3.25** | 20 | 3 |

   Longer formation lookbacks work better here — plausible given the
   cointegration-persistence monitor itself needs ~252 days of clean data
   to reliably tell "broken" from "one noisy month" (see below), so a
   short 12-month formation barely outlives its own detection latency.
   `formation_months=60` (5yr) is the best result found this session for
   *any* multi-pair basket approach, better than the single XOM/CVX pair's
   full-span Sharpe (0.27) though on a smaller sample (3 formation rounds,
   20 pairs ever traded) — worth a second, independent validation window
   before trusting it as a shipped strategy.

**Real, load-bearing finding on rolling-window ADF noise:** even a
genuinely, permanently cointegrated synthetic pair produces occasional
runs of 2-3 consecutive "not cointegrated" monthly checks purely from
finite-sample ADF noise (confirmed both in unit tests and via notebook
17's real XOM/CVX rolling-ADF history). `persistence_checks` must be high
enough to ride that out — 2 was too low (false-stopped a provably-stable
synthetic pair in tests); 4 was not.

**Not yet done:** API route + `/pairs-persistent` (or similar) UI page;
second held-out validation window for the `formation_months=60` result
before calling it more than "promising."

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
