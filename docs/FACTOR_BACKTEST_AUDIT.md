# Factor Backtest — Audit & Chronological Documentation Plan

**Scope:** the Portfolio Simulator page (`/`) and its full chain — `frontend/src/pages/PortfolioSimulator.tsx` → `POST /run-backtest` (`api/routes/strategy.py`) → `core.strategies.run_factor_cross_section_backtest` → `core.backtest.portfolio.{create_signals_from_factor, calculate_portfolio_returns}` → factor artifacts built by `core.data.factors.build_factors`.

**Reviewer:** deep pass with Python repros. Severity labels: CRITICAL / HIGH / MEDIUM / LOW.

---

## 1. Executive summary

The Factor Backtest page exposes seven knobs (factor, rebalance freq, tcost, top/bottom pct, long-only, start, end) and a Run button. The page shows an equity curve, KPIs and a replay. There is **no on-page narrative** — a user cannot tell *why* `mom_12_1` or `beta_60d` should produce a positive long/short return, what the baseline hypothesis is, which assumptions are baked in, what failure modes to expect, or where the evidence comes from. The `/methodology` page has the math (cross-sectional rank, Sharpe, Sortino, MDD) but no economic justification, no references, no empirical validation, and no pointer to notebook 08.

Notebook 08 (`08_factor_cross_section.ipynb`) is a clean pipeline check — distributions, per-date valid counts, equity curves, position counts, outlier-bound comparison — but it is a **plumbing notebook**, not a validation notebook. It answers *"does this run without crashing and produce something"*, not *"is this a real alpha and why?"*. It does not contain: economic rationale, academic references, IC / decile-spread / monotonicity tests, half-life, sub-period stability, long-only vs L/S attribution, multi-factor correlation, sensitivity to `top_pct` / `min_stocks`, regime robustness, turnover and capacity analysis, or multiple-hypothesis correction.

The factor core has four material bugs that the notebook does not catch and that the page does not surface:

1. `momentum_excluding_recent` uses arithmetic `cum_12m − cum_1m` instead of geometric `(1+cum_12m)/(1+cum_1m) − 1`. **Rank flips** occur on realistic names — see §3 Bug 1. Roadmap rates this LOW; the audit rates it MEDIUM.
2. Same function uses `cum.sub(ex_recent, fill_value=0.0)`, which emits a **non-NaN momentum equal to minus the 1-month return** for any stock with less than 12 months of history. This injects spurious signal from IPOs and short-history names — §3 Bug 2.
3. `calculate_portfolio_returns` silently hides delisting losses: NaN returns are `fillna(0.0)`, and the `cash_from_delistings` bookkeeping is computed but never added to `gross_ret`. The portfolio weight on the defaulted name is zeroed out with **no loss recorded** — §3 Bug 4.
4. `run_factor_cross_section_backtest` localises naïve `end_date` to the factor tz (typically UTC) at 00:00, which falls *before* the US close. A request for `end=2024-06-28` silently excludes 2024-06-28 — §3 Bug 5.

There is no `signals.shift(1)` between factor computation and returns. The existing 1-day execution delay in `calculate_portfolio_returns` comes *only* from "today's return uses yesterday's position" — but the factor at close(t) still drives the position that starts earning on day t+1 (MOC-style execution), with no explicit lag and no UI exposure — §3 Bug 3.

Tests (`tests/test_backtest.py`) do not cover momentum formula correctness, delisting handling, IPO / short-history filtering, or signal shift. `test_survivorship.py` covers the universe filter only.

The rest of this document is the audit in full and a **chronological** plan for tying each page to a notebook chapter that actually justifies the strategy, with the bugs folded in at the right step.

---

## 2. What the user sees vs what the user should see

### 2.1 Portfolio Simulator (`/`) — current state

`frontend/src/pages/PortfolioSimulator.tsx`

```
LeftSidebar
  Factor           <select>            (mom_12_1, beta_60d, vol_60d, log_market_cap, mom_6_1, mom_3_1)
  Rebalance freq   Monthly | Quarterly
  Tcost (bps)      10 (free text)
  Top/Bottom %     5..50 (slider)
  Long only        ☐
  Start / End      dates
  [Run Backtest]  [Replay]

Right: KPIs (Sharpe, Sortino, IR, beta, alpha, trading days)
Main : equity curve
Bottom: metrics table + VaR
```

**Missing (every item below is a gap):**

* No prose. No "what is this strategy", no "why does it work", no "what assumption are you making when you hit Run".
* No link to notebook 08 or any validation artefact.
* No factor definition, lookback window, or publication-lag note per selected factor.
* No factor IC, decile spread or t-stat alongside the equity curve.
* No disclosure of the three roadmap-acknowledged risks in the same screen (macro lookahead doesn't apply here, but same-close, delisting fill, IPO spurious signal do).
* No sensitivity sliders beyond top/bottom: `min_stocks=20` is hardcoded to an invisible default.
* No benchmark overlay on the equity curve (IR / alpha / beta are shown numerically but the chart is one line).
* No turnover panel. A 10 bps tcost on 200% notional monthly turnover is a material subtraction, and users cannot see that their parameter choice is turning over 40% vs 300% of the book per month.
* No sub-period breakdown. The single-line equity curve hides regime dependence.
* Transaction cost is flat, with no disclosure: no borrow cost for shorts, no market impact, no bid/ask.

### 2.2 Methodology (`/methodology`) — current state

Has the math (signal rule, Sharpe, Sortino, MDD, Calmar, PnL with tcost, walk-forward). **Missing:** why each factor exists economically, what data preparation is applied, where the 252-day lookback comes from, why "12-1" excludes the recent month (short-term reversal), what the publication lag is on macro vs fundamentals vs price factors, what the known failure modes are, and what parameters were chosen from a grid search vs picked by convention.

### 2.3 Notebook 08 — current state

Eight cells, ~130 lines of code:

1. Setup / data load
2. Factor distribution descriptive stats
3. Per-date valid-stock counts (diagnostic)
4. Run backtests (mom_12_1, beta_60d, vol_60d × ME, QE — 6 runs)
5. Performance summary table
6. Equity curves + drawdowns
7. Position counts over time
8. Outlier-bound comparison (old abs<10 vs new `_infer_abs_bound`)

**What the notebook does:** plumbing check — the pipeline runs, factors have data, min_stocks=20 is not biting.

**What the notebook does not do (this is the heart of your instinct):**

* No **hypothesis statement** per factor. `mom_12_1` is run with no mention of Jegadeesh–Titman (1993) or the rational/behavioral rationale.
* No **Information Coefficient** (cross-sectional rank correlation of factor with forward return).
* No **decile spread / monotonicity** test (do quintiles 1 > 2 > 3 > 4 > 5 on forward return?).
* No **half-life** or decay analysis.
* No **multi-period** robustness — the backtest is one window, one parameter set.
* No **parameter sensitivity** — top_pct, min_stocks, rebalance freq compared only at default.
* No **factor correlation matrix** — are mom/vol/beta independent or redundant?
* No **long-only vs long/short attribution** — does the short book add or subtract?
* No **turnover and capacity** estimate.
* No **sub-period / regime** test (e.g. 2018–2019 vs 2020–2022).
* No **multiple-hypothesis correction** — you are running three factors and two freqs = six strategies; the roadmap currently does not address Bonferroni / BHQ.
* No **bug-catching checks** — a simple check "do the same four names rank top-20 under arithmetic vs geometric formula?" would have caught Bug 1 immediately.

### 2.4 The missing link

There is nothing in the repo that says *"the `/` page is validated by notebook 08 cell N"*. The page and the notebook share code (`create_signals_from_factor`, `calculate_portfolio_returns`) but the user-facing narrative does not reference the notebook, and the notebook does not include a sign-off ("this strategy passed the following tests, therefore you should expect…"). This is the gap you felt when you opened the page.

---

## 3. Bugs — line refs, repros, severity

> File paths are relative to repo root; the full path prefix used during testing was `/sessions/inspiring-stoic-babbage/mnt/quant/`.

### Bug 1 — Arithmetic momentum 12-1 (**MEDIUM**)

`core/data/factors/build_factors.py:17-24`

```python
def momentum_excluding_recent(close: pd.Series, months: int) -> pd.Series:
    ret = close.pct_change(fill_method=None)
    recent = 21
    window = months * 21
    cum = (1 + ret).rolling(window).apply(np.prod, raw=True) - 1.0
    ex_recent = (1 + ret).rolling(recent).apply(np.prod, raw=True) - 1.0
    return cum.sub(ex_recent, fill_value=0.0)          # ← arithmetic
```

Correct 12-1 momentum (Jegadeesh–Titman style) is **P(t−21)/P(t−252) − 1**, equivalently `(1+cum_12) / (1+cum_1) − 1`. The code uses `cum_12 − cum_1`. These are equal only in the limit of small returns.

Repro (clean rank flip on two realistic names):

```
                              cum_12m  cum_1m  mom_arith  mom_geom  rank_arith  rank_geom
Reversal (bounced from -50%)    0.15   -0.50       0.65      1.30           2          1
High-flyer (last month +40%)    1.10    0.40       0.70      0.50           1          2
```

Under the buggy formula you go long the high-flyer (which is exactly the *short-term reversal* name 12-1 is designed to avoid). Under the correct formula you go long the name that had a +130% return *excluding* the recent bounce. This is not a cosmetic issue — it re-introduces the very feature 12-1 is supposed to remove, so the factor you actually run is closer to `mom_12` (worse in the literature) than `mom_12_1`. Roadmap currently rates this LOW; audit upgrades to MEDIUM.

Fix:

```python
return (1 + cum).div(1 + ex_recent) - 1.0
```

and add a test in `tests/test_signals.py` against a hand-built 3-case series.

### Bug 2 — `fill_value=0.0` spurious momentum on short histories (**HIGH**)

`core/data/factors/build_factors.py:24`

`cum.sub(ex_recent, fill_value=0.0)` treats NaN as zero. When 12-month cumulative return is NaN (fewer than 252 observations), the expression collapses to `0 − cum_1m = −cum_1m`. Every stock with less than a year of history gets a momentum score equal to minus its 1-month return.

Repro output (100 trading days of data, 12m never populated):

```
            cum12      cum1  code_mom  correct_mom
2023-05-15    NaN  0.056740 -0.056740          NaN
2023-05-16    NaN  0.093669 -0.093669          NaN
...
Days where code emits a non-NaN momentum: 80/100
Days where correct momentum is non-NaN:   0/100
```

Impact: newly listed names enter the cross-sectional ranking with an *inverted short-term reversal* signal. In a year with heavy IPO supply (2020–2021), this is large enough to bias the factor. Survivorship-bias-free universe filtering does not help — IPOs *belong* in the universe at their listing date, they just should not carry a computed factor yet.

Fix: drop `fill_value=0.0`:

```python
return cum - ex_recent        # NaN propagates correctly
```

or simply use the geometric form (which does not need `sub` in the first place).

### Bug 3 — No explicit signal lag (same-close execution) (**MEDIUM**)

`core/backtest/portfolio.py:239-280` + `core/strategies/factor_runner.py:77-89`

`create_signals_from_factor` reads factor values on date t and emits signal on date t. `calculate_portfolio_returns` writes the new weight into `positions.loc[rebal_date]` and the gross-return loop uses `positions.loc[prev_date]` for the return on date t — so in practice the new weight starts earning on day *rebal_date+1*. This is a *de facto* 1-day delay between factor observation and the first return day, which is standard close-to-close.

Problem: there is no `signals.shift(1)` anywhere, and the factor at close(t) is used to build a weight that is considered "in place" at close(t). In an MOC-execution world this is defensible; in a close-to-open world you would want `signal(t) ← factor(t−1)` explicitly.

Repro (synthetic 300-day panel, 60 names, mom_63, monthly rebalance):

```
Signal uses close(t):          cum_ret = +0.0753  Sharpe = +1.04
Signal shifted by 1 day:       cum_ret = +0.1595  Sharpe = +2.30
```

(The synthetic process is momentum-positive, so here the lag *helps*; on noisier real data the lag typically *hurts* headline returns. Either way, the fact that a 1-day shift moves Sharpe by ~1.3 shows the timing is not innocuous.)

Fix options, from cheapest to most honest:

1. Document explicitly: "execution is MOC at the close used to compute the factor; Sharpes are upper bounds".
2. Add `factor = factor.groupby(level='symbol').shift(1)` before ranking and re-run the roadmap's validation grid.
3. Add a UI toggle "execution = MOC | next-open | next-close" and implement all three.

### Bug 4 — Delisting silently hides the default (**CRITICAL**)

`core/backtest/portfolio.py:237` and `:303-314`

```python
returns = prices.pct_change()                           # default fill_method='pad' → NaN prices ffilled
...
for symbol in pos[pos != 0].index:
    if pd.isna(ret[symbol]) or symbol not in ret.index:
        cash_from_delistings += abs(pos[symbol])        # cash_from_delistings is tracked...
        positions.loc[date:, symbol] = 0.0              # position zeroed going forward

valid_returns = ret[pos.index].fillna(0.0)              # NaN return becomes 0% return
gross_ret = (pos * valid_returns).sum()                 # → no loss for the delisted name
cash = cash_position.loc[prev_date] + cash_from_delistings
if date in rebalance_dates:
    cash = 0.0                                          # ...but cash is never added to gross_ret, and is wiped on rebalance
```

The `cash_from_delistings` value is recorded into a `cash_position` series but **never added to `gross_ret`** anywhere in the function. The net effect is that a name going to $0 vanishes with zero P&L impact.

Repro (portfolio 50% A + 50% B, B goes NaN mid-period):

```
Terminal NAV: 1.0000
Expected (if default realized): 0.5
```

Combined with `pct_change()` default `fill_method='pad'`, consecutive NaN prices become 0% returns rather than -100% shocks, so the bug is silent even in daily diagnostics.

Fix:
1. Drop `fill_method='pad'` in `pct_change` (line 237) — already deprecated, pandas 2.2 will remove it.
2. Either (a) realise −100% when a held position transitions from non-NaN to NaN close (treat delisting as a bankruptcy), or (b) realise `last_close / prev_close − 1` on the last known day and zero the position. Convention: realise bankruptcy as −100% unless a reliable delisting database marks it as a merger with a payout.
3. Add the `cash_from_delistings` into the gross return so that capital accounting closes.
4. Test: `tests/test_backtest.py::test_delisting_realizes_loss` with the repro above.

### Bug 5 — Naïve `end_date` localized to 00:00 UTC drops the last day (**LOW, but silent**)

`core/strategies/factor_runner.py:67-78`

```python
def _align_tz(ts: pd.Timestamp, tz) -> pd.Timestamp:
    if tz is None:
        return ts.tz_localize(None) if ts.tzinfo is not None else ts
    return ts.tz_localize(tz) if ts.tzinfo is None else ts.tz_convert(tz)

...
start_f, end_f = _align_tz(start, f_tz), _align_tz(end, f_tz)
factors_slice = factors[(f_dates >= start_f) & (f_dates <= end_f)]
```

If factors are tz-aware UTC (typical when built from yfinance), `end=Timestamp('2024-06-28')` becomes `2024-06-28 00:00+00:00`. US equity closes are ~20:00 UTC, so the final bar of 2024-06-28 is at 20:00 UTC — **after** the localized end. Pandas slicing `<= end_f` excludes it. The API honors `end_date='2024-06-28'` but the factor panel stops at 2024-06-27.

Repro:

```
Naive end = 2024-06-28 → tz_localize('UTC') = 2024-06-28 00:00:00+00:00
Close that day (20:00 UTC): 2024-06-28 20:00:00+00:00
Is close <= end? False
```

Fix: normalize to the end of day for `end`, or strip the time component from the factor index before slicing.

### Bug 6 — (design, not crash) Transaction cost ignores the short leg (**LOW**)

`core/backtest/portfolio.py:318` `trans_cost = turnover_val * transaction_cost`.

The single scalar `transaction_cost` is applied to (L1 turnover / 2). Real trading has: commission + spread + impact on each trade leg, and for shorts: borrow cost. The knob collapses all of these into one number with no disclosure on the page.

Fix: split into `commission_bps`, `impact_bps`, `borrow_bps_annualized`, charge the short leg `(|short_weight|) * borrow/252` per day. Expose at least commission vs borrow on the UI.

### Bug 7 — Ranking ties use `method='first'` (**LOW**)

`core/backtest/portfolio.py:162` `rank(..., method='first')`.

Deterministic but not statistically clean — the first symbol alphabetically among a tied group wins the long slot. `method='average'` is the standard cross-sectional convention. For momentum on a wide panel the practical effect is tiny; for `log_market_cap` or `vol_60d` on sparse days it can bias.

### Bug 8 — `pct_change(fill_method='pad')` FutureWarning (**LOW, but covers Bug 4**)

Already covered — drop the pad.

### Bug 9 — Floating `min_stocks=20` with no UI (**LOW**)

`core/backtest/portfolio.py:84` defaults `min_stocks=20`; the API does not expose it (`api/routes/strategy.py` does not pass it). On low-history factors or short backtests this silently blanks whole months.

### Bug 10 — Macro lookahead does NOT affect this page (audit finding, not a bug)

The factor backtest uses only price-derived factors (`mom_*`, `vol_*`, `beta_*`) plus optional `log_market_cap`. The FRED macro lookahead issue from `roadmap.txt` applies to ML features, not here. Worth stating on the page so the user knows which bias budget is spent.

---

## 4. Notebook ↔ UI linkage — current gap map

| Surface | What it says | What backs it up |
|---|---|---|
| `/` page title `Portfolio Simulator` | (none) | — |
| Factor dropdown | `mom_12_1 \| beta_60d \| vol_60d \| log_market_cap \| mom_6_1 \| mom_3_1` | No definition, no lookback, no academic reference, no IC. |
| Rebalance freq | `Monthly \| Quarterly` | No defence of why ME is default. |
| Tcost 10 bps | (default) | No link to commission + borrow model. Ignored bug 6. |
| Top/Bottom % | slider 5..50 | No sensitivity chart. |
| Long only | checkbox | No attribution showing what the short leg contributes. |
| Equity curve | single line | No benchmark, no sub-period, no turnover overlay. |
| KPIs | Sharpe, Sortino, IR, beta, alpha | No confidence intervals, no bootstrap. |
| Methodology | rank rule + metric formulas | No economic narrative, no references, no assumptions list. |
| Notebook 08 | plumbing check | Not linked from the page; does not validate the alpha. |

---

## 5. Chronological documentation plan

The goal is: when the user clicks Run, they can follow a chain — page copy → methodology → notebook chapter → core module — where every step is dated, justified, and tested. The sequence below is ordered so that earlier steps unblock later ones.

### Phase A — Fix the ground truth (prerequisite, week 1)

A1. **Fix Bug 1 and Bug 2** (momentum formula + fill_value). Replace the arithmetic form with geometric, drop `fill_value=0.0`. Add two unit tests in `tests/test_signals.py` — one hand-built series with known `(1+cum_12)/(1+cum_1)−1`, one IPO series that should emit all-NaN momentum.

A2. **Fix Bug 4** (delisting). Change `pct_change()` to `pct_change(fill_method=None)`, add realisation of −100% on first NaN close after a held position, and either drop `cash_from_delistings` or actually add it to `gross_ret`. Add `tests/test_backtest.py::test_delisting_realizes_loss`.

A3. **Fix Bug 5** (tz boundary). Normalize `end` to end-of-day in the requested tz, or strip time from the factor index when slicing.

A4. **Add a signal-lag parameter** (Bug 3). Minimum: `signal_lag_days: int = 1` on `run_factor_cross_section_backtest`, applied as `factor.groupby(level='symbol').shift(signal_lag_days)` before ranking. Default 1. Expose on the API request schema.

A5. **Expose `min_stocks`** in `BacktestRequest` with default 20 (Bug 9).

A6. **Rerun notebook 08** — the equity curves will change; save baselines under `notebooks/baselines/08/`. Commit the diff with a note: *"Bug fixes 1/2/3/4 change cumulative returns by X/Y/Z for `mom_12_1`/`beta_60d`/`vol_60d`."*

### Phase B — Rewrite notebook 08 as chronological strategy-validation chapters (week 2)

Rename to `08_factor_cross_section_validation.ipynb` and structure it as a sequence a reader can follow from top to bottom without jumping:

**Chapter 0. Hypothesis.** One paragraph per factor: the claim, the mechanism (rational/behavioral), the reference (Jegadeesh–Titman 1993 for `mom_12_1`; Haugen–Baker / low-vol for `vol_60d`; Fama–French SMB for `log_market_cap`; CAPM residual for `beta_60d`). State the null: "a random-sign portfolio over the same universe has Sharpe 0 ± …".

**Chapter 1. Data lineage.** Exactly which Parquet files, how many rows, date range, survivorship treatment, publication-lag disclosure (zero for price factors).

**Chapter 2. Factor construction.** For each factor, code cell that recomputes it from first principles (no helper), compared to the Parquet value, with a `|diff|.max()` assertion. This is where Bug 1 and Bug 2 live — the hand-built version is the contract.

**Chapter 3. Factor diagnostics.** Distribution (histogram + QQ vs normal), sectional autocorrelation, cross-sectional breadth per date (the existing cell 6/7), factor-factor correlation matrix. Flag days with <50 names.

**Chapter 4. Information Coefficient.** Cross-sectional Spearman rank correlation of factor with forward 1m return, rolling 252d, with 95% CI. This is the single most diagnostic plot for a factor and it is currently absent.

**Chapter 5. Decile spread / monotonicity.** Sort names into deciles each month; plot average forward 1m return by decile; look for monotone slope. A non-monotone spread is a sign of nonlinearity or data issue.

**Chapter 6. Base-case backtest.** Monthly, top/bottom 20%, 10 bps, signal_lag=1. One row per factor. Report: cum return, annual return, ann vol, Sharpe, Sortino, MDD, Calmar, IR vs SPY, alpha, beta, turnover, #names, hit rate.

**Chapter 7. Sensitivity grid.** `top_pct ∈ {0.1, 0.2, 0.3, 0.5}`, `rebalance ∈ {W, ME, QE}`, `signal_lag ∈ {0, 1, 2}`, `min_stocks ∈ {10, 20, 50}`. Heatmap of Sharpe. The stable factor is the one whose Sharpe does not collapse off-default.

**Chapter 8. Sub-period stability.** Split the sample into 3 equal windows and report the Sharpe triple. A factor whose Sharpe goes `+1.5 / +2.0 / −0.5` is not the same factor in different regimes.

**Chapter 9. Attribution.** Long-only vs short-only vs long-short cumulative returns on the same chart. Answer "is this alpha coming from shorts?" explicitly.

**Chapter 10. Multiple-hypothesis adjustment.** If you tested 6 strategies, the 95% threshold t-stat is not 1.96. Apply Benjamini–Hochberg and report which factors survive.

**Chapter 11. Robustness to known bugs.** Repro Bugs 1/2/3/4 and show the bad-formula Sharpe vs the good-formula Sharpe. This is the catch-net that ensures the fixes don't get silently regressed.

Every chapter ends with one sentence in bold: the conclusion the reader should carry to the next chapter.

### Phase C — Make the page linkable to the notebook (week 3)

C1. Add a "Strategy brief" pane to `PortfolioSimulator.tsx` that renders, for the selected factor: one-paragraph hypothesis (from registry), reference citation, IC value (from a precomputed JSON emitted by notebook 08), a link anchor `notebooks/08#chapter-4-mom_12_1`, and the three caveats ("price-only, no borrow cost, signal_lag=1"). This requires extending `core/strategies/registry.py:StrategyMetadata` with fields: `hypothesis`, `reference`, `expected_sharpe_range`, `known_limitations`.

C2. Precompute the IC, decile spread and sub-period Sharpe to `data/factors/diagnostics.json` as part of the daily cron. Stream into `GET /strategies`. The page pulls these and renders alongside the equity curve so the user sees "live IC = 0.08, stable across 3 sub-periods" while looking at their backtest.

C3. Add a turnover chart under the equity curve. The data is already in `calculate_portfolio_returns['turnover']` but thrown away in `api/routes/strategy.py`. Surface it.

C4. Add a benchmark overlay (SPY equal-weight vs strategy) to `EquityCurve.tsx`. Data is in `core/backtest/benchmarks.py` already.

C5. Expose `signal_lag_days` and `min_stocks` as advanced sliders under a "Details" disclosure.

C6. Update `/methodology` with an "Assumptions & known limitations" section listing: MOC execution with configurable lag, no borrow cost, no market impact, ffill-free pct_change, delisting realised as −100% by default.

### Phase D — Tests as the contract (week 4)

D1. `tests/test_factor_formula.py` — hand-built momentum case with rank-flip assertion (Bug 1), IPO short-history NaN assertion (Bug 2).

D2. `tests/test_backtest.py::test_delisting_realizes_loss` (Bug 4).

D3. `tests/test_backtest.py::test_end_date_includes_last_bar` (Bug 5).

D4. `tests/test_backtest.py::test_signal_lag_is_applied` (Bug 3).

D5. `tests/test_factor_runner.py::test_sensitivity_keeps_sign` — a minimal IC sanity that breaks CI if the factor stops predicting forward returns on the default window.

D6. `tests/test_api_strategy.py` — snapshot the `/run-backtest` payload for `mom_12_1, ME, 10bps, top/bot=20%, 2020-01-01..2024-12-31, survivorship_free=True` so pipeline-level regressions are loud.

### Phase E — Propagate the same pattern to the other strategies (later)

Once the factor page has (hypothesis → notebook chapter → diagnostics JSON → page pane → tests), repeat for:

* ML Alpha — notebook 07 stands in for 08 here, but the same gaps apply and the macro-lookahead risk is CRITICAL for this page specifically. Separate audit.
* Sortino Momentum — notebook 09 is closer to a validation notebook than 08 (bootstrap, regime) but still has no decile / IC story.
* Manual Portfolio / ETF Optimizer — these are user-constructed and do not need a hypothesis, but they need the same assumptions pane (delisting, tcost, borrow).

---

## 6. Concrete next PR (if you want to ship one thing this week)

Smallest self-contained PR that moves the needle:

1. Fix Bug 1 + Bug 2 in `core/data/factors/build_factors.py` — ≤ 5 lines.
2. Add `tests/test_factor_formula.py` with the rank-flip case and the IPO-NaN case — ≤ 50 lines.
3. Add a `MOMENTUM_FORMULA_FIX.md` in `docs/` containing the before/after Sharpe table for `mom_12_1 / mom_6_1 / mom_3_1` on ME and QE rebal (regen notebook 08 cell 5 after the fix).
4. Update `roadmap.txt`: move "Momentum formula" from `## Research / decisions` to `## Done / maintained` with a one-line summary.

That PR proves the audit chain (bug → repro → fix → test → docs → roadmap) works. Then Phase A carries on with the delisting and timing fixes.
