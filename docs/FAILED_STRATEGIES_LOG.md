# Failed Strategies Log

What we tried and why it didn't work, with the real numbers. This is a
companion to `docs/ROADMAP.md`, never mixed with it:

- **`ROADMAP.md`** — what we want to build next. Forward-looking only.
- **This file** — what we tried and the evidence it didn't work. Entries
  are never deleted, even after a replacement approach ships — a negative
  result is still evidence, and the next person (or agent session)
  shouldn't re-spend the compute re-discovering it.

See the `strategy-experiment-log` skill for the checklist an agent should
follow after any strategy experiment (which doc to update, and how).

---

## Pairs stat-arb index: rolling multi-pair basket, ranked by price-path proximity

**Status: superseded.** The whole formation approach documented here (rank
candidates by how tightly their prices track, i.e. Gatev SSD or a
statistical-significance proxy for it) is now understood to be conceptually
wrong for a cost-inclusive mean-reversion strategy, not just unlucky in
this dataset — see "Why this whole family of attempts was the wrong shape"
below. Kept here in full because the specific numbers are still useful
evidence, and the next attempt (cointegration + cumulative-return-crossing
selection, trade until cointegration breaks) is a different approach, not
a parameter tweak of this one.

Shipped implementation: `core/strategies/pairs_index.py`,
`POST /run-pairs-index-backtest`, `/pairs-index` UI, registry
`pairs_stat_arb_index`, `notebooks/18_strategy_pairs_stat_arb_index.ipynb`.
Left live and disclosed as a research tool — see that notebook and the
registry's `known_limitations` for the user-facing framing.

### Attempt 1 — Gatev distance (SSD) formation

Rank same-sector candidate pairs each rolling period by sum-of-squared
deviations of normalized prices (smallest = closest tracking); trade the
top 10, re-form every 6 months on a trailing 12-month window.

**Result (2012–2026, real data, 10 sectors):** net Sharpe **-0.27**,
pain ratio **-0.09**. Underperformed the single hand-vetted XOM/CVX pair
(Sharpe 0.27 over the same span). GOOGL/GOOG (Alphabet's own two share
classes) selected in **28 of 28** rolling periods — SSD ranks them
"closest" essentially every time because their prices track near-perfectly
by construction, not because there's a tradeable spread.

### Attempt 2 — Engle-Granger significance ranking

Same rolling schedule; rank by formation-window ADF p-value (most
"statistically significant" cointegration first) instead of SSD.

**Result:** **worse** — net Sharpe **-0.48**. With ~100-250 candidate pairs
screened per period across 10 sectors, an uncorrected `p <= 0.05` threshold
is expected to pass several false positives by chance alone (the multiple-
comparisons problem the single-pair discovery process, notebook 17,
avoided by scoring on a genuinely held-out window — there is no equivalent
held-out window available inside a live rolling formation period).

### Attempt 3 — Minimum-dispersion-filtered SSD

Same as attempt 1, but excludes candidates whose RMS normalized-price
deviation is below a threshold (an attempt to directly exclude degenerate
near-identical pairs like GOOGL/GOOG).

**Result:** marginal improvement over attempt 1 but still negative — net
Sharpe **-0.41**. Removing the single worst offender doesn't fix a
selection criterion that's still ranking on the wrong quantity.

### Attempt 4 — Formation-internal walk-forward OOS Sharpe

Reuse `screen_pairs_walk_forward`'s train/validate split *inside* each
24-month formation window (mini train/validate), rank candidates by their
validate-window OOS Sharpe (using shrunk 126d/30d hedge/z-score windows to
fit inside the shorter internal split), then trade the winners with the
platform's standard 252d/60d windows on the real forward trading window.

**Result:** **worse** — net Sharpe **-1.10**. The internal validation used
different (shrunk) lookback windows than live trading — a parameter
mismatch. Reinforces notebook 17's finding that pairs Sharpe is highly
sensitive to keeping lookback windows consistent between validation and
live trading.

### Attempt 5 — Formation-window gross (0bps) in-sample Sharpe filter

Directly targets the confirmed root cause: require each SSD-ranked
candidate to clear a minimum **gross** (no transaction cost) in-sample
Sharpe over the 12-month formation window before it's eligible to trade,
on the theory that GOOGL/GOOG's near-zero gross edge should fail this gate.

**Result:** **worse** — net Sharpe **-0.74**, and GOOGL/GOOG still got
selected in many periods. A 12-month window is too short to estimate
Sharpe precisely enough to discriminate: `SE(Sharpe) ≈ sqrt(1/T) ≈ 0.08` at
T≈150 trading days, so a 0.1 threshold is barely more than half a standard
error above zero — mostly adds noise to the selection rather than removing
low-quality candidates.

### Attempt 6 — Cointegration-persistence index with COUPLED 60-month screening cadence (2026-07-18)

The replacement approach (crossing-filtered cointegration candidates, trade
each pair until a rolling Engle-Granger monitor breaks it) was first run
with the screening cadence **coupled** to the formation lookback: a
60-month lookback meant slots were only re-filled every 60 months. A
one-off run logged Sharpe **+0.744** / Pain Ratio 3.25 and was recorded in
the roadmap as "positive result."

**It did not survive scrutiny (config: 10 sectors, top 10 pairs, 10 bps,
platform-default windows, 2012→2026):**

- **Non-reproduction:** re-running the documented config produced Sharpe
  **−0.288**, Cid-1 −0.0037, total return −14.6%. The original +0.744 run's
  exact sector list / end date were not recorded; a plausible variation of
  those unrecorded details flips the sign, which is itself the finding.
- **Start-date fragility:** shifting `start` by ±6-12 months (2011-01 →
  2013-01, five starts) swings Sharpe across **+0.54, +0.07, −0.29, −0.97,
  −0.94**. With a 60-month cadence the whole outcome rides on 2-3
  calendar-lucky screening snapshots.
- **Structural dead zones:** every selected pair had stopped by mid-2019
  (first cohort) / late-2023 (second cohort), leaving the index flat for
  years at a time — only 1,031 trading days out of a 14-year span. 7 of 20
  pairs stopped at the earliest possible checkpoint (day 64): a pair can
  pass a 5-year formation window yet fail a 252-day monitor immediately.
- **Deflated Sharpe Ratio** (Bailey & López de Prado 2014, now in
  `core/metrics/deflated_sharpe.py`): counting the full family of
  pairs-basket configurations tried in this repo (attempts 1-5 above plus
  the 12/36/60-month sweep), the expected max Sharpe under pure selection
  luck is **≈0.70 annualized** — the celebrated +0.744 was never
  distinguishable from multiple-testing noise in the first place.

**Root cause:** coupling re-screen cadence to formation lookback. The fix
(decoupled `rescreen_months`, annual re-screening with the same 60-month
lookback) repaired the fragility — all starts positive, no dead zones,
75-85 pairs, market beta ≈ 0 — see `docs/ROADMAP.md` for that follow-up
and its own honest DSR verdict (still below the luck bar; not yet an edge).

**Lesson recorded:** always log the *complete* config (sector list, exact
dates, every parameter) alongside any headline number, and run the
start-shift + DSR checks *before* writing a positive result into the
roadmap, not after.

### Why this whole family of attempts was the wrong shape

All five attempts ranked candidates by some measure of **how tightly
prices track each other** (distance, correlation-flavored significance,
or a coarse in-sample Sharpe check on the same idea). That is backwards
for a strategy that needs to profit *from deviation and reversion, net of
costs*: the tighter two prices track, the smaller the deviations, and the
more transaction costs dominate whatever tiny reversion exists — which is
exactly the mechanism behind GOOGL/GOOG's failure (gross Sharpe ≈ 0.005,
i.e. no edge at all before costs; see notebook 18 §4). A sound selection
criterion should look for pairs whose **cumulative return paths visibly
diverge and cross each other repeatedly** over the formation window (real
oscillation around a shared equilibrium, with tradeable amplitude) — not
pairs that barely move apart at all. See `docs/ROADMAP.md` for the
replacement approach built on this insight.

---

## Regime overlay: exposure gating on the momentum factor book

### Attempt 1 — 2026-07-19: HMM / VIX / MA-200 exposure scalers on mom_12_1 L/S

Roadmap bundle A ("does regime gating improve Pain Ratio / max DD without
killing Sharpe?"). Pre-registered as exactly three overlays with library
defaults, no parameter sweeps.

**Base book:** `mom_12_1` long/short, PIT S&P filter, QE rebalance,
top/bottom 10%, 10 bps ADV-scaled costs, `signal_lag_days=1`,
2015-01 → 2026-07. (Run after the weekend-rebalance and initial-formation
fixes of 2026-07-19; invested ≈100% of days.)

**Overlays** (all causal; applied via `core/backtest/overlay.py::
apply_exposure_overlay`, 1-day signal lag, |Δexposure| charged at 10 bps
on gross 2.0):

1. `hmm` — walk-forward 3-state GaussianHMM (`fit_regime_hmm` defaults:
   5y train window, 21d step, filtered probabilities, diag covariance),
   exposure = `p_risk_on`.
2. `vix` — `vix_threshold_exposure(low=15, high=30)`.
3. `ma200` — `moving_average_exposure(^GSPC, 200d)`.

**Result (full window 2015-01 → 2026-07): every overlay is worse.**

| variant | Sharpe | Pain Ratio | max DD | ann ret | avg exposure |
|---|---|---|---|---|---|
| base    |  0.13 |  0.10 | −54% |  +3.3% | 1.00 |
| hmm     | −0.32 | −0.15 | −62% | −4.8% | 0.27 |
| vix     | −0.08 | −0.04 | −52% | −1.3% | 0.75 |
| ma200   |  0.07 |  0.06 | −54% | +1.6% | 0.81 |

Yearly end-truncation cadence (as-of 2019…2026, per CLAUDE.md): the HMM
and VIX overlays underperform base in **all 8** evaluation years; MA-200
helped only in the 2019/2020 truncations and converges to no-better-
than-base afterward while never improving the max drawdown.

**Why it failed (mechanism, not bad luck):** the base book's −54% hole is
a momentum crash. Momentum crashes are concentrated in sharp *risk-on*
rebounds (the low-momentum short leg rips when the market snaps back —
2020 being the canonical case), which is precisely when any market-level
risk-off gate is at full exposure. Gating on market stress therefore
cannot hedge this book's dominant risk; the HMM variant additionally
sat at 0.27 average exposure, diluting the (already thin) factor return
while paying scaling turnover.

**Not tried (would raise the trial-count bar):** exposure =
`p_risk_on + 0.5·p_neutral`, gating the *short leg only*, or a
momentum-crash-specific signal (e.g. trailing market drawdown as an
*inverse* gate). Any future attempt needs a pre-registered hypothesis and
should count these against the Deflated Sharpe trial family.

Reusable code shipped despite the negative result:
`core/backtest/overlay.py` (`apply_exposure_overlay`) + `tests/test_overlay.py`.
