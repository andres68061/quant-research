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

---

## Cid-1 as a cross-sectional selection criterion (top-500 universe)

### Attempt 1 — 2026-07-30: trailing 252d Cid-1 vs forward quarterly returns

Question: is a stock's trailing Cid-1 ratio (total return ÷ cost-basis
pain, ADR-0009 boundary convention) *relevant* — i.e. does it predict the
next quarter — within the quarterly top-500-by-market-cap universe?
Diagnostic study, not a tradeable backtest: `core/index/cid1_study.py`,
surfaced at `GET /index/top500/cid1-study` and the `/index-top500` page.

**Config:** top 500 by cap at each quarter-end (rebuilt FMP market-cap
panel), 252d trailing window, 1-day execution lag, forward return =
exec-close to next exec-close, 2005-03 → 2026-06 (85 usable quarterly
cross-sections, median 494 symbols each). Controls: `mom_12_1`,
`vol_60d`, `log_market_cap` (z-scored ranks). All t-stats Newey-West.

**Result: persistent, but not predictive.**

- **Persistence (prerequisite): passes.** Cross-sectional rank
  autocorrelation between consecutive rebalances +0.469, t = 22.7 —
  Cid-1 is a stable characteristic of a stock, not noise.
- **Univariate IC: nothing.** Mean Spearman IC vs forward quarter
  −0.013, t = −0.78, p = 0.44.
- **Quintile sort: flat.** Avg forward-quarter returns Q1→Q5:
  3.3%, 4.1%, 3.7%, 3.8%, 3.2% — non-monotonic; Q5−Q1 spread −0.14%/qtr,
  t = −0.21.
- **Fama-MacBeth univariate: nothing** (γ = −0.0009, t = −0.35).
- **Fama-MacBeth with controls: significantly negative** (γ = −0.0054,
  t = −3.41) — *conditional on* momentum/vol/size, the smoother path
  underperforms. Do not read this as a tradeable anti-signal: Cid-1's
  numerator *is* trailing return, so the partial effect is heavily
  collinear with `mom_12_1` (whose own γ is +0.0071, t = 2.17), and the
  sign of the unconditional evidence flips across start years.
- **Start-year sensitivity (annual cadence):** mean IC is negative for
  2005-2009 starts, mildly positive for 2010+ starts; only the 2021 start
  (21 quarters) reaches spread t = 2.39 — exactly the lucky-window
  pattern the multiple-start rule exists to catch. Not significant after
  any multiplicity discipline (ADR-0003).

**Why it fails:** over a common trailing window Cid-1 is ≈ momentum
divided by time-underwater. The momentum part is already a known factor;
the extra "smoothness" information is persistent but carries no
incremental forward-return signal at the quarterly horizon on this
universe. Verdict: keep Cid-1 as a *diagnostic* measure for strategy
curves (its original role); do not use it as a selection criterion.

**Byproduct worth keeping** (shipped, see ROADMAP): the study exposed
that `data/market_caps/historical_market_caps.parquet` had never been
rebuilt from the raw FMP layer — legacy static-shares × price values gave
3Com a $4.5T cap (41% index weight, 2008) and MCI a $1.6e16 cap. After
rebuilding from raw, the simple top-500 index tracks ^GSPC at corr 0.998,
TE 1.18%.

## Factor library screen: 28 new factors, zero Šidák survivors (2026-08-11)

### Attempt 1 — uniform long/short screen of the 2026-08 factor library

**Setup:** every new fundamental/microstructure/surprise factor through the
identical pipeline — `run_factor_cross_section_backtest`, long/short top-bottom
20%, monthly rebalance, 10 bps one-way, S&P 500 PIT membership, T+1, 2000-01-03
→ 2026-08-07. Oriented side only (28 tests). Šidák family-wise α=0.05 →
per-test |t| ≥ 3.12. Full table: `docs/research/factor_screen_20260811.md`;
JSON: `data/quality/factor_screen_20260811.json`.

**Result: none of the 28 passes.** Best three: `amihud_illiquidity` Sharpe 0.47
(t=2.41) — but 2000s-only (decade Sharpes 1.04 / 0.11 / −0.2) and its 10 bps
cost assumption is fictional for the illiquid names it deliberately longs;
`neg_net_operating_assets` 0.42 (t=2.16) — the only factor stable across all
three decades (0.27 / 0.53 / 0.65); `sales_to_price` 0.42 (t=2.16), front-loaded
in the 2000s. Notables on the downside: `piotroski_f` standalone is **negative**
(−0.33) — consistent with Piotroski (2000) being specified WITHIN the value
quintile, not on the whole universe (the registry entry warned exactly this);
`gross_profitability` ≈ 0 net on large caps post-2000.

**Why it failed (read before re-running):** post-publication decay + large-cap
crowding is the base case for every published anomaly on a 774-name S&P
universe after 2000; 10 bps flat costs flatter the liquidity factors and still
nothing passes. The screen is evidence about THIS universe/period/cost model —
not a license to declare the factors dead on small caps or pre-2000.

## PEAD (event-time) on large caps: drift statistically zero (2026-08-11)

> **Superseded 2026-08-13** — not retracted: the large-cap null still stands as
> measured. On the expanded universe with bad-print rejection the drift IS
> significant (all events t=10.05; small caps t=3.60). Two things changed: 4x the
> events, and the discovery that vendor defects up to +100,000,000% were
> destroying the cross-sectional benchmark. See ROADMAP "PEAD validated".

### Attempt 1 — Bernard-Thomas event study, 69,462 announcements 1992-2026

**Setup:** `core/backtest/event_study.py` — day 0 = first tradable day after
announcement, day-0 jump excluded, abnormal = return minus daily equal-weight
universe mean, SUE = (actual−estimate)/prior close, quintiles within calendar
quarter, 60-day horizon. Surface: `GET /backtest/events/pead-study`, `/pead` page.

**Result: Q5−Q1 spread at day 60 = +0.60%, t = 1.50 — not significant.** The
drift exists in the first ~20 days (+55 bps) then flattens; nowhere near the
published 2-4% per quarter. Consistent with the literature: PEAD concentrates
in small caps and low-coverage names, and this universe is the S&P 500.

**Next attempt worth making (logged in ROADMAP):** same study on the expanded
~6,800-symbol earnings dataset once the expanded surprise panel is built against
`prices_fmp.parquet` — that adds the small-cap tail where the anomaly is
supposed to live. A positive there would still need the borrow/cost reality
check for small caps.

## Conditional Piotroski (F-score within the value quintile): no edge (2026-08-13)

### Attempt 1 — Piotroski (2000) as specified, six variants

**Why revisit:** the 2026-08-11 screen ranked `piotroski_f` across the whole
universe (Sharpe −0.33) but the paper applies the F-score *within the high
book-to-market quintile*. The registry entry flagged that mismatch, so the
conditional version had to be tested before calling the factor dead.

**Setup:** `scripts/experiment_conditional_piotroski.py`. F-score masked to NaN
outside each date's top-quintile book-to-market (breakpoint computed per date),
fed to the shared cross-section runner. 2000-01-03 → 2026-08-07, monthly, 10bps,
S&P PIT membership, shells/SPACs excluded, 30% tiers within the quintile.

| Variant | Sharpe | t | 2000s | 2010s | 2020s |
|---|---:|---:|---:|---:|---:|
| A F-in-value, long/short | **−0.39** | −1.99 | −0.57 | −0.19 | −0.23 |
| B F-in-value, long-only | 0.51 | 2.62 | 0.43 | 0.75 | 0.35 |
| C plain book_to_market, long/short | 0.39 | 1.99 | 0.75 | −0.19 | −0.03 |
| D F-score standalone (control) | −0.25 | −1.30 | −0.43 | −0.08 | −0.07 |
| E plain book_to_market, **long-only** | **0.68** | 3.49 | 0.80 | 0.75 | 0.52 |
| F value quintile, F-score ignored, long-only | 0.54 | 2.78 | 0.53 | 0.76 | 0.34 |

**Result: the F-score adds nothing.** B (0.51) is *worse* than F (0.54) — holding
the same value quintile with the score ignored — and much worse than E (0.68),
plain value long-only. Long/short is outright destroyed (A = −0.39, −99% max
drawdown): the short leg is low-F value stocks, i.e. distressed names that
rocket in recoveries.

**Methodological note worth keeping:** the first run of this experiment compared
B (long-only, 0.51) against C (long/short, 0.39) and looked like a win. It is not
a valid comparison — a long-only book's Sharpe is dominated by market beta. E and
F were added specifically as like-for-like controls, and they reversed the
conclusion. Any future long-only result must be compared to a long-only baseline.

**Why it failed:** on this universe the value quintile itself carries whatever
premium exists; the nine accounting tests do not separate winners within it. The
paper's sample was 1976-1996 US small caps with far thinner analyst coverage.

## PEAD long/short: real effect, not tradable after realistic costs (2026-08-14)

### Attempt 1 — overlapping portfolio, flat vs liquidity-scaled costs

**Context:** the event study validated the drift itself (Q5−Q1 +1.70% over 60
days, t=10.05, 267,780 announcements — see ROADMAP "PEAD validated"). That is an
event-time, gross-of-costs research result. This tests the tradable form.

**Construction** (`core/strategies/pead.py`, `scripts/experiment_pead_tradable.py`):
overlapping book — each announcement opens a position the day AFTER the
announcement and holds 60 trading days, so ~1/60th turns over daily; long the top
surprise quintile, short the bottom; gross exposure normalized to 1.0 each day;
shells/SPACs excluded; returns require price ≥ $1 and |ret| ≤ 300%.

| Variant | Gross Sharpe | Net Sharpe | Net ann | Turnover/day | Cost drag |
|---|---:|---:|---:|---:|---:|
| A long/short, flat 10bps | 1.11 | **0.70** | 1.8% | 4.3% | 1.1%/yr |
| B long/short, ADV-scaled costs | 1.11 | **−0.11** | −0.3% | 4.3% | 3.2%/yr |
| C liquid names only (ADV ≥ $20M) | 0.72 | 0.19 | 0.6% | 4.4% | 1.8%/yr |
| D 120-day hold | 0.82 | 0.15 | 0.4% | 2.3% | 1.7%/yr |
| E long-only top quintile | 1.02 | 0.85 | 15.5% | 4.4% | 3.2%/yr |
| F long-only ALL announcers (control) | 0.82 | 0.67 | 11.8% | 4.1% | 2.7%/yr |

**Result: the long/short book is not tradable.** The gross edge is real and
strong (Sharpe 1.11), but a 2.9%/yr gross return cannot carry a 3.2%/yr cost
drag. **The gap between A and B is the entire finding**: a flat 10 bps assumption
turns a dead strategy into an apparently good one (+0.70 vs −0.11). Flat costs
are not conservative for a book whose edge lives in small caps.

**Why the obvious rescues fail:** restricting to liquid names (C) cuts the cost
drag nearly in half but *also* cuts the gross Sharpe from 1.11 to 0.72 — the edge
genuinely lives in the illiquid tail, which is the classic "real anomaly you
cannot reach" shape. Halving turnover with a 120-day hold (D) likewise halves
the gross edge.

**On E (long-only, net 0.85):** read against F, not against A–D. A long-only
book's Sharpe is dominated by market beta, so the honest measure of what the
surprise signal contributes is **E − F = +0.18 Sharpe** (0.85 vs 0.67) over
holding *every* announcer with no signal at all. A real but modest increment,
and it inherits full equity beta and a 3.2%/yr cost drag.

**Status: real effect, no tradable long/short implementation found.** Not a
refutation of PEAD — a refutation of this construction. Ideas not yet tried:
trade only the largest surprises (fewer, higher-conviction positions), enter on
the open rather than the close, or hold a fixed small number of names instead of
every qualifying announcement.
