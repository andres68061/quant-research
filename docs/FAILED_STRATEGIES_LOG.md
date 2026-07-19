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
