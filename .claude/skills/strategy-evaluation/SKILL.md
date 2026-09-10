---
name: strategy-evaluation
description: Use whenever evaluating whether a signal, factor, or strategy is any good — running a backtest, screening factors, comparing variants, or reporting a Sharpe ratio. Defines the standard evaluation: the one-paragraph plain-language framing, the required metric set beyond Sharpe, the mandatory controls, and the multiple-testing significance bar. Triggers on "backtest this", "is this factor any good", "compare these strategies", "what's the Sharpe", or any result that will be reported as evidence.
---

# Strategy evaluation

## Always state the test in one paragraph, in plain language

Before any numbers, say what was actually done, in this form:

> Rank every stock in the universe by X each month. Buy the top 20%, short-sell
> the bottom 20%. Hold until the next rebalance, then repeat. Subtract 10 bps of
> trading cost each way. Did it make money, and was it worth the risk?

This is not decoration. It is the fastest way for a reader to notice that the
test answered a different question than they had in mind, and it forces the
author to name the choices — universe, tier size, frequency, costs — that
determine the answer.

## Report the shape of the edge, not just Sharpe

Sharpe (annualized return ÷ volatility) compresses a whole return distribution
into one number. Two strategies with identical Sharpe can be completely
different things to own. **Always report, at minimum:**

| Metric | What it adds that Sharpe cannot say |
|---|---|
| **Annualized return** (net) | Whether the money is worth the effort at all |
| **Hit rate** | Right often, or rarely-but-hugely? |
| **Win/loss ratio** | Are the wins bigger than the losses? |
| **Profit factor** | Gross gains ÷ gross losses — cushion before costs eat it |
| **Max drawdown** | The worst peak-to-trough hole you must sit through |
| **Sub-period (decade) Sharpes** | Worked throughout, or only in one regime? |

`core.metrics.performance.calculate_performance_metrics` returns all of these.
A hit rate near 50% with a win/loss ratio near 1.0 is the normal signature of a
real cross-sectional factor (many small edges); a Sharpe carried by one 40% day
is not a strategy.

## The controls are not optional

The most common way to manufacture a finding is an unfair comparison. Before
reporting that X beats Y:

- **Compare like with like.** A long-only book's Sharpe is dominated by market
  beta, so comparing it against a long/short baseline measures beta, not skill.
  This exact mistake made conditional Piotroski look like a win until the
  long-only baseline reversed it (`FAILED_STRATEGIES_LOG.md`, 2026-08-13).
- **Include the do-nothing control.** If the strategy conditions on something
  ("F-score *within* the value quintile"), also run the version with the
  conditioning signal replaced by a constant. If the fancy version does not beat
  that, the fancy part contributed nothing.
- **Include the raw baseline** the strategy claims to improve on.
- **Reproduce a known result** as a sanity check when one exists.

## The significance bar

A t-statistic measures how confidently the average return differs from zero. The
conventional bar is |t| ≥ 2 — but that is for **one** test.

When N things are tried, the best of them looks good by luck alone. This repo
uses the **Šidák correction** at family-wise alpha 0.05 (ADR 0003): the per-test
bar becomes

    |t| >= inverse_normal_cdf(1 - alpha_per_test / 2)
    where alpha_per_test = 1 - (1 - 0.05) ** (1 / N)

For N = 28 factors that is **|t| >= 3.12**, not 2.0. Report N, report the bar,
and report **every** result — publishing only the winner reintroduces exactly the
bias the correction removes.

## Universe hygiene

- Exclude non-operating vehicles (shells/SPACs, fund wrappers) via
  `core.data.universe.filters.build_universe_filter`. A pre-merger SPAC has no
  factor exposure, and its artificial near-zero volatility lands it in the traded
  tiers of vol-sensitive factors.
- Compute returns through `core.data.factors.returns`, never bare `pct_change`. The panel
  contains vendor defects up to +100,000,000%, and one of them destroys a
  cross-sectional mean for every symbol on that date.
- State the universe and its size. A number computed on the API's 6,369-symbol
  research universe is not the same as one from the full 8,910-symbol panel.

## Verdict language

End with a verdict, not a number. One of:

- **Validated** — cleared the corrected bar, stable across sub-periods, survives
  realistic costs. Rare; nothing in this repo currently qualifies.
- **Interesting, not validated** — missed the bar but has a property worth
  pursuing (stability, an untested regime). Say what would settle it.
- **No edge** — did not beat its baseline. Log it in
  `docs/FAILED_STRATEGIES_LOG.md` with real numbers; never delete the entry.
- **Real but not tradable yet** — a genuine gross effect with no cost model.
  Say exactly what must happen before it can be called a strategy.

Then follow `strategy-experiment-log` to file the outcome in the right document.
