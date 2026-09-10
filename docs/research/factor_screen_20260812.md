# Factor library screen — 2026-08-12

## What this document is (read first)

A **factor** is a rule for ranking stocks — 'cheapest by book-to-market',
'most profitable', 'least indebted'. A **factor backtest** asks the only
question that matters: if you had ranked every stock by this rule, bought
the top 20% and short-sold the bottom 20%, rebalancing monthly, would you
have made money after trading costs?

Each row below is one factor put through that exact test. **Net Sharpe** is
return per unit of risk after costs (1.0 is very good, 0.5 respectable,
0 means the rule had no predictive value, negative means it predicted
backwards). **t** measures whether the result could plausibly be luck.

### Why the bar is set so high

We tested 1 factors at once. If you test enough rules, the best one
looks impressive purely by chance — flip 28 coins 20 times each and the
luckiest coin looks special. The Šidák correction raises the significance
bar in proportion to how many things were tested: here |t| ≥ 1.96
rather than the usual 2.0. Every factor is published below, winners and
losers, because reading only the winner reintroduces exactly the bias the
correction removes.

### How to read the result

**A factor that fails is not necessarily worthless** — it means this rule,
on this universe, over this period, after these costs, did not clear the
bar. Most published anomalies decay after publication and most were
documented on small caps, so a large-cap null is the expected outcome, not
a surprise. The decade columns matter: a factor that worked only in the
2000s is a decayed factor; one that is flat across all three decades is a
candidate worth more work.

## Setup

long/short top-bottom 20%, monthly rebalance, 10bps one-way costs, S&P 500 point-in-time membership, shells/SPACs excluded, T+1 execution, min 20 stocks.
Period 2000-01-03 -> 2026-08-07. **1 tests**, family-wise alpha 0.05, Šidák per-test threshold |t| ≥ 1.96.

## Results

| Factor | Family | Net Sharpe | Ann ret | t | 2000s | 2010s | 2020s | MaxDD | Šidák |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| roa | profitability | -0.12 | -1.9% | -0.63 | -0.3 | 0.13 | -0.03 | -66% | fail |

Survivors: none.

## Disclosed caveats

From the shared registry (`core/research/caveats.py`) — the same disclosures shown on every research surface:

- **[high] A single flat cost is applied regardless of liquidity** — Backtests charge one basis-point figure per trade for every name. That flatters any strategy that trades illiquid stocks — most of all the liquidity factors, which deliberately go long the hardest-to-trade names. Use the dollar-ADV cost schedule in core/data/factors/liquidity.py before believing a liquidity result.
- **[high] Testing many factors makes the best one look good by luck** — Across N tested factors the maximum Sharpe is inflated by selection. Results are reported against a Sidak-corrected threshold at family-wise alpha 0.05 (ADR 0003) and the FULL cross-section is published — reading only the winner reintroduces exactly the bias the correction removes.
- **[high] The universe contains shell companies and SPACs unless filtered** — About 20% of the expanded universe (1,779 symbols) are pre-merger SPACs or blank-check shells — trust accounts with tickers that sit near $10.00 at near-zero volatility until they merge or liquidate. They have no operations, so every fundamental factor is meaningless for them, and their artificial low-volatility profile puts them in the extreme tiers of vol- and liquidity-sensitive factors. Any cross-sectional result on the full universe must state whether they were excluded.
- **[high] Pre-2000 fundamentals use an imputed 45-day filing lag** — The vendor fills acceptedDate with the period end for pre-EDGAR filings (100% of 1980s rows, ~50% of 1990s). Those rows now carry a conservative 45-day lag and a publication_date_imputed flag. Any result leaning on pre-2000 history must be re-run with imputed rows excluded before it is believed (ADR 0012).
- **[high] Long/short results are gross of borrow cost and short availability** — The short leg assumes every name can be shorted at zero cost. In reality small, distressed, and heavily-shorted names are expensive or impossible to borrow — and those are exactly the names most anomaly short legs select.
- **[medium] One period, one universe, one parameterisation** — Every factor is run with identical untuned parameters over a single window. That avoids per-factor overfitting but means a result is evidence about THIS universe, period, and cost model — not a general claim about the factor.
