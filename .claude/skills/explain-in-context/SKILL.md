---
name: explain-in-context
description: Use when writing ANY user-facing explanation in this repo — reporting a result, summarizing work, answering a question, naming a factor/metric/artifact, or writing a docstring that a non-author will read. Enforces self-contained explanations: no unexplained jargon, no references to prior conversation, no bare identifiers. Triggers whenever a reply or document mentions a Sharpe ratio, a factor name, a file name, an ADR number, a t-stat, a universe, or any repo-specific term.
---

# Explain in context

The reader does not have the conversation you have. They may be the user reading
a summary weeks later, a future session with no memory, or the user right now who
knows the domain unevenly — deep in some places, empty in others, and not
signposted.

**The test:** could someone who has never seen this repo, reading only this
paragraph, act on it correctly? If not, it is not finished.

## The four failure modes

1. **Bare identifier.** "`neg_net_operating_assets` was the only stable one."
   → *What it is, how it is computed, why it might work, and what "stable" meant.*
2. **Orphan reference.** "the 774 panel", "ADR 0011", "the cutover", "the screen".
   → *Say what the thing is at first mention, then use the short name.*
3. **Unglossed jargon.** Sharpe, t-stat, PEAD, point-in-time, survivorship,
   winsorize, cross-section. → *One clause of plain meaning, inline.*
4. **Result without verdict.** "Sharpe 0.42, t=2.16." → *Is that good? Compared
   to what? Should we act on it?*

## The pattern that works

For any result or finding, cover these in order. Two sentences each is usually
enough; do not turn it into a template with headings.

1. **What was asked** — the question in plain language.
2. **What was done** — the method, stated so it could be repeated, including the
   choices that could have gone another way.
3. **What came out** — the numbers, with units and direction.
4. **What it means** — the verdict. Good/bad/inconclusive, and against what
   baseline.
5. **What is still unknown** — the caveat that would change the conclusion.

## Worked example

Bad:

> `neg_net_operating_assets` had the best decade profile (0.27 / 0.53 / 0.65).

Good:

> **Net operating assets** measures how much of a company's balance sheet is tied
> up in operations (receivables, inventory, plant) rather than cash. When the
> figure grows faster than the business, it usually means past reported profits
> were accounting entries rather than cash — and those firms tend to underperform
> later. We therefore rank stocks by the *negative* of it (hence `neg_`), so
> "high" means the side the research says to buy.
>
> Tested over 2000–2026 on our universe, buying the top 20% and short-selling the
> bottom 20%, rebalanced monthly, after 10 bps of trading cost: **Sharpe 0.42**
> (return per unit of risk — 1.0 is very good, 0 is no skill), t-statistic 2.16.
>
> That is *below* our significance bar of 3.12, so it is not a validated edge.
> What makes it the only interesting one of the 28 tested is the decade profile —
> 0.27, 0.53, 0.65 across the 2000s/2010s/2020s. Every other factor decayed after
> publication; this one did not, which is the pattern a real effect leaves.
>
> Unknown: it has not been tested with realistic (liquidity-scaled) costs, nor
> excluding the pre-2000 rows with imputed filing dates.

## Numbers need their scale

Never state a metric without the reader being able to judge it:

- Sharpe → say what good looks like (≈1.0 excellent, 0.5 respectable, 0 no skill)
- t-stat → say the threshold being used and why
- a return → say gross or net, and over what period
- a count → say what population it describes (see `data-inventory-sync`)
- a p-value or correction → say how many things were tested

## What this is not

Not a demand for length. A one-line answer to a one-line question is correct.
This is about *self-containment*, not volume: prefer the extra clause that makes
a sentence stand alone over an extra paragraph that repeats it.

Do not restate the whole history of the repo in every reply. First mention in a
given reply gets the gloss; after that the short name is fine.
