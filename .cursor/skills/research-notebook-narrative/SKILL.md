---
name: research-notebook-narrative
description: Enforces an explicit, plain-English narrative style for research and analysis Jupyter notebooks. Use when creating, editing, or reviewing any .ipynb file, when adding markdown cells that explain methodology, data, regimes/states, signals, exposures, overlays, or backtest results, or when the user asks to make a notebook clearer, more transparent, less implicit, or to disclose definitions, transformations, lookahead handling, or position sizing. Defines required sections (intro question, per-variable audit, state/label definitions, transparency block, overlay disclosure, decision section) and a tone contract.
---

# Research Notebook Narrative Style

## Why This Skill Exists

Research notebooks tend to leave too much implicit: which return, which weighting, which transformation, which lag, what a state label means, how much exposure is allowed and when. This skill forces every notebook to spell those things out in a consistent voice so a reader can audit assumptions without reverse-engineering code.

## When To Apply

Apply this skill on any of:

- creating a new `.ipynb` under `notebooks/`;
- editing markdown cells in an existing notebook;
- adding a new section that introduces a model, signal, factor, regime, overlay, or backtest;
- the user asking for transparency, plain English, definitions, leakage handling, or position-sizing disclosure.

If a notebook already exists and only some sections match this style, bring the rest into alignment instead of leaving a half-rewritten file.

## Tone Contract

- Plain English first, jargon only after definition. Define the term the first time it appears in a notebook.
- Short paragraphs. Bulleted lists for enumerable facts.
- Use a concrete worked example whenever a mechanism is non-obvious (overlay sizing, regime labeling, signal lag).
- Never assume the reader knows project-specific names like `mom_12_1` or `p_risk_on`. Define them on first use, then reuse.
- Distinguish "model belief" from "ground truth" explicitly when describing probabilities or labels.
- Do not soften limitations. Disclose them in their own subsection.

## Required Sections

Every research notebook should contain the following sections, in order. Section numbers can shift to match the notebook's existing pipeline, but the responsibilities must be present.

### 1. Intro: The Question And Plain-English Restatement

- Quote the research question verbatim, indented as a blockquote.
- Restate it in plain English in 1 to 3 sentences.
- Add a worked-example block that illustrates the mechanism. Use a fenced ``` ```text``` block, not pseudocode prose.

### 2. What Exactly Are We Observing?

- Name the primary observed series (e.g. "market returns") and define it precisely: which universe, which weighting, arithmetic vs log vs cumulative, daily vs rolling.
- List the input data sources fed into the model.
- State explicitly what the model does **not** see.

### 3. State / Label Definitions

When the notebook uses categorical labels like `risk_on`, `bull`, `regime_a`, define each label by:

- the rule that produced it (e.g. "highest mean trailing market return inside the training window");
- what economically credible behavior it should show in diagnostics;
- a one-sentence note that these are inferred labels, not ground truth.

### 4. Per-Variable Audit (Mandatory For EDA Sections)

For every column the model observes, produce one short block with the same fields in the same order:

- **Source**: file or module that produced it.
- **Transformation**: the math that turned the raw series into this column.
- **Missing-data handling**: how NaNs and gaps are treated.
- **Publication lag**: any delay applied upstream, or "none" with justification.
- **Standardization timing**: when scaling is fit, on which window, and when applied.
- **Leakage risk**: low / medium / high, with one sentence of justification.

Group near-identical variables (e.g. macro z-scores) into a single block only if every field is genuinely identical.

### 5. Transparency Block: "What Does It Mean When The Model Says X?"

When the notebook outputs a model decision, label, or probability, include a paragraph that reads:

> When the model says "today looks like X", that means \<exact mechanical statement, in features and windows\>.

This is the place to explain filtered vs smoothed probabilities, training-window scope, and label-mapping freeze inside a prediction block.

### 6. Overlay / Sizing Disclosure (When Applicable)

Whenever the notebook applies a signal as a multiplier, gate, or filter on a base strategy, disclose:

- the exact mapping from signal to allowed exposure, in a fenced ``` ```text``` block;
- the lag applied between signal observation and execution;
- the value range (e.g. 0.0 to 1.0) and the meaning of the endpoints;
- a code cell that displays an `exposure.describe()` quantile table and an exposure-by-state breakdown.

### 7. Decision Section

The performance comparison must:

- name what is being compared in plain English (not just column names);
- list each metric as a yes/no question;
- spell out the sign convention for the delta table;
- separate metrics that decide the conclusion from metrics kept only for context.

### 8. Disclosed Risks And Limitations

A dedicated subsection in the summary that lists, in plain language, every approximation, ungated assumption, and known fragility. Avoid burying limitations inside paragraphs.

## Anti-Patterns

- Using `df`, `tmp`, `x` as variable names in code cells. Use domain names.
- Calling something "market return" without specifying universe, weighting, and arithmetic vs log.
- Referring to `risk_off` or "high-risk regime" without describing the rule that produced the label.
- Showing an overlay or scaled-strategy result without disclosing the exposure rule and lag.
- Treating volatility reduction as proof of risk reduction.
- Long prose paragraphs where a table or fenced example would be clearer.

## Worked Example: Acceptable Intro Cell

```markdown
# HMM Regime Detection Signal

This notebook asks a specific research question:

> Can we infer market risk regimes from observable market, factor, macro, and volatility data, and can those regimes reduce downside risk when used as an exposure overlay on a momentum strategy?

In plain English: can we look at data we would have known at the time, decide whether the market looks calm, mixed, or stressed, and use that second opinion to decide how much of a momentum strategy to hold?

Example:

\`\`\`text
Momentum strategy says: be 100% invested in the long/short momentum portfolio.

Regime model says: the market looks risk-on.
Overlay allows: close to 100% of the momentum exposure.

Regime model says: the market looks risk-off.
Overlay allows: much less of the momentum exposure.
\`\`\`
```

## Worked Example: Acceptable Per-Variable Block

```markdown
#### `return_dispersion`

- **Source**: cleaned adjusted-close stock prices (`data/factors/prices.parquet`).
- **Transformation**: daily arithmetic returns per stock with `pct_change(fill_method=None)`, then the cross-sectional standard deviation across symbols on each business day.
- **Missing-data handling**: stocks without a return on a given day are excluded from that day's std. Rows where the entire panel is missing are dropped before standardization.
- **Publication lag**: none. Adjusted closes are observed end-of-day.
- **Standardization timing**: standardized inside each walk-forward training window only. Scaler is fit on the trailing 5-year slice and applied to the next 21-day prediction block.
- **Leakage risk**: low. Same-day construction, trailing-only scaling, and the derived exposure is shifted by one day.
```

## Workflow When Editing An Existing Notebook

1. Skim the cell index to see which required sections are missing or implicit.
2. Add or rewrite the intro using the worked example above.
3. For every variable the model observes, add a per-variable block. Do not collapse fields.
4. For every label, regime, or state, add an explicit definition and a "what it means when the model says X" paragraph.
5. For every overlay or sizing rule, add a fenced text block that states the mapping and the lag, plus a code cell that prints an exposure summary.
6. Rewrite the decision section with explicit yes/no metric questions and a documented sign convention.
7. Run the notebook with the project's environment (e.g. `quant` conda env). If `nbconvert --execute` fails on output schema, run a small repair script that adds missing `name` to stream outputs and missing `metadata` to display/execute_result outputs, then re-execute.
8. Validate with `nbformat.validate` before considering the work done.

## Verification Checklist

Before finalizing the notebook:

- [ ] Intro restates the research question in plain English with a worked example.
- [ ] "What Exactly Are We Observing?" defines every primary input series precisely.
- [ ] Every state or label has a definition and a credibility check.
- [ ] Every model-observed variable has a Per-Variable Audit block with all six fields.
- [ ] Every overlay/sizing rule has a fenced text disclosure plus an exposure summary cell.
- [ ] Decision section lists metrics as yes/no questions with sign convention.
- [ ] Risks and limitations live in their own subsection.
- [ ] Notebook executes end-to-end and passes `nbformat.validate`.
