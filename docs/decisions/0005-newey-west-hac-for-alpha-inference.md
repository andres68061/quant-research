# 0005. Newey-West HAC errors for factor-alpha inference

Date: 2026-07-19
Status: accepted

## Context

"Strategy earned X% at Sharpe Y" is incomplete: the number a quant
reviewer wants is alpha vs known factor premia with a t-statistic that
survives the serial correlation and heteroskedasticity of daily strategy
returns. Plain OLS standard errors overstate significance on such series.

## Decision

`core/metrics/factor_regression.py` regresses strategy returns on FF5
(`mkt_rf, smb, hml, rmw, cma`) via statsmodels OLS with
`cov_type="HAC"`. Default truncation lag is the Newey-West (1994)
plug-in rule `floor(4 * (n/100)^(2/9))` (≈6 for a few years of daily
data), overridable via `hac_lags`. Minimum 30 overlapping observations,
else `DataSchemaError`. Alpha is reported per-period, in bps, and
annualized, alongside betas, t-stats, and R².

## Alternatives rejected

- **Plain OLS errors** — anti-conservative on autocorrelated daily
  returns; exactly the mistake the module exists to prevent.
- **Block bootstrap for the alpha t-stat** — valid and assumption-lighter,
  but slower, and HAC is the literature-standard reviewers expect to see
  named.
- **Fixed `lags=6` default** — right magnitude for daily data but wrong
  for monthly; the n-dependent rule degrades gracefully across
  frequencies.

## Consequences

Timezone reconciliation is the caller's job (equity layer is tz-aware
NY, FF is naive-calendar — per CLAUDE.md, reconcile at the join). Not yet
wired into `POST /run-backtest` output; that needs FF5 loaded at API
startup and is ROADMAP item 10.
