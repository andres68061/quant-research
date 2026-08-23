# 0012. Substitute a 45-day filing lag when acceptedDate is a placeholder

Date: 2026-08-07
Status: accepted

## Context

The entire point-in-time fundamentals layer is built on FMP's `acceptedDate`,
which is supposed to be the EDGAR acceptance timestamp — the moment a filing
became public. Measuring the observed lag (`acceptedDate − period end`) across
400 symbols showed it is frequently not that at all:

| Decade | Rows | Median lag | Rows with lag ≤ 0 |
|---|---|---|---|
| 1980s | 2,597 | −1 day | **100%** |
| 1990s | 8,010 | 20.5 days | **49.5%** |
| 2000s | 11,170 | 39 days | 12.8% |
| 2010s | 13,451 | 35 days | 9.6% |
| 2020s | 9,379 | 35 days | 3.2% |

**21.5% of all statement rows carry an `acceptedDate` on or before the period
end** — the vendor filling the field with the period end for filings that predate
EDGAR electronic acceptance (phased in 1993–1996).

A filing cannot be accepted before the period it reports has finished. Taking
those dates at face value makes a quarter's results knowable on the day the
quarter closed: a ~35 day lookahead, concentrated in the early history where a
backtest has the least data left over to catch the resulting optimism. Because
leaked fundamentals make results look *better*, nothing about the output would
prompt anyone to check.

## Decision

`resolve_publication_date` in `core/data/factors/statement_metrics.py` treats a
publication date as implausible when it is missing or `<= reference_date`, and
substitutes `reference_date + 45 days`. It returns a boolean alongside the dates,
surfaced as a `publication_date_imputed` column that flows into the daily panel
so any result can be recomputed excluding imputed history.

45 days is the SEC's 10-Q deadline for large accelerated filers and sits above
the observed median real lag (32–39 days), so it errs late. The same guard runs
inside `vendor_metrics.build_filing_date_lookup`, since the vendor ratio datasets
inherit their dates from the same statements.

Only `lag <= 0` is treated as implausible. A short-but-positive lag is left alone:
an 8-K earnings release days after quarter end is genuine, and there is no
principled threshold between "suspiciously fast" and "fast".

## Alternatives rejected

- **Trust `acceptedDate` as-is.** The status quo, and wrong for a fifth of the
  panel. This is the single largest remaining leakage source in the repo.

- **Drop rows with implausible dates.** Loses all pre-1996 fundamentals — the
  1980s are 100% affected and the 1990s half. That discards the deepest history
  we have, to avoid a problem a conservative lag already solves.

- **Apply the Fama-French convention (align annual accounting data with returns
  from six months later).** The academically bulletproof choice, and far more
  conservative than needed for the 78% of rows where we have a real filing date.
  Applying it uniformly would throw away five months of genuine timeliness on
  modern data to fix old data.

- **Impute a per-decade or per-filer-size lag.** More accurate on average, but it
  is an estimate dressed up as a measurement, and the added precision does not
  change any ranking — 45 days is already past the deadline for the affected era.

## Consequences

Pre-1996 fundamental factors now lag their period end by exactly 45 days for
every symbol, which introduces a mild artificial synchronisation: names with the
same fiscal year end become visible on the same day, where real filings would
have staggered over weeks. That is visible as clustered rebalancing in early
history and is the accepted cost.

`publication_date_imputed` is carried in the daily panel. Any result that leans
on pre-2000 history should be re-run with imputed rows excluded before it is
believed; a factor that only works on imputed history is an artifact of this
decision, not a finding.
