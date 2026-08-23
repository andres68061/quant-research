# 0013. Cut the canonical panel over from 774 S&P names to the full US universe

Date: 2026-08-12
Status: accepted

## Context

Since inception the platform's canonical price panel (`data/factors/prices.parquet`)
held ~774 symbols: the union of every company that has ever been in the S&P 500
(from the historical membership CSV), plus a few indexes/ETFs. Every backtest,
factor panel, and API surface keys off this file's columns and index.

The 2026-08 backfill staged a survivorship-aware **8,908-symbol** US universe
(live >$300M + delisted names) in `prices_fmp.parquet`, deliberately NOT touching
the canonical file. Two results now make the cutover the blocking step rather
than a nice-to-have: the factor screen (28 factors, zero Šidák survivors on
large caps) and the PEAD event study (t=1.5 on large caps) both point at the
small-cap tail as the only place the next result can come from — and that tail
is exactly what the canonical panel excludes.

User accepted the cutover on 2026-08-12, conditional on index membership being
preserved as an explicit labeling table.

## Decision

1. **Swap**: `prices.parquet` becomes the 8,908-symbol panel (same NYSE trading
   calendar — the expanded panel was already intersected with it, ADR-worthy
   bad-print dates removed). The old panel is archived verbatim as
   `data/factors/archive/prices_sp500_774_20260812.parquet`; any historical
   result can be reproduced against it.
2. **Membership becomes a label, not a universe**: a new
   `data/universe/index_membership.parquet` (symbol, index_name, valid_from,
   valid_to intervals from the S&P historical CSV) is the canonical membership
   table. S&P-restricted research keeps using `sp500_universe_filter` /
   `universe_filter` — membership filters the backtest, not the download.
3. **Every derived panel is rebuilt** against the new canonical file (price
   factors, fundamentals, microstructure/OHLCV, event factors, dollar ADV), and
   the canonical fundamentals build switches to the batched writer above ~1,200
   symbols (the single-pass build is exactly what OOM-killed the first attempt).
4. **Sector labels are extended** from 929 symbols to the whole universe (live
   names from the screener's sector field; delisted names via profile fetch), so
   sector-neutral factors and the sector page cover the new breadth.

## Alternatives rejected

- **Stay on 774 and run expanded research off staging files.** Two parallel
  canonical-ish universes, every module needing a "which panel?" switch, and the
  API showing results from a universe the data layer no longer treats as
  primary. The staging split was right for a week of validation, wrong as an
  end state.
- **Cut over to a filtered subset (e.g. top 3,000 by cap).** Reintroduces a
  today's-cap filter as a universe definition — the exact bias the expanded
  download exists to remove. Filtering belongs at backtest time, where it can be
  point-in-time.
- **Convert the platform to float32 to soften the memory step.** Panel memory
  grows ~35MB → ~750MB (float64) and the factor join to several GB; float32
  would halve it but silently changes numerics platform-wide against the
  CLAUDE.md default. Revisit only if startup memory actually hurts.

## Consequences

- **Every existing backtest number changes.** A top-20% tier of 8,900 names is a
  different portfolio than of 774. Prior results remain reproducible against the
  archived panel; the failure log's numbers are marked by universe implicitly
  via their dates (pre/post 2026-08-12).
- API startup loads a ~750MB price panel and a multi-GB factor join; nightly
  `update_daily` fetches ~8,900 symbols (~18 min at the client throttle) and
  spends longer on rolling factors. Acceptable; measure before optimizing.
- Small-cap data quality now matters: the quarantine/bad-print machinery faces
  names an order of magnitude junkier than S&P constituents. The Data Health
  flaw registry carries this; expect quarantine growth.
- The delisting terminal-loss convention (-100% on the long leg) now triggers
  far more often — which is the honest cost of a survivorship-aware universe.
