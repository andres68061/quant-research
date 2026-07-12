# Roadmap / Next-Steps Log

Working log for the platform build-out. Each entry has enough context to
resume cold ("continue with the roadmap" should be sufficient instruction).
Update this file whenever an item ships or a decision changes.

_Last updated: 2026-07-12 (lifecycle in update_daily; value_quality strategies)._

## Recently shipped

- FMP stack, quarantine, ADV costs, commodities→FMP, Finnhub dropped, BNY,
  STI quarantined (prior commits).
- **S&P Updated CSV trusted** — procedure in [`docs/SP500_MEMBERSHIP.md`](SP500_MEMBERSHIP.md).
  FMP membership = cross-check only.
- **Monthly lifecycle in `update_daily.py`**: rebuild windows every ~30d
  (`build_symbol_lifecycle.py --apply`); re-apply existing windows after any
  price update (`--apply-only`) so FMP refreshes cannot resurrect truncated
  cells. Factors rebuild after lifecycle.
- **`value_quality` / `value_quality_sn` registered** + columns on
  `factors_fundamental.parquet`. Holdout 2018–2025 (ADV costs, S&P filter):
  EY −0.23, ROE +0.33, VQ +0.20, VQ_SN +0.17 — SN does not dominate in this
  window; kept as research strategies with sector-lookahead disclosed.

## 0. Survivorship gap — leave disclosed

Prefer 2015+ windows. Needs Norgate/CRSP/Tiingo to close.

## Open / optional

1. Macro vintages (low priority)
2. Show S&P CSV age on Data Coverage page
3. Ticker-notation normalizer if/when promoting FMP membership
4. Revisit `value_quality_sn` after a second holdout / sector-neutral ROE alone
