# Roadmap / Next-Steps Log

Working log for the platform build-out. Each entry has enough context to
resume cold ("continue with the roadmap" should be sufficient instruction).
Update this file whenever an item ships or a decision changes.

_Last updated: 2026-07-12 (low-priority cleanup: coverage age, notation, vintages, roe_sn)._

## Recently shipped

- FMP stack, quarantine, ADV costs, commodities→FMP, Finnhub dropped, BNY,
  STI quarantined, lifecycle in `update_daily`, `value_quality` / `_sn`
  (prior commits).
- **S&P CSV age on Data Coverage** — `sp500_csv` on `/data-coverage` + KPI /
  coverage-tab banner (file mtime age + last membership row).
- **Ticker notation normalizer** — `normalize_equity_ticker` (`.`→`-`) in
  FMP↔CSV Jaccard; promote still gated / CSV remains canonical.
- **Macro vintages (MVP)** — documented: fixed pub lags only; true ALFRED
  deferred. See [`docs/MACRO_VINTAGES.md`](MACRO_VINTAGES.md);
  `MACRO_USES_TRUE_VINTAGES = False`.
- **`roe_quality_sn` registered** after second holdout 2018–2025 (ADV costs,
  S&P filter): ROE +0.33, **ROE_SN +0.52**, VQ +0.20, VQ_SN +0.17, EY −0.23.
  Sector-neutral ROE dominates raw ROE here; VQ_SN still does not beat VQ.

## 0. Survivorship gap — leave disclosed

Prefer 2015+ windows. Needs Norgate/CRSP/Tiingo to close.

## Open / optional

1. True ALFRED macro vintages (ingest) when a nowcast / revision strategy needs them
2. FMP membership `--promote` only after rename/recycle review (notation already fixed)
3. PIT sector source (replace today's FMP sector labels used in `*_sn` factors)
