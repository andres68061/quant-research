# ADR 0015 — Permanent security identifiers (`qid`), separate from issuer identity

**Status:** Accepted
**Date:** 2026-08-15
**Supersedes:** nothing (new)

## Context

Every table in the platform is keyed by ticker. A ticker is not an identity; it
is a lease:

- it changes on rename (FB → META),
- it is reassigned to an unrelated company after a delisting (**ticker reuse**),
- it is punctuated differently by each vendor (BRK.B / BRK-B / BRK/B),
- it says nothing about which entity survives a merger.

Each of these corrupts data silently. The panel keeps one column, the numbers
stay plausible, and two companies get spliced into one price series. The trigger
for acting now is vendor risk: on a vendor switch, nothing in the repo would
notice that a 30-year history had been attached to the wrong company.

## Decision

Introduce a surrogate key we own, and separate two levels that are easy to
conflate:

| Level | Field | Source | Meaning |
|---|---|---|---|
| Security | `qid` (`Q0000042`) | minted by us | one tradable line, one price series |
| Issuer | `issuer_id` | SEC CIK | the company behind it |

Stored in `data/universe/security_master.parquet` plus an alias interval table
`data/universe/symbol_aliases.parquet` (`qid, vendor, vendor_symbol, valid_from,
valid_to`) — the same interval pattern as index membership (see DATA_MODEL.md).

Rules:

1. **`qid` is opaque and permanent.** It encodes nothing, so nothing about it can
   become wrong. Minted once, never re-minted, never re-used after a delisting.
2. **Ticker is an attribute over an interval**, not an identity.
3. **Match on normalized ticker first**, then on issuer only when unambiguous.
4. **Rebuilds preserve ids.** `build_security_master` takes the existing master
   and mints only for genuinely new securities.
5. **Two securities can never share a `qid`** — enforced by an assertion in the
   builder, not by convention.

## Alternatives rejected

**CIK as the identity.** Tried first, and wrong: CIK identifies an *issuer*. The
first build merged **208 securities across 85 issuers** — American Financial
Group's common stock and its three bond lines became one entity, as did GOOG and
GOOGL. CIK is now the anchor for *continuity* (same security, new ticker), used
only when the issuer names exactly one security on both sides of the rebuild.

**CUSIP / ISIN / SEDOL.** The industry answer, and licensed. Not available.

**FIGI (OpenFIGI).** Free and genuinely security-level — the right external
anchor. Deferred rather than rejected: it needs an API key and a rate-limited
bulk mapping pass. The schema has room for it; `issuer_id` is not load-bearing
for identity, so adding `figi` later does not disturb existing ids.

**Deriving the id from a hash of (ticker, exchange, first date).** Attractive
because it needs no state — and self-defeating, because any input that can be
restated changes the id. The point of a permanent id is that it survives
restatement.

**Deleting punctuation to normalize tickers.** Tried, and wrong on real data:
FMP writes preferred series as `-P`, so stripping punctuation merged `AL-PA`
(Air Lease preferred A) into `ALPA` (Alpha Healthcare Acquisition), and `C-PK`
(Citigroup preferred K) into `CPK` (Chesapeake Utilities). Separators are now
*unified* to `-`, never removed. The governing principle: **when in doubt,
split.** Two companies sharing one id is silent and permanent; two ids for one
company is visible and recoverable.

## Consequences

- 9,011 securities, 3,628 (40%) carrying an issuer id. The gap is mostly
  delisted names — SEC's public ticker file lists current filers only, so
  historical CIK coverage needs a different source.
- Panels stay keyed by ticker for now. `qid` is the join key available when a
  vendor migration or a ticker-reuse audit needs it; migrating the panels
  themselves is a separate, larger decision.
- Ticker reuse is now detectable: a ticker whose new span starts after the prior
  holder's last day mints a second `qid` (`id_source="reused_ticker"`).

## Not covered

Deliberately unresolved rather than guessed, because they need vendor data we do
not hold:

- **Share-class linkage** — GOOG/GOOGL share an issuer but nothing records that
  they are classes of one equity.
- **Merger successor chains** — no link from an acquired entity to its acquirer.
- **Spin-off parentage.**
- **Non-equity entities** — indices, macro series, and commodities are not yet
  minted. FRED `series_id` is already a permanent external id, so those follow
  the same pattern when needed.
