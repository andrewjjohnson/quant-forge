# ADR 0003: Use raw prices with explicit corporate actions

- Status: Accepted
- Date: 2026-08-05
- Jira: Pending — an approved QF key is required before commit

## Context

Multi-year equity backtests cross dividends and sometimes splits. Combining
fully adjusted execution prices with explicit dividend cash double counts total
return. Using raw prices without actions understates returns and breaks share
units across splits. A daily-bar engine also needs deterministic entitlement and
event ordering where intraday and payment-date facts are unavailable.

Tiingo EOD returns raw and adjusted OHLCV, `divCash`, and `splitFactor` together.
Tiingo identifies `divCash` on the ex-dividend date and defines `splitFactor` as
shares after divided by shares before.

## Decision

This ADR extends ADR 0001's chronological next-open state machine and supersedes
only its temporary deferral of corporate-action-bearing execution.

QF-3 preserves the complete Tiingo row in immutable raw JSON but normalizes only
raw OHLCV for execution. Nonzero dividends and non-unit splits become typed,
immutable actions bound to a dataset and action snapshot. Adjusted Tiingo fields
are never execution or mark inputs.

QF-5 requires schema-version-4 unadjusted data with complete action provenance.
Dividend treatment is explicit and independent of the mandatory split policy:
`PRICE_RETURN_ONLY` preserves and discloses dividends without crediting cash,
`CASH_DIVIDENDS` credits entitled cash, and `REJECT_IF_DIVIDENDS` fails closed.
The strict policy is the compatibility default; maintained research examples
select their economic basis explicitly.

Each session uses:

1. prior-close shares determine dividend entitlement;
2. effective splits multiply existing shares before the open;
3. eligible next-open orders execute;
4. dividends are credited, disclosed as ignored, or rejected according to policy;
5. remaining shares mark at the raw close.

Aggregate split cost basis and cash are unchanged; average per-share basis moves
inversely. Fractional split results are rejected until cash-in-lieu exists.
Cash dividends remain separate from price-trade P&L, with attributed income and
total economic trade P&L exposed separately. Price-only estimates are
informational and never affect economics. Strategy and benchmark share the
selected dividend policy and return-basis label.

## Consequences

The approach prevents silent dividend omission and double counting while
retaining provider and event traceability. Provider/action revisions and policy
changes alter deterministic identities. Results include action ledgers,
credited/ignored disclosures, explicit `price_return` or
`total_return_with_cash_dividends` labels, and a matched benchmark. Price-only
mode is suitable for early price-signal research but understates long-period
wealth relative to total-return accounting.

Raw OHLCV remains authoritative for execution and portfolio marks. QF-5 derives
a non-persisted causal strategy-feature view that multiplies OHLC and divides
volume by the cumulative split factor beginning on each effective session. This
prevents artificial indicator crossovers at splits without using a later split
to revise earlier observations; the transformation is fixed and versioned in
the separate split policy.

The daily model credits on ex-date rather than the real payment date. It does
not support DRIPs, tax/withholding, cash-in-lieu, fractional positions, or
intraday action sequencing. Fully adjusted execution datasets remain rejected.

## Alternatives considered

- Fully adjusted OHLCV plus explicit dividends: rejected because dividends
  would be counted twice.
- Adjusted close with raw open/high/low: rejected as an inconsistent basis.
- Raw OHLCV with unlabeled dividend omission: rejected because the economic
  basis would be ambiguous. Explicit price-only mode is accepted with preserved
  events, estimated exclusions, warnings, and unchanged mandatory split handling.
- Silent fractional-share rounding: rejected because it destroys value and
  reproducibility without a defensible cash-in-lieu price.

## Validation

Offline fixtures cover Tiingo mapping, raw-field preservation, dividend and
split parsing, immutable reload, entitlement at buy/sell boundaries,
exactly-once cash, split basis/equity continuity, benchmark parity, fractional
rejection, all three dividend policies, price-only non-effects, deterministic
replay/export, and identity changes. A separately
opted-in live SPY test covers provider ingestion through QF-5 export.
