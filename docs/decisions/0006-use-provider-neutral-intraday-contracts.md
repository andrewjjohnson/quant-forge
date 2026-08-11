# ADR 0006: Use provider-neutral intraday request and bar contracts

- Status: Accepted
- Date: 2026-08-10
- Jira: QF-15

## Context

QF-13 defines canonical timeframe and exchange-session semantics, and QF-14
defines common-source dataset-family provenance. Historical intraday ingestion
needs a typed adapter boundary that consumes both without leaking vendor SDK
objects or altering QF-3's established daily schema and immutable identities.

A display string such as `5m` does not identify an intraday request: exchange
calendar/timezone, regular or extended hours, anchor, feed coverage,
adjustment/corporate-action basis, and timestamp range all affect the meaning
and reproducibility of the observations. Bars also need explicit end and
completion state so a start label cannot expose a future close or volume.

## Decision

QuantForge uses immutable `IntradayBarRequest`, `IntradayBar`,
`IntradayBarBatch`, `IntradayBarProvenance`, and
`IntradayProviderCapabilities` records in the data domain. Requests use
timezone-aware half-open boundaries normalized to UTC and embed the complete
QF-13 intraday `Timeframe`, QF-14 `FeedScope`, and QF-14 `AdjustmentBasis`.

Canonical bars retain explicit start/end timestamps, timeframe, completion,
session identifier, exact Decimal OHLCV, and provider-neutral provenance. They
delegate boundary, anchor, duration, partial-bar, and completion validation to
QF-13's `IntradayBarWindow`. Their retrieval timestamp cannot precede the
observed end. Provider adapters expose declared capabilities and return only a
request-bound canonical batch; its construction rejects out-of-order bars,
duplicate bar keys, and request/provenance mismatches. Vendor response types
remain inside the adapter/raw-artifact boundary.

Feed coverage explicitly distinguishes consolidated, IEX-only single-venue,
provider-defined, and unknown observations. Unknown is not treated as
consolidated. Provider capability validation raises dimension-specific domain
errors for unsupported intervals, feeds, session scopes, and date ranges.

Requests, bars, and capability declarations have canonical sorted-JSON
serialization and SHA-256 identities. Request identities bind every material
timestamp, timeframe/session, feed, and adjustment policy. Bar identities also
bind exact OHLCV, completion, request, snapshot, and adapter provenance.

QF-3 `DailyBar`, schema version 4, daily provider protocol, service, cache,
artifacts, and identities remain unchanged.

## Consequences

Downstream ingestion can switch providers without importing vendor models, and
unsupported requests fail before a network call. Timezone-naive bars and
requests cannot enter the canonical boundary. Completed partial bars remain
distinct from developing bars, preserving causal availability.

The contracts are intentionally verbose, but equivalent instants and
order-independent capability declarations serialize identically. A material
feed/session/adjustment change creates a different identity instead of silently
reusing observations.

QF-15 supplies no HTTP adapter, cache, session-gap validator, aggregation, or
multi-timeframe alignment. Those later stories must preserve raw responses and
bind these identities into their own immutable artifacts.

## Alternatives considered

- **Extend `DailyBar` with optional timestamps.** Rejected because it would
  weaken daily exchange-session semantics and risk changing established QF-3
  artifacts.
- **Represent intervals with vendor strings.** Rejected because those omit
  exchange sessions, anchors, partial/developing completion, and deterministic
  QF-13 identity.
- **Use a new feed enum unrelated to QF-14.** Rejected because request and
  dataset-family provenance must share one feed vocabulary.
- **Return provider responses from the intraday protocol.** Rejected because it
  would couple normalization and downstream code to an adapter's transport or
  SDK model.
- **Infer unknown feeds as consolidated.** Rejected because that can silently
  combine materially different OHLCV and volume observations.

## Validation

Offline tests cover typed serialization and identity sensitivity, timezone
normalization and naive rejection, start/end ordering, full and partial bar
duration, collection ordering and duplicate rejection, OHLCV invariants, feed
distinctions, capability ordering and range metadata, each unsupported-
capability exception, canonical adapter return types, and unchanged daily
schema/protocol behavior. The full repository suite protects existing QF-3
through QF-14 consumers.
