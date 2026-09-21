# ADR 0023: Preserve intraday prediction-input provenance

- Status: Accepted
- Jira: QF-51

## Context

QF-3's session dataset accepts the daily event policy, while canonical intraday
sources can explicitly supply no corporate-action events. QF-11 and future
label checks compare complete adjustment bases. The real QF-45 preflight exposed
this mismatch despite valid acquisition and aggregation.

## Decision

Keep event availability distinct from price compatibility. Retain the existing
intraday unavailable-policy value and introduce a typed, optional, versioned
intraday source reference on the existing session dataset. A generic adapter
validates existing QF-19 session artifacts against their canonical source and
persists them through the existing QF-3 cache. It preserves original adjustment
semantics, explicit unavailable events, source/family/feed/session identity, and
full source evidence. Context and generic outcome dispatch require matching
provenance; their original price-basis equality checks remain intact.

Omit the additive reference for legacy daily serialization so existing daily
identities remain stable. New provenance participates in existing identities
and resume validation. QF-9 validates it observationally in the existing format.

## Consequences

QF-11 retains its session carrier and QF-46/QF-48 retain exact intraday temporal
semantics. This avoids a broad dataset union or prediction-engine rewrite.
The projection intentionally supports only completed one-session aggregates
with explicitly unavailable events; it does not reconstruct actions or make
unknown adjustment semantics comparable. A multi-timeframe study supplies its
existing composed family explicitly. No provider-specific prediction behavior
or QF-45 study code is introduced.

See [the contract](../intraday-prediction-provenance.md).
