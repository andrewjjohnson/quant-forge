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
requires the existing intraday cache, reloads and compares the canonical source,
validates existing QF-19 session artifacts against that source, and persists them
through the existing QF-3 cache. It preserves original adjustment
semantics, explicit unavailable events, source/family/feed/session identity, and
full source evidence. Context and generic outcome dispatch require matching
provenance; their original price-basis equality checks remain intact.

Provenance version 5 embeds immutable family and intraday source manifests so
observational cache and experiment readers can recompute the source dataset,
request, family, and exact-graph identities. Request and raw-snapshot IDs must
match the source manifest; complete family DAG validation reuses the domain
constructor, including members unused by a study.
The declared adjustment basis must match the committed canonical source, and
outcome manifest IDs must match the retained graph. Version-1/2/3/4 intraday projections
must be regenerated from canonical caches; no missing evidence is inferred.

Retain the original QF-19 session manifest and normalized session bars in immutable
session evidence. The QF-19 and QF-3 byte formats differ, so their digests cannot be
compared directly. Readers rebuild the typed session artifact and verify its
original identity before comparing the canonical projected OHLCV digest with the
recorded dataset fingerprint. This preserves QF-19 schemas. Projected decimals use
exact canonical strings so
the same values have a reproducible fingerprint. Retaining the session evidence
increases manifest size but permits standalone QF-9 checks without source I/O.

Retain source bar observations with shared timeframe/provenance templates. This
lossless representation reconstructs the canonical intraday batch and must match
both its existing batch ID and data digest. Each session's ordered constituent
IDs must equal those of the authenticated source session. An ID list alone
cannot prove membership under the existing full-record batch hash. This adds
intraday observations to manifests, but does not change the source/QF-19 schemas
or rerun provider normalization. Reconstruct the typed request, bars, and batch
with the ingestion decoder, verifying canonical serialization as well as hashes.
Verify each retained session's OHLCV from its ordered source bars with the same
reduction used by QF-19. Volume summation is exact and independent of the ambient
Decimal context; existing exact totals retain their identities, while previously
rounded session volumes require regeneration. This bounded derivation check
requires no source I/O or research execution and adds no provenance fields.

Namespace projection IDs as `intraday-projection-<sha256>`, keeping their existing
256-bit content digest. The namespace and adapter version independently require
intraday provenance, even if the declared event policy changes. Cache readers
also check the immutable raw extract's origin. Removing the namespace changes
the dataset reference; standalone checks prove consistency with that reference,
not external authenticity of an entirely replaced artifact. Daily IDs stay bare
SHA-256 values and remain unchanged.

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
