# ADR 0026: Retain Massive pages in logical acquisition snapshots

- Status: Accepted
- Date: 2026-09-24
- Jira: QF-54

## Context

Massive historical stock aggregates paginate long timestamp ranges. The existing
`IntradayFetchResult` expects ordered contiguous raw coverage envelopes, and bars
must lie in their referenced envelope. A wire page is a provider-selected result
partition; it may contain only extended-hours observations or no retained bars.
Treating every wire page as a canonical time chunk would invent coverage boundaries
from transport details and complicate empty-page handling.

## Decision

Keep the current provider-neutral API and cache schemas. One logical Massive
range uses one `IntradayRawSnapshot` whose existing JSON-compatible record list
contains all ordered pages. Each page preserves the safe URL, retrieval timestamp,
and response payload, with credentials removed/redacted. The snapshot's existing
SHA-256 binds the complete page sequence. Canonical bars point to this single
snapshot; its retrieval instant is the final page's retrieval instant. The raw
coverage envelope is exactly the original logical request.

Massive alone owns pagination, URL/authentication safety, row mapping, and XNYS/RTH
retention. Missing aggregates remain absent, including when no qualifying trade
occurs. Existing diagnostic coverage/cache behavior preserves sparse and empty
datasets; strict consumer policies determine whether coverage is acceptable.
There is no Massive-specific complete-session invariant. Canonical bar, coverage,
cache, aggregation, prediction-provenance, and bounded-view implementations stay
unchanged. The adapter accepts explicit raw or split-adjusted OHLCV while declaring
corporate-action event records unavailable.

The setup factory adds named Massive and Tiingo selection without changing the
existing injected-provider or credential-free `provider_name` service API.

## Consequences

Logical request identity remains independent of page boundaries. Acquisition
identity continues to bind raw layout and retrieval metadata, as existing
contracts require. There is no Massive cache namespace or parallel provenance
system. Failed pagination never yields a persistable success; restarting a failed
acquisition refetches the logical range. Existing completed datasets replay offline.

All pages are held until the result validates, so peak memory scales with the
requested history, as does the current immutable dataset serializer. Page-level
checkpoints and streaming persistence would require a separate generic contract
and are not introduced here. REST is sufficient for the initial SPY history path;
bulk-universe flat files remain a separate future story.

## Validation

Synthetic HTTP fixtures verify complete pagination, credential redaction, unsafe
URL rejection, failed-acquisition atomicity, exact boundaries, sessions/early
closes, both adjustment modes, missing-interval preservation, existing strict
consumer rejection, and immutable replay of complete, sparse, and empty datasets.
Massive-backed fixtures pass unmodified QF-51/QF-52 validation and the existing
1m-to-2m and daily aggregators. Existing provider regressions remain unchanged.
