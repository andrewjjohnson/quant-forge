# ADR 0007: Cache intraday source snapshots separately

- Status: Accepted
- Date: 2026-08-11
- Jira: QF-16

## Context

QF-15 returns provider-neutral canonical batches, while QF-16 must also retain
every Tiingo transport response, support bounded chunking, reload identical
requests without credentials, and preserve QF-3's established daily schema
version 4. A long intraday request can be backed by several raw responses, and
bars near inclusive provider date boundaries can appear in adjacent responses.

The canonical bar provenance already requires a source snapshot ID. That ID
must refer to immutable raw content rather than a mutable latest-response file
or provider object.

## Decision

QuantForge uses a separate intraday dataset schema and filesystem namespace.
Every bounded raw chunk is canonicalized, content-addressed, and written once.
Each normalized bar names the exact raw chunk hash from which it was derived.
The normalized QF-15 batch and a manifest are content-addressed as one dataset;
only the provider/request pointer is mutable and it advances atomically on an
explicit refresh.

The manifest binds the complete QF-15 request, feed/session/source-interval
semantics, adapter and capability versions, ordered chunk ranges and retrieval
timestamps, all raw hashes, and the normalized batch/hash. Chunk membership is
half-open by bar start, so overlapping provider date boundaries cannot create
two canonical observations. Remaining duplicates fail closed.

Cache lookup requires only provider name and the complete request identity.
Provider construction and credentials occur only after a miss or explicit
refresh.

## Consequences

Repeated studies can run offline, prior provider revisions remain reloadable,
and one canonical bar can be traced to one exact raw response. Tiingo response
types and credentials do not cross into strategies or predictions. Daily QF-3
artifacts and identities remain byte-for-byte compatible.

The design stores one raw artifact per chunk and therefore uses more small
files than a mutable aggregate download. A dataset-ID load also requires the
original typed request so cached primitives can be verified against the QF-15
contract without adding a second timeframe deserializer.

## Alternatives considered

- **Extend QF-3 daily schema version 4.** Rejected because timestamped intraday
  bars and multi-chunk raw provenance have different semantics and would change
  established daily identities.
- **Store only the merged canonical batch.** Rejected because provider records,
  chunk boundaries, and normalization evidence would be lost.
- **Let the Tiingo adapter write directly to a cache.** Rejected because it
  would couple transport to storage and make the QF-15 canonical method harder
  to test independently.
- **Fall back from consolidated to IEX automatically.** Rejected because feed
  coverage is a material request and dataset-family policy, not a transport
  retry.
- **Overwrite a latest snapshot.** Rejected because corrections would silently
  change reproducibility.

## Validation

Offline fixture tests cover deterministic chunk ordering, overlapping boundary
de-duplication, exact content hashes, cache-only replay without a provider or
key, immutable refresh, credential absence, feed/endpoint distinctions,
1-minute and 5-minute support, malformed responses, duplicate rejection, and
downstream provider independence. A fixed SPY integration test is skipped
unless the explicit live flag and API key are present.
