# ADR 0037: Reuse authenticated canonical preparation within a load session

- Status: Accepted
- Date: 2026-09-29
- Jira: [QF-65](https://frostfiredigital-37308542.atlassian.net/browse/QF-65)

## Context

After QF-62–QF-64, QF-45 startup (cached inputs to the first scheduled decision)
took 354.6 s, and reading plus parsing the cache took under 1 s. The time went to
repeating immutable work. The 96,960-bar canonical source was fully re-decoded
from the cache twice. It was re-authenticated inside each aggregation, and its
retained QF-51 evidence was rebuilt six times through `validate_market_dataset`.
Daily bars were derived twice. Every bar construction resolved its exchange
session through the calendar (1.07 M times). Every bar serialization re-hashed
its timeframe (6.3 M `bar_id` evaluations). Reading and verifying persisted bytes
was not the problem.

## Decision

Add an explicit, execution-local `CanonicalPreparation`
(`data.prepared_canonical`). The outermost `canonical_preparation()` scope owns
it and clears it on exit, including on failure. Inner scopes join it. Generic
consumers consult the active preparation and otherwise run the unchanged
reference path. The QF-45 script owns one session for loading, configuration
checks and research. `load_inputs` and `PredictionEvaluator.preparation_scope`
join that session or open their own.

- **Trust boundary kept.** `IntradayMarketDataCache.load` still performs every
  check on first read: bytes, schema, request, raw snapshots, manifest identity,
  per-bar identities, batch identity, coverage and fetch bindings. Each identity
  is now computed once. An identical later load from the same root re-reads
  and re-hashes every artifact file. Only if every file matches does it reuse
  the decoded, validated dataset.
- **Content-proven reuse.** Aggregation and cache-binding consumers reuse the
  authenticated batch and bar identities only for a presented source whose
  request and metadata are equal by value. Every bar field and every nested
  record field must also equal its pristine authenticated value (the QF-63
  integrity technique). Otherwise the reference validation runs and fails closed.
- **Derived artifacts.** Each QF-18/QF-19 artifact passes its full `validate()`
  once per session. A derivation of an authenticated source is recorded as a
  relationship, so `prediction_dataset_from_intraday` does not re-derive the
  same daily bars. A mismatch triggers independent re-derivation.
- **QF-3 prediction inputs.** An intraday-derived dataset's validation verdict
  is reused only for byte-identical content. The key binds every metadata and
  provenance field, the SHA-256 of every evidence snapshot's canonical bytes and
  the canonical bar serialization. The first validation of a new canonical input
  still rebuilds its QF-51 source evidence independently.
- **Calendar and identity memos.** Exchange sessions and timeframe configuration
  identities are resolved once per complete frozen value inside the scope, and
  never keyed by object identity.

## Consequences

QF-45 startup falls from a mean of 355.8 s to 43.1 s (-87.9%). All scientific identities are
unchanged, including source, batch, 2m, daily, family, prediction input, plan,
universe and QF-52 view identities. So are every cache format, persisted
schema, QF-63 feature and QF-64 receipt. Invalid, changed, mutated or rebound
inputs fall back to independent validation and fail as before.

Prepared state is never persisted or process-global. It is not needed to
validate artifacts after a restart: a new process authenticates again.
Offline consumers (QF-9/QF-40/QF-41/QF-57 in their own processes) never see a
preparation. A session retains only its own authenticated datasets' identity
tuples and integrity snapshots. Datasets that research already holds are not
duplicated.

A binary or columnar persisted format is not introduced. Measured I/O and
parsing remain below 1 s, and the rest of startup is now dominated by the one
trust-boundary decode and the one independent evidence rebuild. See
[prepared canonical loading](../prepared-canonical-loading.md).
