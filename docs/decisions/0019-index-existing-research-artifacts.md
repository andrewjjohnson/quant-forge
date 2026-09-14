# ADR 0019: Index existing research artifacts

- Status: Accepted
- Date: 2026-09-14
- Jira: [QF-9](https://frostfiredigital-37308542.atlassian.net/browse/QF-9)

## Context

QF-5/6, QF-7/29, QF-11/32/42 and QF-39/40 already own execution identities,
configuration snapshots, result schemas and local persistence. A common
experiment record must describe them without becoming another research engine.
QF-40 exclusively owns final-holdout exposure state.

## Decision

Add a separate `quantforge.experiments` consumer. Adapters read existing JSON
metadata and files; they accept no executable study factories. Reuse canonical
primitive serialization and SHA-256. Preserve upstream identities and capture a
namespaced study identity over their material provenance, with distinct explicit
execution IDs and content-addressed manifest IDs.

Embed a typed artifact index in version-1 manifests. Hash file bytes, reference
embedded records with JSON pointers, and represent relationships with typed edges.
Use local relative paths, strict reading, explicit unknown fields and atomic
no-clobber writes. Unknown schema versions require a future explicit migration.
Current holdout state comes only from `HoldoutLedger.state`; manifests do not
reserve or consume holdouts and never infer pristine state from a QF-8 plan.

## Consequences

Existing execution, result schemas, numeric semantics, historical backend
configurations and producer persistence remain unchanged. Backtests retain costs
and timing; predictions and features need no trading metrics. A manifest links
results to validation evidence without accessing future prices or recomputing
partitions, indicators, outcomes or metrics.

Callers must retain execution-time code/dependency metadata and keep the artifact
root/layout and original files. Missing historical fields stay unavailable.
New code-provenance capture rejects dirty working trees because a dirty flag
cannot fingerprint uncommitted source; callers must commit before execution.
Historical version-1 dirty flags remain readable with that reproducibility limit.
Hashes provide consistency, not signatures or independent provider authenticity.
Old manifests are historical holdout snapshots, so current reports must still
consult QF-40. QF-41 rendering and a universal replay system remain separate.

## Alternatives considered

Reconstructing study objects would risk triggering context/indicator validation
and execution. A universal result model would impose trading fields on prediction
research. Rewriting all producers for uniformity would broaden QF-9 and jeopardize
existing identities. A second holdout ledger would conflict with QF-40 authority.

## Validation

Offline producer fixtures followed by disabled execution/backend entry points;
material identity, round-trip, hash/tamper, schema, relationship, credential,
permanent holdout consumption and failed-evaluation tests. See
[`experiment-manifests.md`](../experiment-manifests.md).
