# ADR 0027: Normalize historical-window evidence

- Status: Accepted
- Jira: [QF-55](https://frostfiredigital-37308542.atlassian.net/browse/QF-55)

## Context

QF-42 requires identical market metadata and study configuration in every
decision. QF-45 measured 3.26 MB of repeated QF-52 evidence per decision across
5,760 decisions. Embedded serialization therefore repeats roughly 18.8 GB of
immutable evidence before decision-specific context and outcomes.

## Decision

Add explicit window representation schema `"2"` while preserving v1 execution,
artifacts, identities, and validation. Store the complete immutable scientific
window identity once, with a versioned content hash binding its source, exact
schedule, bounded ancestry, configuration and environment. Normalized decisions
reference this scope and retain all context, signal, row, diagnostic and original
QF-11 identity evidence. Do not pool context snapshots that can differ or record
rejected sources.

Use finalized canonical JSONL: header, shared evidence, ordered decisions. The
window result hash remains ordinary canonical JSON/SHA-256 over normalized
ordered records; stream its array encoding without a collection allocation.
Verify original QF-11 identities by copying their common SHA-256 prefix state.
Reuse semantic decision validation directly, without expanding v1 objects.

Expose a common version-aware reader. Structural integrity and scientific
validation are explicit: reading a header does not certify its unread body.
QF-52 subset validation still requires independent canonical-parent metadata.
Unknown versions and incompatible evidence fail closed. No persisted v1 file
is rewritten and no provider authenticity claim is added.

## Consequences

Shared evidence occupies O(1) storage per correct window scope. Exact membership
remains O(N), and readers retain only one decision payload beyond the shared
scope and schedule. Legacy documents retain their original materialization
cost. Conversion from retained v1 results is transitional representation work;
incremental execution/persistence is QF-56. Downstream consumers are QF-57.

Numerical prediction/outcome semantics, temporal boundaries, and holdout behavior
are unchanged. Tests prove strict versions, corruption rejection, identity
equivalence, genuine bounded-input ancestry, all intraday outcome families,
legacy regressions and the 5,760-decision serialized growth shape.

See [the compact contract](../compact-prediction-windows.md) for the exact fields,
identities, API, limits and measured fixture sizes.
