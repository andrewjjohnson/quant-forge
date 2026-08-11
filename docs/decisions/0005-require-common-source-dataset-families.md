# ADR 0005: Require common-source dataset families

- Status: Accepted
- Date: 2026-08-10
- Jira: QF-14

## Context

Multi-timeframe research can silently combine candles that share a symbol but
not the same observations or semantics. Tiingo IEX five-minute volume is not
the same feed as consolidated volume. Provider-native EOD, intraday, and
resampled bars can also differ in session scope, anchoring, adjustment basis,
source revision, and aggregation rules. Treating those series as one coherent
context can create irreproducible features and misleading conclusions.

QF-13 supplies provider-neutral timeframe/session semantics, but it deliberately
does not define dataset lineage. Existing QF-3 schema-version-4 daily manifests
are immutable scientific artifacts and cannot be retrofitted without changing
their identities.

## Decision

Every ordinary multi-timeframe context must use datasets from one deterministic
`DatasetFamily`. The family is defined by one immutable canonical source
snapshot plus symbol, provider, feed scope, complete QF-13 source timeframe,
adjustment/corporate-action basis, and a snapshotted aggregation-policy
reference. The aggregation reference records name, version, and primitive
configuration but does not implement aggregation in QF-14.

Every source and derived dataset is recorded in a validated single-parent DAG.
Each entry points directly to the canonical source snapshot and records its own
timeframe, parent, and children. Missing parents, mismatched reverse links,
duplicates, disconnected graphs, and cycles fail construction.

The deterministic `family_id` covers canonical source compatibility and is
stable as new derived datasets are recorded. A separate `manifest_id` covers
the exact sorted lineage graph. Dataset references carried into a study include
the family ID and canonical source snapshot. The default source-consistency
validator requires both to match across every reference.

Consolidated, single-venue, and provider-defined feed coverage are explicit
provider-neutral values. IEX-only and consolidated data therefore receive
different family IDs even when symbol and provider match. Provider-specific
response or SDK types do not cross this boundary.

An identity-bearing `ExternalBarValidationPolicy` protocol is reserved for a
future explicit validation implementation. QF-14 ships no concrete policy.
Mixed references fail closed unless a caller supplies such a policy and it
validates the fixed reference tuple without changing its configuration.

Family manifests are new, embeddable artifacts. They do not modify legacy QF-3
cache manifests or dataset identities.

## Consequences

Multi-timeframe consumers can prove that every derived interval shares one
source feed, revision, session policy, adjustment basis, and aggregation policy.
An independently reconstructed equivalent family receives the same ID, while a
material policy difference receives another ID. Adding a derived dataset keeps
existing members compatible but changes the exact manifest identity.

The contract is more verbose than comparing provider and interval strings, and
callers must retain a family manifest alongside future derived artifacts.
Single-parent lineage excludes transformations that genuinely combine multiple
source datasets; those require later explicit semantics rather than an
accidental merge.

Existing daily research remains byte-for-byte compatible. Intraday retrieval,
aggregation, and multi-timeframe alignment remain later QF-12 work.

## Alternatives considered

- **Match only symbol and provider.** Rejected because one provider can expose
  materially different venue, interval, adjustment, and session products.
- **Use the canonical source snapshot ID alone.** Rejected because policy and
  feed differences must participate in compatibility and audit output.
- **Include the lineage graph in `family_id`.** Rejected because recording a new
  derived dataset would make older members appear incompatible despite an
  unchanged canonical source. The graph instead has a separate `manifest_id`.
- **Retrofit QF-3 manifests.** Rejected because it would change immutable daily
  dataset identities and prematurely mix QF-14 with ingestion scope.
- **Permit mixed sources by default.** Rejected because validation is not
  transitive from matching labels. A future explicit policy must demonstrate
  and record compatibility.

## Validation

Offline unit and invariant-style parameterized tests cover complete valid
families, order-independent deterministic serialization, identity sensitivity
to provider/feed/source interval/session/adjustment/aggregation policies,
family-versus-manifest identity, canonical-source pointers, parent/child
agreement, IEX/consolidated rejection, source-interval rejection, cycles, and
the identity-bearing external-policy extension point.
