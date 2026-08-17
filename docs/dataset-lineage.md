# Dataset-family lineage and source consistency

QF-14 defines the provider-neutral provenance contract that groups canonical
source bars and QuantForge-derived timeframes into one auditable dataset family.
It does not itself retrieve or aggregate bars or align multi-timeframe contexts.
QF-18 consumes this contract for derived intraday datasets, and QF-19 consumes
it for exchange-session daily and exchange-weekly datasets.

The public implementation is `quantforge.data.lineage`, re-exported from
`quantforge.data`.

## Common-source rule

Every ordinary multi-timeframe study must use datasets from one deterministic
family. The family's higher-timeframe datasets all trace through parent links to
one immutable canonical source snapshot.

This is internally consistent:

```text
Tiingo consolidated 5m source snapshot
    -> QuantForge 15m
    -> QuantForge 1h
    -> QuantForge 4h
    -> QuantForge daily
    -> QuantForge weekly
```

These provider-native datasets are not one family merely because their symbol
or provider matches:

```text
Tiingo IEX 5m
+ Tiingo EOD daily
+ provider-native 4h
```

An IEX-only feed is structurally `single_venue` with market center `IEX`; a
consolidated feed is structurally `consolidated`. Their family IDs therefore
differ. Provider-native bars from another interval have a different canonical
source snapshot and source interval, so they also fail the common-family check.

`validate_source_consistency()` accepts `DatasetFamilyReference` records created
by `DatasetFamily.reference()`. Without an external policy, every reference must
carry both the same family ID and the same canonical source snapshot ID. A mixed
set raises `MixedDatasetFamilyError`; callers must not catch that error and
silently continue.

QF-27 also carries the family's typed feed scope on each compact reference.
Timeframe-bound volume indicators compare their declared feed scope with that
reference and fail binding on a mismatch, so a consolidated series cannot be
labeled as IEX-only (or the reverse) merely by constructing different indicator
parameters.

## Canonical source contract

One `DatasetFamily` records:

- canonical symbol;
- provider name as provenance, without retaining a provider response model;
- typed feed coverage and optional venue/provider-scope name;
- canonical source snapshot ID;
- the QF-13 source timeframe, including interval, session scope, exchange
  calendar/timezone, anchoring, labels, and developing-bar policy;
- OHLC, volume, adjustment, adjusted-field, and corporate-action basis;
- a snapshotted aggregation-policy name, version, and primitive configuration;
- every canonical and derived dataset in the lineage graph.

`AggregationPolicy` is an identity-bearing reference only. QF-14 does not
define OHLCV aggregation behavior. A later aggregation implementation owns the
policy's meaning and must supply its complete primitive configuration.

`FeedScope` is provider-neutral. The canonical coverage categories are
`consolidated`, `single_venue`, `provider_defined`, and explicitly `unknown`.
QF-15 adds `FeedScope.iex_only()` as the canonical single-venue IEX shorthand
and `FeedScope.unknown()` for observations whose coverage has not been
established. Unknown never aliases consolidated coverage. A provider adapter
may map its vocabulary into these values, but downstream lineage code never
accepts provider SDK or response types. See
[`intraday-market-data.md`](intraday-market-data.md).

`AdjustmentBasis` follows the existing QF-3 invariant: unadjusted data must use
`raw_provider` for both OHLC and volume, while every adjusted mode uses
`split_adjusted` for both. Contradictory combinations fail construction rather
than becoming trusted family provenance.

## Lineage invariants

Each `DatasetLineage` entry stores its dataset ID, complete timeframe, canonical
source snapshot ID, one optional parent ID, and all child IDs. Construction
fails unless:

- exactly one entry is the canonical source snapshot and it has no parent;
- every derived entry names a parent contained in the family;
- every entry points directly to the family canonical source snapshot;
- following any derived entry's parents terminates at that source;
- the graph has no cycles;
- dataset IDs and child IDs are unique;
- child lists exactly match the reverse of parent links.

Dataset entries and child IDs are sorted before serialization. Caller input
order therefore cannot alter manifest bytes or identities.

## Identities and manifests

Dataset-family schema version 1 has two related SHA-256 identities:

- `family_id` identifies the canonical source and the policies that define
  compatibility. It includes the symbol, provider, feed scope, source snapshot,
  complete source timeframe/session policy, adjustment/corporate-action basis,
  aggregation policy, and common-source rule.
- `manifest_id` identifies one exact family manifest, including the complete
  current lineage graph and `family_id`.

Adding another derived dataset changes `manifest_id` but preserves `family_id`.
That makes datasets produced at different times from the same immutable source
remain compatible. Changing any source or semantic policy changes `family_id`.

`DatasetFamily.to_manifest()` returns the complete primitive manifest, including
both IDs and every lineage entry. `serialize_manifest()` produces canonical
sorted JSON bytes. Equivalent families built independently, including with
different mapping or tuple input order, produce equal IDs and bytes.

Family manifests are new artifacts for QF-14 and future QF-12 consumers. They do
not rewrite QF-3 schema-version-4 daily cache manifests or their dataset IDs.
Future derived-dataset, multi-timeframe-study, and report manifests must embed
the family manifest or an immutable link to it plus the applicable
`DatasetFamilyReference` and `SourceConsistencyValidation`.

## External-bar validation extension point

QF-14 ships no external-bar validation implementation. The
`ExternalBarValidationPolicy` protocol reserves an explicit extension point for
a later ticket. A future policy must:

- expose a complete primitive configuration and matching deterministic ID;
- validate the fixed tuple of dataset references or reject it;
- remain unchanged while validation runs;
- cause the returned `SourceConsistencyValidation` to record the external
  policy ID.

Supplying an explicit policy is the only path by which
`validate_source_consistency()` can accept mixed family references. Merely
matching symbols, provider names, or display interval labels is never enough.

## Deliberate limitations

QF-14 does not provide:

- intraday provider retrieval;
- bar aggregation or derived dataset construction;
- source-bar completeness rules beyond the opaque aggregation policy reference;
- multi-timeframe as-of alignment;
- a concrete external-bar validator;
- migration of legacy QF-3 cache manifests.

QF-18 supplies intraday-to-larger-intraday aggregation and QF-19 supplies
intraday-to-daily/weekly aggregation by consuming these contracts. Each QF-19
artifact records only the canonical source and its selected daily or weekly
child. Multi-timeframe as-of alignment and external-bar validation remain later
QF-12 work and must not be implemented implicitly here.
