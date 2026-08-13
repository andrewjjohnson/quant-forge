# Completed-bar multi-timeframe context

QF-20 provides the shared, provider-neutral as-of alignment boundary in
`quantforge.data.multi_timeframe`. It consumes QF-13 timeframes, QF-14 dataset
family references, and canonical QF-15/QF-18/QF-19 bars. It does not calculate
indicators, call a provider, integrate prediction rules, or simulate trades.

## Public contract

`build_multi_timeframe_context()` accepts:

- one timezone-aware `as_of` decision timestamp;
- one primary or decision `Timeframe`;
- ordered `ContextTimeframeRequirement` declarations for contextual
  timeframes;
- `TimeframeBarSeries` inputs constructed from validated source or derived
  dataset artifacts;
- the default `ContextCompletionPolicy.COMPLETED_BARS_ONLY` policy.

Requirements may set a positive `maximum_age`. This limit is a caller-defined
freshness expectation measured from the latest visible bar end to `as_of`; it
does not fill, discard, or reinterpret the bar.

```python
from datetime import timedelta

from quantforge.data import (
    ContextTimeframeRequirement,
    TimeframeBarSeries,
    build_multi_timeframe_context,
)

five_minute_series = TimeframeBarSeries.from_source_dataset(
    source_dataset,
    family=dataset_family,
    cache=intraday_cache,
)
four_hour_series = TimeframeBarSeries.from_aggregated_intraday_dataset(
    four_hour_dataset,
    family=dataset_family,
)
daily_series = TimeframeBarSeries.from_aggregated_session_dataset(
    daily_dataset,
    family=dataset_family,
)
weekly_series = TimeframeBarSeries.from_aggregated_session_dataset(
    weekly_dataset,
    family=dataset_family,
)

context = build_multi_timeframe_context(
    as_of=decision_timestamp,
    primary_timeframe=five_minute,
    required_timeframes=(
        ContextTimeframeRequirement(weekly, maximum_age=timedelta(days=7)),
        ContextTimeframeRequirement(daily, maximum_age=timedelta(days=2)),
        ContextTimeframeRequirement(four_hour),
    ),
    series=(five_minute_series, four_hour_series, daily_series, weekly_series),
)
```

The primary timeframe appears first in `context.timeframes`. Contextual
requirements are sorted by complete timeframe configuration ID, so changing
the input tuple order cannot change the context or its identity. Bars inside a
series are likewise ordered by completion timestamp, start timestamp, and bar
ID. Duplicate boundaries and overlaps fail construction.

`TimeframeBarSeries` cannot be initialized from a bare family reference and bar
tuple. Its public constructors validate the concrete immutable dataset first
and derive the reference from that dataset's exact ID and embedded family. The
source-dataset constructor requires both the QF-14 family containing that
canonical source and its `IntradayMarketDataCache`. It reloads the exact
content-addressed artifact through the cache, which verifies the manifest
identity, canonical normalized and raw paths, raw payload checksums, normalized
content digest, quality report, request, and batch identity. The reloaded value
must equal the supplied dataset before its symbol, provider, feed, adjustment,
and source timeframe are checked against the family. This prevents a forged
in-memory dataset ID or coherently altered bars and metadata from authenticating
bars that the referenced immutable artifact does not contain.
When QF-18 and QF-19 artifacts carry their own partial lineage manifests, pass
the one context family containing all exact derived dataset IDs to each
constructor as shown above. The constructor verifies that the artifact ID and
timeframe occur in that family and that its canonical source, symbol, provider,
feed, adjustment basis, and source timeframe match the artifact's validated
embedded family. A composed context family must use the
`quantforge_context_artifact_set` version 1 aggregation policy and list every
embedded `artifact_family.manifest_id` in its
`artifact_family_manifest_ids` configuration. Each constructor verifies its
own exact manifest membership, so a fabricated common policy cannot relabel
unrelated validated artifacts as one family.

`TimeframeContext` and `MultiTimeframeContext` are likewise result-only models:
their constructors are not public. `build_multi_timeframe_context()` is the
only supported construction boundary, so callers cannot bypass artifact
validation by attaching arbitrary bars directly to an otherwise valid family
reference.

## Causal visibility

A bar is visible only when both conditions hold:

1. its completion state is not `DEVELOPING`;
2. its explicit `end_timestamp` is less than or equal to `as_of`.

The end timestamp is the availability timestamp even when a timeframe uses a
start label. Completed leading or terminal partial-duration intraday bars are
eligible after their actual end. A Tuesday intraday decision therefore cannot
see Tuesday's eventual session bar or the current week's eventual trading-week
bar. Supplying those future bars to the builder does not expose them.

The context retains only the causally visible in-memory bar tuple for each
declared timeframe. Additional bars after `as_of` within the same validated
dataset-family inputs cannot change the serialized context or `context_id`.
A newly persisted content-addressed dataset or family is a different provenance
input and intentionally changes the identity, even when its OHLCV prefix is
numerically equal. QF-20 records exact family and bar identities rather than
treating observations from distinct retrievals or source snapshots as
interchangeable.

## Availability and access

Every declared timeframe has a `TimeframeContext` with:

- the complete interval and timeframe configuration;
- its source or derived dataset ID when a series was supplied;
- `AVAILABLE`, `STALE`, or `MISSING` availability;
- latest completed bar end timestamp and completion state;
- exact age at the decision timestamp;
- the ordered visible bar IDs.

An empty series is `MISSING` while retaining its dataset ID. An omitted series
is `MISSING` with no dataset ID. No earlier value is forward-filled. A visible
bar older than a declared `maximum_age` is `STALE` and remains inspectable so a
consumer can apply its own explicit policy.

`metadata_for()` permits inspection of missing and stale state.
`bars_for()` and `latest_bar_for()` raise `UnavailableTimeframeError` for a
declared missing timeframe. All accessors raise `UndeclaredTimeframeError` when
the requested timeframe was not declared. These are data-domain errors below
the prediction and backtesting packages.

## Source and policy compatibility

All supplied series pass through `validate_source_consistency()`. Their family
ID and canonical source snapshot must agree. Feed scope, adjustment basis,
source timeframe, source session policy, source revision, and aggregation
policy are already bound into that family identity. Mixing consolidated and
IEX-only references therefore fails closed.

Every declared timeframe must also use the primary timeframe's exact exchange
session policy. Regular-hours and extended-hours series, calendars, or exchange
timezones cannot be combined merely because their symbols match. QF-20 ships no
external validation escape hatch and accepts only completed-only timeframe
configurations.

## Identity and serialization

`context_id` is the canonical sorted-JSON SHA-256 identity of:

- the UTC `as_of` timestamp;
- the primary timeframe;
- the sorted contextual requirements and freshness limits;
- the completed-bars-only policy;
- common-family validation evidence and family identity;
- each timeframe's dataset reference, availability, age, latest completion,
  and ordered visible bar IDs.

`serialize()` returns canonical JSON bytes. It references immutable bars by
their IDs instead of duplicating OHLCV payloads. Reproducing a context therefore
requires the referenced immutable datasets as well as the serialized context.

The schema version is `1`.

## Deliberate limitations

QF-20 does not implement developing higher-timeframe reconstruction, indicator
calculation, prediction-rule declarations, feature export, or execution. Those
remain sibling stories under QF-12. It also does not silently combine
provider-native higher-timeframe products with QuantForge-derived datasets.
