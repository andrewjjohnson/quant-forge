# Timestamp prediction membership (QF-48)

QF-48 connects the existing QF-42 schedule, QF-46 temporal contracts, QF-8
partition helpers, and QF-39 prediction adapter. It adds no scheduler, validation
engine, outcome arithmetic, selection policy, or metrics.

## Explicit observation source

Legacy `ValidationPlan` construction retains its existing session behavior and
serialized fields. Intraday feature inputs alone do not activate timestamp
membership. To opt in, capture `PredictionMembershipSource.capture(schedule,
primary_series)` and pass it as `ValidationPlan(prediction_membership=source)`.
The plan exposes the existing typed `BoundaryAxis` through `membership_axis`;
`axis` continues to describe its boundary coordinates. Existing QF-8 timestamp
boundaries for daily backtests retain their session-close mapping.

The immutable source binds the exact QF-42 `PredictionDecisionSchedule`, its
identity, canonical `DatasetFamilyReference`, and complete family manifest ID.
Timeframe, calendar, timezone, session scope, completion policy, feed, adjustment
basis, and source lineage already belong to those contracts. It stores no prices
and introduces no additional market-data fingerprint. Capture requires a
completed primary bar for every scheduled timestamp with the same session label.
Missing scheduled observations fail closed; observed bars never redefine or
shorten the schedule. Each scheduled instant is a distinct observation even when
many decisions share an exchange session.

Capture also builds a read-only timestamp-to-session index, so resolving all
retained decisions takes linear total lookup work. This derived index does not
change serialized evidence or membership identity and is rebuilt on copy or
deserialization through Python's object protocol.

The schedule must cover the study windows and their primary warm-up. Plan
construction requires every window endpoint in that schedule and enough preceding
scheduled observations for each window's declared primary warm-up, including the
final holdout. Extra artifact history outside the captured schedule cannot satisfy
that requirement. Source prefixes are accepted only
when they are exact prefixes, and both endpoints of the selected window must be
present. Appending unrelated future bars under the same immutable source
reference does not change captured historical membership. A different source
snapshot gets its own identity.

## Boundaries, purge, and embargo

Development, optional selection, test, and final holdout continue to use closed,
inclusive `ValidationInterval` boundaries. A decision exactly at test or holdout
start belongs to that window. An instant immediately before the start does not.
Aware timestamps are normalized to UTC; session labels come from QF-42 rather
than a timestamp's calendar date.

Timestamp outcomes declare QF-46 `OutcomeTemporalConfiguration.elapsed_duration`.
Their `required_future_duration` delegates to that configuration's conservative
reach. `OutcomeProvenance.capture_timestamp` verifies the delegation when a typed
temporal configuration is present. QF-46 currently includes one full observation
interval beyond the nominal horizon for alignment. Neither same-session overflow
nor known missing data shortens the declared reach.

Use `temporal.future_temporal_reach` for `PurgePolicy.label_horizon` and
`TemporalOffset.duration(...)` for embargo. QF-8's existing rule purges an earlier
observation when:

```text
decision timestamp + conservative label reach + embargo >= protected start
```

Development protects selection (or test when selection is absent); selection
protects test; each test protects the next test or final holdout. Equality is
purged. No elapsed duration is converted to a session count, and timestamp
wrappers omit `required_future_sessions` entirely.

## Prediction execution and replay

QF-39 passes QF-8's retained timestamp tuple into QF-42 and verifies exact equality
with the restricted schedule. Minimum sample sizes count decisions. Frozen
selections and OOS partitions persist both `evaluation_timestamps` and their
session labels, plus the complete membership source. Selection and backtest
semantics remain unchanged.

A `PredictionStudy` using elapsed outcomes supplies a canonical `outcome_source`
and a `TimestampStudyOutcomeLabeler`. Its `label_request(dataset, request,
source=..., resolution=...)` receives QF-46's original exact anchor after causal
predictions have been fixed. The runner supplies only the last anchor-side source bar and
same-session future bars through the declared conservative reach. The source
identity is bound into study/grid identities. QF-46 resolution and availability
metadata are persisted on each generic outcome record; unavailable outcomes
require an explicit row, rather than a zero or a silently omitted prediction.
Availability resolution uses the complete source artifact to distinguish a missing
required observation from the actual end of the dataset. The callback receives
that same typed `OutcomeResolution` and must use it for availability and unavailable
reasons; resolving its bounded source alone loses coverage evidence. The resolution
contains either no price bar or the expected endpoint already inside the bounded
slice, so it exposes no later prices. Dispatch validates request/source agreement,
and the runner rejects callback mutation of the detached resolution.
Concrete labelers/evaluators remain responsible for their typed value schemas.

For this path the separate QF-3 dataset supplies established identity and
adjustment metadata; its bounded projection contains only daily bars completed
before the first permitted decision. It does not determine intraday membership
or supply same-day future prices to a rule. QF-20/QF-28 feature context remains
bounded by each original decision and its timeframe-specific warm-up. A prior
completed daily observation is currently required by the existing QF-11 dataset
contract.

Elapsed horizons do not waive strategy warm-up. Direct studies still validate the
signal's dataset observation count. When a validated primary context is supplied,
its completed bars must satisfy the declared warm-up instead of the separate
daily metadata projection. A signal absent from that projection is rejected unless
validated primary context supplies the count. Contextual QF-7 outcome replay keeps
the original context provider and requirements so it can certify that history
again; the elapsed horizon or captured timestamp alone is not warm-up evidence.

QF-7 `SignalFeatureCandidate.decision_timestamp` captures the original context
instant before replay. Fixed replay preserves it verbatim, keys outcomes by that
instant, and distinguishes multiple candidates in one session. Missing exact
anchors cannot be reconstructed from dates or candidate order. Contextual daily
features see only bars completed by the anchor; already captured QF-29 features
are checked against their original context timestamp. Timestamp-aware exports
add a `decision_timestamp` identity column. Date-only exports keep their existing
columns; original legacy labelers retain positive session horizons.

## Persistence and QF-40

The opt-in membership source enters plan identity, frozen membership, validation
lineage, and holdout provenance. Changes to axis, schedule, boundaries, source,
timeframe/session policy, or elapsed reach invalidate material identities. The
existing persistence comparisons reject incompatible artifacts even when copied
into a new study directory. Compatible completed runs resume without evaluation.
No new cache or automatic migration is introduced.

Standalone QF-11 results and non-window QF-32 trials use the same QF-46 resolution
checks as window results when their temporal contract is elapsed. Read-only
inspection validates exact anchors, context warm-up, source identity, endpoint
boundaries, and availability without calling research components. Same-session
rows are ordered and unique by their original timestamps. Elapsed outcomes always
require explicit labeled rows, including unavailable endpoints; legacy session
artifacts retain their session-horizon checks.

QF-40 verifies captured timestamps, source lineage, and QF-46 request metadata;
it does not rerun QF-8 partition/purge calculations during OOS aggregation. Its
existing holdout adapter retains exact decisions whose conservative reach stays
inside the reservation. QF-48 execution never consumes the holdout. Explicit
consumption remains QF-40's permanent ledger operation; its exposure guard
conservatively covers the corresponding whole exchange sessions so an axis
change cannot restore a viewed session. The existing QF-9 holdout reader accepts
this additional provenance without research callbacks.

Deterministic tests use metadata-only labelers. Intraday forward returns belong
to QF-49; MFE/MAE, target-stop paths, and event outcomes belong to QF-47 or later
work. QF-48 includes none of those calculations and no QF-45 study logic.
