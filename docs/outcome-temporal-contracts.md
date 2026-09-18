# Outcome anchors, horizons, and observation resolution

QF-46 defines future-label **time semantics**. It adds no intraday return,
excursion, target/stop, price selection, or path calculation. The dependency
order is **QF-46 → QF-48 → QF-49 / QF-47 → QF-45**. The historical Jira link
showing QF-48 blocking QF-46 is stale.

## Anchors and requests

`OutcomeAnchor` carries an explicit `OutcomeAnchorKind`, a `signal_session`
exchange trade-date label, and an optional `decision_timestamp`:

- `SESSION` preserves the legacy date anchor. It may carry the original exact
  decision as supplementary metadata without changing the anchor semantics.
- `TIMESTAMP` requires a timezone-aware instant, canonicalized to UTC. Naive
  timestamps are rejected. The supplied exchange trade-date is never inferred
  from the UTC or local calendar date; overnight exchanges can differ.

`OutcomeEvaluationRequest` adds the typed temporal configuration, complete
outcome configuration ID, prediction dataset ID/fingerprint, and an optional
QF-14 `DatasetFamilyReference` identifying the intraday observation source.
Its deterministic `request_id` binds all these inputs. The resolver checks the
source reference and timeframe against the actual validated series.

The QF-11 session runner constructs this request **after predictions are fixed**.
When context exists it copies `PredictionRuleContext.as_of`, including QF-42's
original scheduled instant. It never creates another schedule or converts a
session date into an intraday decision.

`evaluate_outcome_request()` calls an explicitly provided `label_request(dataset,
request)` method, or passes the identical `(dataset, signal_session)` arguments
to the original `label()` method. It checks dataset and configuration identity
before dispatch. `RequestOutcomeLabeler` is the opt-in protocol for future
request consumers. Exact-anchor requests cannot fall back to date-based labelers.
Dataset validation and fixing causal predictions remain the caller's job.

QF-48 now integrates explicit timestamp membership and fixed-candidate replay.
`TimestampStudyOutcomeLabeler` adds a runner-bounded canonical source to request
execution; it reuses these temporal contracts. Legacy session paths retain their
positive session counts, and elapsed paths omit the count rather than setting
it to zero. See [timestamp prediction validation](timestamp-prediction-validation.md).

## Typed horizons and material identity

```python
from datetime import timedelta
from quantforge.prediction import (
    ExchangeSessionHorizon,
    ElapsedDurationHorizon,
    OutcomeTemporalConfiguration,
)
from quantforge.timeframes import IntradayInterval, Timeframe

ExchangeSessionHorizon(count=5)
ElapsedDurationHorizon(duration=timedelta(minutes=60))

source_timeframe = Timeframe.us_equity(IntradayInterval(timedelta(minutes=2)))
temporal = OutcomeTemporalConfiguration.elapsed_duration(
    timedelta(minutes=60), source_timeframe
)
```

Both variants require strictly positive values. Session counts reject booleans,
floats, and strings; elapsed horizons require `timedelta`. Durations serialize
as exact integer microseconds without floating-point conversion. 10, 30, 60,
120 minutes, and longer elapsed durations can be represented; representing
24 hours does not permit it to cross the session boundary.

The versioned temporal configuration serializes anchor kind, typed horizon,
full observation timeframe (calendar, session hours, anchoring, bar label,
completion exposure), alignment, and session policy. `configuration_id` uses
QuantForge's existing canonical configuration hash. Use
`PrimitiveMappingSnapshot.capture(config.to_primitive())` for canonical JSON;
`OutcomeTemporalConfiguration.from_primitive()` validates the exact schema and
round trip, rejecting unknown policies and extra/noncanonical fields.

Components opt in with a `temporal_configuration` field in their configuration.
`outcome_temporal_configuration()` reads this or interprets historical
`parameters.future_sessions` / `required_future_sessions` declarations as
exchange-session counts. Caller-supplied session requirements must agree.
It does not mutate or enrich legacy serialized objects. Legacy component,
study, feature dataset, window, and resume identities therefore remain unchanged.
One session, 24 elapsed hours, 30 minutes, and 60 minutes cannot collide. Unknown
alignment or cross-session policies fail rather than alias an implemented policy.

## Canonical future endpoints

`resolve_future_observation(request, source)` consumes a validated
`TimeframeBarSeries`, created through its existing artifact constructors.
Its `DatasetFamilyReference` must exactly match the request's source reference.
No provider is called. No price is selected or synthesized.

The only alignment policy is
`FIRST_COMPLETED_AT_OR_AFTER`: resolve the first **expected** completed bar end
at or after `decision_timestamp + duration`. It reuses
`intraday_session_windows()` and canonical exchange-session boundaries, including
clock-anchored leading partial bars and normal/early-close terminal partial bars.
The requested target never changes. Bar labels do not determine availability;
explicit bar ends do. Calendars with intraday recesses remain unsupported,
matching the shared window contract.

For five-minute observations, a target of 12:20 resolves to 12:20; 12:21 resolves
to the expected 12:25 boundary. If the 12:25 observation is absent but 12:30
exists, the result remains unavailable. The resolver never selects 12:30.

The sole session policy is `SAME_SESSION_ONLY`. Decisions must be inside the
supplied session. A target after session close returns overflow, even if a later
session has data. Targets exactly at close may use a completed terminal partial
bar; this aligns an in-session target and does not shorten an overflowing horizon.

| Status | Meaning |
| --- | --- |
| `AVAILABLE` | The exact expected completed endpoint exists. |
| `SESSION_OVERFLOW` | The requested duration ends after the permitted close. |
| `MISSING_OBSERVATION` | The required endpoint is absent inside observed coverage. |
| `INCOMPLETE` | The required interval exists only as a developing observation. |
| `DATASET_END` | The required endpoint is beyond the final observed boundary, or the source is empty. |

A source may explicitly permit developing bars in its timeframe; the resolver
still selects **only completed** observations. A developing required interval
is reported incomplete. No developing price is returned. A trailing absent
observation without later coverage is classified as dataset end; this contract
does not infer whether an upstream provider omitted that trailing observation.

`OutcomeResolution` retains the request, requested target, expected endpoint,
actual resolved endpoint and observation ID, status, and availability. Unavailable
results have `observation=None` and `resolved_observation_timestamp=None`.
They are never numeric zero. Resolving an endpoint does **not** certify the
intervening path; QF-47 must validate path coverage separately.

## Conservative temporal reach and metadata

For an elapsed horizon, `required_future_duration` is the nominal duration plus
one nominal observation interval. This deliberately conservative bound covers
ceiling alignment, including partial windows, without consulting market values.
For example, 60 minutes on five-minute bars exposes 65 minutes of future reach.
It is not clipped to session close or a particular observation's availability.
`future_temporal_reach` returns the existing QF-8 `TemporalOffset.duration(...)`;
session horizons return `TemporalOffset.sessions(...)`. A timestamp component
can delegate `required_future_duration` to the temporal configuration and use
`OutcomeProvenance.capture_timestamp(component)` unchanged. No new purge model
or timestamp observation-membership engine is introduced.

`OutcomeResolution.metadata_primitive()` and `outcome_resolution_fields()` form
an opt-in flat schema hook for QF-7 CSV and QF-29 Parquet exports. Fields include
anchor kind, original decision timestamp/session, horizon kind/duration, requested
and expected timestamps, resolved timestamp/observation ID, status, availability,
and complete outcome/temporal configuration IDs. All fields use
`FUTURE_OUTCOME`, remaining separate from causal feature columns. Existing schemas
and daily exports are unchanged. The full resolution retains source lineage.
QF-48 owns preserving the original anchor during fixed-candidate replay.

QF-9's generic outcome-contract validation understands typed configuration and
requires its wrapper to agree exactly with the component declaration. Session
wrappers still require a positive matching session count. Elapsed wrappers must
omit `required_future_sessions`; zero, null, conflicting legacy counts, malformed
horizons, unsupported policies, and mismatched timeframes are rejected. Market
field and schema checks are preserved. QF-9 does not perform temporal membership,
endpoint research, or intraday arithmetic. QF-48 supplies row-level timestamp execution and validation through the explicit
membership source.

QF-39/QF-40 provenance already retains immutable component snapshots and hashes;
no execution changes are needed here. Changing material configuration creates a
new identity for existing persistence/resume comparisons. Appending unrelated
future observations cannot change an already-resolved endpoint or its temporal
configuration. A different immutable source artifact retains its own identity.
