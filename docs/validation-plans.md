# Leakage-safe validation plans

QF-8 implements study-neutral chronological partition contracts in
`quantforge.validation`. The package defines scientific boundaries and safe
membership only. It does not run optimization, train models, execute backtests,
combine out-of-sample results, consume the final holdout, or render reports.

Prediction and trading/backtest research use the same partition vocabulary but
retain separate study types and scientific configurations. A validation plan
does not require predictions, trades, an equity curve, or a metric model.

## Temporal boundaries

Every plan uses exactly one ordered axis:

- `ExchangeSessionBoundary` identifies an actual session under one immutable
  QF-13 `ExchangeSessionPolicy`. Weekends and exchange holidays are rejected.
- `TimestampBoundary` requires a timezone-aware timestamp and normalizes it to
  UTC. This supports intraday and other timestamp-indexed studies.

`ValidationInterval` is closed and inclusive at both ends. Its boundaries are
observation membership keys, not wall-clock availability claims. Session and
timestamp boundaries cannot be mixed, and session policies cannot change within
a plan.

All plan windows use the same QF-13 session policy as every configured research
timeframe when the plan is session-indexed. This prevents a validation interval
from being described as XNYS regular sessions while its inputs use another
calendar, timezone, or session scope.

## Folds and final holdout

One `ValidationFold` contains:

- one `development_training` window;
- an optional `validation_selection` window;
- one `walk_forward_test` window.

Within a fold, development must finish before selection, and selection must
finish before test. Without selection, development must finish before test.
Test windows across folds are strictly chronological and disjoint.

`TrainingWindowMode.EXPANDING` requires later development windows to keep the
same start and advance their end. `TrainingWindowMode.ROLLING` requires both
start and end to advance. The windows are explicit definitions; QF-8 does not
execute them.

Every plan has one `FinalHoldout`. Its metadata marks the interval as reserved
and explicitly states that consumption is outside QF-8. The holdout must follow
every development, selection, and test window and cannot overlap them. A holdout
consumption ledger is intentionally deferred to a later story.

## Horizon purging and embargo

`PurgePolicy` fixes two identity-bearing offsets on the plan's temporal axis:

- `label_horizon` is the maximum future reach of any configured outcome used by
  training or parameter selection;
- `embargo` is additional separation required beyond that label reach.

For exchange sessions, both values are actual configured exchange-session
counts, not weekdays or calendar-day counts. For timestamp plans, both are exact
elapsed durations serialized as integer microseconds.

`purge_partition_observations()` compares an earlier partition with its next
protected interval: development with selection/test, selection with test, and a
test segment with the next test or final holdout. The common
`purge_development_observations()` helper selects the development path. An
observation is purged when:

```text
observation + label horizon + embargo >= protected start
```

The equality case is purged: a label ending on the first protected observation
has crossed the boundary. Exchange-session distance is resolved from the QF-13
calendar, so holidays and weekends do not weaken or extend the configured
horizon accidentally.

Composite studies must configure `label_horizon` to the maximum future reach of
all outcome configurations that influence training or selection. A trading
study with no forward observation label uses a zero horizon. This does not make
test or holdout results available for selection; it only states that individual
development observations have no forward label window to purge.

Each `OutcomeProvenance` pairs the exact versioned outcome configuration with
its typed maximum future reach. Plan construction requires the purge horizon to
equal the maximum declared outcome horizon and rejects axis mismatches. A
shorter unsafe purge horizon and a longer identity-equivalent over-purge cannot
silently diverge from the fixed outcome definitions.

Appending later observations cannot change historical membership. The plan,
window boundaries, calendar policy, label reach, and embargo are fixed identity
inputs, and observations outside the closed development interval are ignored by
the purge result.

## Indicator warm-up

Each `ValidationWindow` declares a non-negative `warm_up_observations` count.
`select_window_observations()` returns two structurally separate tuples:

- `warm_up_context` contains the exact preceding observations used only to
  calculate causal indicators;
- `study_observations` contains the observations eligible for that window's
  training, selection, test, or holdout role.

The serialized selection explicitly records
`warm_up_eligible_for_selection=false`. Warm-up rows must precede the window and
never enter study membership. Insufficient history fails closed instead of
silently shortening the declared warm-up.

The contract supplies observation membership only. It does not attach outcomes
to warm-up rows or calculate indicators. Consumers continue to use the existing
backend-neutral QuantForge indicator architecture.

## Fixed research environment and provenance

Every `ValidationPlan` owns exactly one `ResearchEnvironment`, shared by all
folds and the holdout. It preserves:

- QF-3 dataset IDs and data fingerprint;
- QF-14 dataset-family references, family ID, canonical source snapshot ID, and
  the complete exact family manifest when a family is used;
- complete QF-13 timeframe and session-policy configurations and IDs;
- versioned aggregation-policy configurations;
- indicator configuration, implementation version, normalized backend ID,
  wrapper/library version, mapped function, and native runtime version;
- prediction rule or trading strategy configuration and implementation version;
- every applicable outcome-label configuration;
- execution and cost configuration when applicable.

Every QF-14 family reference must match one of the environment's exact QF-13
timeframe configuration IDs. A family member cannot be relabeled as another
session count, intraday duration, or session policy inside a validation plan.
Construction also recreates every compact reference from the supplied complete
family and requires exact equality. A valid-looking but unrelated manifest ID
therefore cannot be paired with persisted references, and the complete verified
manifest participates in environment and plan identity.

`ResearchStudyType.PREDICTION` and
`ResearchStudyType.TRADING_BACKTEST` identify the consumer without changing the
partition model. Prediction environments need not define execution, while
trading environments need not define outcomes.

`IndicatorProvenance.capture()` snapshots both the existing indicator
configuration and its resolved QF-35 backend identity. For an historical native
indicator, `legacy_native_configuration=true` and the original pre-explicit-
backend configuration remains unchanged. An explicitly configured `native_v1`
indicator has a different configuration and identity. Capturing a validation
plan never migrates one to the other.

All caller-owned primitive mappings are captured through
`PrimitiveMappingSnapshot`. Later mutation of the original configuration cannot
change an existing environment or plan.

## Identity and serialization

Schema version 1 defines deterministic SHA-256 identities for:

- validation windows;
- folds;
- final-holdout metadata;
- research environments;
- complete validation plans;
- purged development membership;
- warm-up/study observation selections.

`serialize_validation_plan()` writes the complete manifest with QuantForge's
canonical sorted compact JSON policy. Scientific timestamps are UTC, durations
are integer microseconds, exchange sessions are ISO dates, and set-like
provenance collections are sorted by stable identity.

Persistence or cache consumers must call
`validate_validation_plan_manifest(plan, content)` before reuse. It rejects
invalid JSON, non-canonical bytes, a changed plan ID, or any content that differs
from the current fixed plan. QF-8 defines this validation boundary but does not
introduce a separate persistence store.

Changing any dataset family or fingerprint, timeframe/session or aggregation
policy, indicator configuration/backend/version, rule/strategy, outcome,
execution/cost configuration, boundary, warm-up count, horizon, embargo, fold
mode, or holdout reservation produces a different plan identity.

## Deliberate limitations

QF-8 does not provide:

- walk-forward optimization or model-training orchestration;
- aggregation of out-of-sample windows;
- final-holdout consumption or access tracking;
- report or manifest rendering beyond canonical plan serialization;
- machine-learning pipelines;
- indicator, prediction, outcome, backtest, or metric recomputation.

Those consumers must use the immutable QF-8 plan and existing research artifacts
rather than redefining partitions or scientific provenance independently.
