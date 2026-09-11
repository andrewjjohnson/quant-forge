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

For exchange sessions, both values count the validated observations used by the
session-indexed outcome component, not weekdays or synthesized calendar rows.
Each observation must still be an actual session under the configured exchange
calendar. This keeps purging aligned with outcome labelers that advance through
the observed bar sequence when a dataset records a missing session. For
timestamp plans, both values are exact elapsed durations serialized as integer
microseconds.

`purge_partition_observations()` compares an earlier partition with its next
protected interval: development with selection/test, selection with test, and a
test segment with the next test or final holdout. The common
`purge_development_observations()` helper selects the development path. An
observation is purged when:

```text
observation + label horizon + embargo >= protected start
```

The equality case is purged: a label ending on the first protected observation
has crossed the boundary. Exchange-session distance is resolved from the
supplied validated chronology, so a missing observation cannot make an
observed-bar label reach silently cross into the protected partition. When the
combined horizon and embargo is positive, the chronology must include a
protected-window observation; otherwise purging fails closed.

Composite studies must configure `label_horizon` to the maximum future reach of
all outcome configurations that influence training or selection. A trading
study with no forward observation label uses a zero horizon. This does not make
test or holdout results available for selection; it only states that individual
development observations have no forward label window to purge.

Each `OutcomeProvenance` pairs the exact versioned outcome configuration with
its typed maximum future reach. It cannot be constructed from an independent
configuration reference and offset. `capture_exchange_sessions()` reads
`required_future_sessions` from a typed session outcome component, while
`capture_timestamp()` reads `required_future_duration` from a typed timestamp
outcome component. Plan construction then requires the purge horizon to equal
the maximum captured outcome horizon and rejects axis mismatches. A shorter
unsafe purge horizon and a longer identity-equivalent over-purge cannot silently
diverge from the fixed outcome definitions.

Appending observations after the first protected observation cannot change
historical membership. Correcting observations before that boundary can change
the observed label reach and therefore the purge result; the corresponding
dataset fingerprint also changes. The plan, window boundaries, calendar policy,
label reach, and embargo remain fixed identity inputs.

## Indicator warm-up

Each single-timeframe `ValidationWindow` may declare a non-negative scalar
`warm_up_observations` count. A multi-timeframe window instead declares one
`TimeframeWarmUpRequirement` for every configured source timeframe; mixing the
two forms is rejected. Counts remain in their own source-bar units, so a weekly
count is never compared with or selected from a daily chronology.

`select_window_observations()` accepts the exact source timeframe when a window
uses timeframe-specific warm-up and returns two structurally separate tuples:

- `warm_up_context` contains the exact preceding observations used only to
  calculate causal indicators;
- `study_observations` contains the observations eligible for that window's
  training, selection, test, or holdout role.

The serialized selection explicitly records
`warm_up_eligible_for_selection=false`. Warm-up rows must precede the window and
never enter study membership. Insufficient history fails closed instead of
silently shortening the declared warm-up.

`IndicatorProvenance.capture()` binds each indicator's existing
`warm_up_observations` contract to the exact QF-13 source timeframe on which the
count is expressed. The domain-specific
`ResearchRuleProvenance.capture_prediction()` and `capture_trading()` factories
verify the component's own canonical configuration type, then capture its own
warm-up and the exact configuration IDs of its required indicators. When a
rule declares QF-12 multi-timeframe context, provenance also captures every
required `(source timeframe, indicator configuration)` binding from that
canonical context. The environment must contain all of those exact bindings;
one daily instance cannot satisfy the same indicator required on weekly bars.
A multi-timeframe rule must also identify the source timeframe for its own
warm-up. Because the
first study observation supplies the final input needed for its own result,
every development, selection, test, and holdout window must declare at least
`warm_up_observations - 1` preceding rows for each indicator and rule in that
component's source timeframe. Plan construction rejects missing or undersized
per-source context. Consumers call `select_window_observations()` independently
for each source chronology, preventing daily and weekly bar counts from being
interchanged.

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

The configured timeframe set must exactly equal the timeframe set supplied by
dataset provenance. Every QF-14 family reference names one exact QF-13
timeframe; no referenced timeframe may be omitted and no unavailable timeframe
may be added. `DatasetProvenance.from_market_dataset()` binds the legacy QF-3
daily dataset to its canonical one-session exchange timeframe, including its
calendar and the calendar's authoritative exchange timezone, so it cannot be
relabeled as intraday or advertised as multiple timeframes. The provider's
serialization timezone is preserved in dataset metadata but is not substituted
for exchange timezone. `DatasetProvenance` is factory-only: standalone identity
must come from a validated `MarketDataset`, while family identity must come from
`from_dataset_family()`. Family construction recreates every compact reference
from the supplied complete family and requires exact equality. Its dataset
fingerprint is derived from the exact family manifest ID plus the selected
artifact references; callers cannot supply an unrelated hash. A valid-looking
but unrelated fingerprint, timeframe, or manifest cannot be paired through a
public constructor, and the verified manifest or standalone timeframe
participates in environment and plan identity.

Every primary and contextual rule requirement must have selected family
references on its exact timeframe with the same `FeedScope`, including inputs
that declare no indicators. Matching compares the complete existing scope
contract (coverage, venue, and provider scope); IEX-only data cannot satisfy a
consolidated requirement. A missing family reference or missing/contradictory
scope is rejected before the environment is accepted. The check reads the
rule's immutable configuration snapshot, so later edits to the component cannot
change the captured requirements. Existing snapshots and family references
already serialize these fields and include them in environment/plan identity;
no additional persistence schema is introduced.

Aggregation provenance is derived from the same selected dataset lineage. When
any selected QF-14 member is derived, the environment must contain exactly the
typed `AggregationPolicy` recorded by that family; a generic reference, changed
policy configuration, missing policy, or additional policy is rejected.
Standalone QF-3 datasets and selections containing only the canonical family
source require no aggregation reference because no derivation was applied.

`ResearchStudyType.PREDICTION` and
`ResearchStudyType.TRADING_BACKTEST` identify the consumer without changing the
partition model. Prediction environments need not define execution. Trading
environments need not define outcomes, but must provide explicit execution and
cost provenance so implicit fill, fee, or slippage defaults cannot alias under
one environment identity.

The semantic type of `research_rule` is also fixed by study type: prediction
uses `prediction_rule`, while trading/backtest research uses
`trading_strategy`. Rule provenance is factory-captured from the typed component,
including its own warm-up, required-indicator identities, and available
multi-timeframe indicator bindings. A prediction rule
cannot stand in for the strategy provenance required to trace trades, a trading
strategy cannot be mislabeled as a prediction rule, and neither can silently
depend on an indicator missing from the fixed environment.

Trading execution provenance accepts and factory-captures only an existing
validated `BacktestConfig`, not a structural lookalike or freely tagged generic
reference. Its immutable
snapshot includes execution timing and price, commission, transaction fees,
slippage, sizing, rejection/accounting policies, capital, and engine/result
versions. Consequently an unrelated configuration cannot satisfy the backtest
environment contract, and any changed execution assumption changes the
environment and plan identity.

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
