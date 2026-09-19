# Intraday excursion and target/stop outcomes (QF-47)

QF-47 adds one direction-neutral `IntradayPathOutcomeLabeler`, reused by
`IntradayExcursionEvaluator` and `IntradayTargetStopEvaluator`. The existing
session `ExcursionOutcomeLabeler`, `DirectionalExcursionEvaluator`,
`TargetStopOutcomeLabeler`, and `TargetStopEvaluator` keep their configurations,
formulas, units, and daily ambiguity behavior unchanged.

## Composition

```python
from datetime import timedelta
from decimal import Decimal
from quantforge.prediction import (
    IntradayExcursionEvaluationValues,
    IntradayPathValues,
    PredictionStudy,
    SignalFeatureCandidate,
    build_signal_feature_dataset,
    intraday_excursion_outcome,
    intraday_target_stop_outcome,
)

# source is an existing validated canonical TimeframeBarSeries, e.g. 2m bars.
excursion = intraday_excursion_outcome(timedelta(minutes=60), source)
target_stop = intraday_target_stop_outcome(
    timedelta(minutes=60), source, Decimal("0.003"), Decimal("0.002")
)
study = PredictionStudy[
    SignalFeatureCandidate, IntradayPathValues, IntradayExcursionEvaluationValues
].create(rule, excursion.labeler, excursion.evaluator, outcome_source=source)
result = build_signal_feature_dataset(
    dataset=prediction_metadata_dataset,
    prediction_study=study,
    contextual_features=(),
    outcomes=(excursion, target_stop),
    context_provider=context_provider,
    output_root=output_root,
)
```

The default export namespaces include elapsed microseconds, such as
`intraday_mfe_mae_3600000000us` and `intraday_target_stop_3600000000us`.
Supply distinct `namespace=` values when comparing sources or target/stop
thresholds at the same horizon. There is no new cache or outcome engine.

## Reference and exact path interval

The reference price **P** is the close of the exact completed canonical source
bar ending at the original decision timestamp. This reuses QF-49's
`completed_decision_observation_close` convention and shared reference/basis
validation. Daily metadata prices and nearby intraday closes are never substitutes.
A missing, developing, or between-boundary reference is invalid input.
Every used bar must match the prediction dataset's symbol and adjustment basis.

The decision bar's high and low are excluded. The first eligible future bar
**starts at the decision timestamp** and ends at the next expected canonical
completion boundary. Every expected canonical window thereafter is required,
through and including QF-46's resolved endpoint. Thus the completed-bar-end
interval is `(decision, resolved endpoint]`; the underlying price intervals
start at the decision boundary. No earlier or later bar contributes an extreme
or threshold event. `path_start_timestamp` records the decision boundary;
`path_end_timestamp` records the expected endpoint, even if unavailable.
The typed label retains each complete path bar's start/end, ID, high, and low.

QF-46 alone resolves the elapsed target, using the first **expected** completed
boundary at or after `decision + duration`. For 11:20 on 2m bars, 60m ends at
12:20 and includes bars ending 11:22 through 12:20 (30 bars). A 31m request
ends nominally at 11:51 and resolves to 11:52: that entire final canonical bar
is included. This is the existing ceiling policy, not a truncated synthetic bar.
The 11:54 bar cannot affect it. Requested, expected, and resolved timestamps
remain separately serialized. UTC is used for timestamps; exchange-session
labels and actual normal/early-close boundaries come from the configured calendar.

## MFE and MAE

Let H be the maximum future high and L the minimum future low in the full path.

| Direction | MFE | MAE |
| --- | --- | --- |
| UP | H / P − 1 | L / P − 1 |
| DOWN | 1 − L / P | 1 − H / P |

These are the existing daily decimal-ratio formulas: `0.003` means **+0.30%**,
not 0.003%. Arithmetic uses QuantForge's fixed 34-digit Decimal context.
Tied extremes select the earliest completed bar. Results record its completion
timestamp and observation ID.

As in the daily evaluator, values are **not clamped to zero** and the reference
price is not inserted as a synthetic path extreme. A flat path at P yields zero
for both. A path wholly above P can have positive UP MAE; a path wholly below P
can have negative UP MFE. This deliberately preserves historical conventions.

## Target and stop

Both threshold inputs are positive decimal ratios strictly between zero and one.
For target distance T and stop distance S:

| Direction | Target price | Stop price |
| --- | --- | --- |
| UP | P × (1 + T) | P × (1 − S) |
| DOWN | P × (1 − T) | P × (1 + S) |

The QF-45 example uses `T=0.003`, `S=0.002`, and 60m. Touches are inclusive.
The earliest eligible bar touching either threshold determines `target_first`
or `stop_first`. A full path with no touch yields `neither`.

If that earliest bar touches both, the result is `both_same_bar`, an additive
`TargetStopLabel` member. `SameBarConflictPolicy.AMBIGUOUS` is the sole supported
intraday policy. The result retains ambiguous high/low, bar ID, and completion
timestamp; unambiguous event fields remain null. A bar end identifies when the
range became observable, **not the exact intrabar hit time**. Neither open price
nor an assumed OHLC sequence resolves the collision. Later collisions cannot
change an earlier unambiguous result. The legacy daily opt-in conservative
stop-first policy remains available only on its existing session evaluator.
Unsupported intraday policies are rejected rather than silently coerced.

## Completeness and availability

The runner supplies QF-46 resolution from full source coverage before bounding
future label inputs. The path labeler consumes that exact resolution unchanged.
Endpoint failure takes precedence and preserves `session_overflow`,
`missing_required_observation`, `incomplete_future_data`, or `dataset_end`.
A 60m request with only 44m observed, or one exceeding the actual normal/early
session close, never becomes a shortened success and never rolls into tomorrow.
An endpoint at the actual close may include a completed terminal partial bar.

An available endpoint does not certify interior coverage. The labeler enumerates
expected windows using the same `intraday_session_windows()` calendar authority.
The first missing interior interval yields `missing_required_observation`;
a developing interval yields `incomplete_future_data`. Its expected end is
recorded as `missing_observation_timestamp`. All partial ranges are withheld.
Even an early observed target hit is unavailable if the configured path cannot
be certified in full. Nothing is filled, skipped, or synthesized.

`endpoint_status` / `endpoint_available` retain QF-46's endpoint result;
`status` describes full-path availability using the existing status enum.
The original QF-46 request/resolution remains attached to the generic outcome.
Evaluated `available` additionally requires a prediction direction. A
missing direction leaves the path status intact, sets `available=false`, and
records `candidate_direction_unavailable`. Numeric outcomes are null when
unavailable; target/stop uses `unavailable`, distinct from `neither`.

## Identity, export, and validation

Path labeler identity binds the typed elapsed configuration, full source
timeframe, reference convention, interval and completeness policy. Its temporal
reach delegates unchanged to QF-46: nominal duration plus one nominal observation
interval (62m for a 60m/2m path). QF-48/QF-8 uses that conservative duration for
exact timestamp membership, purge, and embargo. No session-count proxy or new
purging algorithm is introduced.

Direction-neutral path configuration is intentionally shared by the two
evaluators. Thresholds, formulas, units, ambiguity policy, and tie/event-time
conventions bind evaluator identity. The composed study/export binds **both**
labeler and evaluator plus immutable source reference/family manifest. Direction
and exact anchor enter per-row identity. Changing target, stop, horizon, timeframe,
source, or supported policy semantics therefore changes material identity and
prevents unsafe reuse. A future alternate ambiguity policy must have a distinct
configuration; this release does not implement one.

QF-7 CSV and QF-29 Parquet retain the exact anchor, duration, endpoint resolution,
path boundaries/status, reference price/convention, direction, source reference,
labeler/temporal/evaluator configuration IDs, extrema and extreme-bar IDs, or
thresholds/levels and event/ambiguity evidence. The normal adapter also records
composed outcome identity. Every path output column is `FUTURE_OUTCOME`.
Predictions and causal features are fixed before label access. Safe completed
resume checks existing artifacts without label callbacks or duplicate rows.

QF-39 persists typed path outcomes and evaluations in existing timestamp OOS
artifacts. QF-40 reads stored `mfe_percentage`, `mae_percentage`, and `label`
through its existing generic summaries; it does not traverse paths again.
QF-9 indexes and checks generic component, row, source, and artifact identities
without research callbacks or new integrity schemas. None of these three
production packages changes. Historical daily manifests and exports keep their
original representations; the QF-49 reference helper extraction preserves its
configuration, result schema, and numerical behavior.

## Scope and limits

These are future research labels, not executable fills or profitability claims.
Only canonical same-session completed OHLC paths are supported. No lower-resolution
ambiguity resolution, tick reconstruction, event-until-crossback outcomes,
new forward-return calculation, EMA/QF-45 rule or study, trading execution,
options, machine learning, or reporting framework is added.
