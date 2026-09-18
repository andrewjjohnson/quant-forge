# Intraday forward returns (QF-49)

`IntradayForwardReturnOutcomeLabeler` is a concrete implementation of the existing
QF-48 `TimestampStudyOutcomeLabeler` contract. It evaluates one future endpoint.
The legacy `ForwardReturnOutcomeLabeler(horizon_sessions)` and its serialized
session configuration, numerical policy, and date semantics are unchanged.

## Prices and units

The reference is the close of the **completed canonical source observation ending
exactly at the original decision timestamp**, in the decision's exchange session.
It is not the QF-3 daily close, bar open, previous available bar, or a simulated
fill. A missing, developing, or unaligned reference observation raises
`InvalidPredictionDataError`; it never silently changes the anchor. If primary
and outcome timeframes differ, the outcome source must still contain an exact
completed observation at the decision instant.

The future price is the close of the observation supplied by QF-46 resolution.
Both prices share the same canonical source and adjustment basis. Since the
outcome stays inside one exchange session, the legacy multi-session split guard
is not applied to the separate daily metadata dataset. Canonical source
construction and existing contextual study validation retain their price,
symbol, adjustment, feed, and family checks.

```text
raw_return = future_close / reference_close - 1
```

Returns use the existing 34-digit, half-even Decimal arithmetic policy and exact
decimal-string serialization. They are dimensionless arithmetic ratios:
`0.003` means `+0.30%`, and equal prices produce numeric zero. The result is
direction independent, including rejected or directionless fixed candidates.
There are no transaction costs, fills, orders, or executable-return claims.

## Resolution and unavailable results

The runner calls QF-46 `resolve_future_observation` on the complete immutable
source, then supplies its typed resolution and a bounded source to the labeler.
The labeler adds the reference lookup and arithmetic only. It neither rounds
timestamps nor re-resolves the bounded slice (which would lose coverage evidence).

`FIRST_COMPLETED_AT_OR_AFTER` uses the first **expected** canonical bar end at or
after the requested target. On two-minute bars, 11:20 + 30m resolves to 11:50;
11:20 + 31m requests 11:51 and resolves to 11:52. If 11:52 is missing, a present
11:54 bar is never substituted. Bar end/completion determines availability,
independently of the bar label convention.

All QF-46 statuses are preserved: `available`, `missing_required_observation`,
`incomplete_future_data`, `dataset_end`, and `session_overflow`. Unavailable
results retain the valid reference price and resolution metadata, while
`outcome_price` and `raw_return` are null. They remain explicit typed rows.

`SAME_SESSION_ONLY` uses the actual exchange calendar, including early closes.
A target beyond close is unavailable even if tomorrow's bars exist. The horizon
is never shortened or rolled into the next session. A target exactly at the
actual close may use a completed terminal observation. Missing intervening bars
do not invalidate an otherwise valid endpoint return: this result makes no
claim about path completeness.

## Study and feature-dataset composition

```python
from datetime import timedelta
from quantforge.prediction import (
    IntradayForwardReturnValues,
    PredictionStudy,
    SignalFeatureCandidate,
    build_signal_feature_dataset,
    intraday_forward_return_outcome,
)

# outcome_source is a validated canonical TimeframeBarSeries, e.g. two-minute bars.
outcomes = tuple(
    intraday_forward_return_outcome(timedelta(minutes=minutes), outcome_source)
    for minutes in (10, 30, 60, 120)
)
primary = outcomes[1]
study = PredictionStudy[
    SignalFeatureCandidate, IntradayForwardReturnValues, IntradayForwardReturnValues
].create(rule, primary.labeler, primary.evaluator, outcome_source=outcome_source)
result = build_signal_feature_dataset(
    dataset=prediction_metadata_dataset,
    prediction_study=study,
    contextual_features=(),
    outcomes=outcomes,
    context_provider=context_provider,
    output_root=output_root,
)
```

The adapter uses the existing typed QF-7/QF-29 composition and QF-46 metadata
fields. Default namespaces contain the exact duration in microseconds, e.g.
`intraday_forward_return_1800000000us` for 30m. Supply `namespace=` when comparing
multiple sources at the same horizon. Each duration has a distinct identity and
namespace; configured ordering and existing replay/chunk ordering are deterministic.

CSV and contextual Parquet exports include the original decision timestamp,
anchor/horizon kinds, elapsed duration, requested/expected/resolved timestamps,
availability/status, reference and future observation IDs, both price conventions,
prices, ratio, outcome/temporal configuration IDs, and source reference. The
source reference includes family, source snapshot, feed, and timeframe identity;
the family binds the adjustment basis, and the full timeframe is retained in
outcome configuration.
All these columns are `FUTURE_OUTCOME` fields, never contemporaneous features.
Per-row resolution defaults are nullable in the generic adapter schema; actual
elapsed execution always supplies an explicit QF-46 resolution, including for
unavailable outcomes.

## Validation, provenance, and persistence

QF-48 preserves the original anchor during fixed-candidate replay, so multiple
same-session decisions remain distinct. Predictions and causal features are fixed
before labelers receive future data. `required_future_duration` delegates directly
to QF-46's nominal duration plus one observation interval. Existing
`OutcomeProvenance.capture_timestamp`, QF-8 purging, and QF-39 membership consume
that reach without session counts or another validation algorithm.

The component identity binds the elapsed temporal configuration (including
alignment, session policy, and full observation timeframe), fixed close-price
conventions, formula, units, and implementation/schema versions. Study/export
identities additionally bind the immutable source reference and family manifest.
There are no alternative reference-price or alignment policies in this release.
Changing material configuration/source prevents unsafe reuse through the existing
QF-7/QF-32/QF-39 persistence checks; no new cache is introduced. Adding observations
strictly after a complete horizon under the same relevant source/configuration
semantics cannot change that historical value. A different immutable artifact
still intentionally has distinct provenance.

QF-39 stores the typed outcome in its existing timestamp-keyed OOS artifact.
QF-40 reads that artifact without recalculating returns or adding metrics.
QF-9 indexes and validates the existing generic configuration, temporal resolution,
and source provenance without executing research. Its checkpoint reader orders
same-session feature rows by exact decision timestamp before checking CSV/Parquet
bytes, rather than relying on hash-filename order. No producer engine/schema
version or legacy artifact migration is needed.

QF-47 exclusively owns intraday MFE/MAE, high/low path coverage and traversal,
target/stop order, and same-bar ambiguity. QF-49 implements none of them, and
does not implement event-terminated outcomes or QF-45 strategy logic.
