# Signal-level feature datasets and outcome labels

QF-7 extends the generic QF-11 prediction-study contracts. It does not add a
second prediction engine, feature store, backtester, or execution model.

```text
QF-3 MarketDataset
        |
        v
QF-4 trailing indicators and completed-session context
        |
        v
QF-11 PredictionRule -> fixed SignalFeatureCandidate disposition
        |
        +------ causal feature/outcome boundary ------+
                                                     |
                                                     v
                                     QF-11 OutcomeLabeler + evaluator
                                                     |
                                                     v
                              QF-7 flattened, resumable analytics rows
```

The first `PredictionStudy` run fixes and guards the complete candidate tuple
before a future-bearing labeler runs. `PredictionStudyResult.signals` exposes a
detached copy of that already-guarded tuple, including candidates whose declared
outcome is unavailable at the end of the dataset. It is an additive runtime
property: QF-11 study identities and serialized study output are unchanged.
QF-7 replays fixed, context-enriched candidates through one generic QF-11 study
per configured labeler/evaluator pair and joins the results by logical candidate
identity.

## Public composition

```python
from decimal import Decimal
from pathlib import Path

from quantforge.prediction import (
    OvernightGapPredictionParameters,
    OvernightGapSignalFeatureRule,
    PredictionStudy,
    SignalFeatureCandidate,
    build_signal_feature_dataset,
    default_overnight_gap_contextual_features,
    excursion_outcome,
    forward_return_outcome,
    target_stop_outcome,
)
from quantforge.prediction.feature_outcomes import ForwardReturnValues

rule = OvernightGapSignalFeatureRule(OvernightGapPredictionParameters())
primary = forward_return_outcome(1)
study = PredictionStudy[
    SignalFeatureCandidate,
    ForwardReturnValues,
    ForwardReturnValues,
].create(rule, primary.labeler, primary.evaluator)

result = build_signal_feature_dataset(
    dataset=dataset,
    prediction_study=study,
    contextual_features=default_overnight_gap_contextual_features(),
    outcomes=(
        primary,
        forward_return_outcome(2),
        forward_return_outcome(5),
        forward_return_outcome(10),
        forward_return_outcome(20),
        excursion_outcome(5),
        target_stop_outcome(5, Decimal("0.01"), Decimal("0.005")),
    ),
    output_root=Path("reports/features"),
)
```

The supplied outcome list must contain the labeler/evaluator composition used
by the initial `PredictionStudy`. This makes the study a real composition rather
than a prediction-only shortcut. The builder is provider neutral and consumes
only a validated QF-3 `MarketDataset`.

## Candidate identity and disposition

`SignalFeatureCandidate` is a QF-11 `PredictionRecord`. It contains no future
price, return, excursion, threshold touch, or correctness value. Its stable
dispositions are:

- `accepted`: the source rule selected a direction;
- `rejected`: an identifiable rule veto or no directional rule match;
- `blocked`: a known eligibility filter prevented acceptance;
- `overlapping`: available for rules that genuinely identify an occupied
  study-defined slot.

The overnight-gap adapter records Friday exclusions as `blocked`, ADX-above-max
vetoes and directionless dojis as `rejected`, and never fabricates
`overlapping`. A rejected or blocked candidate may have no direction. Raw
forward returns remain available for it; direction-dependent MFE/MAE and
target/stop evaluations explicitly report `candidate_direction_unavailable`.
Every accepted candidate must carry both its selected direction and selected
rule reason. Whenever any disposition carries a selected reason, that reason
must be first in the matched-rule trace; contradictory records fail at model
construction. Directionless blocked or overlapping candidates may omit both a
direction and selected reason.

The adapter calls the same `evaluate_overnight_gap_rules()` decision trace used
by the original QF-11 baseline. It records the selected highest-priority reason
and every matched rule in documented priority order. The original
`OvernightGapPredictionStrategy` output is unchanged.

One `candidate_id` is derived from the bar fingerprint, symbol, session, source
rule/configuration/version, candidate-rule configuration, and complete parameter
snapshot. One `row_id` hashes the complete canonical flattened row payload under
the QF-7 dataset identity. It therefore binds the candidate, features,
disposition, provenance, and outcomes to the checkpoint filename and detects
valid-JSON payload mutations during resume. Accepted and rejected views of the
same opportunity cannot become two rows. Row order follows the fixed QF-11
candidate order. Summary counts expose candidate, accepted, rejected, blocked,
and overlapping totals separately.

## Contemporaneous feature schema

Every candidate contains all baseline decision inputs:

- current and previous RSI;
- current and previous ADX;
- current and previous +DI and -DI;
- completed-session open and close;
- Python weekday (`Monday=0`);
- the selected direction and complete reason trace.

Candidate strategy-feature names must match their declared schema exactly.
Every strategy input field must use the `contemporaneous_feature` category;
missing, undeclared, or temporally miscategorized inputs fail closed instead of
being omitted from or contradicting the flattened artifact, even when the rule
produces no candidates.

The documented unused baseline context is:

- `atr_percentage_of_close`: QF-4 Wilder ATR(14) divided by completed close,
  chosen to describe prevailing range/volatility scale;
- `volume_ratio`: completed volume divided by its trailing 20-session QF-4 SMA,
  chosen to describe unusual participation;
- `trend_distance_percentage`: completed close divided by its trailing
  20-session QF-4 SMA minus one, chosen to describe trend location.

These features are exploratory; none changes the baseline rule. Every
`ContextualFeature` owns a stable configuration, schema definition, and
`value_from_history()` implementation. The three reviewed built-in contexts have
an explicit optimized integration that provides one full-dataset aligned series
calculated through causal QF-4 indicators; the builder computes each series once
and selects the value at the candidate session. Changing a later bar cannot
change an earlier aligned value. Custom
features always receive only a market-history prefix ending at the candidate
session, even if they expose a similarly named aligned-series method, so they
cannot inspect a later bar or later corporate action. Warm-up values remain
`null`; they are never backfilled. A contextual feature declared non-nullable
must return a value; schema/value contradictions fail closed.

To add a context feature, implement `ContextualFeature`, preferably by composing
a reusable QF-4 `Indicator`, document its type/unit/source/timing in
`SchemaField`, and add the object to `contextual_features`. Custom contexts are
required to declare the `contemporaneous_feature` category and are evaluated
through `value_from_history()` against a causal prefix. A full-dataset aligned
optimization requires an explicit reviewed integration into the bundled context
set; merely exposing `values_for_dataset()` does not grant access to future bars.
Candidate rules must leave `SignalFeatureCandidate.contextual_features` empty;
the builder exclusively owns contextual enrichment and rejects pre-populated
values rather than silently replacing them.

## Forward returns

`ForwardReturnOutcomeLabeler(H)` declares an exact QF-11 horizon of `H` exchange
trading sessions. It uses the dataset's chronological session sequence, not
calendar-day arithmetic. Each outcome study builds its session lookup once and
reuses it for every candidate rather than rescanning all bars per label:

```text
forward_return(H) = close[t+H] / close[t] - 1
```

The flattened values include availability, horizon, actual outcome session,
reference close, outcome close, and arithmetic return. A missing `t+H` session
beyond the dataset boundary produces explicit unavailable fields. The same
adjustment/basis provenance applies to both prices. Raw unadjusted datasets with
incomplete corporate-action provenance or recorded stock splits fail closed
because an unknown or mechanical split is not a research return. Split-adjusted
datasets do not require a complete raw corporate-action snapshot for these
labels.
The default supported composition uses 1, 2, 5, 10, and 20 sessions; arbitrary
positive session horizons are configurable.

## MFE and MAE

`ExcursionOutcomeLabeler(H)` captures the maximum high and minimum low from
sessions `t+1` through `t+H`, inclusive. The evaluator then orients them using
the already-fixed candidate direction and reference close `P`:

```text
UP:   MFE = maximum_high / P - 1
      MAE = minimum_low / P - 1

DOWN: MFE = 1 - minimum_low / P
      MAE = 1 - maximum_high / P
```

The exact sessions of both extrema are retained. Ties use the earliest session.
MFE and MAE are descriptive research labels. A daily high or low does not imply
that an order could have executed there.

## Target versus stop

`TargetStopOutcomeLabeler` retains every future daily high/low through its
configured horizon. `TargetStopEvaluator` applies inclusive percentage levels:

```text
UP:   target = P * (1 + target_percentage)
      stop   = P * (1 - stop_percentage)

DOWN: target = P * (1 - target_percentage)
      stop   = P * (1 + stop_percentage)
```

Labels are `target_first`, `stop_first`, `neither`, `both_same_session`, and
`unavailable`. The default `ambiguous` policy records `both_same_session` if one
daily bar contains both levels. It preserves that session, high, low, target,
and stop and makes no intraday-order assumption. The optional explicit
`conservative_stop_first` policy labels the event `stop_first` while still
preserving the ambiguity fields. Threshold equality counts as a touch.

This is not an execution engine. It creates no order, fill, position, fee,
slippage, or P&L record and does not modify QF-5 behavior.

## Flattened schema and identity

`schema.json` defines every column with semantic category, data type, unit,
nullability, calculation/source, and temporal availability. Identity columns
include deterministic row/candidate/study IDs, exact QF-3 dataset ID and bar
fingerprint, provider and adjustment provenance, source and candidate rule
identities, implementation version, full parameters and their identity, and
feature/outcome schema versions.

Strategy and contextual feature values, outcome evaluator values, and explicit
end-of-data defaults are checked against each declared schema type and
nullability before export. Decimal values must be finite exact decimal text,
dates must be canonical ISO dates, and primitive boolean, integer, string,
object, and array types must match their declarations. Any schema data type
outside `boolean`, `integer`, `decimal`, `date`, `string`, `object`, and `array`
is rejected when the field is constructed, including for an empty dataset.
Nullable string fields reject empty-string values because CSV represents both an
empty string and null as an empty field; producers must use null for absence and
a nonempty string for a present value so the flat artifact remains unambiguous.

Causal columns use `feature_`; future fields use `outcome_<namespace>_`. Decimal
values are exact decimal text in CSV and can be inferred or explicitly parsed
by pandas/Polars without decoding a feature JSON object. Parameter and reason
collections remain canonical JSON columns.

The QF-7 dataset ID hashes the exact immutable QF-3 dataset ID and complete
prediction provenance, including retrieval timestamp, requested and actual
ranges, corporate-action snapshot, bar fingerprint, and price/volume basis. It
also hashes the complete prediction-study template, rule and parameter
configuration, strategy-feature schema, contextual feature configurations,
complete contextual field definitions, every QF-11 labeler/evaluator
configuration, and feature/outcome/engine versions.
Chunk size does not participate. Equivalent configurations over the same exact
QF-3 cache entry receive the same identity; a refreshed cache entry or any
material feature or outcome change receives a new one.

## Incremental persistence and resume

Artifacts are written below `<output-root>/<feature-dataset-id>/`:

```text
manifest.json
schema.json
summary.json
features.csv
rows/
  <row-id>.json
```

Generation begins with an `in_progress` manifest and fixed schema. Each complete
row checkpoint is written to a temporary file, flushed, and atomically renamed.
Temporary/partial files are not checkpoints, and the payload-bound row ID path
prevents duplicates and fails closed on valid-JSON row corruption. On resume,
the builder validates the source dataset, complete configuration, schema, row
payload identities, candidate population, and QF-11 study IDs. Regenerated
causal candidates are enriched and compared with completed protected rows before
any checkpoint is skipped, so one artifact cannot combine two generations. A
candidate-only QF-11 boundary fixes and validates the population without running
the configured future outcome over completed rows. One validated, detached QF-11
dataset session is reused while each missing candidate chunk is labeled,
evaluated, and checkpointed before the next chunk starts; completed candidates
are never relabeled.
If startup is interrupted before the manifest is created, an empty or
schema-only checkpoint-free destination is safely reinitialized. Manifest-less
state containing row checkpoints or unknown artifacts fails closed instead.
`features.csv`, `summary.json`, and the final `complete` manifest are written
atomically; the manifest is last. A complete resume validates exact deterministic
CSV/JSON bytes and returns without rerunning the prediction rule or per-candidate
outcomes. For an empty dataset, it independently recomputes QF-11 study identities
from the configured outcome compositions so manifest corruption cannot replace
their provenance. Corrupt or incompatible state fails clearly and is never
silently reconciled.

CSV is required and implemented. Parquet is intentionally not emitted because
the repository has no existing Parquet dependency; QF-7 does not add one solely
for a duplicate representation.

## Exploratory three-feature example

Run the cached, provider-neutral example with an immutable QF-3 dataset ID:

```bash
uv run python scripts/analyze_signal_features.py --dataset-id <dataset-id>
```

It builds/resumes the baseline candidate dataset, then defines a winner as a
strictly positive five-session raw close return. It compares ATR/close, volume
ratio, and trend distance for winners and losers using sample count, mean,
median, population standard deviation, first/third quartiles, and disclosed
fixed bins with winner rates. Empty bins remain present. The split is
configurable through `WinnerDefinition`; it is not a universal definition of a
winner. The analysis API accepts only contemporaneous-feature schema fields as
features and a future-outcome schema field as the outcome, preserving the causal
feature/outcome boundary. Analyzed features must declare numeric `decimal` or
`integer` types. `DECIMAL_GREATER_THAN_ZERO` likewise requires a numeric outcome;
`VALUE_EQUALS` accepts only scalar outcome types and compares decimal outcomes by
numeric value rather than serialization scale. Boolean equality requires the
canonical `true` or `false` winner value and compares against the declared
boolean value rather than Python string casing. Integer equality parses and
normalizes integer text; date equality requires canonical ISO `YYYY-MM-DD` text.
When an outcome namespace declares an `available` flag, rows where that flag is
false are excluded before winner/loser classification, including non-null
sentinel labels such as `unavailable`. Availability flags are eligibility
metadata and cannot themselves be selected as analysis outcomes. They must be
non-nullable booleans whose unavailable-row default is exactly `false`. When a
custom outcome omits that flag, a non-null end-of-data default is ambiguous with
a legitimate matching value, so analysis rejects the configuration rather than
silently filtering rows. Such outcomes must provide an explicit boolean
`available` field; nullable outcomes whose unavailable value is null remain
unambiguous because null values are never classified.

The output is explicitly exploratory. It does not cherry-pick bins, select a
filter, modify the rule, claim causality, or establish tradability. Any candidate
relationship requires untouched out-of-sample validation and appropriate
multiple-comparison controls.

## Adding an outcome

Implement a typed QF-11 `OutcomeLabeler` with an exact positive session horizon
and sorted required market fields, then a typed evaluator that receives only the
fixed candidate and typed outcome. Wrap them with
`PredictionStudyOutcome.create()`, supply sorted `SchemaField` definitions and
explicit end-of-data defaults, and add the composition to `outcomes`. When an
`available` field is declared, it must be a non-nullable boolean and its
end-of-data default must be `false`; the builder enforces this for both typed
compositions and direct `ConfiguredOutcome` implementations. The QF-7 builder
does not need outcome-specific branches. The builder independently validates
every flattened available and unavailable value against these declarations,
including values returned by custom `ConfiguredOutcome` implementations. Before
checkpointing, it also validates every identity, disposition, feature, and
outcome value against the complete row schema. Each configured outcome must
return a canonical lowercase SHA-256 QF-11 study ID; invalid provenance is
rejected before any row is persisted. Returned outcome session keys must belong
to the current candidate chunk; unknown keys fail closed instead of silently
censoring a candidate's available outcome. Candidate source-rule configuration
IDs must likewise be canonical SHA-256 provenance values. The builder also
routes completed empty-dataset outcome revalidation through the same checked
execution boundary, so mutable direct outcomes cannot bypass configuration
identity checks during resume. It normalizes the
protocol's public namespace, fields, and unavailable defaults into dataset
configuration, so downstream analysis does not depend on a concrete outcome
implementation's private configuration shape. Outcome configuration identity is
rechecked immediately after every execution callback so chunk rows cannot be
persisted under transient or restored undeclared settings. Direct protocol
implementations must expose sorted, unique fields categorized exclusively as
future outcomes, including when the candidate population is empty.
