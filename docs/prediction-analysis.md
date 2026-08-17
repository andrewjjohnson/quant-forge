# Prediction studies and overnight gap analysis

QF-11 adds `quantforge.prediction`, a provider-neutral research boundary for
testing causal predictions against outcomes observed later. It does not place
orders, simulate fills, or change QF-5's execution safeguards. The original
overnight-gap implementation remains the concrete QF-11 baseline on top of the
generic study contracts.

```text
PredictionStudy
    |
    +--> prediction rule --> fixed causal predictions
    |                              |
    |                 feature/outcome boundary
    |                              |
    +--> outcome labeler --> typed future outcomes
                                   |
    +--> evaluator -------> typed evaluations
                                   |
                                   v
                         generic study rows
```

`PredictionStudy` composes three independently configured and identified
components. A prediction rule sees the QF-3 dataset and emits typed records
using only information available at the signal timestamp. Only after those
records are fixed does the outcome labeler read later observations. The
evaluator receives a fixed prediction and a typed outcome; it does not receive
the dataset. Generic study rows and manifests do not require a direction,
correctness flag, next-open value, or gap field.

`PredictionStudyResult.signals` exposes detached copies of the complete fixed
prediction tuple, including end-of-data predictions without an available label.
This runtime extension supports QF-7 candidate datasets without changing QF-11
study identities or the serialized generic result. See
`docs/signal-feature-datasets.md`.

The concrete overnight-gap study composes an existing directional prediction
strategy with `NextSessionOpenGapOutcomeLabeler` and
`OvernightGapDirectionEvaluator`. For that study, the signal close is a label
anchor, not a claimed executable fill. A strategy that requires the completed
daily close cannot also claim it purchased at that exact close.

## Generic study contracts

The reusable contracts are:

- `PredictionRule` and `PredictionRuleOutput`, which generate typed causal
  prediction records;
- `OutcomeLabeler`, which declares its required future-session horizon and
  market fields and returns typed outcome values;
- `PredictionEvaluator`, which compares one fixed prediction with one already
  generated typed outcome;
- `PredictionStudy`, `PredictionStudyRow`, and `PredictionStudyResult`, which
  compose the components and retain provenance without imposing study-specific
  metrics.

The study identity includes the dataset provenance, prediction-rule identity
and configuration, labeler identity and configuration, future-session horizon,
required market fields, evaluator identity and configuration, feature
configuration, and result-schema versions. Changing any of these inputs creates
a distinct study identity. Component configurations are captured before the run
and checked immediately after every callback, as well as afterward, so a mutable
component cannot temporarily change and restore the meaning of individual rows.

The runner requires each strategy's warm-up declaration to be a positive
integer and rejects any signal emitted before that many dataset observations
have completed. The declaration and its enforced semantics therefore agree with
the study provenance.

The runner also verifies that a returned outcome session is exactly the
labeler's declared number of future sessions after the signal within the
validated QF-3 session sequence. A missing outcome is accepted only when that
declared future session is beyond the dataset boundary, preventing a labeler
from selectively censoring available outcomes. Every prediction is snapshotted
before the future-bearing dataset is exposed to labeler validation, then
rechecked after validation and around each label call. Prediction and outcome
primitives are also checked around evaluation. Outcome and evaluation identities
are derived from the same immutable value snapshots retained by their
authoritative records, never from a second component serializer call, and
component-owned values are checked again after all evaluations, preventing
immediate or delayed evaluator mutation. Returned rows contain detached typed
payloads for adapter compatibility, while their authoritative serialization is
an immutable primitive snapshot. Reusing a stateful component therefore cannot
rewrite an earlier result. Prediction records and typed outcome/evaluation
values must support a component-independent `copy.deepcopy`; the runner rejects
copies that alias their top-level component-owned object or serialize
differently. The complete fixed prediction payload, including contemporaneous
features, participates in each generic row identity.

To add a future prediction experiment, define a typed prediction record and
rule when an existing one is unsuitable, define typed outcome values and an
`OutcomeLabeler`, define the corresponding typed `PredictionEvaluator`, compose
them with `PredictionStudy.create(...)`, and call `run_prediction_study(...)`.
Study-specific aggregation and export belong outside the generic runner. For
example, a multi-session close-return study can declare a two-session horizon
and a future-close outcome without adding next-open or gap logic to the core.

## Original baseline rules

The original QF-11 strategy is retained unchanged as the
`combined_original` baseline. New hypotheses are separate prediction strategy
classes and must not silently replace this baseline. The baseline owns Wilder
RSI with period 2 and Wilder +DI, -DI, and ADX with period 5. Indicator values
use only the current and earlier completed bars.
Wilder smoothing starts from an initial full-period mean or sum and then uses
the recursive Wilder update. The first ADX is the mean of the first five DX
values.

Friday signal sessions are excluded by default. Python weekday numbers are used
in configuration (`Monday=0` through `Sunday=6`), so the default excluded value
is `4`.

Rules are evaluated in this order:

1. If current ADX is greater than 60, emit no prediction.
2. If ADX is inclusively between current +DI and -DI and was not inclusively
   between the prior session's DI values:
   - +DI above -DI predicts an upward gap;
   - -DI above +DI predicts a downward gap.
3. An RSI stab from above occurs when prior RSI was above both prior DI values
   and current RSI is at or below the larger current DI value. It predicts up.
4. An RSI stab from below occurs when prior RSI was below both prior DI values
   and current RSI is at or above the smaller current DI value. It predicts
   down.
5. RSI below 15 predicts up; RSI above 85 predicts down.
6. RSI equal to 15 or 85 is in the middle range. Within the inclusive middle
   range, a bullish candle predicts up and a bearish candle predicts down. A
   doji emits no prediction.

The ADX zone-entry rule has priority over the RSI-stab rule, which has priority
over the base RSI/candle rule. An ADX value greater than 60 is an absolute veto.

## Comparison configurations

The maintained comparison evaluates four stable configurations independently:

- `combined_original` runs the complete original QF-11 implementation above.
- `focused_rules` retains only RSI below the lower threshold (UP), ADX entering
  the positive-DI-controlled zone (UP), RSI stabbing the DI zone from above
  (UP), and RSI stabbing it from below (DOWN). It excludes the negative-DI ADX
  entry, RSI above the upper threshold, and both middle-range candle rules.
- `rsi_oversold_up` predicts UP only when completed-session RSI is strictly
  below its configured lower threshold. Equality is not a prediction. Its
  defaults are RSI(2), threshold 15, and Friday exclusion.
- `always_up` predicts UP on every eligible session. It is a structural-bias
  control, not a proposed trading strategy.

Every configuration uses the same QF-3 dataset, immediate-successor outcome
label, raw-price basis, weekday eligibility, and flat-gap-as-incorrect policy.
An optional weekday-inclusion tuple supports explicit future subset studies;
the default report still shows Monday through Thursday and does not remove a
weekday because it looked weak.

SPY has historically exhibited an upward overnight tendency, so directional
accuracy alone can make an UP-heavy rule look more informative than it is. The
report therefore includes two always-UP comparisons. The all-eligible-session
row describes each configuration alongside the full eligible baseline. The
matched-session row recomputes always-UP statistics on the exact dates when the
other configuration predicted. For an UP-only rule, matched accuracy is simply
the upward-gap frequency on those dates. A DOWN prediction records the
always-UP signed return and the strategy-minus-baseline incremental result for
that same session.

## Outcome and metrics

For signal session `t` and its immediate exchange-calendar successor `t+1`:

```text
overnight gap = open[t+1] / close[t] - 1
gap size      = abs(overnight gap)
signed return = overnight gap       for an up prediction
                -overnight gap      for a down prediction
correct       = signed return > 0
```

All percentage-named fields are decimal ratios: `0.01` means 1%. Any positive
or negative gap is evaluated. An exactly flat gap is counted as incorrect. The
summary records prediction, correct, and incorrect counts; accuracy; average
absolute gap size for correct and incorrect predictions; and average signed
prediction return for each group. A final-session signal is preserved in the
generated-signal count but cannot receive a label and is counted separately.

The comparison study adds raw and signed return means, medians, population
standard deviation, absolute gap magnitude, positive and adverse tail
frequencies at 0.10%, 0.25%, 0.50%, and 1.00%, and correct/incorrect magnitude
splits. It reports 95% Wilson score intervals for accuracy with their sample
counts. These intervals communicate binomial uncertainty; they are not p-values
or evidence that a rule is tradable.

Prediction-level streak statistics include longest correct and incorrect
streaks, counts of incorrect streaks at least three and five predictions long,
and maximum decline in the ordered cumulative signed-gap series. The last is
explicitly labeled a prediction-study statistic, not portfolio drawdown. The 20
best and 20 worst available outcomes per configuration remain visible so an
average cannot conceal dependence on a few large gaps.

## Causal derived features and range summaries

The comparison export exposes typed completed-session columns for RSI, prior
RSI, RSI change, ADX, prior ADX, ADX change, +DI, -DI, current and prior DI
spread, Wilder ATR, ATR divided by close, OHLC, candle return, volume, trailing
average volume, volume ratio, and weekday. Definitions are:

```text
RSI change       = RSI - previous RSI
ADX change       = ADX - previous ADX
DI spread        = positive DI - negative DI
candle return    = close / open - 1
ATR percentage   = Wilder ATR(14) / close
volume ratio     = volume / trailing 20-session average volume
```

All rolling values are trailing and use the current and earlier completed
sessions only. Outcome labels are attached after every strategy has generated
its predictions and cannot influence these features or a rule decision.

Default bins are deterministic and configurable. RSI uses 0–5, 5–10, 10–15,
15–20, then 10-point ranges through 90–100. ADX uses 10-point ranges through
50–60 and 60+. DI spread uses `<-20`, `-20–-10`, `-10–10`, `10–20`, and `20+`.
ATR/close uses `<0.50%`, `0.50–1.00%`, `1.00–1.50%`, `1.50–2.00%`,
`2.00–3.00%`, and `3.00%+`. Volume ratio uses `<0.75`, `0.75–1.00`,
`1.00–1.25`, `1.25–1.50`, `1.50–2.00`, and `2.00+`.

Intervals are left-inclusive and right-exclusive (`[lower, upper)`) except the
final bounded RSI bin, which includes 100. Unbounded labels state their exact
interval convention. Each row reports both eligible observation count and
prediction count; `adequate_sample` is false until the prediction count reaches
the configurable minimum, 30 by default. Empty and inadequate bins are retained
rather than presented as useful ranges.

## Exploratory thresholds and chronological periods

The simple RSI strategy is rerun at the fixed ordered thresholds 5, 10, 15, 20,
25, and 30 while RSI period remains 2. Results include full-period, annual, and
configured chronological-period rows. This study does not mutate the default
threshold or automatically select a winner.

The default periods are development (2020–2022), exploratory validation
(2023–2024), and observed 2025. The periods are nonoverlapping and configurable.
Because all of them, including 2025, have already been inspected, none is a
pristine holdout and the full report remains exploratory in-sample analysis.
Apparent RSI strength must be evaluated for sample adequacy and consistency
across these periods, then tested on untouched data.

## Public API

```python
from pathlib import Path

from quantforge.data import MarketDataCache
from quantforge.prediction import (
    OvernightGapPredictionParameters,
    OvernightGapPredictionStrategy,
    export_prediction_analysis,
    run_prediction_analysis,
)

dataset = MarketDataCache(Path("data/market-data")).load("<dataset-id>")
strategy = OvernightGapPredictionStrategy(OvernightGapPredictionParameters())
result = run_prediction_analysis(dataset, strategy)
artifact_path = export_prediction_analysis(result, Path("reports/predictions"))
```

This original API is a backward-compatible overnight-gap adapter. Its
`PredictionRow`, `PredictionMetrics`, `PredictionAnalysisResult`, and CSV
columns remain the QF-11 gap-specific schema. Internally it delegates ordering
and provenance to `run_prediction_study`, then adapts the typed gap outcome and
directional evaluation back into that public result. Existing callers remain
runnable and numerical results remain directly comparable.

### Native/TA-Lib overnight-gap compatibility example

QF-38 uses the unchanged `OvernightGapPredictionStrategy` as the first
end-to-end backend-impact example. Omitting `backend_id` still produces the
exact legacy native configuration. The comparison helper instead constructs
two explicit strategies with the same logical parameters and dataset, then
reports RSI and directional/ADX value differences together with prediction
dates present on only one backend, matched directions, changed directions,
accuracy, and average prediction-signed return.
Date and direction counts include generated end-of-data signals; metric sample
counts remain limited to predictions with an available next-session outcome.
The helper binds each backend label to the backend identities in its analyzed
required-indicator configurations. It compares the complete logical strategy
and analysis configurations after removing only those backend identity objects.
The in-memory `PredictionAnalysisResult.generated_signals` retains the complete
fixed signal set from the same generic study run used to construct its rows and
metrics. An additional immutable in-memory record snapshot binds every signal,
including end-of-data signals, and every labeled outcome row to that source
run. The backend comparison verifies the snapshot before including signals or
deriving metrics; it never performs a second strategy generation. Each strategy
consumes the exact normalized indicator fields already captured for the value
comparison, and
each computation must match the resolved backend's complete identity rather
than only its stable backend ID, including when the strategy indicators are
constructed later. The prediction rule name, implementation version, and
parameters are snapshotted into both the result and its comparison identity.
Reuse of the precomputed RSI and directional fields is private to the
backend-comparison call: it passes only the computations captured by that
call's source-bound indicator comparisons into its two internal strategy runs.
Standalone precomputed evidence and supplied-output analysis arguments are not
part of the public prediction API, so normal QF-11 analysis identity and export
semantics remain unchanged.
A custom indicator backend registry requires an explicit backend ID so the
strategy configuration records the resolved backend identity; omitting both
continues to preserve the legacy native configuration.

```python
from decimal import Decimal
from pathlib import Path

from quantforge.data import MarketDataCache
from quantforge.indicators import IndicatorComparisonTolerances
from quantforge.prediction import (
    export_overnight_gap_backend_comparison,
    run_overnight_gap_backend_comparison,
)

dataset = MarketDataCache(Path("data/market-data")).load("<dataset-id>")
result = run_overnight_gap_backend_comparison(
    dataset,
    tolerances=IndicatorComparisonTolerances(
        absolute=Decimal("1e-12"),
        relative=Decimal("1e-12"),
    ),
)
artifact_path = export_overnight_gap_backend_comparison(
    result, Path("reports/backend-comparisons")
)
```

The example writes deterministic JSON, CSV, and text artifacts. Its signed
return and accuracy differences remain prediction-study statistics, not fills,
orders, option returns, or evidence that one backend is superior. It performs
no migration and does not change existing native study identities or results.

The pre-merge schema-v2 integrity fix adds QF-3 retrieval time and provider
timezone to prediction manifests. The legacy gap engine/result versions and the
generic study engine therefore advance to `2`; the generic engine also enforces
exact horizons, detects evaluator-input mutation, and creates feature-complete
row identities. The current generic integrity correction advances only the
generic study engine to `3`: a labeler cannot omit an available declared
outcome, and component values are revalidated after all evaluations to detect
delayed mutation. The latest generic-only correction advances that engine to
`4`: warm-up declarations are enforced and returned rows are detached with
immutable primitive snapshots. The current generic-only correction advances
the engine to `5`: signals are fixed and guarded before any labeler receives the
future-bearing dataset. The latest generic-only correction advances the engine
to `6`: outcome and evaluation identities derive directly from their canonical
value snapshots. The next generic-only correction advances the engine to `7`:
signal primitives are captured before validation, and parameter validation reads
from that canonical snapshot. The generic engine advances to `8` to execute
components against an isolated dataset copy and reject any mutation relative to
the validated pristine snapshot. The generic engine advances to `9` to capture
the generated signal tuple once for validation, evaluation, and record counts.
A legacy-adapter identity correction advances only
the legacy gap engine to `3`: each prediction ID includes the complete fixed
causal signal snapshot, while its result schema remains at `2`. A
comparison-only integrity correction advances the comparison engine/result
versions to `3`: custom-period stability labels are chronology-neutral, and
valid zero-prediction results export deterministic header-only CSV artifacts.
The comparison engine then advances to `4` while retaining result schema `3` so
weekday summaries include every observed eligible session day after configured
filters, including weekends on `24/7` calendars. The market-data provenance
correction adds volume basis, adjusted-field usage, and corporate-action policy
to every prediction manifest. It advances the legacy gap engine/result to
`4`/`3`, the comparison engine/result to `5`/`4`, and the generic study engine
to `10` so the identity and schema changes are explicit. The subsequent
coverage-provenance correction includes the complete requested-range
missing-session tuple and advances those versions to legacy `5`/`4`, comparison
`6`/`5`, and generic engine `11`. The generic engine advances to `12` to verify
labeler configuration after dataset validation and every label callback and
evaluator configuration after every evaluation callback. The provenance
corrections intentionally changed legacy analysis and prediction IDs; generic
engine corrections change generic study and row IDs even when numerical
predictions and metrics are unchanged. Existing older immutable artifacts remain
historical records at their original paths; current runs write new identity
paths rather than reusing or overwriting them.

New experiments should use the generic composition API and add their own
study-specific result summaries rather than widening the legacy gap schema:

```python
from quantforge.prediction import PredictionStudy, run_prediction_study

study = PredictionStudy.create(
    strategy=prediction_rule,
    outcome_labeler=outcome_labeler,
    evaluator=evaluator,
    feature_configuration={"feature_schema_version": "1"},
)
study_result = run_prediction_study(dataset, study)
```

The maintained SPY command wraps the same API:

```bash
TIINGO_API_KEY=... uv run python scripts/run_spy_gap_prediction.py
```

By default it requests raw, unadjusted Tiingo EOD bars for SPY from 2020-01-01
through 2025-12-31, stores the immutable QF-3 dataset under
`data/market-data`, and exports the QF-11 analysis under
`reports/predictions`. The JSON summary prints direction counts, accuracy, and
average absolute gap size for correct and incorrect predictions. Repeating the
same run verifies and reuses the exact immutable export.

Reuse a previously cached QF-3 dataset without making a provider request:

```bash
uv run python scripts/run_spy_gap_prediction.py --dataset-id <dataset-id>
```

Use `--refresh` only when intentionally retrieving a new immutable Tiingo
snapshot. Use `--cache-root` and `--output-root` to override the default
locations. The analysis manifest records schema-v4 corporate-action provenance,
including dividend and split counts and the corporate-action snapshot ID. The
baseline period contains no special options handling; all labels use underlying
SPY close-to-next-open prices.

Tiingo ingestion uses raw, unadjusted OHLCV. Ex-dividend price effects therefore
remain part of the observed underlying-price gap; this analysis does not convert
them into total returns. A raw dataset containing a stock split fails closed
because the mechanical price change must not be counted as a predicted gap.
Split-adjusted QF-3 datasets are not subject to that raw-price rejection. These
policies and the action counts are recorded in the manifest.

Run the multi-configuration comparison against the same fixed Tiingo request or
an existing immutable dataset:

```bash
TIINGO_API_KEY=... uv run python scripts/analyze_spy_gap_predictions.py
uv run python scripts/analyze_spy_gap_predictions.py --dataset-id <dataset-id>
```

For either maintained script, a cached dataset must match the exact Tiingo SPY
request: raw unadjusted prices requested from 2020-01-01 through 2025-12-31.
The commands reject cached IDs from another provider, adjustment basis, symbol,
calendar, requested range, or actual coverage rather than silently running a
different experiment. The first and last bars must match the expected XNYS
boundary sessions and no expected session may be missing. The provider-neutral
Python APIs remain available for explicitly configured studies on other QF-3
datasets.

Weekday summaries use the weekdays observed among eligible outcome-bearing
sessions after applying `included_weekdays` and `excluded_weekdays`. XNYS SPY
studies therefore remain on their observed exchange weekdays, while supported
`24/7` datasets retain Saturday and Sunday summaries instead of silently
omitting them. A filter with no matching eligible sessions still produces the
documented header-only empty summary artifact.

The concise JSON output includes each configuration, matched always-UP results,
strongest and weakest years and weekdays, the threshold table, and the export
location. It never prints the API key.

## Export schema

Each legacy overnight-gap analysis exports atomically to
`<output-root>/<analysis-id>/`:

- `manifest.json` records engine and schema versions, complete strategy and
  indicator configuration, QF-3 dataset identity, bar fingerprint, retrieval
  timestamp, provider timezone, OHLC and volume bases, adjusted-field usage,
  corporate-action policy, missing requested sessions, counts, metrics, and
  limitations.
- `predictions.csv` contains one labeled row per evaluable signal.

Prediction rows record:

- stable prediction, dataset, strategy, and strategy-configuration identities;
- QF-3 provider, OHLC and volume bases, adjusted-field usage, and
  corporate-action snapshot/policy and missing-session provenance;
- symbol, signal session, and outcome session;
- direction, originating rule, full parameter set, and contemporaneous feature
  values;
- signal close and next open;
- overnight gap, absolute gap size, prediction-signed return, and correctness.

Exports never contain orders, fills, quantities, option prices, or portfolio
profit and loss. The results measure direction prediction only and remain
in-sample descriptive evidence until evaluated on untouched data.

The generic `PredictionStudyResult` instead serializes component provenance and
typed prediction, outcome, and evaluation payloads. It deliberately has no
universal accuracy or gap metrics because those concepts do not apply to every
prediction study. A new study supplies its own aggregation and immutable export
schema when it becomes reportable.

Each comparison study similarly exports atomically and immutably to
`<output-root>/<comparison-study-id>/` with `manifest.json`, `metrics.json`, and
CSV files for configuration, prediction, rule, weekday, annual, period,
threshold, feature-bin, baseline, best-outcome, and worst-outcome records.
Configuration and dataset fingerprints participate in the study identity.
Repeated runs validate exact bytes before reusing an existing directory.
When a valid configuration produces no prediction rows, row-oriented artifacts
retain their complete CSV headers and zero data rows. Failed exports remove
their temporary directory before returning an error.

Underlying close-to-next-open gaps do not model option spreads, implied
volatility changes, strike selection, contract multipliers, theta, liquidity,
or fills. Neither a favorable average gap nor high directional accuracy proves
option profitability. In particular, RSI below 15 is an exploratory candidate,
not a validated or profitable rule.
