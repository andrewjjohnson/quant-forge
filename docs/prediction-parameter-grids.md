# Deterministic prediction parameter grids

QF-32 adds `quantforge.prediction.PredictionGridStudy` for deterministic,
in-sample comparison of QF-28/QF-31 multi-timeframe prediction studies. It
reuses QF-6 finite search spaces, parameter constraints, candidate-index
neighborhoods, and stability thresholds without routing prediction research
through QF-5 orders, fills, costs, or portfolio accounting.

Grid rankings are exploratory research results. They are not validated
profitability, financial advice, or a substitute for walk-forward and untouched
holdout validation.

## Boundaries

One grid fixes all of the following before any trial executes:

- one immutable QF-3 prediction dataset and explicit QF-14 dataset-family
  fingerprint;
- one versioned prediction-study factory;
- one versioned analyzer that produces metrics and comparison records;
- one identified QF-28 context-provider environment;
- one normalized QF-35 indicator backend environment; and
- ranking, minimum-sample, outcome-quality, and stability policies.

The finite search space may vary indicator periods, thresholds, condition
enablement, compatible timeframe choices, completed/developing context policy,
and outcome horizon or labeler configuration. The factory must express those
choices in the built `PredictionStudy` configuration. Invalid assignments may
be rejected by a QF-6 `ParameterConstraint` or by raising
`InvalidPredictionGridParametersError` during factory construction. Both paths
produce persisted `excluded` trials before prediction execution.

Backend choice is not an ordinary search parameter. Every normalized standard
indicator declared by every built trial must match the fixed
`PredictionIndicatorBackendEnvironment`. Backend ID, wrapper/library version,
runtime library version, function identity, indicator configuration IDs, and
the fixed backend configuration are retained in study/trial identity. Use the
separate backend-comparison workflow for native-versus-TA-Lib comparisons.

## Determinism and identity

Parameter order comes from `PredictionStudyFactory.parameter_order`, not mapping
insertion order. The rightmost searched parameter changes fastest. Candidate,
study, and trial identities use canonical JSON and SHA-256.

The study identity includes the dataset and family fingerprint, complete search
space, pre-execution constraints, factory and analyzer configurations, context
environment, fixed backend environment, ranking rules, and stability rules. A
trial additionally binds its full built prediction definition:

- rule identity, rule parameters, and configuration;
- context requirements and completion policy;
- indicator configuration IDs and complete normalized backend identities;
- outcome-labeler identity and horizon configuration;
- evaluator and feature configuration; and
- dataset-family fingerprint.

Changing any backend or indicator configuration produces a different identity.
It cannot resume an old study or use an incompatible indicator-cache entry.
Factory and analyzer configurations are captured as detached immutable
snapshots before identity hashing. Their names, versions, parameter contracts,
configuration IDs, and configuration content are revalidated during candidate
construction and after analysis, before a successful artifact can be persisted.

## Persistence and resume

The store is `<output-root>/<study-id>/`:

```text
manifest.json
summary.json
trials/<trial-id>.json
artifacts/<trial-id>/prediction-study.json
```

The manifest is written before candidate construction. Candidates are built in
deterministic order and each is immediately persisted as `pending` or
`excluded`, so an interruption during a later factory build retains all earlier
grid state for `resume()`. Each executable trial then transitions through
`running` to `succeeded` or `failed`. Writes use a same-directory temporary file,
`fsync`, and atomic replacement. A failed trial records a sanitized exception
type/message and does not stop later trials. `resume()` skips completed and
excluded trials; failed trials are skipped unless `retry_failed=True` was fixed
in the original manifest. Before retrying, the completed failed attempt is
archived with its sanitized diagnostic and timestamps, so a later success or
interruption cannot erase the prior failure history. Raw exception messages are
not persisted because they may contain provider credentials or account data;
the same safe diagnostic policy applies to factory and trial-definition
exclusions.

Every successful artifact retains the generic QF-11 result together with the
analyzer output. Resume and result loading parse each artifact and verify its
schema, grid study/trial identity, underlying prediction-study identity, and
analyzer output against the immutable trial record. A truncated or modified
artifact is rejected rather than ranked from duplicated trial metadata. The
trial record separately retains a canonical SHA-256 fingerprint of the complete
artifact content, binding prediction rows and analysis evidence to that trial.
Result loading also requires every deterministic candidate to have a terminal
`succeeded`, `failed`, or `excluded` record; interrupted `pending` or `running`
trials must be resumed before any rankings can be produced.

`PredictionTrialAnalysis` requires rankable numeric metrics and
retains period, weekday, matched-baseline, and analyzer-specific artifact
records. Ranking first enforces `minimum_prediction_count`, then all configured
`PredictionMetricConstraint` values. Undefined objectives or quality metrics are
ineligible rather than converted to zero. `PredictionRankingConfig` requires an
explicit baseline name, and every nonempty successful analysis must retain
period, weekday, and matched-baseline records using that baseline.

## Safe reuse

Each execution owns a `PredictionGridExecutionCache`.

- Context bars reuse a key containing the dataset-family fingerprint, context
  provider environment, timeframes, feed/session policies, staleness limits,
  and completed/developing policy. Indicator declarations are intentionally
  excluded from that context-bar key. Before a returned context is cached, its
  QF-14 source-consistency evidence must identify that same dataset family.
- Normalized indicator output reuse additionally binds the QF-20 context ID,
  timeframe, completion policy, indicator configuration ID, complete backend
  function identity, and fixed backend configuration.

The cache calls the existing QF-35-normalized indicator contract. Grid
orchestration does not import or call TA-Lib directly. Cache hits first
revalidate the live indicator metadata against its captured declaration, so a
mutated indicator cannot receive output computed for its earlier configuration.
Every fresh or cached output is then checked against the exact restricted bar
IDs, end timestamps, completion states, dataset-family lineage, timeframe,
indicator/configuration/backend identity, source and output fields, and warm-up
metadata before strategy logic can access it. Custom and stale caches therefore
cannot introduce bars beyond the causal decision boundary.

## Multiple comparisons

Every result warns that the grid searched a stated number of parameter
combinations without a multiple-comparison correction. Stability summaries,
minimum sample sizes, and outcome-quality constraints are research safeguards;
they do not control the false-discovery rate. Treat ranked combinations as
hypotheses until they pass separately configured walk-forward and untouched
holdout validation.

Stability summaries also compare each ranked center with its eligible-neighbor
median in the configured ranking direction. The configured isolated-peak
absolute drop, relative drop, top-rank fraction, neighbor pass fraction, and
boundary rules are all applied. A center that meets the isolated-peak criteria
is explicitly identified and cannot retain a `stable` classification.

## Cached SPY example

The example assumes the caller already loaded an immutable SPY daily prediction
dataset and a local QF-14/QF-20 multi-timeframe context provider. No provider
credentials or network access are needed during trials.

```python
from decimal import Decimal
from pathlib import Path

from quantforge.optimization import (
    BooleanValues,
    FloatValues,
    IntegerValues,
    ParameterSearchSpace,
    StabilityConfig,
    ThresholdOperator,
)
from quantforge.prediction import (
    PredictionContextEnvironment,
    PredictionGridConfig,
    PredictionGridStudy,
    PredictionIndicatorBackendEnvironment,
    PredictionMetricConstraint,
    PredictionRankingConfig,
)

# `confluence_factory` builds a QF-31 PredictionStudy for each mapping. Its
# parameter contract includes period, threshold, enablement, completion-policy,
# timeframe, and outcome-horizon fields. `comparison_analyzer` emits full-period,
# chronological-period, weekday, and matched-baseline records.
grid = PredictionGridStudy(
    dataset=spy_prediction_dataset,
    dataset_family_fingerprint=spy_context_family.family_id,
    study_factory=confluence_factory,
    analyzer=comparison_analyzer,
    context_provider=cached_spy_context_provider,
    context_environment=PredictionContextEnvironment.create(
        "cached_spy_qf20_context",
        "1",
        {
            "dataset_family_id": spy_context_family.family_id,
            "dataset_family_manifest_id": spy_context_family.manifest_id,
        },
    ),
    indicator_backend=PredictionIndicatorBackendEnvironment.create(
        backend_id="native_v1",
        library_name="quantforge",
        library_version="0.1.0",
        contract_version="1",
        configuration={"selection_policy": "fixed_for_ordinary_grid"},
    ),
    config=PredictionGridConfig(
        label="SPY weekly/daily/4h confluence grid",
        search_space=ParameterSearchSpace(
            {
                "daily_rsi_period": IntegerValues((10, 14, 20)),
                "daily_rsi_ceiling": FloatValues(("65", "70", "75")),
                "require_relative_volume": BooleanValues((False, True)),
                "outcome_horizon_sessions": IntegerValues((1, 3, 5)),
            }
        ),
        ranking=PredictionRankingConfig(
            "accuracy_delta_vs_matched_baseline",
            "always_up",
            minimum_prediction_count=30,
            outcome_quality_constraints=(
                PredictionMetricConstraint(
                    "outcome_availability_rate",
                    ThresholdOperator.GREATER_THAN_OR_EQUAL,
                    Decimal("0.95"),
                ),
            ),
        ),
        stability=StabilityConfig(minimum_eligible_neighbors=2),
        output_root=Path("reports/prediction-grids"),
    ),
)

result = grid.run()
resumed = grid.resume()  # completed trials are not rerun
loaded = grid.load_result()  # rebuild ranking/stability without execution
```

The placeholders are application composition objects, not hidden framework
globals. The factory owns how searched fields create typed QF-31 rules and
outcome labelers; the analyzer owns the domain meaning of quality metrics and
the explicit baseline. This keeps orchestration generic while preserving the
full typed QF-11/QF-28 study definition in every trial.
