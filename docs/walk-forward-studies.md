# Deterministic walk-forward studies

QF-39 adds `quantforge.walk_forward`: chronological orchestration that ends at
**typed per-fold out-of-sample artifacts**. It consumes the explicit QF-8
`ValidationPlan`; it does not create calendar splits or change either evaluator.
QF-40 owns combining OOS results and final holdout consumption. QF-9's generic
manifest/index infrastructure and QF-41's reports are outside this implementation.

## Study contracts and entry points

`WalkForwardStudy(config, evaluator, output_root)` validates the plan against a
finite `CandidateUniverse` before execution. `run()` starts a new study;
`resume()` verifies existing state and completes compatible unfinished folds.
Both return `WalkForwardResult`, containing ordered `FoldResult` records with
status, optional `FrozenSelection`, typed OOS artifact, and failure history.

```python
from pathlib import Path
from quantforge.walk_forward import (
    BacktestEvaluator,
    SelectionPolicy,
    WalkForwardConfig,
    WalkForwardStudy,
)

# dataset, strategy_factory, grid_config, and plan are existing QF-3/QF-6/QF-8
# objects. grid_config.backtest must have no preassigned evaluation interval.
config = WalkForwardConfig(
    name="moving-average walk-forward",
    plan=plan,
    selection_policy=SelectionPolicy.BEST_ELIGIBLE,
    minimum_training_observations=20,
    minimum_test_observations=5,
    continue_on_failure=True,
    retry_failed=True,
)
study = WalkForwardStudy(
    config,
    BacktestEvaluator(dataset, strategy_factory, grid_config),
    Path("reports/walk-forward"),
)
result = study.run()
# Reconstruct the identical configuration after interruption:
resumed = study.resume()
```

The prediction adapter takes the existing window analyzer, factory, grid
configuration and fixed backend environment:

```python
from quantforge.walk_forward import PredictionEvaluator

adapter = PredictionEvaluator(
    dataset=prediction_dataset,
    series=validated_context_series,
    primary_timeframe=primary_timeframe,
    study_factory=prediction_factory,
    analyzer=window_analyzer,
    indicator_backend=fixed_backend_environment,
    grid_config=prediction_grid_config,
)
prediction_study = WalkForwardStudy(
    prediction_walk_forward_config, adapter, Path("reports/walk-forward")
)
```

The inputs are already loaded immutable artifacts. No provider request is made.
The QF-8 plan must capture the separate QF-3 prediction dataset and context
family. The latter must contain exactly one selected source for each configured
timeframe. The adapter uses QF-8's existing session-indexed two-input contract;
QF-42 schedules intraday primary-bar ends inside those permitted sessions.
Backtests retain the QF-5 daily-data contract, including QF-8 timestamp boundaries
when those keys are the observed daily exchange closes.

## Candidate universe and selection

The universe captures the existing QF-6/QF-32 Cartesian combinations, constraints,
factory name/version/configuration, complete executable candidate definitions,
ranking and stability policies, and fixed backend provenance. Excluded combinations
remain exclusions in the underlying grid; they cannot be frozen for OOS.
QF-32 exposes an additive `PredictionGridStudy.candidates` property to enumerate
its existing validated definitions without executing a prediction. Its
window-specific schedule is bound separately from each logical configuration.

The QF-8 research environment declares the union of **permitted** indicator
configurations and outcome definitions. Its captured rule defines the family and
implementation version; candidate parameters may differ. Each candidate's exact
indicator bindings, context counts and rule warm-up are checked through QF-8
using that candidate's rule provenance. Outcome definitions must be present in
the plan, whose purge horizon already equals their maximum reach. An undeclared
indicator, larger context requirement or different outcome fails construction.

All indicators in the permitted universe must retain one fixed backend identity
(including library/runtime versions). Backend-named search axes are rejected.
Native legacy configurations remain explicit; they are not silently migrated.
Use QF-38 for an intentional backend comparison.

There are two explicit policies, each selecting **one** configuration:

- `BEST_ELIGIBLE` selects the first candidate in the existing eligible objective
  ranking, after its sample-size/quality/risk constraints and deterministic ties.
- `FIRST_STABLE` scans that same order for a candidate classified `STABLE` and
  non-isolated by the existing stability calculation. No eligible stable
  candidate means failure; there is no best-return fallback.

With an optional selection window, ranking uses only that window. Otherwise it
uses development. Development and selection membership are both recorded and
purged separately. These are finite, preconfigured rules and strategies, without
model fitting: development and selection scores are not pooled or averaged.
Historical development observations may supply only the explicitly declared
pre-selection context. Test metrics never enter a ranking callback.

## Windows and temporal boundaries

QF-8 owns fold validation, ordering, closed intervals and mode semantics.
Expanding folds keep the development start fixed and advance its end. Rolling
folds advance both boundaries, using the explicitly configured bounded history.
QF-39 iterates `plan.folds` in that order and never derives a second schedule.

For each role, `select_window_observations()` supplies the exact separate context
and study membership, and `purge_partition_observations()` supplies retained and
purged observations. This includes test-tail purging against the next test or
reserved final holdout. Missing endpoints, insufficient context, empty retained
partitions and configured minimum-count failures fail closed.

A small in-memory QF-3 projection retains unchanged bars from the first permitted
warm-up observation through the last retained observation plus the plan's maximum
outcome horizon. QF-3 canonical bar/action hashing and validation rebind its
metadata to that range. Protected prices, actions, counts and range metadata are
absent from evaluator inputs. Synthetic retrieval time is the epoch unavailable
sentinel, and adapter provenance identifies `qf39-bounded-projection-v1`.
Synthetic canonical paths do not claim to be stored provider responses. Original
source metadata, family and fingerprints remain in the enclosing study/freeze.

Prediction context uses `select_prediction_context_observations()` independently
at every QF-42 decision. Only its exact completed context/study source bars reach
QF-20/QF-28. Existing QF-21 source evidence is retained for declared developing-as-of
reconstruction; QF-21/QF-28 still own that causal computation. The provider rejects
a request outside the permitted QF-42 schedule. A schedule containing a session
absent from the QF-3 membership fails explicitly instead of silently dropping it.

QF-32 analyzes the entire permitted QF-42 collection for each candidate. The test
adapter reconstructs the frozen logical definition and invokes
`run_prediction_window()` with the following test schedule. Original ordered
QF-11 study, context, signal and row identities remain intact. A test collection
without any labeled, non-rejected signal fails explicitly; skipped and rejected
records otherwise remain in the original window evidence.

Backtest grids receive the bounded dataset and a QF-43 `EvaluationInterval`.
The selected strategy then runs through QF-5 with the test interval. Context bars
initialize indicator/strategy history; QF-43 keeps them out of signals eligible
for orders, fills, accounting, benchmark and metrics. Every test starts with
configured capital and no positions. Next-open execution, commissions, fees,
slippage and corporate-action policies are unchanged. No account is carried
between windows.

The final holdout participates only in QF-8's reservation and boundary checks.
Its bar chronology/provenance may certify a purge cutoff; its prices, labels,
features and metrics never enter an evaluator input or OOS artifact.

## Frozen selection and identity

A `FrozenSelection` contains a deeply immutable canonical snapshot of:

- study, validation-plan, fold and candidate-universe IDs;
- the entire fixed study/adapter/research definition and exact QF-8 membership;
- selected combination and training trial IDs, complete searched parameters and
  executable rule/strategy/outcome/evaluator/indicator configuration;
- original source fingerprints, timeframe/session/aggregation/backend policy;
- ranking/stability evidence and backtest execution/cost/reset policy.

The snapshot is persisted before test evaluation. It is compared with the
candidate universe before reconstruction and checked again after execution.
Changing test results cannot revise it. Component/configuration drift is checked
around adapter calls; QF-5/QF-11/QF-32 retain their own callback mutation guards.
As with those engines, user-supplied Python callbacks must be trusted code:
input boundaries do not sandbox unrelated global variables or external I/O.

Study identity uses canonical JSON and SHA-256 and includes plan, explicit
schedule/mode, universe, selection/minimum-count/failure policies, adapter and
underlying engine versions. Paths are not scientific inputs. Frozen selection
and OOS content identities additionally bind exact membership and underlying
result identities. A new immutable source revision requires a new plan/study.
Exposing a longer prefix of the same immutable context source preserves historical
fold membership, configurations and results.

## Artifacts, persistence and failure behavior

```text
<output-root>/<study-id>/manifest.json
<output-root>/<study-id>/folds/<qf8-fold-id>/
    state.json
    selection.json
    selection/<qf6-or-qf32-study-id>/...  # Existing grid persistence
    test/<qf5-run-id>/...                # Backtest's existing immutable export
    oos.json
```

QF-39 JSON records are canonical payloads with SHA-256 envelopes. Writes use a
same-directory temporary file, flush/fsync and atomic replacement. Selection and
OOS payloads are immutable. This is a local, single-writer study store, not the
future QF-9 generic manifest service; concurrent writers are unsupported.

The state progression is `pending → selecting → selection_frozen → evaluating_test
→ completed`, with explicit `failed` states and retained numbered failure
attempts. `selection.json` and its state transition are durable before any test
callback. An interruption between the snapshot and state writes can recover the
verified snapshot. An interrupted or partially persisted result is never inferred
to be completed merely because an artifact exists.

Compatible completed folds verify frozen selection, content fingerprints and
underlying artifact integrity without rerunning selection or test. Prediction
loads reuse QF-42's offline window validator with the expected schedule,
configuration, sessions and provenance. Backtests validate the existing QF-5
immutable export and its integrity fingerprint. Incompatible/corrupt completed
artifacts raise an error instead of being silently invalidated or replaced.

Failed folds retain safe exception type/stage diagnostics; raw external exception
messages are omitted. With `continue_on_failure=True`, later folds proceed.
Otherwise the call stops with the failed fold recorded. Failed folds retry only
when `retry_failed=True` was configured. A frozen selection is reused on a test
retry; selection never runs again for that fold. An existing stale OOS artifact
cannot turn a failed attempt into success: test execution must succeed again and
match any already-persisted immutable evidence.

No eligible candidate, all failed trials, insufficient training/test observations,
missing endpoints/context, an empty prediction test, evaluator failure and write
failure are explicit failures, without a favorable fallback or previous-fold
selection. Underlying grid trials retain their own failure/exclusion records and
retry policy. If even the failure-state write fails, the persistence error
propagates, leaving the last trustworthy state incomplete for recovery.

No cross-window equity, combined prediction statistic, final performance
conclusion, holdout ledger, report renderer or new optimizer is produced.

QF-40 now provides the separate consumer described in
[`oos-holdout-aggregation.md`](oos-holdout-aggregation.md). Its explicit holdout
operation reuses the adapters' additive `evaluate_partition()` and
`validate_partition_artifact()` entry points with validated final-holdout
membership. Ordinary QF-39 `evaluate()` and `validate_artifact()` delegate to those
same paths with their original test membership; no QF-39 state/schema/identity or
selection behavior changes. The structural `EvaluationPartition` contract accepts
QF-39's unchanged `PermittedPartition` or QF-40's separate `HoldoutPartition`. The
latter records its own tail policy instead of inventing a QF-8 purge against a
later protected window.
