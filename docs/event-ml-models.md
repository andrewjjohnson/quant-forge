# Chronological event ML models (QF-68)

> One model is fitted on **one QF-8 fold's development rows only**, frozen into
> an inspectable envelope, and only then applied to that fold's walk-forward
> test rows. The final holdout needs the existing QF-40 consumed-ledger
> authorization. Outputs are descriptive classification evidence on a
> conditional event population and a raw-return target without costs. They
> are not evidence of an edge or of profitable trading.

`quantforge.ml.modeling` consumes persisted QF-67 datasets (see
[event ML datasets](event-ml-datasets.md)) through their public interface:
`feature_columns`, `numeric_feature_rows()`, `target`, `partition_membership`
and the documented scientific manifest. It never reads prediction windows,
re-runs strategies or parses strategy-specific fields. See
[ADR 0040](decisions/0040-train-event-models-chronologically.md).

```text
persisted QF-67 dataset ──read_event_dataset (offline validation)──┐
exact QF-8 plan + fold_id ─────────────────────────────────────────┤
                                                                   ▼
fit_event_model: bind plan/fold/windows ─ development rows only ─ exclusions
   └─ preprocessing + L2 logistic regression fitted on the same fitting rows
                                                                   ▼
FittedEventModel (unfrozen) ── predict_selection (selection rows)
          │
freeze_event_model ──► <root>/<model-id>/model.json (selection evidence bound)
          │
read_event_model ──► FrozenEventModel (envelope re-validated at every use)
          ├─ predict_out_of_sample (walk-forward test rows)
          └─ predict_final_holdout (+ workspace ledger + consumed HoldoutEvaluation)
                                                                   ▼
export_prediction_set ──► <root>/<prediction-set-id>/{manifest.json, predictions.jsonl}
reproduce_prediction_set ──► offline regeneration within the numerical contract
```

## Public API

```python
from decimal import Decimal
from pathlib import Path

from quantforge.ml.modeling import (
    ModelConfiguration,
    export_prediction_set,
    fit_event_model,
    freeze_event_model,
    predict_final_holdout,
    predict_out_of_sample,
    predict_selection,
    read_event_model,
    reproduce_prediction_set,
)

configuration = ModelConfiguration(
    "conditional_logistic",  # name and version: part of identity
    "1",
    minimum_training_observations=20,  # fitting rows needed (default 20)
    class_threshold=None,  # optional fixed threshold, e.g. Decimal("0.5")
)
dataset = Path("reports/ml-datasets") / dataset_id  # a QF-67 export
result = fit_event_model(
    dataset, plan=plan, fold_id=plan.folds[0].fold_id, configuration=configuration
)
if not result.fitted:  # an explicit TrainingStatus and detail, never a fallback
    raise SystemExit(f"{result.status}: {result.detail}")
selection = predict_selection(result.model, dataset)
model_path = freeze_event_model(result.model, dataset=dataset, output_root=root / "models")
model = read_event_model(model_path)  # the only way to obtain a FrozenEventModel
oos = predict_out_of_sample(model, dataset)
path = export_prediction_set(oos, root / "predictions")
report = reproduce_prediction_set(path, model=model, dataset=dataset)  # offline
# After explicit QF-40 consumption and a QF-67 rebuild with final_holdout=...:
holdout = predict_final_holdout(
    model, extended_dataset, workspace=Path("."), evaluation=consumed_evaluation
)
```

`plan` is the exact QF-8 plan the dataset was built under. Errors raise
subclasses of `EventModelError`; QF-67 dataset failures raise `EventDatasetError`.

| Error | Meaning |
| --- | --- |
| `ModelConfigurationError` | Invalid or unsupported configuration |
| `ModelBindingError` | Dataset, plan, fold, window, schema/order, target, population or model mismatch |
| `ModelLeakageError` | A row outside its window, or a training label reaching the cutoff |
| `ModelHoldoutError` | Holdout rows without consumed-ledger authorization |
| `ModelInputError` | A value that is not finite binary64 (before or after scaling) |
| `ModelFreezeError` | Out-of-sample or holdout inference without a validated frozen model |
| `ModelArtifactIntegrityError`, `PredictionIntegrityError` | Corrupt, duplicated, missing, misassigned or wrongly bound artifacts |

Data limitations are not exceptions. They return a `TrainingResult` without a
model and with one of these statuses:

- `no_training_observations`, including the dataset's recorded source
  exclusion (for example `role_not_executed_by_qf39_selection`);
- `insufficient_training_observations`;
- `single_class_training_labels`;
- `feature_entirely_missing_in_training`;
- `all_features_constant_in_training`;
- `estimator_did_not_converge`;
- `estimator_failed`.

The estimator type is never changed.

## Chronological membership

A model binds one fold of the plan. Roles are fixed:

| Role | Use |
| --- | --- |
| `development_training` of the fold | Fits preprocessing and the estimator |
| `validation_selection` of the fold | Selection predictions and metrics (nothing is selected) |
| `walk_forward_test` of the fold | Out-of-sample predictions after freezing |
| `final_holdout` | Only with consumed-ledger authorization |

There is no random split, no shuffling and no role reassignment. Rows of other
folds are ignored. Multiple folds mean one model per fold; with QF-8
`expanding` windows, later development windows already contain earlier data.

**Binding checks** (each fails closed):

- The dataset's `plan_id` and `final_holdout_id` equal the plan's.
- Each source of the fold, and the holdout source, has the plan's
  `qf8_window_id` for its role and fold index.
- Every row of the fold and of the holdout lies inside its role's closed plan
  window.
- The target's outcome reach does not exceed the plan's purged label horizon.

**Training cutoff.** Training labels must be available before the fold's first
protected instant. Timestamp order alone is insufficient; the existing QF-8
purge rule applies:

```text
decision + label_horizon + embargo < protected_start   (equality is leakage)
```

`protected_start` is the selection window's start, or the test window's start
when the fold declares no selection window. `label_horizon` is the plan's
purged maximum outcome reach, which covers the target's reach. The same reach
must also end before the final holdout. QF-67 builds rows only from QF-8
purged membership, so a violation means a mismatched plan or a corrupt
artifact: training raises `ModelLeakageError` and does not drop the row.

**Fitting rows.** Every development row of the fold is either fitted or
excluded with a recorded reason:

- `target_unavailable` (detail: the QF-46 status). Unavailable labels stay
  unavailable and never become negatives.
- `null_feature_value` (detail: the null features), under
  `exclude_observation` only.

Preprocessing and the estimator fit on exactly the same fitting rows. Their
observation IDs and decision timestamps are persisted, together with every
exclusion, its reason, and the dataset's source exclusions for this fold's
development role.

## Preprocessing and estimator

All learned state comes from the fitting rows only. It is plain JSON:

| Step | Rule |
| --- | --- |
| Numeric view | QF-67 `numeric_feature_rows()`: decimals `float(Decimal(text))`, integers exact, booleans 1.0/0.0, nulls `None` |
| Missing features | `exclude_observation` (default): excluded from fitting; an evaluation row stays unscored. `impute_training_mean`: the fitting mean of the non-null values, in raw units, replaces a null in any partition. A feature with no non-null fitting value fails training. |
| Scaling | `(x - center) / scale`, with the fitting mean and population standard deviation (`ddof = 0`) |
| Constant features | Identical over fitting rows: center = that value, scale = 1, coefficient fixed at exactly `0.0`, not passed to the estimator. If all features are constant, training fails. |
| Non-finite values | Rejected (`ModelInputError`), for example a decimal beyond binary64 range or an overflow after scaling. Nothing is clipped, winsorized, zero-filled or forward-filled. |
| Feature selection, encoding, reduction | None |

Preprocessing arithmetic is pure-Python binary64. It uses `math.fsum` and
`math.sqrt`, which are correctly rounded, so the fitted preprocessing state is
identical across platforms.

**Estimator.** scikit-learn `LogisticRegression` with fixed hyperparameters:

- `lbfgs`, L2 penalty (`l1_ratio = 0`);
- default `C = 1`, `tol = 1e-8`, `max_iter = 1000`;
- fitted and unpenalized intercept, no class weights;
- `random_state = 0`, recorded and passed (`lbfgs` uses no randomness);
- one BLAS thread during the fit, via `threadpoolctl`.

scikit-learn only fits. The persisted intercept and coefficients (standardized
units, in model column order) drive every probability, in pure Python:

```text
linear = fsum([intercept, coef_1 * z_1, ..., coef_k * z_k])
p = 1 / (1 + exp(-linear))           if linear >= 0
p = exp(linear) / (1 + exp(linear))  otherwise
```

A test checks this against scikit-learn's own `predict_proba` within 1e-12.
There is no hyperparameter or threshold search. An optional `class_threshold`
is fixed in the configuration before fitting: `p >= threshold` is the positive
class.

## Identities

All identities are SHA-256 of canonical JSON:

| Identity | Binds |
| --- | --- |
| `ModelConfiguration.configuration_id` | Estimator hyperparameters and seed, preprocessing policy, minimums, threshold, calibration bins, log-loss clip |
| `model_configuration_id` | The configuration plus the dataset ID, population ID, explicit feature schema and column order, bound target, fold chronology (plan, fold, windows, horizon, embargo, cutoff), and material runtime versions (Python major.minor, NumPy, SciPy, scikit-learn) |
| `fitted_state_id` | Fitting membership, preprocessing state, estimator state and baseline only, independent of every non-training row |
| `model_id` | The complete fitted model (binding plus fitted state); also the frozen directory name |
| `prediction_set_id` | Bindings, authorization, evaluation settings, ordered records (SHA-256), summary, metrics and baseline |

The operating system and CPU are provenance only (`provenance.platform` in the
envelope). Changing any selection, test or holdout value changes the dataset
ID and therefore `model_id`. It never changes `fitted_state_id`; a test proves
this.

## Freezing and the model envelope

```text
<output-root>/<model-id>/
  model.json   {"payload": {...}, "fingerprint": sha256(payload)} (canonical line)
```

The payload holds the following:

- `component`, `schema_version`, `model_id`;
- `lifecycle` (`state: frozen`);
- `model`: the bound configuration, the training membership with every fitting
  observation, the preprocessing state, the estimator state, the baseline and
  their identities;
- `selection`: the selection prediction set ID, summary, metrics and baseline
  metrics, regenerated from the dataset at freeze time;
- `provenance` and `interpretation`.

`freeze_event_model` writes and fsyncs in a private staging directory and
validates the staged bytes. It then atomically renames the directory to
`<model-id>` and reuses an identical existing directory. Output is refused
inside `reports/holdout-ledger` and for `.rapid.json` paths. No pickle, joblib
or library object is ever written or read.

`read_event_model` re-checks the following:

- the fingerprint, the component and the identity against content and
  directory name;
- every nested identity (configuration, dataset binding, schema and target
  identities);
- constant features having a zero coefficient;
- every fitting observation against the training cutoff.

It is the only constructor of `FrozenEventModel`. Out-of-sample and holdout
inference re-read the envelope at its path on every call and require equality.
An unfrozen `FittedEventModel`, a directly constructed or `replace`d object, or
a deleted or edited envelope raises `ModelFreezeError`. A frozen model never
refits or selects anything; any change is a new model.

## Predictions

`predict_selection` accepts a fitted or frozen model. `predict_out_of_sample`
accepts only a frozen model and the model's exact training dataset. Each set
scores every row of its partition, in dataset row order. Each record holds:

- `row_index`, `source_observation_id`, `decision_timestamp`,
  `partition_role` and `fold_id`;
- `score_status`: `scored`, or `unscored_null_feature` under
  `exclude_observation`;
- `probability`, and `predicted_class` (only with a frozen threshold);
- `imputed_features`;
- `target` and `target_status`, recorded where evaluation is permitted.

An unavailable label does not prevent scoring. The summary reports scored
rows, unscored rows by reason, labeled scored rows (the metric rows) and
unlabeled scored rows by status, separately.

```text
<output-root>/<prediction-set-id>/
  manifest.json      canonical fingerprinted payload (bindings, summary, metrics, baseline)
  predictions.jsonl  one canonical record per line
```

`read_prediction_set` validates the following offline:

- the fingerprint, file hash, record types and probabilities in `[0, 1]`;
- unique observations and strictly increasing row order;
- a single role and fold, and holdout authorization;
- classes against the threshold;
- the recomputed summary, metrics and identity.

`reproduce_prediction_set` regenerates the set from the frozen model and
dataset. It fails on a missing, extra, duplicated or reassigned row, on a
binding to another model, dataset, fold or role, or on a status, class or
label difference.

## Metrics and baseline

Metrics are computed in pure Python over scored rows with available labels:

| Metric | Convention |
| --- | --- |
| Log loss | Mean of `-(y ln p + (1-y) ln(1-p))`; `p` clipped to `[1e-15, 1 - 1e-15]` in binary64 |
| Brier score | Mean of `(p - y)^2`, unclipped |
| ROC AUC | Mann-Whitney with ties counted 1/2, from exact doubled integer ranks; undefined without both classes |
| Calibration | `B` equal-width bins on `[0, 1]` (default 10; bin `min(floor(pB), B-1)`) with count, mean probability and observed rate (null if empty), plus mean probability minus prevalence |
| Precision and recall | Only with a frozen threshold; undefined with no predicted positives or no actual positives |
| Observed prevalence, mean probability | Plain means |

Undefined metrics are explicit and are never numbers:

| Reason | When |
| --- | --- |
| `no_labeled_observations` | No labeled scored rows |
| `insufficient_labeled_observations` | Fewer than `minimum_evaluation_observations` (default 1) |
| `single_class_labels` | ROC AUC with one class |
| `no_frozen_threshold` | Precision and recall without a threshold |

**Baseline.** A constant probability equal to the training prevalence (fitting
positives / fitting rows). It is evaluated on exactly the same labeled scored
rows with the same metrics. It is a reference only: nothing asserts that the
model must beat it, and neither classification metric says anything about
returns after costs.

## Final holdout

Training and selection never use holdout rows, even rows of an already consumed
holdout that are physically present in the dataset; a test shows training on
such a dataset yields the same `fitted_state_id`. `predict_out_of_sample`
never returns holdout rows.

`predict_final_holdout(model, dataset, workspace=..., evaluation=...)` requires
all of the following:

1. A `FrozenEventModel`, re-validated from its envelope.
2. A dataset compatible with the model. The population, feature schema and
   order, target, plan, holdout and outcome reach must be equal, and the
   QF-67 logical-row digest of every row outside the holdout must equal the
   training dataset's. The intended lifecycle is to freeze first, consume the
   holdout, then rebuild the same QF-67 dataset with
   `final_holdout=evaluation`.
3. The workspace's existing permanent ledger (never created) and a
   `HoldoutEvaluation`. `HoldoutLedger.result(evaluation)` must succeed, so the
   ledger must already record that exact request as consumed.
4. The dataset's holdout source must carry that consumption's `request_id`,
   and the evaluation must belong to the model's plan, holdout and population.

A role string or the physical presence of rows is never authorization. Holdout
inference and its reproduction only read the ledger. They never reserve,
consume, reset or recreate state, and the ledger's existing locks still apply.

The ledger records the strategy holdout's consumption. It does not record an
ML-specific exposure or the order of model freezing relative to consumption, so
freeze models before consuming a holdout and never retrain after inspecting
holdout predictions. QF-72 rapid results cannot reach this layer: dataset,
output and evaluation inputs are refused as non-authoritative.

## Offline reproduction and the numerical contract

Persisted datasets, model envelopes and prediction sets regenerate predictions
with no network, provider or library deserialization:

- **Storage.** Probabilities, coefficients and scaling state are JSON
  binary64, in shortest round-trip text, so reading them back is exact.
- **Same platform.** Regenerated probabilities are bitwise identical, so the
  prediction set ID is identical too.
- **Across platforms.** Every operation except `exp` is correctly rounded;
  libm `exp` may differ by one unit in the last place. Regenerated
  probabilities must therefore match within an absolute `1e-12`
  (`PROBABILITY_TOLERANCE`), and `PredictionReproduction.exact` reports
  bitwise agreement. A probability within the tolerance of a frozen threshold
  could change its class; that mismatch is reported, not ignored.
- **Retraining.** Repeating training on identical inputs gives identical
  identities with the same library versions on one platform (tested).
  `lbfgs` coefficients may differ in the last bits across BLAS or CPU
  platforms. That changes `fitted_state_id` and `model_id`, but not
  `model_configuration_id`.
- **Forgery.** A fully re-identified forgery of coefficients is detected only
  by retraining with the same dataset, plan and configuration.

## Example artifact (synthetic fixture)

This example uses the unit fixture: a synthetic QF-67 dataset on the QF-48
timestamp plan fixture, fold 0, 34 development, 10 selection and 10 test
rows. Selected fields:

```text
<root>/c603a681e08390a1298e290a62ea133dce05ae689d6816f1dc67570be1e74ac6/model.json  18,701 bytes
<root>/88179094755b0a6155ebb4858eb652cde36f22c9b860a33a1a8e183294f21df6/  manifest 5,708 + predictions 4,146 bytes

chronology: development 2024-07-08T14:00..15:55Z, selection 16:00..16:55Z,
  test 17:00..17:55Z, label_horizon 35m, embargo 0, training_cutoff 2024-07-08T16:00:00Z
runtime: {"python": "CPython 3.13", "numpy": "2.5.1", "scipy": "1.18.1", "scikit_learn": "1.9.1"}
preprocessing: momentum center -0.008823529411764706 scale 0.19609068589156806
estimator: intercept -0.33212116107819906, coefficients momentum 1.0902453818438151,
  range_width -0.2637084746556541, bar_count 0.3003502327306231, gap_up 0.7032928811119984 (9 iterations)
baseline: 15 / 34 = 0.4411764705882353
training membership: 34 rows, 34 fitting (15 positive, 19 negative), no exclusions

predictions.jsonl (first record):
{"decision_timestamp":"2024-07-08T17:00:00+00:00","fold_id":"9a98640f…","imputed_features":0,
 "partition_role":"walk_forward_test","predicted_class":null,"probability":0.43373905941352575,
 "row_index":44,"score_status":"scored","source_observation_id":"f14a27f4…","target":false,
 "target_status":"available"}

test metrics:     log loss 0.4717, Brier 0.1526, ROC AUC 0.8333 (10 labeled)
baseline metrics: log loss 0.7238, Brier 0.2652, ROC AUC 0.5
```

These synthetic labels follow the synthetic features by construction, so the
numbers show only that the plumbing works.

## Real QF-45 data: no training role

The real QF-45 12/60 event dataset (`69babfca…`, built by QF-67) has 18
selection and 12 test rows and **no development rows**. QF-39 executed its
trials on the selection window, so development was never run; QF-67 records
the exclusion as `role_not_executed_by_qf39_selection`.

The ignored harness `reports/qf68-event-models/qf68_real.py` checks this. It
loads the QF-45 plan from cached inputs, without network, and attempts
`fit_event_model` on the persisted dataset:

- The plan, fold and window binding pass.
- The result is `no_training_observations` with that exclusion reason, and no
  model.
- The permanent ledger is never opened; it is `stat`-identical before and
  after.

QF-68 does not relabel selection rows as training rows, train on test rows or
change the frozen QF-45 study to create training data. A real ML study, with a
study design that produces training rows, belongs to QF-69.

## Measurements

Synthetic scale, not evidence. The ignored harness
`reports/qf68-event-models/qf68_scale.py` ran on one Apple-silicon host with
Python 3.13 and scikit-learn 1.9.1. The dataset had 27,000 rows: 17,000
development, 5,000 selection and 5,000 test.

| Step | Seconds |
| --- | ---: |
| QF-67 build, export and validation | 4.26 |
| `fit_event_model` (includes QF-67 offline read, about 1 s) | 2.46 |
| `freeze_event_model` (includes selection scoring) | 1.54 |
| `read_event_model` | 0.12 |
| `predict_out_of_sample` (5,000 rows) | 1.33 |
| `export_prediction_set` | 0.17 |
| `reproduce_prediction_set` (exact) | 1.71 |

Sizes: `model.json` 2.4 MB (it lists every fitting observation),
`predictions.jsonl` 2.1 MB, peak RSS 0.5 GB. Most of the remaining cost is
QF-67's offline dataset read, which every entry point repeats.

## Limitations

- **Conditional and descriptive.** Models describe a conditional event
  population against a raw-return target without costs. Classification
  metrics are not trading performance, and no result is evidence of an edge.
  Every relationship remains a hypothesis until untouched out-of-sample
  validation.
- **One estimator.** One fixed L2 logistic regression and probability-first
  evaluation. There is no model zoo, hyperparameter or threshold search,
  calibration fitting or feature selection.
- **Binary targets only.** QF-67 supports only the binary forward-return
  target. A continuous kind would add its own estimator configuration, record
  value type and metric set. The lifecycle, chronology, freeze, holdout
  authorization and identities would stay as they are.
- **Selection evidence.** Selection predictions are persisted with
  `export_prediction_set`. The frozen envelope records their ID, summary and
  metrics, so an exported selection set can be checked against it, but
  freezing does not write the set itself.
- **Labels are not independent.** Intraday events can share outcome windows,
  so metrics have no sampling-error estimates.
- **Small partitions.** Selection or test partitions can be tiny or
  single-class; their metrics are explicitly undefined rather than estimated.
- **Holdout exposure.** The ledger has no ML-specific exposure record (see
  [Final holdout](#final-holdout)).
- **Cross-platform fits.** Coefficients may differ in the last bits across
  BLAS or CPU platforms.
- **Out of scope:** QF-69 studies, QF-70 dense datasets, QF-71 discovery, broad
  hyperparameter search, AutoML, neural networks, live inference, execution,
  position sizing, options and brokers.
