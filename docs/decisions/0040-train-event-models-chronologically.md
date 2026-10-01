# ADR 0040: Train event models chronologically behind a persisted freeze

- Status: Accepted
- Date: 2026-10-01
- Jira: [QF-68](https://frostfiredigital-37308542.atlassian.net/browse/QF-68)

## Context

QF-67 produces verified conditional event datasets. Each row has causal
features, a target that may be unavailable, and QF-8 role and fold membership.
QF-69 (conditional ML) and QF-71 (dense ML) need a shared way to train,
evaluate, freeze and reproduce models on those datasets. The risks are:

- **Preprocessing or label leakage.** Scalers, imputers or labels fitted
  across partitions, or training labels that resolve inside a protected
  window.
- **Role drift.** Random splits, shuffling, or relabeling selection or test
  rows as training rows to create data.
- **Out-of-sample use before freezing.** A caller convention such as
  `frozen=True` that does not prove what was frozen.
- **Holdout contamination.** Rows physically present in a dataset being
  treated as permission to evaluate them.
- **Opaque or unsafe artifacts.** Pickle or joblib payloads that cannot be
  inspected and are unsafe to load.
- **Platform-dependent predictions.** Library scoring paths that reproduce
  only approximately.

The real QF-45 dataset has selection and test rows only: QF-39 did not execute
its development window.

## Decision

Add `quantforge.ml.modeling` above the QF-67 dataset boundary.

- **Inputs.** Persisted QF-67 dataset directories are validated by
  `read_event_dataset` at every entry point. Together with the exact QF-8 plan
  and one `fold_id`, these are the only data inputs. Only the public dataset
  interface is used. QF-72 rapid inputs are refused.
- **Fixed roles.** The fold's `development_training` rows train the model. Its
  `validation_selection` rows yield selection predictions. Its
  `walk_forward_test` rows are out-of-sample. The plan's final holdout is
  scored only with authorization.
- **Chronology.** The dataset's plan, holdout and window IDs must match the
  plan. Every row of the fold must lie inside its role's window. Training rows
  must satisfy the QF-8 purge rule
  `decision + label_horizon + embargo < protected_start`; a violation fails
  closed rather than dropping the row.
- **Fitting rows.** Unavailable labels and, by default, rows with null features
  are excluded with recorded reasons. Labels are never converted, and nothing
  is zero-filled. Preprocessing and the estimator fit on exactly the same
  persisted fitting observations.
- **Preprocessing and estimator.**
  - Preprocessing is pure-Python standardization with optional training-mean
    imputation; constant features get a coefficient of exactly zero. Its
    arithmetic is correctly rounded, so the state is identical everywhere.
  - The estimator is one fixed L2 logistic regression from scikit-learn
    (`lbfgs`, seed recorded, one BLAS thread). scikit-learn is added as an
    explicit dependency, together with `threadpoolctl`.
- **Transparent scoring.** Only the fit uses scikit-learn. Every probability is
  computed in pure Python from the persisted intercept and coefficients.
- **Explicit statuses.** Data limitations return a `TrainingStatus`; no other
  estimator is substituted.
- **Identities.** Four distinct identities:
  - a configuration ID (fixed choices);
  - a model configuration ID, which binds the dataset, population, schema and
    order, target, chronology and material runtime versions;
  - a fitted-state ID, which no non-training row can change;
  - a model ID (the whole model).

  Prediction sets have their own ID.
- **Freeze.** `freeze_event_model` atomically publishes an inspectable JSON
  envelope (`<model-id>/model.json`) with the regenerated selection evidence.
  `read_event_model` is the only constructor of `FrozenEventModel`.
  Out-of-sample and holdout inference re-read and compare the envelope on
  every call.
- **Predictions.** Prediction sets are separate immutable directories: a
  manifest plus canonical JSON lines, every row of the partition, unscored and
  unlabeled rows reported apart. Metrics are pure Python and explicitly
  undefined when degenerate. The baseline is a constant training prevalence.
- **Holdout.** Holdout inference requires all of:
  - a frozen model;
  - a dataset whose non-holdout rows are identical to the training
    dataset's;
  - the workspace's permanent ledger;
  - a `HoldoutEvaluation` that `HoldoutLedger.result` validates as consumed,
    whose request ID matches the dataset's holdout source.

  The ledger is only read.

Rejected alternatives:

- **Pickle or joblib model payloads.** They are opaque and unsafe to
  deserialize, and unnecessary for a linear model.
- **Using scikit-learn's `predict_proba` for persisted predictions.** BLAS
  reduction order would make reproduction approximate on a single platform.
- **A caller-supplied freeze flag or role.** Neither can be trusted.
- **Training on selection rows when development is absent.** That is role
  reassignment and would change the frozen QF-45 study's meaning.
- **Requiring holdout rows in the training dataset.** That would force
  freezing after consumption; compatible holdout extensions allow freezing
  first.
- **A general model zoo, threshold or hyperparameter search.** Out of scope;
  the selection policy would need its own integrity design.

## Consequences

- QF-69 and QF-71 can train, freeze and evaluate through one generic contract
  without parsing strategies or windows. Multiple folds mean one model per
  fold.
- The real QF-45 dataset reports `no_training_observations` with QF-67's
  `role_not_executed_by_qf39_selection` reason. QF-69 must design a study that
  produces training rows.
- Fitted preprocessing state is platform-independent. Coefficients may differ
  in the last bits across BLAS or CPU platforms; reproduced probabilities are
  exact on one platform and within `1e-12` across platforms.
- Model envelopes grow with the number of fitting observations (about 140
  bytes each).
- A fully re-identified forgery of coefficients is detected only by
  retraining, as for QF-67 feature forgeries.
- The ledger records strategy holdout consumption but not ML exposures, so
  freezing before consumption remains a procedural rule.

See [event ML models](../event-ml-models.md).
