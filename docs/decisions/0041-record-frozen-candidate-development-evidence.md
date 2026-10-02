# ADR 0041: Record frozen-candidate development evidence for ML studies

- Status: Accepted
- Date: 2026-10-01
- Jira: [QF-69](https://frostfiredigital-37308542.atlassian.net/browse/QF-69)

## Context

QF-68 fits a model only on one QF-8 fold's development rows. QF-39 executes
its QF-32 trials on the fold's selection window when the fold declares one,
otherwise on development. Development, selection and test evidence therefore
never coexist for one fold:

- With a selection window, development is never executed, and QF-67 records
  `role_not_executed_by_qf39_selection`. The real QF-45 dataset is this case.
- Without one, there are no selection rows. Selection predictions are then
  empty.

QF-69 needs authoritative development, selection and test observations of
the same plan, fold and frozen candidate. Declaring three windows alone does
not produce them. Relabeling selection or test rows, building rows from
exports, or training across folds would each break the QF-67/QF-68 contracts.

## Decision

Add one post-freeze QF-39 operation and its verifier.

- **Execution.** `WalkForwardStudy.evaluate_development(fold_id)` requires a
  fold that declares a selection window and holds its durable frozen
  selection. It executes the **frozen** candidate on the exact development
  membership frozen in `selection.json`. Execution goes through
  `PredictionEvaluator.evaluate_partition`, the same path as test and holdout
  windows, with no predictions required, because an empty in-sample window is
  valid evidence. The result is validated with `validate_partition_artifact`.
- **Record.** An immutable, fingerprinted `folds/<fold-id>/development.json`
  (`DevelopmentEvidence`) binds:
  - the study, plan, fold and selection IDs;
  - the frozen candidate;
  - the development membership identities (QF-8 window, membership
    selection, purge result, retained count);
  - the window reference.

  The window lives in `folds/<fold-id>/development/`. An existing record is
  verified, never recalculated. Fold state, selection and test evaluation are
  untouched.
- **Verification.** QF-40 `load_prediction_development_window` checks the
  record against the verified source: study, plan, fold, frozen selection,
  candidate and membership identities. It then applies the same offline
  partition checks as test and trial windows, with development's protected
  window (the fold's selection). These cover configuration, context
  partition, schedule, bounded canonical lineage, scientific window
  validation and outcome reach.
- **Admission.** QF-67 adds the verified window as development rows. With no
  evidence recorded, the existing `role_not_executed_by_qf39_selection`
  exclusion stays, so existing dataset IDs are unchanged. A fold that froze
  another candidate records `frozen_selection_is_another_candidate`. A
  `development/` directory without its record fails closed.

Rejected alternatives:

- **Run the QF-32 grid on development as well.** That changes QF-39's
  selection semantics and study identity, and runs every candidate.
- **A new QF-39 fold state.** It changes the state machine and the persisted
  schema that QF-45 and other studies resume from.
- **A second study without a selection window.** That is a different plan and
  fold, so its rows cannot train a model bound to this one.
- **Relabeling selection rows as development.** That is role reassignment.

## Consequences

- Conditional ML studies obtain real development rows of the frozen
  candidate, and selection and test rows, from one plan and fold.
- Existing QF-39 studies, QF-45 artifacts, study identities and QF-67 dataset
  identities are unchanged.
- The development window is executed after selection is frozen, and in QF-69
  after the test window. Nothing is selected from it, and QF-68's
  `fitted_state_id` does not depend on test rows. The execution order of
  strategy evidence is therefore immaterial to model leakage.
- Only the frozen candidate gets development evidence. Comparing candidates
  on development stays outside QF-39 selection.
- QF-9 does not index the development record; QF-40 and QF-67 verify it.

See [conditional ML smoke study](../conditional-ml-smoke-study.md) and
[walk-forward studies](../walk-forward-studies.md).
