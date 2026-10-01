# ADR 0039: Assemble conditional event ML datasets from verified observations

- Status: Accepted
- Date: 2026-09-30
- Jira: [QF-67](https://frostfiredigital-37308542.atlassian.net/browse/QF-67)

## Context

QF-64 schema-4 windows persist each generated signal as a rich observation:
the signal, its causal features and its labeled row. Every scheduled decision
also gets a small coverage receipt. QF-39 studies place those windows under
QF-8 partitions; QF-40 verifies test windows and owns the permanent holdout
ledger. The next step (QF-68) needs a training-ready dataset: given that a
strategy triggered, which triggers were stronger or weaker.

There are four risks:

- **Leakage through field selection.** Outcome, provenance and partition
  fields sit next to causal values. Selecting `feature_*` columns or every
  persisted field would leak them.
- **No-trigger decisions as negatives.** That would answer a different
  question and change the population.
- **Exploratory or protected evidence entering silently.** QF-72 rapid results
  could be promoted by type coercion, and holdout rows exist on disk.
- **Unverified selection windows.** QF-40 verifies only test windows offline.
  Selection and development trial windows had no plan-bound offline verifier.

## Decision

Add `quantforge.ml` as one event-dataset boundary.

- **Verified sources only.**
  - Test windows come from `load_oos_source`.
  - Selection and development windows come from a new QF-40
    `load_prediction_trial_window`. It applies the existing test-window checks
    (generalized by role) under the fold's frozen QF-8 membership, plus the
    QF-32 grid identity, trial record and wrapper fingerprints. Test-window
    behavior is unchanged.
  - Final-holdout rows come only from `HoldoutLedger.result` of an explicitly
    consumed holdout.
  - `EventSourceWindow` can be built only by these loaders.
- **One population.** The caller names one universe candidate and a
  disposition policy. The rule, parameters, outcome, evaluator, context,
  backend, source, symbol and schedule come from verified artifacts.
  - Every window and every signal is re-checked against the population.
  - A fold that froze another candidate contributes no test rows; the
    exclusion is recorded with its reason.
- **Rows are observations.** Rows are read from
  `iterate_decision_receipts()`. Bare receipts are counted as coverage and
  never expanded, so no-trigger decisions are never rows or negatives.
- **Explicit causal feature schema.** An ordered allowlist of definitions reads
  only the persisted signal's causal `features` mapping. Each definition has an
  approved scalar type, an explicit null policy, decision-time availability, a
  source timeframe and a version.
  - Outcome, availability, ambiguity, identifier, provenance and partition
    names are refused.
  - Decimals stay exact strings; there is an explicit binary64 view.
- **Explicit target.** QF-49 `raw_return > threshold` (default 30 minutes,
  zero threshold). It is bound by reconstructing the exact labeler and
  evaluator configurations and checking the plan's outcome and purge.
  - Unavailable outcomes are null labels with their QF-46 status.
  - Zero is a negative.
  - There is no direction adjustment.
- **Membership.** Each row has a role and fold (`None` for the holdout). The
  order is timestamp, then fold, role, signal index and observation ID.
  Duplicate observations and same-fold role conflicts fail closed.
- **Holdout and ledger.**
  - The builder takes the workspace root and requires its permanent ledger.
  - It holds the ledger's shared lock while sources are read.
  - Fold rows must stay outside the plan holdout (horizon plus embargo) and
    every reserved or consumed exposure scope for the symbol.
  - There is no override.
- **Artifact.**
  - Layout: `<dataset-id>/manifest.json` (fingerprinted canonical envelope),
    `rows.parquet` (pyarrow, already a dependency) and an optional
    `rows.csv`.
  - The scientific `dataset_id` hashes the population, schemas, bound target,
    partition plan, column layout and logical rows.
  - Physical identities, QF-39 selection/study IDs and paths are kept apart.
  - Export validates staged bytes before an atomic rename.
  - `read_event_dataset` re-validates everything offline.
- **Rapid refusal.** Objects declaring `authoritative=False` or
  `mode="exploratory"`, `quantforge.rapid` types and `.rapid.json` paths are
  refused at every input.

Rejected alternatives:

- **Re-run the strategy or reconstruct no-trigger decisions.** This is slower
  and unnecessary, because persisted evidence is sufficient.
- **Re-label from market data.** That is a second outcome implementation.
- **Float64 storage.** It is lossy and cannot be verified offline.
- **Per-row provenance envelopes.** Shared provenance belongs in the manifest.
- **A caller-supplied role or ledger object.** Neither can be trusted.

## Consequences

- QF-68 can read `feature_columns`, `target` and `partition_membership`
  without parsing prediction-window internals.
- Dataset identity is independent of the physical window schema (2, 3 or 4)
  and of paths. Determinism holds across rebuilds and source order.
- Offline validation detects corrupted or partially rehashed artifacts. A label
  that contradicts its stored outcome value is detected even under a complete
  consistent rehash. A full forgery of features with recomputed identities is
  detected only by rebuilding from the source evidence (as for QF-64 bare
  receipts).
- Fold rows in the same exchange session as a reserved holdout scope are
  refused, a conservative session-granular rule shared with QF-72.
- Selection and development windows are now verifiable offline, but not every
  trial is checked: only the population's trial in each fold is read.
- Dense datasets (QF-70), training (QF-68) and studies (QF-69) remain separate.

See [event ML datasets](../event-ml-datasets.md).
