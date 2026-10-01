# Conditional event ML datasets (QF-67)

> A conditional event dataset answers one question: **given that this exact
> strategy configuration triggered, which trigger instances were stronger or
> weaker?** Every row is a persisted generated signal. Timestamps where the
> rule did not trigger are not negative examples and never become rows. A model
> trained on these rows is valid only conditional on this population.

`quantforge.ml` assembles one frozen strategy population's verified,
persisted observations into a deterministic, versioned dataset. It contains:

- an explicit causal feature schema;
- an explicit versioned target over existing outcomes;
- verified chronological partition membership.

It does not train models, fit imputers, scalers or encoders, select features,
tune strategies or claim profitability. See
[ADR 0039](decisions/0039-assemble-conditional-event-ml-datasets.md).

```text
QF-39 study store                      permanent ledger (reports/holdout-ledger)
  folds/<id>/selection/<grid>/...       shared lock held while building
  folds/<id>/test/...                   consumed holdout result only
        │                                          │
  QF-40 load_oos_source (test)          HoldoutLedger.result (consumed)
  QF-40 load_prediction_trial_window    (never consume, never reserve)
        │                                          │
        └──────────── EventSourceWindow (loader-only) ─────────┘
                                 │
         iterate_decision_receipts(): count every decision,
         expand only evaluated/skipped receipts, never no-trigger ones
                                 │
     population check ─ explicit feature schema ─ bound target ─ holdout scopes
                                 │
                  EventDataset (ordered rows, identities, summaries)
                                 │
              <dataset-id>/manifest.json + rows.parquet (+ rows.csv)
```

## Public API

```python
from pathlib import Path

from quantforge.examples.spy_ema_ml_dataset import (
    ema_event_feature_schema,
    ema_forward_return_target,
)
from quantforge.ml import (
    DispositionPolicy,
    build_event_dataset,
    export_event_dataset,
    read_event_dataset,
)

dataset = build_event_dataset(
    plan=config.plan,  # the exact QF-8 plan the QF-39 study used
    study_path=Path("reports/walk-forward") / study_id,
    workspace=Path("."),  # research workspace; its permanent ledger must exist
    combination_id=combination_id,  # one candidate of the study's universe
    feature_schema=ema_event_feature_schema(),
    target=ema_forward_return_target(),
    disposition_policy=DispositionPolicy.ACCEPTED_ONLY,  # default
    # roles=frozenset({...}) selects fold roles (default: all three fold roles)
    # final_holdout=evaluation adds rows of an already consumed holdout only
)
path = export_event_dataset(dataset, Path("reports/ml-datasets"), include_csv=True)

loaded = read_event_dataset(path)  # validates everything offline first
loaded.feature_columns  # ordered model-input names
loaded.numeric_feature_rows()  # explicit binary64 view, None for nulls
loaded.target  # EventTargetColumn: values (True/False/None), statuses
loaded.partition_membership  # (row_index, role, fold_id, fold_index) per row
loaded.rows_for(PartitionRole.WALK_FORWARD_TEST, fold_id=fold_id)
```

A generic consumer needs only `feature_columns`, `numeric_feature_rows()`
(or exact `feature_values(name)`), `target` and `partition_membership`.
Nothing in this interface exposes prediction-window internals.

Lower-level pieces:

- `load_study_event_sources`, `EventPopulation` and `assemble_event_dataset`
  accept only loader-built `EventSourceWindow` values.
- `validate_event_dataset` performs offline validation only.
- `EventFeatureDefinition`/`EventFeatureSchema` and
  `ForwardReturnBinaryTarget`/`BoundEventTarget` define the schemas.
- Every failure raises a subclass of `EventDatasetError`:
  - `EventFeatureSchemaError`
  - `EventTargetError`
  - `EventPopulationError`
  - `EventPartitionError`
  - `EventHoldoutError`
  - `NonAuthoritativeSourceError`
  - `EventDatasetIntegrityError`

## Sources and population

Sources are verified by existing readers. Callers cannot assert a role.

| Role | Verifier | Notes |
| --- | --- | --- |
| Development/selection (QF-8 `DEVELOPMENT`/`SELECTION`) | `quantforge.oos.load_prediction_trial_window` (new) | The candidate's QF-32 trial in a fold with a frozen selection. QF-39 executes trials on the selection window, or on development when the fold declares no selection window. The role it did not execute is recorded as an exclusion. |
| Walk-forward test | `quantforge.oos.load_oos_source` (unchanged) | Only when the fold froze this candidate. Otherwise the exclusion `frozen_selection_is_another_candidate` is recorded. |
| Final holdout | `HoldoutLedger.result(evaluation)` | Only an explicitly consumed holdout of this candidate; see below. |

`load_prediction_trial_window` is the QF-40 test verifier generalized by role.
Test windows keep exactly the same checks. For trials it requires:

- the frozen selection's grid study and a `succeeded` trial status;
- the QF-32 grid identity and trial record;
- the trial wrapper fingerprints and decision schedule.

It then applies the same checks as a test window:

- candidate configuration;
- context partition equal to the frozen QF-8 membership of that role;
- schedule from the retained membership;
- bounded canonical lineage;
- `validate_prediction_window_reader`;
- outcome reach before the next protected window.

No factory, market data or execution is needed. Only the population's trial
in each fold is read.

The **population** is one universe candidate plus a disposition policy. It
binds:

- the plan ID;
- the candidate's executable definition: rule name, version, configuration ID
  and parameters, outcome labeler and evaluator, feature configuration,
  context requirements and indicator backend;
- the symbol, primary timeframe, schedule ID, source reference and family
  manifest.

Every source window's manifest and every signal's prediction identity are
re-checked against it, so another configuration's window or signal fails with
`EventPopulationError`. `ACCEPTED_ONLY` (default) keeps accepted and
ordinary signals and counts the others. `ALL_GENERATED_SIGNALS` keeps rejected,
blocked and overlapping candidates too. The policy is part of the identity.

QF-72 rapid results are exploratory and are refused at every input with
`NonAuthoritativeSourceError`. This covers `RapidScanResult`, `RapidEvent`,
any object with `authoritative=False` or `mode="exploratory"`, and
`.rapid.json` exports. A promising rapid configuration must first be reproduced
through QF-32/QF-39/QF-40 and then built from that study.

## Row unit and no-trigger coverage

Rows are read through `iterate_decision_receipts()` for every window schema
(1 to 4):

- Every scheduled decision is counted by status (`evaluated`, `no_prediction`,
  `skipped`) in the source's coverage record.
- Only receipts with retained rich evidence are expanded.
- Schema-4 bare receipts never become rich objects or rows. A test counts
  exactly two `CompactPredictionWindowDecision` constructions per evaluated
  decision and none for 566 no-trigger receipts.
- The strategy is never re-run and no decision is reconstructed.

Each row carries `decision_sequence`, `signal_index`, `context_id`,
`prediction_study_id`, `source_index` and `source_observation_id`, which bind
it to its exact persisted observation. `source_observation_id` is the SHA-256
of:

- the QF-42 scientific window identity;
- the sequence and signal index;
- the decision timestamp;
- both decision identities;
- the full signal;
- the row ID.

## Feature schema

`EventFeatureSchema(name, version, features)` is an ordered allowlist. Column
order is declaration order, and nothing is discovered: there is no `feature_*`
selection. Each `EventFeatureDefinition` declares the following.

| Field | Meaning |
| --- | --- |
| `name` | Canonical lowercase snake_case model column |
| `source_field` | A key of the persisted signal's causal `features` mapping, the only place values are read from |
| `value_type` | `decimal`, `integer` or `boolean` |
| `missing_values` | `reject` (a null fails the build) or `preserve_null` (explicit null) |
| `availability` | `known_at_decision_timestamp_from_completed_inputs` (the only admitted value) |
| `timeframe` | Material source timeframe; it must be declared by the plan |
| `unit`, `description`, `version` | Definition identity (`definition_id`) |

**Refused as `name` or `source_field`:**

- every field of persisted QF-11 rows and of QF-46/QF-47/QF-49 (and legacy
  QF-7/QF-11) outcome, evaluation and resolution records, for example
  `raw_return`, `outcome_price`, `mfe_percentage`, `label`, `status`,
  `available` and `resolved_observation_timestamp`;
- names starting with `outcome`, `evaluation`, `evaluator`, `target`, `label`,
  `future`, `forward`, `fold`, `partition`, `holdout`, `selection`,
  `experiment`, `result`, `provenance`, `lineage`, `mfe` or `mae`;
- names ending in `_id`, `_ids`, `_sha256`, `_fingerprint`, `_hash`, `_digest`
  or `_timestamp`;
- the reserved dataset columns.

**Also refused:** duplicate names or source fields, unsupported types or
policies, unknown definitions, and an undeclared timeframe.

**Values** must be the exact persisted scalar:

- **Decimal:** a canonical finite decimal string. Floats, structures, `NaN`
  and non-canonical text are refused.
- **Integer:** an `int`, never a `bool`.
- **Boolean:** a `bool`.

A source field the population did not persist fails with "the feature schema
does not match this population".

**Encoding.** Decimals are stored as their exact persisted text (Arrow
`string`), so serialization never changes a value. `numeric_feature_rows()`
returns `float(Decimal(text))`, which is correctly rounded. Integers are
`int64` and booleans `bool`. Nulls stay null; nothing is imputed,
zero-filled, forward-filled or scaled.

The QF-45 example schema (`spy_ema_ml_dataset.ema_event_feature_schema`) maps
the rule's six declared contemporaneous inputs:

| Column | Source field | Timeframe |
| --- | --- | --- |
| `ema_fast_previous` | `previous_fast` | 2m |
| `ema_slow_previous` | `previous_slow` | 2m |
| `ema_fast` | `current_fast` | 2m |
| `ema_slow` | `current_slow` | 2m |
| `daily_close` | `daily_close` | daily |
| `daily_ema50` | `daily_ema50` | daily |

## Target

`ForwardReturnBinaryTarget(horizon=30 minutes, threshold=0)` labels a row
`raw_return > threshold`, using the existing QF-49 intraday forward close
return:

- **Return convention:** `outcome_close / reference_close - 1`, an exact
  decimal arithmetic ratio over the same session and source. There are no
  costs or fills.
- **Comparison:** strictly greater than. An available return equal to the
  threshold (`0`) is a **negative**.
- **Direction:** none. The raw return is not sign-adjusted for DOWN signals.
- **Missing labels:** an outcome whose QF-46 status is not `available` is
  null with that status, never `False` and never zero. The statuses are
  `session_overflow`, `missing_required_observation`,
  `incomplete_future_data` and `dataset_end`.

**Binding.** `target.bind(definition, plan)` reconstructs
`IntradayForwardReturnOutcomeLabeler` and `IntradayForwardReturnEvaluator` from
the persisted temporal configuration. It requires:

- byte-equal configurations and IDs;
- the configured horizon;
- the same-session policy;
- a timestamp plan that declares this outcome, with a purge horizon covering
  its reach (30m plus one 2m alignment interval).

Any other outcome, horizon or evaluator fails with `EventTargetError`.

**Per-row checks.** Each row's persisted outcome must match the bound
configuration IDs, the decision timestamp and the horizon. Its evaluation
values must equal the outcome values. Availability, status and return must
agree. A missing QF-49 availability row is corrupt evidence. Each row stores:

- `target` (bool or null);
- `target_status`;
- `target_source_value` (the exact `raw_return`, or null);
- `target_outcome_id`.

These target columns are separate from the features. The `kind` field reserves
room for later continuous targets; no general target framework is
implemented.

## Partitions, ordering and holdout

`partition_role` (QF-8 `PartitionRole` value), `fold_id` and `fold_index` come
from the verified source. Final-holdout rows have no fold. Partition columns
are never features.

**Ordering** is by:

1. decision timestamp (UTC);
2. fold index in plan order (final holdout last);
3. role (development, selection, test, holdout);
4. signal index;
5. `source_observation_id`.

Input order and file traversal never matter.

**Failures:**

- duplicate source windows, roles or observation IDs;
- one fold holding the same event in two roles;
- a decision outside its plan window;
- a session differing from the plan schedule;
- a schedule differing from its QF-8 retained count.

**Holdout and ledger.** The builder takes the workspace root and opens
`reports/holdout-ledger`. It never creates one; a missing ledger fails, and
there is no override or substitutable ledger object. It holds the ledger's
shared lock while sources are read and rows are checked, so a concurrent
reserve or consume fails closed until it finishes. Fold rows must satisfy:

- decision plus label horizon plus embargo is before the plan's final holdout;
- the decision session is outside every reserved or consumed exposure scope for
  the symbol (conservative and session-granular, as for QF-72).

Final-holdout rows require `final_holdout=HoldoutEvaluation` naming a holdout
the ledger already records as consumed. `HoldoutLedger.result` validates the
consumed request and artifact. The rows are read from the ledger's own copy and
must stay inside the reserved interval under the QF-40 tail rule:

- a reserved but unconsumed holdout fails with `EventHoldoutError` and stays
  reserved;
- a holdout evaluated for another candidate fails with `EventPopulationError`;
- building never calls `consume`.

## Artifact layout and identities

```text
<output-root>/<dataset-id>/
  manifest.json   {"payload": {...}, "fingerprint": sha256(payload)} (canonical line)
  rows.parquet    pyarrow, zstd, no dictionary encoding, field group metadata
  rows.csv        optional, inspection only (empty cell = null, true/false)
```

**Manifest `payload`:**

- `component`, `schema_version` (`"1"`), `dataset_id`;
- `scientific`: population, feature schema, bound target, partition plan with
  per-source coverage and exclusions, column layout, and row count, ordering
  and logical-row SHA-256;
- `summaries`: rows, positive, negative and unavailable counts by status,
  overall and per role/fold, plus disposition and direction counts;
- `provenance`: QF-39 study and selection IDs, physical window schema, window
  result and evidence IDs, and relative paths;
- `interpretation`;
- `files`: SHA-256 and bytes for each file.

**Columns** (each Arrow field records `quantforge_column_group`):

| Group | Columns |
| --- | --- |
| metadata | `row_index`, `source_observation_id`, `source_index`, `decision_timestamp` (timestamp µs UTC), `signal_session` (date), `decision_sequence`, `signal_index`, `context_id`, `prediction_study_id`, `direction`, `disposition` |
| partition | `partition_role`, `fold_id`, `fold_index` |
| feature | the schema's columns, in order |
| target | `target`, `target_status`, `target_source_value`, `target_outcome_id` |

**Scientific identity.** `dataset_id` is the SHA-256 of `scientific`. It is
identical across rebuilds, source order and the physical window schema (a
schema-2 and a schema-4 study of the same science produce the same ID).
Paths, timings, QF-39 study/selection IDs and physical window identities are
provenance only.

**Physical integrity.** The manifest fingerprint plus each file's SHA-256 and
size. `export_event_dataset` writes and fsyncs files in a private staging
directory, then fully validates it. Only then does it rename the directory and
fsync the parent, so an unverifiable dataset never appears under its ID. An
existing identical directory is reused. Output inside a holdout ledger is
refused.

**Offline validation** (`read_event_dataset`) checks:

- the fingerprint, schema version and component;
- the ID against the content and the directory name;
- schema and target identities;
- file hashes and sizes, and no unlisted `rows.csv`;
- the Arrow schema with metadata;
- per-row types and null policies;
- label consistency: `target` must equal `target_source_value > threshold`,
  and unavailable must be null;
- the logical-row hash, ordering and unique observation IDs;
- membership against the source table;
- the summaries;
- the CSV.

A fully and consistently rehashed forgery of feature values is detectable only
by rebuilding from the source evidence. Label forgeries that contradict their
stored return are detected even then.

## Example artifact (synthetic fixture)

This is the dataset built by the integration fixture (one fold, frozen 8/48,
schema-4 windows, synthetic prices). The selected columns of its `rows.csv`
follow; the other columns are omitted here.

```text
<output-root>/0a367763f90b7381cfc6107d2d8f4e1420826a1f0b1730ad63baf329db8c4a30/
  manifest.json  35,488 bytes
  rows.parquet   16,021 bytes
  rows.csv        2,477 bytes

row_index,decision_timestamp,partition_role,fold_index,ema_fast,daily_ema50,target,target_status,target_source_value
0,2024-12-23T16:00:00+00:00,validation_selection,0,163.89434908779208,162.5,true,available,0.001595744680851063829787234042553
1,2024-12-24T17:40:00+00:00,validation_selection,0,164.89399354227666,163.61764705882354,,session_overflow,
2,2024-12-26T16:00:00+00:00,validation_selection,0,165.89432514474328,164.6130334486736,false,available,0
3,2024-12-27T16:00:00+00:00,walk_forward_test,0,166.89434908779208,165.55764705882353,false,available,-0.0015706806282722513089005235602094
```

Coverage: selection has 375 scheduled decisions (3 evaluated, 372 no-trigger
receipts) and test has 195 (1 evaluated, 194 no-trigger). Development is
recorded as excluded with `role_not_executed_by_qf39_selection`. Row 1's
12:40 trigger on the 13:00 early close cannot reach its 30-minute target, so it
is unavailable (null), not negative. Row 2's exactly flat return is a
negative.

## Measurements

All figures come from one 14-core Apple-silicon host with a warm file cache.
Python 3.13 and pyarrow 21 ran with no concurrent test run. The harnesses are
ignored: `reports/qf67-event-datasets/qf67_real.py` and `qf67_synthetic.py`.

### Real evidence: completed QF-45 study (plumbing only)

The population is the QF-45 frozen 12/60 configuration. Its sources are the
fold-0 selection trial window (2025-05-01 to 05-30) and the June OOS window
(2025-06-02 to 06-27). Both are completed schema-2 windows of about 2.3 GB and
1.9 GB. The study was cloned with APFS `cp -c` and its Finder `.DS_Store`
files removed (see Limitations). The real permanent ledger was read under its
shared lock. Its reserved QF-45 holdout (2025-06-30 to 07-31) was neither read
nor consumed.

| Item | Value |
| --- | ---: |
| Scheduled decisions (selection / test) | 4,095 / 3,705 |
| Evaluated (rows) / no-trigger receipts | 18 + 12 = 30 / 4,077 + 3,693 |
| Labels: positive / negative / unavailable | 19 / 11 / 0 |
| QF-45 input load (QF-65 authenticated) | 35.9 s |
| Test-window verification (`load_oos_source`) | 77.4 s |
| Selection-trial verification | 80.2 s |
| Row assembly (re-reading both windows) | 58.1 s |
| Build total / export / offline read | 219.8 s / 0.03 s / < 0.01 s |
| Artifact: Parquet / CSV / manifest | 21,725 / 16,991 / 35,483 bytes |
| Peak RSS (whole process, mostly cached inputs) | 2.39 GB |

All 30 rows equal QF-45's independent native QF-7 candidate exports: decision
timestamp, all six features and the 30-minute `raw_return`. The cost is
dominated by verifying and decoding schema-2 windows (about 600 KB per
no-trigger decision). Schema-4 windows remove that per-decision payload.
Thirty events are not evidence of ML adequacy or an edge.

### Synthetic benchmark (not evidence)

The benchmark uses `tests/performance/event_dataset_scale.py` over 200
synthetic sessions. A QF-49 rich observation from a real QF-39 trial window is
the template. Its features, timestamps and outcomes are rewritten per event,
rotating positive, zero, negative and unavailable labels. The schema-4 windows
are streamed through the unchanged QF-64 contracts and read by the normal
reader. They cannot pass QF-39 scientific validation, so the benchmark measures
reading, assembly, writing and offline reading, not plan-bound verification.

| Measure | 10,765 rows (1 in 3 decisions) | 32,295 rows (every decision) |
| --- | ---: | ---: |
| Scheduled decisions / window bytes | 32,295 / 336.9 MB | 32,295 / 975.3 MB |
| QF-64 reader alone (`iterate_observations`) | 21.6 s | 58.2 s |
| Assembly including reading | 23.0 s (468 rows/s) | 61.1 s (529 rows/s) |
| Assembly excluding reading (approx.) | 1.4 s (~7,900 rows/s) | 2.9 s (~11,000 rows/s) |
| Export: render, staging validation, publish | 1.06 s | 1.93 s |
| Offline read and full validation | 0.50 s | 1.48 s |
| `rows.parquet` / `rows.csv` / manifest | 0.88 MB / 5.6 MB / 34 KB | 2.54 MB / 17.0 MB / 34 KB |
| Parquet bytes per row | 81.8 | 78.5 |
| tracemalloc peak: assembly / export / read | 191 / 43 / 43 MB | 223 / 127 / 127 MB |
| Peak RSS (whole process, mostly fixture inputs) | 1.98 GB | 2.03 GB |

What the figures show:

- **Reading dominates.** Decoding about 30 KB of rich QF-64 evidence per
  observation takes about 95% of build time. Feature extraction, labels,
  identities and ordering add about 0.1 ms per row.
- **Writing and reading the artifact cost seconds** at tens of thousands of
  rows.
- **Memory grows with retained rows, not with scheduled decisions.** No-trigger
  receipts are never expanded. The committed regression test
  (`test_event_ml_dataset_scale.py`, about 2,500 rows) reports the same
  quantities without timing thresholds.

## Limitations

- **Population scope.** The dataset is conditional on one population. It is
  not a sample of market timestamps, and no-trigger decisions are coverage
  only. Combining configurations requires separate datasets.
- **Not evidence of anything.** QF-45's frozen sample is tiny (tens of events
  per window). It is plumbing, not evidence of ML adequacy, robustness or an
  edge. All relationships remain hypotheses until untouched out-of-sample
  validation.
- **Labels are not independent.** Intraday events can share outcome windows;
  labels are not independent samples.
- **Target support.** Only the QF-49 binary forward-return target is
  supported. The plan must be timestamp-based with canonical metadata.
- **Holdout sessions.** Fold rows in a session that holds a reserved or
  consumed holdout scope are refused (session granularity).
- **Trials checked.** Only the population's trial is verified per fold; other
  trials are left to QF-9.
- **Finder metadata.** `load_oos_source` treats any unexpected entry under
  `folds/` as corruption, including Finder `.DS_Store` files. Remove such
  files, or copy the study, before building.
- **Out of scope:** dense or all-bars datasets (QF-70), training and
  evaluation (QF-68), studies (QF-69), live inference and execution.
