# Experiment manifests and artifact indexes

QF-9 adds `quantforge.experiments`, a local provenance and artifact-index layer.
It answers which code, dependencies, data, configuration, backend, validation
lineage, and files produced a result. It reads existing structured exports. It
does not run indicators, predictions, outcome labelers, feature generation,
grids, backtests, walk-forward selection, OOS aggregation, holdout evaluation,
or charts. QF-41 owns subsequent report presentation.

## Public contracts

| Contract | Purpose |
| --- | --- |
| `StudyType` | Prediction, prediction window, feature dataset, parameter study, backtest, optimization, walk-forward, OOS validation, holdout validation |
| `StudyProvenance` | Producer identity, detached material configuration, and separate observed status/references/counts |
| `CodeProvenance` | QuantForge version, Git commit/dirty state, lock SHA-256, Python and relevant installed dependency versions |
| `ExecutionProvenance` | Explicit execution ID, UTC creation timestamp, optional original execution start, code and random seeds |
| `StudyArtifacts` | Observed study provenance plus an `ArtifactIndex` |
| `ExperimentManifest` | Immutable combination with logical `study_id` and content `manifest_id` |
| `ArtifactEntry` | Typed artifact category/format, schema version, root-relative path, optional JSON pointer, byte SHA-256, producer study/artifact/run IDs, metadata and required status |
| `ArtifactRelationship` | Typed directed edge between indexed artifact IDs |
| `ArtifactIndex` | Deterministically ordered entries and relationships with its own content ID |
| `IntegrityReport` / `IntegrityIssue` | Explicit verification failures and deliberately absent optional artifacts |

Snapshots use the existing `PrimitiveMappingSnapshot` canonical serialization
and `configuration_identity` SHA-256 helpers. Mappings returned to callers are
detached. There is no database, provider client, remote registry, or new research
execution abstraction.

## Identity and time

Producer IDs are retained verbatim as `producer_study_id`. QF-5's deterministic
`run_id` is such a producer identity: its historical name does not make it an
individual execution ID. QF-42's `window_id` identifies the configuration;
`window_result_id` remains an observed result reference. QF-40 aggregate IDs are
content references, separate from their source walk-forward study ID.

The namespaced QF-9 `study_id` hashes the producer ID, study type, complete
captured material configuration, supplied code environment and random seeds.
Adapters separately verify supported producer identities against their recorded
inputs using canonical hashing, preserving historical versions and optional-field
semantics. Inconsistent producer IDs are rejected. The additional experiment
binding includes execution-time code and seeds. Dataset/family, source and derived
timeframe, feed/session/aggregation/completion/adjustment, backend/library,
rule/outcome, search, execution and validation changes therefore change the
experiment identity wherever these fields are recorded by the source.

Paths, result bytes, observation counts, timestamps, and current holdout state
do not define this logical study identity. The caller supplies a distinct
`ExecutionProvenance.run_id` for each execution that should remain distinguishable.
QF-9 does not invent an execution timestamp for an older result. Creation and
execution timestamps must be timezone-aware and serialize in UTC. Exchange
session boundaries and bar completion policies retain the producer's semantics.

The `manifest_id` hashes the entire manifest including execution metadata and
the artifact index. Different executions, result bytes, paths, or newly observed
holdout state produce a different manifest ID. Exact repeated input produces
identical bytes. Artifact IDs include their full reference metadata and byte
hash. Duplicate IDs, duplicate producer artifact identities within an execution,
conflicting references at the same path/pointer, duplicate edges, and unresolved
edge endpoints are rejected.

## Index an existing prediction or backtest

These examples consume already-produced files. No strategy or dataset object is
passed to the manifest layer:

```python
from datetime import UTC, datetime
from pathlib import Path
from quantforge.experiments import (
    CodeProvenance,
    ExecutionProvenance,
    StudyType,
    create_manifest,
    inspect_study,
    read_manifest,
    write_manifest,
)

root = Path(".").resolve()  # Common root for all referenced artifacts.

# This should be retained from the original execution. Explicit None fields
# disclose provenance that was not captured; never substitute today's runtime.
original_code = CodeProvenance()
execution = ExecutionProvenance(
    run_id="research-execution-001",
    created_at=datetime.now(UTC),
    code=original_code,
    execution_started_at=None,
)

# Existing JSON containing PredictionStudyResult.to_primitive():
prediction = inspect_study(
    StudyType.PREDICTION,
    root / "reports/prediction-result.json",
    artifact_root=root,
)
prediction_manifest = create_manifest(prediction, execution)
path = write_manifest(
    prediction_manifest,
    root / "reports/experiments",
    artifact_root=root,
)
assert (
    read_manifest(path, artifact_root=root).manifest_id
    == prediction_manifest.manifest_id
)

# Existing export_backtest_result directory; its original integrity.json is
# verified before indexing, without running execution or metrics.
backtest = inspect_study(
    StudyType.BACKTEST,
    root / "reports/backtests/<producer-run-id>",
    artifact_root=root,
)
write_manifest(
    create_manifest(backtest, execution),
    root / "reports/experiments",
    artifact_root=root,
)
```

For new executions, call `capture_code_provenance(repository)` at execution
time and retain the returned record. This reads Git status, `uv.lock`, Python
version and package metadata; it imports no numerical backend and reads no
environment variable values. Relevant packages default to TA-Lib, NumPy,
exchange-calendars and PyArrow. Additional distribution names may be supplied
explicitly. Installed dependency versions do not override historical indicator
backend/library/runtime/function fields recorded by the study.
Public experiment imports, provenance capture, manifest assembly and manifest/index
read/write/hash verification do not import NumPy, TA-Lib, PyArrow or research
producer packages. Producer-specific validators load when `inspect_study` or
`inspect_validation` inspects existing research exports. They check persisted
metadata without executing research calculations.

Capture rejects a dirty working tree, including staged, unstaged and untracked
files. Commit the intended source before execution; a commit and a dirty flag
alone cannot identify different uncommitted implementations. Ignored files are
outside Git provenance and must not supply research code. Missing Git metadata
remains explicitly unknown. Existing version-1 records remain readable, including
historical dirty flags, but those flags do not identify the uncommitted source.

## Producer coverage

| Input to `inspect_study` | Provenance and indexed output |
| --- | --- |
| `PREDICTION`: QF-11 result JSON or its manifest | Complete rule/version/parameters, outcome/evaluator definitions and horizons, feature configuration, dataset metadata, optional QF-28 context; existing rows stay in JSON |
| `PREDICTION_WINDOW`: QF-42 result JSON | Original schedule, window configuration/result IDs, context/data/backend requirements and decision collection |
| `FEATURE_DATASET`: QF-7/QF-29 directory or result JSON | Candidate/outcome configuration, feature schema, QF-29 contextual timeframe/backend/family provenance, source fingerprint, contributing prediction IDs, counts; CSV and optional Parquet |
| `PARAMETER_STUDY`: QF-32 directory | Search space, constraints, factory/analyzer, fixed backend, context family, ranking/stability and optional QF-42 schedule; summary, all persisted trials, successful prediction/window result files |
| `BACKTEST`: QF-5 directory or its `manifest.json`; detached manifest under another filename | Full strategy/indicators, capital, costs/fees/slippage, execution, corporate actions, QF-43 context/evaluation interval, benchmark configuration and source data; export inputs include original CSV tables and integrity record; detached manifests index metadata only |
| `OPTIMIZATION`: QF-6 directory | Existing scientific identity inputs, grid/constraints, ranking/stability, counts and trial IDs; summaries, complete `ranking.json`, trials and original successful QF-5 export files |

Study types never acquire another type's metric requirements. Prediction and
feature manifests require no capital, transaction costs, fills, or equity.
The manifest retains material configurations, not result tables. Schema fields
are copied from their producer; an engine version is not an artifact schema.
QF-5 artifacts retain their recorded `result_schema_version` and `run_id` in
both `producer_study_id` and `producer_run_id`, whether indexed standalone or
inside an optimization, fold or holdout. Nested backtest files use the same
indexing helper and retain `DERIVED_FROM` relationships to the enclosing trial,
window or holdout result. The outer study's manifest and trial records retain
their own producer identity and schema.
Passing a QF-5 export's `manifest.json` selects the same complete artifact set
and integrity-sidecar validation as passing its directory. Missing or modified
sibling files fail inspection, including a missing sidecar. For metadata-only
inspection, save a detached manifest under another filename, such as
`backtest.json`; this does not assert export completeness or table integrity.

QF-11 study IDs are verified against the original engine, market data, complete
study configuration and optional prediction context. For both complete results
and manifest-only inputs, `generated_predictions` must equal
`labeled_rows + unavailable_outcomes`; all three counts must be nonnegative
integers, excluding booleans. When rows are available, their length must also
match `labeled_rows`. Each row must reference that study and dataset, match the recorded
rule/outcome/evaluator metadata, and retain valid outcome, evaluation and row
identities. Duplicate row identities are rejected. These checks hash stored
fields; they do not regenerate predictions, labels or evaluations. Manifest-only
inputs retain producer-declared counts without asserting row verification.
The rule, labeler and evaluator configuration IDs also bind their complete stored
definitions. Row and generated-signal checks share the same symbol, component,
parameter, session and warm-up validation. QF-11 requires contiguous observed
daily sessions: the recorded calendar, first/last session and bar count establish
that session sequence using the existing calendar helper. Outcome sessions must
be exactly the configured number of exchange sessions after their signals,
including holidays and weekends. No label values or price metrics are calculated.
Standalone QF-11 results and manifest-only inputs also reconcile QF-28 context
status and requirements, source identity, selected bars and indicator metadata.
Available contexts bind the primary bar's exchange session and reject selected
bars after that bar's decision boundary. The source may have a later `as_of`;
staleness remains measured at that recorded capture time. Skipped contexts require
the explicit skip policy, a reason and zero generated predictions. Their rejected
source evidence is retained and checked internally, without treating it as usable
rule input; a missing provider result may have no source snapshot.
Direct inputs and QF-32 trials without a decision schedule use the same QF-11
manifest validator for identity, counts and context. Refreshing a nested study's
row IDs and artifact fingerprints cannot bypass these context checks.
QF-5 run IDs are verified against their documented market-data
reference, bar fingerprint, strategy and execution inputs, including the strategy
configuration hash. Full export metadata, such as initiation time and warm-up
diagnostics, does not become a new run-ID input.
Execution provenance must include position sizing. The supported QF-5 version-4
buy-and-hold benchmark configuration and deterministic benchmark ID must agree
with recorded capital, costs, corporate-action policies, snapshot and optional
evaluation interval. These checks derive only fixed metadata; no benchmark,
fills or performance metrics are calculated, and absent historical evaluation
intervals remain absent.
The shared nested-export indexer applies the same benchmark contract to QF-6
trials and QF-39/QF-40 fold and holdout backtests. Refreshing file hashes,
captured export fingerprints or enclosing record hashes cannot bypass this
reconciliation, including optimization directories without a final summary.

QF-7/QF-29 dataset IDs must match the producer's hash of its complete recorded
`configuration`. Result JSON with rows must reconcile `candidate_count` and every
accepted/rejected/blocked/overlapping count against the rows' fixed dispositions,
including any embedded summary. Counts must be nonnegative integers. Directory
inputs retain declared row counts without loading CSV or Parquet into research
objects. Their required `summary.json` must exactly match the manifest's
`record_counts`: every disposition count is a nonnegative integer and their sum
equals `candidate_count`. The index binds these summary counts to their stored
JSON fields. Equal totals with redistributed dispositions are still rejected.
Direct result rows additionally bind dataset/source fingerprints, candidate-rule
configuration, parameters, schema versions and namespace-specific prediction
study references. Candidate/row IDs, unique ordered sessions and disposition
evidence are verified. The schema must match the persisted feature/outcome
definitions, and every row must have exactly those columns with valid types and
nullability. Directory schema metadata receives the same definition check.
Every feature manifest, including directory inputs, requires one valid contributing
prediction-study digest per configured outcome; rows are not needed to check this
lineage declaration.
These checks reuse producer row hashing and schema-value validation; they do not
evaluate causal features or future outcomes.

Completed QF-6 exports must include `ranking.json` and `stability.json`. They are
indexed as required artifacts, so missing files and changed content fail integrity
verification. Both `ranking.json` and `stability.json` must declare the
inspected study's ID; that ownership is also retained as an index metadata binding.
Their configurations must match the study and summary. All ranking and stability
entries must reference unique compatible successful trials, including entries
beyond the summary's top ten. Eligible and ineligible lists must cover the saved
successful trials; stability must cover the eligible list. Stored objective
values, ranks, counts, top-ten projections and selected trial references must
agree across the artifacts, trial records and summary. Empty eligible lists are
valid. Objective ranks must follow the configured direction, each configured
tie breaker's metric/direction, then ascending combination ID. Undefined tie-break
metrics sort last in either direction. Stability ranks must follow descending
stored stability score, then objective rank and combination ID. The robust
recommendation must be the first stability-ranked record that is classified
stable, is not isolated, and lies within the configured ceiling-rounded fraction
of eligible objective ranks. A zero fraction or absence of qualifying records
requires a null recommendation. These checks compare existing metrics, scores
and classifications; they do not rerun eligibility, calculate neighbor statistics,
reclassify trials, sort/rewrite artifacts or invoke the ranking/stability engines.

For QF-32 and QF-6 grids with a persisted summary, inspection reconciles the
trial-file total and status counts against that summary. Missing or extra trial
files and contradictory counts are rejected, including failed and excluded
trials. A final summary requires one persisted trial at every Cartesian position:
the total is the product of the serialized axis lengths, and positions must be
unique and bounded. QF-6's manifest total/valid/excluded declarations and summary
total must agree with that grid and the recorded statuses. Deleting failure or
exclusion evidence cannot be hidden by decrementing summary or manifest counts.
This counts axes and checks saved coordinates without enumerating candidates or
reapplying constraints. Without a summary, incomplete grids remain indexable,
positions must still be unique and bounded, and final trial counts remain unknown.
The index describes
only the persisted records; it does not assert completion. Successful trials
must retain their result reference. Failed trials require nonempty diagnostic
type and message (plus QF-6's failure category); excluded trials require their
exclusion code and reason. Outcome fields must agree with the trial status.
Archived `failed_attempts` must be an array of producer-specific diagnostic records
for every current trial status. Existing QF-6 and QF-32 attempt readers validate
their fields; text fields are also checked explicitly. An omitted history retains
the producers' legacy empty-history default, and QF-6 nullable timestamps remain
supported. History is observed without retrying trials.
Prediction result fingerprints must match
both their content and the fingerprint in the trial record; the recorded analysis
and schema must also agree. A QF-32 summary must retain the study's schema and
partition all successful trials between unique ranking and ineligible references,
with no failed/excluded trials or foreign combination IDs. Eligible counts,
configured objective names, values from saved trial analyses, consecutive ranks,
declared objective order and combination-ID tie breaks must agree. Stability
references must cover the ranked trials in objective order and retain their ranks
and objective values. Ineligible references require recorded reasons; an empty
ranking with all successful trials ineligible remains valid. These checks do not
reapply eligibility constraints or recalculate neighborhood/stability statistics.
QF-6's advertised `study_schema_version` must match the version in its hashed
`identity_inputs`; the manifest index binds that same recorded schema field.
This applies to both directory and manifest-only inputs, preserving the original
version rather than substituting the installed producer's version.
QF-6 trial metrics, dataset, execution configuration
and strategy provenance must match the linked
QF-5 manifest, including its strategy-configuration hash and the grid's recorded
engine/schema versions. These comparisons use stored values only and do not
recalculate metrics or reconstruct strategies.

Every QF-6 and QF-32 trial also matches its declared Cartesian position in the serialized
search space. The combination ID binds those parameters to the recorded strategy
factory; the trial ID binds the combination to the study, resolved strategy
metadata, dataset and historical engine/schema versions. These checks include
failed and excluded trials and use the original axis/value order and primitive
types. They decode one saved position without enumerating the grid or invoking
the factory to recreate strategy parameters.
QF-32 uses its own factory/combination/trial identity format and checks the recorded
trial definition, backend, dataset-family fingerprint and indicator configuration
IDs. Historical schema versions and excluded candidates' null definitions are
preserved; prediction factories and candidate generation are never invoked.
Every trial's `schema_version` must equal the study's recorded version, including
failed/excluded records and directories without a final summary. Successful QF-32
trials require a complete canonical analysis record, validated by the producer's
pure deserializer. Counts, finite numeric metrics, comparison records and artifact
metadata must remain readable without discarded or defaulted fields. Missing,
null or malformed analysis is rejected even when the trial and result agree and
their fingerprints have been refreshed. Failed and excluded trials retain null
analysis. No analyzer or research metric calculation is invoked.
Successful plain-prediction and window artifacts must also match the trial's
recorded component definitions, context requirements, feature configuration and
result schema, and the grid's dataset and backend. Window results additionally
match the grid's schedule, context environment, dataset family and window engine.
Each artifact's prediction-study/window-result reference must match its nested
result. Rehashing a result from another candidate does not establish that binding.

QF-42 inspection requires the complete result snapshot, including decisions,
and explicitly supports window and schedule schema version `1`. Unknown, future
or corrupt versions are rejected before their contents can be indexed; a new
version requires an explicit adapter or migration.
It verifies the result identity over the recorded window ID and ordered decisions,
then checks all stored record counts using QF-42's primitive count helper. The
same checks apply to QF-32's nested windows and captured fold/holdout windows.
Changed or truncated decisions
cannot retain a stale result identity. Ordered decision timestamps must exactly
match the recorded schedule, even if the result hash and counts have been updated.
The separate `schedule_id` must equal the hash of the complete recorded schedule;
missing or stale schedule identities are rejected in standalone and grid windows.
The recorded timeframe and closed start/end interval are validated with QF-42's
pure `PredictionDecisionSchedule` calendar contract. Every completed primary-bar
boundary must be present, including leading/terminal partial-duration bars;
deleting a timestamp and its decision cannot be hidden by refreshing all hashes.
Regular/extended-hours policies, clock anchors, holidays, empty intervals,
overnight trade-date labels and terminal bars at midnight retain the producer's
semantics. Available contexts and their signals must use the resulting exchange
session. Only expected calendar boundaries are derived: no market observations,
contexts, decisions, predictions, outcomes or research metrics are generated.
Each nested QF-11 manifest must also match the window's recorded configuration,
market data and prediction-engine version, and its study ID must match the
decision's `prediction_study_id`. A self-consistent result from another study
cannot be substituted by updating the window's result hash or counts.
Each decision's context ID must match its retained QF-20 source snapshot, and its
requirements must match the window configuration. Available contexts must match
the decision timestamp and dataset family; existing offline producer validators
check source/selected timeframe, completion, bar-time, dataset and indicator
metadata without constructing market context or calculating indicators.
Skipped decisions retain rejected source evidence, which may have the wrong
timestamp/family or be absent, but require the explicit skip policy and empty
signals/rows. That evidence remains subject to internal source validation.
Decision status and generated counts must agree with the context and signals.
Every labeled row's stored features and prediction must match a distinct
generated signal; duplicate signals and substitutions are rejected.
Unlabeled end-of-data signals receive the same provenance/session checks as
labeled signals. A missing row is permitted only when the configured future
session lies beyond the recorded dataset range.

Embedded source-dataset entries reference the producer's recorded dataset
metadata. Their byte hash verifies that metadata file, not an unavailable source
cache. To verify original raw/canonical data bytes, explicitly add their files
with `index_artifact` as shown below.

Historical native configuration snapshots stay byte-equivalent as primitive
objects. Some legacy QF-5 configurations store native arithmetic/indicator
definitions without resolved backend version fields. QF-9 retains these original
definitions and IDs; absent fields remain absent/unavailable. It does not use
the installed version to fill them. Explicit `native_v1` and `talib_v1` metadata,
including wrapper/runtime versions and mapped functions, is retained exactly.
No TA-Lib call or indicator construction is needed to index either format.

## QF-8, QF-39 and QF-40 lineage

```python
from quantforge.experiments import inspect_validation
from quantforge.oos import HoldoutLedger, load_oos_source

# plan is the original QF-8 plan. This existing QF-40 reader verifies persisted
# source records; it has no evaluator or factory and performs no aggregation.
source = load_oos_source(plan, existing_walk_forward_directory)
validation = inspect_validation(
    source,
    existing_walk_forward_directory,
    artifact_root=root,
    aggregate_path=existing_oos_aggregate_json,
    ledger=HoldoutLedger(existing_permanent_ledger_directory),
    study_type=StudyType.OOS_VALIDATION,
)
write_manifest(
    create_manifest(validation, execution),
    root / "reports/experiments",
    artifact_root=root,
)

# prediction must be an inspected result captured in this validation bundle.
# QF-42 windows match by window_result_id; backtests match by their QF-5 run_id.
linked = create_manifest(prediction, execution, validation=validation)
```

Validation attachments require a prediction or backtest primary study of the
matching research family. Its result ID must match an indexed completed fold or
consumed holdout result in the validation bundle; a shared configuration, symbol
or dataset is insufficient. QF-42 uses its result ID, not its window configuration
ID. A standalone QF-11 result is not a captured QF-42 window. Feature datasets,
grids and validation studies cannot receive a `validation=` attachment.
The captured fold/holdout entry explicitly `VALIDATES` the primary configuration.
When both bundles reference the same QF-5 export files, the combined index reuses
the primary entries and redirects validation edges to them. Shared files must
agree on content hash, schema and producing study/run; all nested bindings must
already be retained by the primary entry. Conflicting aliases are rejected.

`inspect_validation` records the original plan/environment, boundary and warm-up
definitions, purging/embargo, walk-forward study, lineage ID, ordered fold
statuses, frozen selections and typed test-window artifacts. Completed artifacts
must match the supplied immutable source snapshots and their stored fold state.
Each complete persisted fold-state record must match the corresponding captured
`OOSSource.references` state, including selection, artifact and failure fields.
Changed records are rejected even when their envelopes have valid new hashes.
The presence or absence of a pending state record must also match the snapshot.
Backtest tables must match their existing QF-5 integrity sidecar, and the sidecar's
original text must match the captured QF-39 export fingerprint. This applies to
both fold and holdout exports, preventing rehashed table changes from being
attributed to the captured OOS result.
Nested backtest files retain their QF-5 manifest's `result_schema_version` and
producing `run_id`, matching standalone and optimization inspection; the
surrounding QF-39/QF-40 envelope version and study identity describe the enclosing
validation records.
Consumed holdout artifacts must match the frozen selection retained in their
permanent request and the captured source study. Prediction results receive the
same nested window checks as standalone results, bind their result IDs and stored
candidate definitions, and match the request's recorded partition, bounded dataset
and allowed decision sessions. Backtests must match the frozen candidate's full
strategy definition and the plan's recorded execution configuration, engine and
result schema. Their QF-43 evaluation interval must use the request's first/last
evaluation sessions and the producer's boundary contract. Dataset IDs and QF-3
`data_sha256` must match the requested bounded dataset; QF-5's independently
serialized `bars_fingerprint` is a different hash. Backtest result snapshots
must also match the indexed export manifest. The final backtest holdout summary
must exactly match the captured manifest's `performance`, as copied by QF-40;
changed, missing or extra metrics remain invalid under a refreshed ledger envelope.
A self-consistent replacement run cannot change those frozen inputs by retaining
the original selection ID.
These checks do not prepare/evaluate a holdout, rebuild membership
or recompute the holdout summary; consumed state remains authoritative.
Failed or absent folds stay failed or absent; stale files do not become OOS
observations. No partition memberships are calculated again.

An existing QF-40 aggregate must have the expected content ID and exact source
plan/study/lineage/fold references, and its family must match the source study.
Backtest `native_windows` must match the complete captured fold payloads, frozen
selections, export locations and fingerprints in source order. Normalized-equity
fold/run/session membership and window-start markers must match native equity;
per-window performance copies must match the captured backtest manifests.
Prediction observations must retain the source fold, selection, window result,
decision/context/study references, generated signals and matching stored rows.
Missing, duplicate, reordered or foreign records are rejected even under a new
aggregate content hash. Window summary membership and availability must preserve
failed and missing folds, including partial and empty aggregates.
Its stability section receives an index entry using a JSON pointer. Normalized
index values, prediction summary statistics and stability calculations remain
producer-owned; these checks bind copied evidence without recalculating metrics.
The graph links aggregate to contributing test artifacts, test artifacts to
frozen selections, selections to the plan, and the plan to source provenance.

`HoldoutLedger.state(source)` is the sole holdout authority. A supplied ledger
must already have the reservation; QF-9 never creates/reserves/consumes it. The
manifest references the reservation, permanent consumption marker, original
consumption run/time/request, and result when available. A failed evaluation
remains consumed with no result. Regenerating through the same ledger retains
consumption. State is checked again after reading to reject a concurrent
transition during indexing. There is no second ledger.

Without a ledger, ordinary validation indexing says **unknown** (`state: null`),
never pristine or unconsumed. `HOLDOUT_VALIDATION` requires a ledger. QF-8's
reservation metadata is not evidence of current holdout state. An older immutable
manifest is a historical snapshot, not a live pristine claim: QF-41 and other
current-state consumers must query the permanent QF-40 ledger. Missing/corrupt
ledger evidence raises its existing integrity error; it is never reset here.

## Additional artifacts and relationships

`ArtifactType` supports source data/quality, feature schemas and rows, outcomes,
prediction/backtest/trial results, validation plans, fold state/selections,
OOS/stability, holdout records, inspection, chart and future report categories.
Formats include JSON, CSV, Parquet, HTML, SVG, PNG and explicitly declared binary.
Arbitrary URLs, remote stores and downloads are not supported in version 1.

```python
from quantforge.experiments import (
    ArtifactRelationship,
    ArtifactType,
    RelationshipType,
    index_artifact,
)

chart = index_artifact(
    root,
    path="reports/inspection/<report-id>/report.html",
    artifact_type=ArtifactType.INSPECTION,
    schema_version="1",
    producer_study_id=prediction.provenance.producer_study_id,
    producer_artifact_id="inspection-html",
)
prediction_result = next(
    entry
    for entry in prediction.index.entries
    if entry.artifact_type is ArtifactType.PREDICTION_RESULT
)
linked = create_manifest(
    prediction,
    execution,
    additional_artifacts=(chart,),
    relationships=(
        ArtifactRelationship(
            chart.artifact_id,
            RelationshipType.VISUALIZES,
            prediction_result.artifact_id,
        ),
    ),
)
```

The same API attaches canonical source files, quality reports or a feature
dataset to a prediction (`USES_FEATURES`), without rendering/rebuilding them.
Relationships are caller-declared provenance unless supplied by a producer
adapter. Do not assert an association merely because files share a directory.
Optional known row/count/shape or creation metadata can be supplied through
`metadata`; no counts are computed from CSV/Parquet. For embedded JSON records,
`json_pointer` uses RFC 6901 and `bindings` maps JSON pointers to exact expected
stored metadata, such as an ID or schema version.

## Integrity, immutability and schema evolution

`verify_artifacts(index, artifact_root)` verifies normalized relative paths,
root containment (including symlink resolution), SHA-256 of exact file bytes,
JSON pointer resolution and metadata bindings. Missing referenced files,
hash mismatch, invalid JSON or incompatible metadata are explicit issues.
`require_valid()` raises on failures. Verification never replaces hashes.
A missing optional file is allowed only when it was absent at indexing and has
`sha256: null`; a later appeared or disappeared file requires a new index.

`write_manifest` verifies artifacts, writes/fsyncs a temporary file, publishes it
using an atomic no-clobber hard link, and fsyncs the directory. It writes
`<output-root>/<manifest-id>.json`. An existing destination is reused only for
exact identical bytes. Changed content is never silently overwritten. All
normal output belongs under ignored `reports/`; inputs may reside in existing
immutable `data/` caches. Keep the artifact root and relative layout together
when moving an experiment. File bytes are not copied or packaged by QF-9.

Manifest and artifact-index schemas both start at **version "1"**. The index is
embedded in the manifest and its content ID is verified; the manifest ID binds
every indexed artifact without a circular self-hash. The strict reader rejects
duplicate JSON keys, noncanonical bytes, unknown fields, modified IDs, missing
fields, and unsupported versions. There is no earlier QF-9 schema to migrate.
Existing producer manifests are inputs to adapters, not prior QF-9 versions.
Future migration must explicitly define material defaults and identity changes;
version 1 guesses none. Optional unknown execution metadata serializes as null.

Hashes detect corruption relative to a trusted manifest; they do not authenticate
a provider or an attacker who can replace both artifacts and all hashes. Keep
producer exports quiescent while indexing. File hashing reads bytes and JSON
metadata; there is no metric calculation. QF-9 does not validate statistical
correctness or recreate a lost source dataset.

### Producer contract verification

The adapters check related invariants together. Regression tests change individual
fields, replace records with outputs from other real studies, and recompute
enclosing hashes so that consistency checks are exercised beyond hash mismatch.

| Contract family | Stored evidence checked | Boundary |
| --- | --- | --- |
| QF-11 | Study/component IDs, dataset/parameter provenance, ordered unique signal sessions, warm-up, outcome horizons, row/outcome/evaluation IDs and counts | Calendar metadata validation only; no labeling or evaluation |
| QF-42 / nested QF-32 | Window/result/schedule identities and coverage, decision/context lineage and status, generated-signal provenance, distinct signal-to-row membership and unavailable outcomes | Reuses offline source/context checks; no context or indicator execution |
| QF-7/QF-29 | Dataset/candidate/row IDs, source/rule/schema versions, prediction-study references, schema definitions/types, dispositions and matching manifest/directory-summary counts | Existing values are validated structurally; features and outcomes are not recalculated |
| QF-6/QF-32 grids | Coordinates, candidate/trial identity, status payloads, result bindings, recorded metrics, ranking/stability references, eligibility coverage/counts and summary projections | No factory construction, candidate enumeration or research selection |
| QF-5 standalone / optimization / validation exports | Run provenance, original integrity sidecar, captured fold/holdout fingerprints, original schema versions and producing QF-5 run IDs on every backtest file | No execution or accounting |
| QF-8/QF-39/QF-40 lineage | Captured source and fold state, selections, aggregates and permanent holdout-ledger state | No partition, aggregate or holdout computation |
| Credential metadata | Recursive normalized field-name and recognizable-value rejection at construction, JSON and binding boundaries | Conservative field filtering cannot discover arbitrary disguised secrets |

Some producer facts require original bars, strategy objects or research execution
to verify. They are not inferred from a self-consistent replacement of all records.
For example, missing source-rule definitions remain unavailable; schema-correct
feature values are not evidence that the feature calculation itself was correct.

## Secrets and reproducibility

Only explicit primitive research snapshots are accepted. No provider object,
object `__dict__`, process environment, or raw exception text is copied. A
recursive guard rejects credential-related keys, bearer/private-key material,
and authenticated/query-bearing URLs in metadata, bindings and read JSON.
Errors do not echo credentials. Provider/feed identities remain ordinary
provenance. The exact `account_id` field permits only QF-5's fixed simulation
labels `benchmark` and `strategy`; other account identifiers remain prohibited.
Account identifier/number/name/reference aliases, including `account_number`,
`broker_account` and `trading_account`, are rejected across construction, JSON
reading and metadata bindings. This does not reject accounting-policy fields
such as `account_initialization` and `corporate_action_accounting`.
The field-name guard conservatively rejects normalized keys containing `token`,
including compound names such as `api_token`, `auth_token` and `session_token`,
with the same policy for nested records and metadata bindings.
Access-key aliases such as `access_key` and `aws_access_key_id` are also rejected
after normalization of case and separators.
This guard cannot discover an arbitrary secret disguised as an
unrelated free-text value; callers must supply research configuration only.

A fully supplied manifest identifies the original code, dependencies, dataset
and material configuration for reproduction through that study's existing
execution API. Unknown historical execution metadata limits this claim and must
be recovered from trustworthy records, not inferred from today's runtime.
QF-9 implements no universal replay engine, report renderer, new chart, metric,
ML registry, strategy or broker functionality.

## Offline acceptance examples

```bash
uv run pytest tests/unit/experiments/test_adapters.py
uv run pytest tests/unit/experiments/test_validation.py
uv run pytest tests/unit/experiments
```

These tests first produce tiny synthetic QF-11/QF-7/QF-29/QF-32/QF-42/QF-5/QF-6
and two-fold QF-39/QF-40 artifacts, then disable research entry points and
native/TA-Lib computation before manifest generation. They cover successful and
failed holdout consumption, regeneration, integrity, schemas, relationships,
historical backend preservation and immutable serialization. Existing QF-34
HTML is indexed directly. No live API, credential or profitability claim is used.
