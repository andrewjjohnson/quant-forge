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
| `BACKTEST`: QF-5 directory or manifest | Full strategy/indicators, capital, costs/fees/slippage, execution, corporate actions, QF-43 context/evaluation interval, benchmark configuration and source data; original CSV tables and integrity record |
| `OPTIMIZATION`: QF-6 directory | Existing scientific identity inputs, grid/constraints, ranking/stability, counts and trial IDs; summaries, complete `ranking.json`, trials and original successful QF-5 export files |

Study types never acquire another type's metric requirements. Prediction and
feature manifests require no capital, transaction costs, fills, or equity.
The manifest retains material configurations, not result tables. Schema fields
are copied from their producer; an engine version is not an artifact schema.

QF-11 study IDs are verified against the original engine, market data, complete
study configuration and optional prediction context. When rows are available,
their length must match `labeled_rows`, and `generated_predictions` must equal
`labeled_rows + unavailable_outcomes`; all three counts must be nonnegative
integers. Each row must reference that study and dataset, match the recorded
rule/outcome/evaluator metadata, and retain valid outcome, evaluation and row
identities. Duplicate row identities are rejected. These checks hash stored
fields; they do not regenerate predictions, labels or evaluations. Manifest-only
inputs retain producer-declared counts without asserting row verification.
QF-5 run IDs are verified against their documented market-data
reference, bar fingerprint, strategy and execution inputs, including the strategy
configuration hash. Full export metadata, such as initiation time and warm-up
diagnostics, does not become a new run-ID input.

QF-7/QF-29 dataset IDs must match the producer's hash of its complete recorded
`configuration`. Result JSON with rows must reconcile `candidate_count` and every
accepted/rejected/blocked/overlapping count against the rows' fixed dispositions,
including any embedded summary. Counts must be nonnegative integers. Directory
inputs retain declared row counts without loading CSV or Parquet into research
objects.

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
valid. Inspection compares persisted metadata and metrics without rerunning
eligibility, ranking, stability statistics or recommendation rules.

For QF-32 and QF-6 grids with a persisted summary, inspection reconciles the
trial-file total and status counts against that summary. Missing or extra trial
files and contradictory counts are rejected, including failed and excluded
trials. Without a summary, trial counts remain unknown and the index describes
only the persisted records; it does not assert completion. Successful trials
must retain their result reference. Failed trials require nonempty diagnostic
type and message (plus QF-6's failure category); excluded trials require their
exclusion code and reason. Outcome fields must agree with the trial status.
Prediction result fingerprints must match
both their content and the fingerprint in the trial record; the recorded analysis
and schema must also agree. QF-6 trial metrics, dataset, execution configuration
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
Successful plain-prediction and window artifacts must also match the trial's
recorded component definitions, context requirements, feature configuration and
result schema, and the grid's dataset and backend. Window results additionally
match the grid's schedule, context environment, dataset family and window engine.
Each artifact's prediction-study/window-result reference must match its nested
result. Rehashing a result from another candidate does not establish that binding.

QF-42 inspection requires the complete result snapshot, including decisions.
It verifies the result identity over the recorded window ID and ordered decisions,
then checks all stored record counts using QF-42's primitive count helper. The
same checks apply to QF-32's nested window results. Changed or truncated decisions
cannot retain a stale result identity. Ordered decision timestamps must exactly
match the recorded schedule, even if the result hash and counts have been updated.
This reads existing signals and rows;
it does not rebuild schedules, contexts, predictions, outcomes or metrics.
Each nested QF-11 manifest must also match the window's recorded configuration,
market data and prediction-engine version, and its study ID must match the
decision's `prediction_study_id`. A self-consistent result from another study
cannot be substituted by updating the window's result hash or counts.

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

# An explicit attachment binds validation configuration/lineage into a
# prediction or backtest experiment identity and preserves the same index.
linked = create_manifest(prediction, execution, validation=validation)
```

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
Failed or absent folds stay failed or absent; stale files do not become OOS
observations. No partition memberships are calculated again.

An existing QF-40 aggregate must have the expected content ID and exact source
plan/study/lineage/fold references. Its stability section receives an index entry
using a JSON pointer; neither stability nor any aggregate metric is recalculated.
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

## Secrets and reproducibility

Only explicit primitive research snapshots are accepted. No provider object,
object `__dict__`, process environment, or raw exception text is copied. A
recursive guard rejects credential-related keys, bearer/private-key material,
and authenticated/query-bearing URLs in metadata, bindings and read JSON.
Errors do not echo credentials. Provider/feed identities remain ordinary
provenance. The exact `account_id` field permits only QF-5's fixed simulation
labels `benchmark` and `strategy`; other account identifiers remain prohibited.
The field-name guard conservatively rejects normalized keys containing `token`,
including compound names such as `api_token`, `auth_token` and `session_token`,
with the same policy for nested records and metadata bindings.
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
