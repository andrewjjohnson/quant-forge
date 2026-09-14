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
It does not recreate any producer's identity algorithm. The additional binding
prevents changed supplied configuration from aliasing an experiment even if a
producer ID was inadvertently reused. Dataset/family, source and derived
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
| `OPTIMIZATION`: QF-6 directory | Existing scientific identity inputs, grid/constraints, ranking/stability, counts and trial IDs; summaries, trials and original successful QF-5 export files |

Study types never acquire another type's metric requirements. Prediction and
feature manifests require no capital, transaction costs, fills, or equity.
The manifest retains material configurations, not result tables. Schema fields
are copied from their producer; an engine version is not an artifact schema.
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
