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
| `OPTIMIZATION`: QF-6 directory | Existing scientific identity inputs, grid/constraints, ranking/stability, counts and trial IDs; JSON summaries, seven reconciled native CSV tables for completed studies, trials and original successful QF-5 export files |

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
study configuration and optional prediction context. Every standalone, manifest-only
or nested QF-11 manifest must retain the exact producer `feature_outcome_boundary`
declaration: predictions are fixed before outcome labeling, and evaluators receive
only a fixed signal and an already-generated outcome. This required declaration
remains outside study identity; refreshing enclosing hashes cannot replace it.
For both complete results
and manifest-only inputs, `generated_predictions` must equal
`labeled_rows + unavailable_outcomes`; all three counts must be nonnegative
integers, excluding booleans. When rows are available, their length must also
match `labeled_rows`. Each row must reference that study and dataset, match the recorded
rule/outcome/evaluator metadata, and retain valid outcome, evaluation and row
identities. Duplicate row identities are rejected. These checks hash stored
fields; they do not regenerate predictions, labels or evaluations. Manifest-only
inputs retain producer-declared counts without asserting row verification.
Every prediction-rule wrapper requires a positive integer warm-up, excluding
booleans. When the captured rule configuration also declares warm-up, the two
values must agree. This shared check covers standalone, manifest-only and nested
QF-32/QF-39/QF-40 results, including window headers without decisions. Generic
rule configurations that omit a duplicate declaration retain the validated
wrapper value; inspection never constructs a rule to infer missing metadata.
Outcome wrappers likewise require a positive integer future-session horizon,
sorted unique nonempty market-field names and a nonempty result schema. Horizons
must agree with captured `parameters.future_sessions` or `required_future_sessions`
declarations; market fields and outcome/evaluator result schemas must agree with
their captured declarations when present. These shared checks cover manifest-only,
standalone and nested results, including empty window headers. Generic components
without duplicate declarations retain their validated wrapper metadata.
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
The manifest must retain every producer field, including nullable initiation time,
performance, record counts, warnings and limitations. Market provenance, strategy
warm-up, performance counters/finite decimal strings, benchmark metadata and
corporate-action declarations must retain their complete schemas and primitive
domains. Empty disclosure arrays and producer-nullable metrics remain valid.
These checks apply to standalone and nested backtests; refreshing the integrity
sidecar cannot certify an incomplete manifest. Performance is not recalculated.
For complete exports, every `record_counts` value must also match its captured
CSV table. Logical CSV records are counted with quoted newlines preserved;
`trades.csv` is partitioned by its required `is_open` boolean so an unchanged
total cannot hide redistributed completed/open counts. The same captured bytes
must match the sidecar and the final index. Header-only tables remain valid for
zero counts. Detached manifests retain declared counts without table verification.
Execution provenance must include the complete supported position-sizing record:
`model: "discrete_target_weight"`, boolean `whole_shares_only: true` and boolean
`rebalance_existing_position: false`, with no missing or additional fields.
Rehashed run/benchmark IDs do not make malformed sizing valid, and inspection
does not fill in defaults. The complete execution configuration must also retain
QF-5's fixed next-session-open market-order record, split/dividend timing policies,
arithmetic policy, version bindings and long-only/no-forced-liquidation flags.
Capital must be positive, the finite annual risk-free rate must exceed -1, and the
annualization factor must be a positive integer. An optional evaluation interval
must retain its complete producer contract and ordered session dates; historical
absence remains absent. Native version-1 costs require their exact schemas and
nonnegative decimal parameters, with slippage below 10,000 basis points. Custom
cost records retain their producer-owned shapes and explicit implementation
versions; commissions and fees must record the nondecreasing buy-cost guarantee.
These checks compare pure configuration metadata without constructing a
`BacktestConfig` or invoking any cost or execution callback. The supported QF-5 version-4
buy-and-hold benchmark configuration and deterministic benchmark ID must agree
with recorded capital, costs, corporate-action policies, snapshot and optional
evaluation interval. These checks derive only fixed metadata; no benchmark,
fills or performance metrics are calculated, and absent historical evaluation
intervals remain absent.
Benchmark order/signal/fill IDs must derive from the recorded benchmark ID.
The order must match the run, canonical symbol and benchmark strategy; the fill
must match its order, signal, symbol, strategy, quantity and execution session.
The fixed buy/market/long metadata and filled/rejected states are checked, with
zero requested quantity and absent fill for insufficient-cash rejections. These
checks validate saved relationships without calculating affordable quantities,
fill prices, costs or performance, and apply to detached and nested manifests.
The shared nested-export indexer applies the same benchmark contract to QF-6
trials and QF-39/QF-40 fold and holdout backtests. Refreshing file hashes,
captured export fingerprints or enclosing record hashes cannot bypass this
reconciliation, including optimization directories without a final summary.

QF-7/QF-29 dataset IDs must match the producer's hash of its complete recorded
`configuration`. Every direct or directory manifest requires the exact ten-field
producer schema, including its fixed `feature_outcome_boundary` declaration and
`limitations` array of strings. The declaration requires candidate dispositions
and causal features to be fixed before any QF-11 outcome labeler is invoked.
Missing fields, extra fields and malformed disclosures are rejected. Empty
limitation arrays, empty strings, duplicates and original order remain intact;
inspection does not invent disclosures or add them to dataset identity.
Direct result JSON requires the complete producer envelope:
`manifest`, `rows`, `schema` and `summary`, including an explicit empty row array
for an empty dataset. Rows must reconcile `candidate_count` and every
accepted/rejected/blocked/overlapping count against the rows' fixed dispositions,
including the required embedded summary. Counts must be nonnegative integers.
Directory inputs validate persisted `rows/*.json` checkpoints without generating features
or outcome labels. Their required `summary.json` must exactly match the manifest's
`record_counts`: every disposition count is a nonnegative integer and their sum
equals `candidate_count`. The index binds these summary counts to their stored
JSON fields. Equal totals with redistributed dispositions are still rejected.
Both direct result rows and directory checkpoints additionally bind dataset/source fingerprints, candidate-rule
configuration, parameters, schema versions and namespace-specific prediction
study references. Candidate/row IDs, unique ordered sessions and disposition
evidence are verified. The schema must match the persisted feature/outcome
definitions, and every row must have exactly those columns with valid types and
nullability. Directory schema metadata receives the same definition check.
Each outcome's `configuration_id` must equal the canonical hash of its saved
`component_configuration` before schema derivation. Refreshing the outer dataset
ID, row IDs and table bytes cannot hide a contradictory component identity.
Every feature manifest, including directory inputs, requires one valid contributing
prediction-study digest per configured outcome; rows are not needed to check this
lineage declaration. Outcome namespaces must be unique even when their field names
are disjoint, so two configured outcomes cannot collapse into one lineage entry.
Directory inputs require the checkpoint directory, including for an empty dataset.
Checkpoint filenames must equal their row IDs, and checkpoint counts must match
the manifest. CSV bytes must equal the native serialization of the validated rows
in session order. QF-29 Parquet is decoded from one captured byte buffer and
compared logically with the producer's table: ordered column names and types,
nullability, row order, values and nulls must match. Its QuantForge dataset ID and
parsed schema metadata must also match. Compression, dictionary encoding,
statistics, row-group layout, writer annotations and JSON metadata formatting
may differ without changing those scientific records. A neighboring Parquet file
in a QF-7 export is not a producer artifact and is omitted. Every consumed
checkpoint is indexed with its row/study binding;
tables and summary entries reference those rows through `DERIVED_FROM` edges.
The read-set check binds both checkpoints and table bytes through inspection.
After final byte verification, inspection also rechecks the captured `rows/*.json`
file set and the required checkpoint directory, including for empty exports.
Concurrent checkpoint additions, removals or renames require a fresh inspection;
unowned temporary files do not change checkpoint membership.
Parquet inspection retains the original file's hash and never invokes a Parquet
writer. Historical exports need not reproduce the current writer's bytes, but
must remain readable by the installed decoder. Inspection never rewrites exports.
These checks reuse producer serialization, row hashing and schema-value
validation; they do not evaluate causal features or future outcomes.

Completed QF-6 summary exports require the complete `StudyResult` envelope plus
the producer's ranking/stability configuration, top-trial and parameter-summary
extensions. Required `warnings` and `limitations` must be string arrays; missing
or extra fields and malformed disclosures are rejected. Valid disclosure text,
order, duplicates and empty arrays remain unchanged.
Completed QF-6 exports must include `ranking.json` and `stability.json`. They are
indexed as required artifacts, so missing files and changed content fail integrity
verification. Both `ranking.json` and `stability.json` must declare the
inspected study's ID; that ownership is also retained as an index metadata binding.
Their configurations must match the study and summary. All ranking and stability
entries must reference unique compatible successful trials, including entries
beyond the summary's top ten. QF-6 stability entries must retain the complete
`StabilitySummary` field set. They share QF-32's checks for integer counts/ranks,
finite decimal strings, nullable statistics, boolean flags, classification and
isolation reasons. The eligible-neighbor count must match the objective array and
not exceed the valid-neighbor count; standard deviation must be nonnegative and
constraint fractions and stability scores must lie within `[0, 1]`. Completed
entries require an assigned positive stability rank. Updating summary and CSV
projections does not make an incomplete or malformed record valid.
Eligible and ineligible lists must cover the saved
successful trials; stability must cover the eligible list. Stored objective
values, ranks, counts, top-ten projections and selected trial references must
agree across the artifacts, trial records and summary. The summary's
`objective_distribution` must have exactly a nonnegative integer count and the
minimum/maximum of the saved eligible objective values; empty eligible lists
require count zero and null extrema. No objective metric is recalculated. Objective ranks must follow the configured direction, each configured
tie breaker's metric/direction, then ascending combination ID. Undefined tie-break
metrics sort last in either direction. Stability ranks must follow descending
stored stability score, then objective rank and combination ID. The robust
recommendation must be the first stability-ranked record that is classified
stable, is not isolated, and lies within the configured ceiling-rounded fraction
of eligible objective ranks. A zero fraction or absence of qualifying records
requires a null recommendation. These checks compare existing metrics, scores
and classifications; they do not rerun eligibility, calculate neighbor statistics,
reclassify trials, sort/rewrite artifacts or invoke the ranking/stability engines.
Completed QF-6 exports also require all seven native CSV tables: `trials.csv`,
`failures.csv`, `exclusions.csv`, `eligible_rankings.csv`, `ineligible_trials.csv`,
`stability.csv` and `parameter_summary.csv`. Each must match the native CSV
serialization of the validated saved JSON records, including headers, row order,
JSON-valued cells, nulls and diagnostics. Inspection projects stored records only;
it does not recompute metrics or ranking/stability results or rewrite any file.
Before CSV projection, each parameter-summary record must retain exactly the
producer's eight fields. Parameter names are nonempty strings, values retain
their string/integer/boolean types, counts are nonnegative integers with eligible
counts no greater than successful counts, and constraint fractions are finite
decimal strings in [0, 1]. Objective statistics are finite decimal strings when
eligible samples exist and null otherwise. Empty categorical strings remain
valid values. Regenerating matching blank CSV cells cannot hide missing fields.
The expected bytes are checked against both the index and the file before return.
The completion decision uses the captured summary; its removal during inspection
cannot bypass table reconciliation.
Without `summary.json`, the directory remains resumable and potentially stale CSVs
and derived `ranking.json`/`stability.json` files are omitted from the index.
The producer writes these before its completion summary, so either or both may
be absent, stale or interrupted without preventing inspection of saved trials.
Once all saved trials are terminal and the captured completion summary exists,
both JSON summaries and all seven CSVs are required and reconciled. Unrecognized
neighboring files are always omitted. QF-6 summary indexing allows only these
seven CSV filenames and `summary.json`, `ranking.json` and `stability.json`.
QF-32 emits only `summary.json` as a top-level summary. Neighboring CSVs,
`result.json` and other unrecognized files are not treated as grid summaries,
whether the directory is complete or resumable.

QF-32 inspection supports study schema `"1"`, matching the producer's explicit
schema contract. Unsupported, missing, or non-string versions are rejected before
study/trial validation, including empty resumable directories and manifest-only
inputs; rehashing a future schema's identity does not make it supported.

For QF-32 and QF-6 grids with a current persisted summary, inspection reconciles the
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
QF-32 and QF-6 can retain old summaries while retrying a failed trial. When saved
trials include a pending or running record, inspection omits that stale summary and
derived exports and leaves final counts unknown. Trial validation and indexing use
the same captured bytes that establish this state; changes during inspection are
rejected. A completed retry's replacement summary is validated and indexed normally.
The owned `trials/*.json` file set is captured before trial reads and checked again
after final byte verification. Concurrent additions or removals require a fresh
inspection, including initially empty directories and resumable grids without a
summary. Temporary files outside that producer-owned pattern remain unindexed.
Inspection does not remove or rewrite producer files or retry any trial.
The index describes
only the persisted records; it does not assert completion. Successful trials
must retain their result reference. Failed trials require nonempty diagnostic
type and message (plus QF-6's failure category); excluded trials require their
exclusion code and reason. Outcome fields must agree with the trial status.
Every trial must also pass its producer's pure record deserializer. Required
nullable keys must be present even when their value is null, including QF-6
`metrics` and QF-32 `analysis` on pending/running trials. Documented producer
defaults for omitted `failed_attempts` and non-success QF-32 artifact fingerprints
remain supported; inspection does not replace missing required fields with defaults.
Failed QF-32 trials additionally require nonempty string `started_at` and
`finished_at` values, including in directories without a final summary, because
the producer needs both to archive a retry. Inspection preserves these timestamps
without retrying the trial. Other statuses and QF-6 retain their existing contracts.
Archived `failed_attempts` must be an array of producer-specific diagnostic records
for every current trial status. Existing QF-6 and QF-32 attempt readers validate
their fields; text fields are also checked explicitly. An omitted history retains
the producers' legacy empty-history default, and QF-6 nullable timestamps remain
supported. History is observed without retrying trials.
Prediction result fingerprints must match
both their content and the fingerprint in the trial record; the recorded analysis
and schema must also agree. A current QF-32 summary must retain the producer's
exact top-level field set, including required `warnings` and `limitations` string
arrays. Missing or additional fields, non-array disclosures and non-string members
are rejected. Valid disclosure text, order, duplicates and empty arrays are
preserved without generating or rewriting disclosures. Stale retry summaries
remain unindexed. A QF-32 summary must retain the study's schema and
its exact `PredictionGridCacheStatistics` record: `context_hits`, `context_misses`,
`indicator_hits` and `indicator_misses` must all be nonnegative integers, not
booleans or numeric strings. Cache usage remains recorded diagnostics and is not
reconstructed from trials. The summary must
partition all successful trials between unique ranking and ineligible references,
with no failed/excluded trials or foreign combination IDs. Eligible counts,
configured objective names, values from saved trial analyses, consecutive ranks,
declared objective order and combination-ID tie breaks must agree. Stability
references must cover the ranked trials in objective order and retain their ranks
and objective values. Each stability record must have exactly the fields emitted
by `PredictionStabilitySummary`, with nonnegative integer neighbor counts,
finite decimal strings (or null for optional statistics), boolean flags, a known
classification and a nullable nonempty isolation reason. Ranks must be positive,
dispersion nonnegative, and constraint fractions within `[0, 1]`. Eligible-neighbor
counts must match the stored objective array and not exceed valid-neighbor counts;
an isolated peak requires a reason. These checks validate recorded types and
internal consistency without computing neighbors, statistics or classifications.
Ineligible references require recorded reasons; an empty
ranking with all successful trials ineligible remains valid. These checks do not
reapply eligibility constraints or recalculate neighborhood/stability statistics.
QF-6 inspection supports study schema `"1"`, matching the producer's explicit
contract. Its hashed `identity_inputs` must declare that supported version, and
the outer `study_schema_version` must match it; the index binds that same field.
This applies to directory and manifest-only inputs, including empty resumable
stores. Rehashing unsupported, missing, or non-string versions does not make them
supported. Inspection does not migrate or substitute a recorded schema version.
An indexed QF-6 completion summary must retain that same `study_schema_version`;
missing, malformed, or different versions are rejected. Stale summaries omitted
during pending/running retries do not supply a schema contract for the index.
QF-6 trial metrics, dataset, execution configuration
and strategy provenance must match the linked
QF-5 manifest, including its strategy-configuration hash and the grid's recorded
engine/schema versions. These comparisons use stored values only and do not
recalculate metrics or reconstruct strategies.
Pending, running, failed and excluded QF-6 trials must retain exactly the study's
dataset mapping, matching the native resume contract; additional dataset fields
are rejected. Successful trials retain the enriched QF-5 market-data mapping
and are checked against their linked backtest export.

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
New QF-32 candidates capture the existing QF-11 configuration snapshot as trial
definition contract version `2`. This includes rule warm-up, outcome horizon and
market fields, and outcome/evaluator result schemas even when generic component
configurations omit duplicate declarations. The grid records
`trial_definition_version: "2"` in its identity; the definition records the same
`contract_version`. All executable trial records require those wrapper fields,
and successful nested results must match their frozen values exactly. Rehashing
a changed result cannot change its candidate contract.
Version `2` compares the entire prediction-rule, outcome-labeler and evaluator
wrapper, including its key set; extra fields, even null fields, are rejected in
plain and windowed results. Projection onto captured fields is reserved for
legacy version `1` definitions.
Absent version markers denote legacy version `1`. QF-9 reads legacy wrappers
from explicit declarations in the saved component configuration, including
`parameters.future_sessions`; it never constructs a component to infer them.
Legacy records without enough captured information raise `ManifestError` naming
the unavailable contract field. Historical files and identities are not migrated
or rewritten. New producer grids have distinct study/trial identities and cannot
resume a legacy store across this boundary; use its recorded producer version.
QF-9 manifest schemas and prediction result-row schemas remain unchanged.
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
the producer component `quantforge_prediction_window`, and explicitly supports
window and schedule schema version `1`. A missing, malformed or foreign component
is rejected even when the window and result identities have been rehashed.
Unknown, future or corrupt versions are rejected before their contents can be
indexed; a new version requires an explicit adapter or migration.
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
Standalone, optimization, fold and holdout inspections capture the sidecar bytes
before producer validation and bind every named file digest to both the index and
the final bytes on disk. Atomic CSV or sidecar replacements during inspection
therefore fail even after the initial producer check; identical-byte replacements
remain valid. Captured fold/holdout fingerprints use that same sidecar snapshot.
Standalone and nested backtest indexes include only the producer's exact filename
allowlist, which the captured sidecar must cover. Files created after validation
are left unindexed; they cannot acquire trusted backtest provenance through a
directory-listing race. Missing or changed allowlisted files still fail inspection.
Nested backtest files retain their QF-5 manifest's `result_schema_version` and
producing `run_id`, matching standalone and optimization inspection; the
surrounding QF-39/QF-40 envelope version and study identity describe the enclosing
validation records. Fold and holdout backtest run IDs must also equal QF-5's hash
of the recorded code/schema versions, dataset, strategy and backtest configuration.
Renaming an export and refreshing its sidecar and parent fingerprints cannot
substitute an arbitrary run ID.
Both prediction and backtest final-holdout results must retain the producer's
top-level `schema_version: "1"`, `kind: "final_holdout_result"` and
`state: "consumed"` before indexing. Missing, malformed or unsupported values
remain invalid after refreshing envelope hashes and ledger result references.
Rejecting incompatible evidence never changes the ledger's consumed state.
Before indexing a consumption marker, QF-9 requires the complete producer envelope,
its supported schema/state/transition constants, source lineage and request hash,
a nonblank execution ID, and an aware UTC consumption timestamp. Its entire
exposure scope must equal the reservation already checked by `HoldoutLedger.state()`
against the source, including the symbol and exchange-session range. Extra scope
or marker fields are rejected. Interrupted attempts receive the same checks;
refreshing a completed result's `consumption_sha256` cannot bypass them. Valid
execution metadata is preserved without normalization or a new ledger transition.
Every consumed request must retain the producer's exact top-level fields and
supported schema, operation and tail-policy constants. Its complete lineage,
study definition, study/plan/lineage IDs and final-holdout reservation must match
the captured source; its full frozen selection must match a captured fold.
These checks run before indexing consumption, including interrupted attempts
without a result. Refreshing request IDs and result-envelope hashes cannot change
the declared holdout boundary or lineage. No partition or holdout is prepared.
The saved `evaluation_membership` must retain its full partition and selection
schemas, source/window/timeframe identities, bounded-dataset digests, explicit
null purge, and false warm-up eligibility. Captured observations are ordered,
unique and bound to the reserved window, with the declared warm-up kept before
it. Evaluation sessions must match the captured observations after the recorded
outcome horizon and meet the source minimum. Missing result files do not bypass
these checks; validation never loads prices or prepares a partition.
Consumed holdout artifacts must match the frozen selection retained in their
permanent request and the captured source study. Prediction results receive the
same nested window checks as standalone results, bind their result IDs and stored
candidate definitions, and match the request's recorded partition and bounded dataset.
Version `2` frozen candidates require exact rule, labeler and evaluator wrappers,
including their key sets, using the same comparison as successful grid trials.
Extra fields remain invalid after refreshing nested study/window identities and
the ledger result hash. Only legacy version `1` projects the captured fields.
The full prediction schedule must equal QF-40's calendar-derived schedule from
the first requested session's open through the last requested session's close,
using the frozen primary timeframe. A canonical subset, empty replacement or
narrowed interval with unchanged decisions is insufficient, even after rehashing.
Backtests must match the frozen candidate's full
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
For prediction holdouts, QF-40 now captures its already-computed summary in the
artifact wrapper's `holdout_summary` extension (`schema_version: "1"`,
`window_result_id`, `summary`). The existing `artifact_sha256` covers this evidence
and binds it to the captured window. QF-9 requires the supported extension, matching
window identity and exact top-level summary equality. This is a metadata export
hook. Both copies must satisfy the base QF-40 prediction-summary schema, including
typed counts, accuracy/interval, outcome, baseline and event records. Shared
aggregate validators reconcile counts and availability with the stored decisions
and labeled rows using QF-40's fixed holdout metric fields. Matching malformed
copies remain invalid under refreshed hashes; statistics are not recomputed. The
QF-40 calculation, QF-42 window, prediction rows, ledger state and outer
result schema remain unchanged. A legacy prediction result without this extension
reports `holdout prediction summary evidence is unavailable`; QF-9 neither infers
its metrics nor writes a migration or reevaluates the holdout. Retain the historical
producer/environment for legacy reproduction. The authoritative ledger continues
to expose that lineage as consumed even when QF-9 cannot validate its result.
Failed or absent folds stay failed or absent; stale files do not become OOS
observations. No partition is prepared from market data during inspection.

An existing QF-40 aggregate must have the expected content ID and exact source
plan/study/lineage/fold references, and its family must match the source study.
Backtest `native_windows` must match the complete captured fold payloads, frozen
selections, export locations and fingerprints in source order. Normalized-equity
fold/run/session membership and window-start markers must match native equity;
per-window performance copies must match the captured backtest manifests.
Every normalized-equity row must retain all ten producer fields. Index and
window-start values are finite nonnegative decimal strings; strategy and benchmark
drawdowns are finite decimal strings in `[-1, 0]`. Window-start flags are booleans
and timing remains `exchange_session_close`. These checks do not rebuild the
normalized return chain or drawdowns.
Backtest aggregate summaries require every producer field, including completeness,
stitching semantics, returns, drawdowns, trade counts, costs, exposure and warnings.
Counts are nonnegative integers, decimal statistics are finite strings with their
supported domains, and metrics without completed windows or samples retain nulls.
Completeness uses the same schema checks as prediction aggregates. These checks
do not recompute backtest performance or costs.
Aggregate trade, winning, losing and open-trade counts must equal the sums of
their captured per-window performance counters. `oos_session_count` must equal
the number of captured native equity rows. Missing or failed folds contribute no
counts; partial and empty aggregates retain these same checks. Updating the
aggregate content hash cannot replace these totals. Economic metrics remain
producer-owned and are not recalculated.
Prediction observations must retain the source fold, selection, window result,
decision/context/study references, generated signals and matching stored rows.
Missing, duplicate, reordered or foreign records are rejected even under a new
aggregate content hash. Window summary membership and availability must preserve
failed and missing folds, including partial and empty aggregates.
For both aggregate families, the entire completeness record must equal QF-40's
projection of the captured plan, fold states and references. Expected/completed
counts, the complete flag, ordered missing/failed/incomplete fold lists and the
interpretation must agree. Missing references remain distinct from persisted
pending folds. Refreshing an aggregate's content hash cannot present partial or
failed validation as complete. This comparison reads status metadata only; it
does not aggregate research results or calculate performance.
Prediction summaries require the complete top-level and per-window producer
schemas, including counts, direction distributions, nullable accuracy and outcome
metrics, Wilson intervals, matched baselines, event counts/rates, field bindings,
window consistency, completeness and warnings. Nested records validate exact keys,
integer counts, finite decimal strings, supported statuses and value domains.
Unavailable samples retain null statistics; custom metric-field bindings remain
supported. No summary statistics or confidence intervals are recalculated.
Prediction frequency is non-null exactly when `scheduled_decisions` is positive;
accuracy is non-null exactly when `accuracy_sample_count` is positive. These
availability checks also apply to completed-window and final-holdout summaries,
and preserve recorded decimal values without calculating either ratio.
Aggregate and per-window sample counts must also match the captured observations
and completed decision schedules. Checks cover generated, eligible, excluded and
labeled signals, direction distributions, accuracy and interval sample counts,
paired baseline samples, metric availability, and event counts. Custom field
bindings determine which stored evaluation fields supply this metadata. Counting
existing evidence does not rerun predictions, label outcomes, calculate rates,
estimate performance, or rebuild confidence intervals.
The stability section must match `ConfigurationStabilitySummary` before receiving
an index entry using a JSON pointer. Transition/selection/parameter-change counters
must be nonnegative integers, frequency records must have their supported nullable
ratio domains, and the descriptive interpretation must be retained. Its windows,
candidates and neighborhood evidence must match the captured frozen selections.
Normalized index values, prediction summary statistics and stability calculations
remain producer-owned; these checks validate schema and copied evidence without
recalculating metrics or configuration turnover.
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
For JSON artifacts, the hash, pointer and bindings are checked against one
captured byte buffer, so replacing a file between hashing and parsing cannot
combine evidence from different versions. This also applies during manifest
publication. Hash mismatches take precedence over JSON parse errors.
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

Study and validation inspection hash each consumed JSON record from the same byte
buffer used to parse and validate it, including QF-39/QF-40 envelopes. Repeated
reads must agree. Before returning, every consumed record must be indexed with
that exact hash and its file must still match. Concurrent trial completion or
summary/result replacement raises `ManifestError` and requires a fresh inspection;
even formatting-only changes are detected. Atomic replacement with identical
bytes remains valid. These checks do not lock producer directories or prevent
changes after inspection; publication and later consumers still verify artifacts.

### Producer contract verification

The adapters check related invariants together. Regression tests change individual
fields, replace records with outputs from other real studies, and recompute
enclosing hashes so that consistency checks are exercised beyond hash mismatch.

| Contract family | Stored evidence checked | Boundary |
| --- | --- | --- |
| QF-11 | Study/component IDs, dataset/parameter provenance, ordered unique signal sessions, warm-up, captured outcome horizons/fields/schemas, row/outcome/evaluation IDs and counts | Calendar metadata validation only; no labeling or evaluation |
| QF-42 / nested QF-32 | Window/result/schedule identities and coverage, decision/context lineage and status, generated-signal provenance, distinct signal-to-row membership and unavailable outcomes | Reuses offline source/context checks; no context or indicator execution |
| QF-7/QF-29 | Dataset/candidate/row IDs, source/rule/schema versions, prediction-study references, schema definitions/types, dispositions, counts, native CSV equality and logical Parquet equality with indexed checkpoints; original table hashes retained | Existing values are validated and serialized; features and outcomes are not recalculated |
| QF-6/QF-32 grids | Coordinates, candidate/trial identity, status payloads, result bindings, recorded metrics, ranking/stability references, eligibility coverage/counts and summary projections; QF-6 CSVs match saved JSON records | No factory construction, candidate enumeration or research selection; native CSV serialization only |
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
