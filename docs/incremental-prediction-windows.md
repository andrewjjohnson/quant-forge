# Incremental historical prediction windows (QF-56)

QF-56 executes the existing QF-42 schedule sequentially, validates each original
QF-11 result, compacts it with QF-55, commits it, and releases the completed full
result before executing the next decision. It does not change prediction math,
outcomes, exact membership, ranking, validation partitions, or OOS methodology.

This is explicit opt-in. Existing v1 execution and artifacts remain unchanged.
[QF-55](compact-prediction-windows.md) remains authoritative for compact schema,
hash composition, provenance, and reading.

## Execution and writer APIs

`quantforge.prediction.window_execution.run_incremental_prediction_window_in_session`
accepts the prepared QF-11 dataset session, study, exact QF-42 schedule, context
provider, family/environment identities, and destination `path`. It returns a
validated `PredictionWindowReader`, including when reusing an existing final.
QF-52 bounded inputs require independent `canonical_metadata` for ancestral subset
verification. That metadata never enters the rule or individual result records.

`IncrementalPredictionWindowWriter` in `window_incremental` exposes:

- `open(path, validator=...)`: initialize or verify/resume compatible work.
- `append(compact_decision)`: validate and durably commit the exact next record.
- `checkpoint()`: inspect already committed progress; append persists it.
- `completed_count` and `finalized`: progress and lifecycle.
- `finalize()`: verify completeness and publish canonical QF-55 JSONL.

`PredictionWindowDecisionValidator` factors existing QF-55 scientific checks for
append, resume, finalization, and offline reading. Construct it from independently
trusted identity, schedule, outcome sessions, strategy parameters, and canonical
ancestry. Original study/context/signal/row/outcome IDs, source bindings and
QF-46/49/47 temporal provenance are still checked.

QF-42's `iter_prediction_window_decisions` executes a schedule suffix with the
same independent component copies and `run_prediction_study_in_session` calls
as legacy execution. Supply nonzero `start_sequence` only after verifying a
durable prefix. There is no second scheduler or observed-bar scan. Zero-candidate
and policy-authorized skipped decisions remain decisions.

## Staging and durability

~~~text
prediction-window.jsonl.in-progress/
  shared.json
  decisions.jsonl
  checkpoint.json
~~~

Staging file count is constant: **three** per window. Checkpoint granularity and
physical file granularity are separate.

`shared.json` is precisely the QF-55 evidence record, stored once.
`decisions.jsonl` contains only canonical compact decisions.
`checkpoint.json` is a small content-hashed envelope containing operational
checkpoint version `"1"`, compact schema `"2"`, window/evidence/schedule IDs,
completed count, committed journal byte offset, last decision ID, QF-55 canonical
ordered-prefix hash, and aggregate counts. It contains no completed payloads or
shared evidence body.

Every append validates the record, appends/fsyncs its line, then writes/fsyncs a
temporary checkpoint, atomically replaces the checkpoint and syncs the directory.
Checkpoint publication is the commit boundary. This cadence bounds lost
uncommitted work to the current decision. Frequency, offsets, temporary names,
and storage layout do not enter scientific identity.

Initialization atomically publishes a complete zero-decision staging directory.
An uncertain I/O failure or interruption requires reopening the writer; the
checkpoint on disk determines whether the current decision committed.

## Prefix validation and recovery

Open compares the complete trusted scientific scope with shared evidence:
schema, rule parameters/combination, market/source evidence, outcomes,
context/validation environment and exact schedule. It then streams the committed
journal prefix, verifying each compact hash, scientific record, reference,
sequence and scheduled UTC bar-end timestamp. Recomputed QF-55 counts, ordered
hash and last decision ID must match every checkpoint field.

Only after those checks pass may recovery truncate bytes **after** the committed
offset. A partial or complete line beyond that offset was never committed and
may be retried. Truncation inside the prefix, incorrect offset/count/hash, gaps,
duplicates, reordering, foreign records, incompatible versions or checkpoint
corruption fail closed. An invalid checkpoint never triggers tail truncation.

A compatible 3,000-decision prefix resumes at sequence 3,000 (decision 3,001).
It revalidates persisted evidence but reruns no completed predictions.
Incompatibility preserves existing evidence and raises; it never resets or mixes
results. Identity-addressed grid directories separate configurations. Stale work
may be explicitly archived by its owner; recovery does not delete it.

Failures during context construction, execution, validation, conversion or
persistence preserve the committed prefix. The failed decision is not skipped
or replaced with an empty success. `PredictionWindowDecisionError` identifies
the sequence, exact timestamp, window and exception type. Durable grid/fold
diagnostics omit raw exception text. Existing explicit retry policies apply.

## Finalization and canonical identity

The staging directory is in-progress and has no finalized header. Finalization
requires exact complete membership and revalidates the durable prefix. It
streams the existing QF-55 header, shared record and decision journal into a
temporary file. QF-55 structural and scientific validation exhaust that file
before atomic non-overwriting publication and filesystem sync.

A failed publication preserves resumable work. Successfully published finals are
immutable under the API. Cleanup removes only successfully finalized staging
files. Temporary-file and staging cleanup are best-effort: filesystem cleanup
errors do not turn a durably published window into a failed trial. A cleanup
error or crash may leave redundant files (including partially removed staging);
the validated final remains authoritative on resume.

Shared evidence is never re-embedded per decision. Atomic assembly temporarily
has one staging copy and one final-file copy; successful completed storage has
one copy. Journal copying uses a bounded buffer.

`WindowResultIdentity` and `WindowRecordCounts` preserve QF-55's canonical
algorithms. A constant-state hash copy supports transaction rollback. Final bytes
and IDs equal `CompactPredictionWindowResult.from_window` for the same ordered
records. There is no new scientific schema or alternate hash chain.

These are local, single-writer artifacts requiring atomic replacement,
same-directory hard links, and file/directory `fsync`. Concurrent writers to the
same path are unsupported. No database or distributed execution is introduced.

## QF-32 integration

Set `PredictionGridConfig(window_schema_version="2")` and supply a
`CompactPredictionWindowAnalyzer.analyze_compact_window(reader)` implementation.
Legacy `analyze_window` remains supported with schema `"1"`. The compact analyzer
must preserve its domain's observation eligibility, numerical ordering, metrics
and baseline comparisons, exhaust the reader, and avoid retaining expanded
results. The grid does not invent averages of per-decision metrics.

Representation choice participates in grid artifact identity; logical candidate
definitions and combination IDs remain unchanged. Each trial has:

~~~text
artifacts/<trial-id>/
  prediction-window.jsonl
  prediction-window.json
~~~

The small JSON wrapper binds grid/trial IDs, analysis, a versioned relative
reference and the QF-55 header/result hash. Reserved `prediction_window_sources`
references the normalized header; exact original decision/result/context IDs
remain in JSONL records. Completed-trial reuse validates both files against the
current candidate. An interrupted trial resumes its own prefix. Analyzer
interruption can reuse a finalized window without rerunning predictions.
Existing `retry_failed` policy and failure history remain authoritative.

Ranking, eligibility, tie-breaking and stability are unchanged. Compact execution
does not populate the legacy accumulating context/indicator caches; QF-11 still
performs normal indicator calculation and validation. This intentionally trades
cross-trial cache reuse for bounded retained state.

## QF-39 integration and QF-57 boundary

The same grid switch drives development/selection and frozen test execution.
Existing QF-8 partitions, QF-48 membership, QF-52 bounded inputs, candidate universe
and frozen logical configuration remain authoritative. With explicit selection,
only selection is scored; development supplies permitted context, as before.

Each test directory contains `prediction-window.jsonl`.
`PredictionOOSArtifact.snapshot` holds the versioned reference
`{schema_version: "2", path: "prediction-window.jsonl", header: ...}`.
The path is relative to the existing test output root. Reuse validates the fixed
path, exact header/result ID and trusted fold/configuration. Study envelopes and
freeze lifecycle are unchanged.

[QF-57 consumers](compact-window-consumers.md) support this finalized reference
through QF-40 OOS/holdout, QF-9 indexing/integrity and QF-41 reports.
Existing v1 workflows remain available. No QF-45-specific runner is introduced.

## Scale and scientific evidence

The deterministic scale diagnostic uses 5,760 QF-42 timestamps and 3,263,326
synthetic shared-evidence bytes. It appends inexpensive normalized skipped
records with valid QF-11 identities, interrupts after 3,000, adds a partial
trailing write, resumes at 3,001 and finalizes. It never allocates the legacy
18.8 GB representation.

| Measurement | Result |
| --- | ---: |
| Decisions / decision checkpoints | 5,760 / 5,760 |
| Final artifact bytes | 55,420,985 |
| Shared record bytes | 3,448,976 |
| Compact decision bytes, min–max | 9,020–9,023 |
| Mean decision bytes | 9,022.8073 |
| Interruption / resume ordinal | 3,000 / 3,001 |
| Committed decisions recomputed | 0 |
| Retained completed writer records | 0 |
| Current record buffer | 1 |
| Staging files | 3 |
| Local elapsed time, descriptive | approximately 15 seconds |

Schedule metadata grows with membership as QF-42 requires; completed-result state
does not. The test inspects retained writer fields at every append, exact byte
accounting, one shared marker, recovery and canonical identity equality.
A separate real-QF-11 weak-reference test fails if a completed full result remains
alive when the next execution begins. No wall-clock/RSS threshold is asserted.

Real integration fixtures cover QF-52 ancestry, QF-48 membership, QF-49
10/30/60/120-minute returns, QF-47 excursions/target-stop, and bounded QF-39
production. Small v1/v2 comparisons cover final bytes, rows, analysis, rankings,
ties and frozen logical configuration without live APIs.
