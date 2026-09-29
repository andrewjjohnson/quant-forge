# Sparse event windows (QF-64)

QF-64 adds opt-in compact window schema `"4"`. It separates three things:

- **decision coverage:** did each scheduled decision execute, and with which
  disposition;
- **rich research observations:** the generated signals, their causal features
  and their outcomes;
- **shared evidence:** stored once per window.

Every scheduled decision gets exactly one small, authenticated coverage receipt.
Complete evidence is retained only for decisions that produced generated signals
or that retained skipped-context evidence.

Schemas 2 and 3 are unchanged and remain readable. QF-45's frozen configuration
keeps schema 2. New event studies opt in with
`PredictionGridConfig(window_schema_version="4")`, which the QF-39 adapter
passes through. The decision is recorded in
[ADR 0036](decisions/0036-separate-decision-coverage-from-observations.md).

## Existing semantics this builds on

| Concept | Source | Meaning, unchanged |
| --- | --- | --- |
| Decision eligibility | QF-42 schedule, QF-48 membership | The authoritative ordered decision timestamps. Never inferred from observed bars or records. |
| `evaluated` | QF-42 | At least one generated signal of any disposition (accepted, rejected, …), labeled or not. |
| `no_prediction` | QF-42 | An available causal context and no signal. |
| `skipped` | QF-42, QF-28 SKIP policy | A rejected context (for example foreign, stale or missing primary bar), retained for audit. It may lack a source context. |
| Failed decision | QF-56 | Never persisted. The window stops at its committed prefix and the failure is recorded against the trial. |
| Observation | QF-11 generated signal | Prediction plus contemporaneous features, and the labeled row if its outcome was available. |
| Outcome availability | QF-46/47/49 | Stays inside the observation (`temporal_resolution`, unlabeled signals). |

Current QF-42 validation allows at most one generated signal per decision
session. The QF-11 result and the persisted structure are zero-to-many.

## What schema 3 stored for an ordinary no-trigger decision

A QF-45 no-candidate schema 3 record averaged 21,703 bytes. It held the decision
timestamp, `status`, `context_id` and `prediction_study_id`. It also held the
complete QF-28 context, now with membership ranges: requirements, source
context, rule timeframes and indicator manifests. On top of that came empty
`generated_signals` and `rows`, zero record counts, the feature/outcome boundary
text, catalogue segments, and the physical `sequence`, `shared_evidence_id` and
`decision_id`.

| Field | Where it comes from in schema 4 |
| --- | --- |
| Timestamp and session | The authenticated schedule, at `sequence` |
| Market data, configuration, engine, requirements | Shared evidence (unchanged since QF-55) |
| Empty signals/rows, zero counts | Implied by `status: no_prediction`; the reader and the counts check both enforce it |
| `shared_evidence_id` | The window result identity, which binds `window_id` |
| `context_id`, `prediction_study_id` | Persisted: the decision's scientific identities |
| Context body and membership | Not persisted; checked when the decision executes (see below) |

## Schema 4 representation

```text
{"component":"quantforge_prediction_window",...,"record_type":"header","schema_version":"4",...}
{"record_type":"shared_evidence","schema_version":"4","window_identity":{...}}
{"context_id":"…","observation_count":0,"prediction_study_id":"…","receipt_id":"…","record_type":"decision_receipt","sequence":0,"status":"no_prediction"}
{"context_id":"…","decision":{…complete schema 3 decision…},"observation_count":1,"prediction_study_id":"…","receipt_id":"…","record_type":"decision_receipt","sequence":1,"status":"evaluated"}
```

The header has the same fields as schema 3, including `membership_catalogues`.
The body has exactly one `decision_receipt` line per scheduled decision, in
schedule order.

- **Receipt fields.** `sequence`, `status`, `context_id`, `prediction_study_id`,
  `observation_count` and `receipt_id`. `receipt_id` is the hash of the receipt
  without itself.
- **Status rules.**
  - `no_prediction`: `observation_count` is 0 and there is no `decision`.
  - `evaluated`: `observation_count` is at least 1 and equals the nested
    decision's number of generated signals.
  - `skipped`: `observation_count` is 0, a nested `decision` is required, and
    `context_id` may be null.
- **Nested decision.** It is the unchanged schema 3 record with QF-62 catalogue
  segments and ranges. Its `sequence`, `status`, `context_id` and
  `prediction_study_id` must equal the receipt's. It must reference this
  window's shared evidence and the scheduled timestamp. Catalogues therefore
  hold only bars visible to some retained decision.
- **Observations.** Readers number observations across the window
  (`observation_index`), in schedule order and then signal order. Each
  receipt's observations are `[observation_start, observation_start +
  observation_count)`. The start is derived, not stored.

## Identities

| Layer | Schema 4 behavior |
| --- | --- |
| Scientific | Unchanged: `schedule_id`, the nested QF-42 window identity, every decision's `context_id` and `prediction_study_id`, and every row, outcome and evaluation ID and signal in retained decisions. |
| Physical | New for schema 4: shared-evidence ID, `window_id`, `receipt_id`, nested `decision_id`, `window_result_id` (the QF-55 algorithm over the ordered receipt records), and `catalogue_content_id`. `catalogue_id` still hashes only its binding. |
| Operational | QF-56 checkpoint version `"1"` with `"schema_version":"4"`. `last_decision_id` holds the last `receipt_id`. |

## Validation boundary

**Rich decisions** are validated exactly as in schema 3, with exact expanded
membership: on append, on resume, at finalization and offline. QF-20/QF-11 IDs
are recomputed, contexts and sources are checked, and rows, outcomes and
temporal resolution are checked.

**No-prediction decisions** skip the rich snapshot. The executor decodes the
in-memory QF-28 context once and calls
`PredictionWindowDecisionValidator.validate_no_prediction` before building the
receipt. It checks:

- the context is available;
- its as-of time and session equal the scheduled ones;
- the window family, prediction dataset, symbol and adjustment basis match;
- the shared QF-52 provenance matches (family manifest, feed scopes, per-timeframe
  dataset references);
- the configured requirements and primary timeframe match;
- the record counts are zero;
- `context_id` is the hash of the source context;
- `prediction_study_id` is `StudyIdentity.for_context(context)` under the
  window's authenticated scope.

The full per-timeframe QF-20/QF-28 structural re-validation is not repeated for
these contexts. Context construction already enforces it: QF-20, QF-28, and the
QF-59/QF-63 prepared checks. The validator re-checks it for persisted contexts
only.

**Offline**, a bare receipt is checked for shape, identity format, receipt
digest, exact coverage and order, header counts, and its binding into
`window_result_id`. Its context was not persisted, so it cannot be re-validated
offline. Re-executing that decision deterministically reproduces both
identities. A forgery that rewrites the whole file consistently, header
included, is caught by scientific validation for rich decisions. For bare
receipts it is caught only by re-execution.

## Reader API

| Method | Versions | Yields |
| --- | --- | --- |
| `iterate_decision_receipts()` | 1-4 | `PredictionDecisionReceipt` for every scheduled decision: sequence, schedule timestamp, status, identities, observation range, `decision` (the retained rich record, with its catalogue view) and `record` (the persisted record) |
| `iterate_observations()` | 1-4 | `PredictionWindowObservation` per generated signal: indices, timestamp, identities, `signal()`, `row()` (`None` if the outcome was unavailable) |
| `iterate_decisions()` | 1-3 | Unchanged; raises for version 4 |
| `verify_integrity()` | 1-4 | Exhausts coverage |

Every traversal checks each record before yielding it. Totals, the ordered
result identity and the catalogue contents are checked at exhaustion.
`iterate_observations()` checks every receipt but builds rich objects only for
retained decisions. No-prediction receipts never become rich objects, and no
legacy decision object is ever reconstructed.

## Execution and QF-56 durability

`run_incremental_prediction_window_in_session(schema_version="4")` iterates
`iter_prediction_window_results`. That is the same QF-42 execution and component
isolation, but it yields typed QF-11 results without building the decision
snapshot. `sparse_decision_receipt` then does one of two things:

- for signals or a skipped context, it builds the `PredictionWindowDecision` and
  its schema 3 form and nests them;
- otherwise it validates the context as above and builds a bare receipt.

The commit boundary is unchanged. Each append validates the receipt, then appends
and fsyncs one journal line. It then writes and fsyncs a temporary checkpoint,
atomically replaces the checkpoint and syncs the directory. A rich receipt's
catalogue growth travels in the same line. No fsync or checkpoint batching is
introduced.

Resume re-accepts every committed receipt: nested evidence is validated
scientifically and bare receipts structurally. Resume then checks the
checkpoint's count, offset, last receipt ID, prefix hash and counts, and only
after that truncates an uncommitted tail. It restarts at the first decision
with no committed receipt.

- **Executed with no observation:** a committed `no_prediction` receipt.
- **Never executed:** no committed receipt for that sequence.

Interrupted and uninterrupted runs produce byte-identical finals, and match
`CompactPredictionWindowResult.from_window(..., schema_version="4")`.

## Downstream consumers

| Consumer | Change |
| --- | --- |
| QF-40 `iter_prediction_observations` | Streams `iterate_observations()`; every emitted field and all ordering are unchanged |
| QF-40 protected-window reach | Reads rows of retained decisions |
| QF-39 test-window prediction check | Reads rows of retained decisions |
| QF-32 `EmaWindowAnalyzer` (QF-45) | Reads rows of retained decisions |
| QF-45 candidate export | Replays only retained decisions with signals |
| QF-9 `index_compact_window` | Checks metadata safety of every persisted record |
| QF-9 inspection, QF-41 reports, holdout, grid, walk-forward references | Accept `"4"` through the existing version tuple; QF-41 still never walks decisions |

All other semantics are unchanged: eligibility, ordering, metrics, ranking,
ties, selection, protected reach, manifests and reports.

## Measurements

All figures were measured on the same host with the frozen real QF-45 inputs:
48,480 2m bars and 250 daily bars, with the fixed 8/48/daily-50 study on the
fold-0 selection window (4,095 decisions, 25 candidates). The ignored harness is
`reports/qf64-sparse-decisions/qf64_profile.py`. "Before" is `main` (`b62d1d2`)
writing schema 3, QF-64's predecessor; "after" is this branch writing schema 4.
Every executed decision was compared with the completed QF-45 record:

- a receipt's status and identities must equal the stored record's;
- a rich decision's schema 2 form must match the stored bytes.

All matched. No research was recomputed for the window-level figures: the
stored decisions were replayed through the production writer with the trusted
QF-45 validator.

### Storage (fixed window, bytes including line feeds)

| Category | Schema 2 (QF-45) | Schema 3 (QF-62) | Schema 4 (QF-64) |
| --- | ---: | ---: | ---: |
| Header | 656 | 1,062 | 1,062 |
| Shared evidence | 4,369,822 | 4,369,822 | 4,369,822 |
| No-candidate decisions (4,070) | 2,431,493,767 | 88,330,787 | 1,411,184 (receipts) |
| Candidate decisions (25), including catalogue growth | 17,352,458 | 748,998 | 1,038,039 |
| **Total** | **2,453,216,703** | **93,450,669** | **6,820,107** |

- **Reduction.** 92.70% against schema 3 (13.7× smaller) and 99.72% against
  schema 2 (360×).
- **Receipts.** An ordinary no-observation receipt is 344-347 bytes, 346.7 on
  average, against a mean of 21,703 bytes in schema 3.
- **Observation receipts.** They average 41,522 bytes (30,524-70,537). That
  includes 284,766 bytes of catalogue growth, now carried only by rich
  decisions: the 2m and daily catalogues hold 4,086 and 70 bars, against 4,156
  and 72. Without catalogue growth, an observation costs about 30.1 KB: the
  schema 3 candidate record (29,960 bytes) plus the receipt envelope.
- **No index.** The format needs no separate observation index. Coverage is the
  receipt stream itself (about 1.4 MB), read line by line.

### Scaling

`tests/performance/test_sparse_window_scale.py` builds 4,095 decisions with
QF-62's QF-45 membership pattern and a 34 KB fixture observation, at 0, 25, 100
and 4,095 observations. These are exact bytes:

| Observations | Total | No-observation receipts | Observation receipts | Catalogue growth |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1,559,524 | 1,419,855 | 0 | 0 |
| 25 | 2,693,718 | 1,411,186 | 1,142,090 | 292,642 |
| 100 | 5,224,740 | 1,385,177 | 3,699,119 | 301,326 |
| 4,095 | 140,043,091 | 0 | 139,902,648 | 763,791 |

- **Fixed and marginal costs.** Header plus shared evidence are 140,443 bytes;
  each receipt is 344-347 bytes, and each extra observation adds 33,747 bytes.
  Storage is therefore O(decisions × about 347 B) + O(observations × rich
  payload) + shared evidence, with catalogues bounded by the bars observed.
- **All decisions observed.** At 4,095 observations the receipt envelope around
  each schema 3 record is 354 bytes, the only overhead over schema 3.
- **Projection for the real window** (estimate, not measured): about 5.8 MB at
  0 observations, about 9.1 MB at 100, and about 94.9 MB at 4,095 (schema 3
  plus the envelope).

### Runtime (seconds; uninstrumented means unless noted)

| Operation | Schema 3 (main) | Schema 4 |
| --- | ---: | ---: |
| Late no-candidate decision, sequences 4040-4079 | 0.0452 | **0.0221** |
|   QF-42 iterator (QF-11 run; schema 3 also builds the decision snapshot) | 0.0270 | 0.0178 |
|   Snapshot decode + compact conversion | 0.0023 + 0.0069 | — |
|   Receipt construction (decode + binding checks) | — | 0.0036 (checks 0.0027) |
|   Append (validation, fsync, checkpoint) | 0.0084 (validation 0.0062) | 0.0005 |
| Candidate decision, 7 samples, sequences 460-4024 | 0.066-0.088 | 0.067-0.097 |
| Writer replay append, per decision (4,095) | 0.0051 | 0.00037 |
| Reopen and committed-prefix validation (4,095) | 19.88 | 0.84 |
| Finalization (prefix check, stream, validate, publish) | 45.54 | 2.43 |
| Structural `verify_integrity()` | 5.98 | 0.77 |
| Offline scientific validation (prepared QF-52 lineage) | 24.86 | 4.53 |
| QF-40 observation iteration (25 observations) | 5.98 | 0.15 |
| QF-9 `index_compact_window` | 9.63 | 0.37 |
| QF-32 `EmaWindowAnalyzer.analyze_compact_window` | 5.89 | 0.15 |

- **No-trigger decisions** run 2.0× faster, because they no longer build,
  convert, hash, normalize, re-validate or write a rich record.
- **Candidate decisions** keep the full rich path. They are within about 10% of
  schema 3; the added cost is one context decode and the receipt wrapper. Each
  isolated sample also introduces its entire catalogue history.
- **Profiled remaining cost** (20 late decisions, with profiler overhead, per
  decision):
  - QF-11 study-identity hashing of the ~1.1 MB context (`_stable_id`): 6.1 ms;
  - context preparation: 4.4 ms;
  - the QF-63 prepared-values guard: 3.0 ms;
  - receipt binding checks: 2.7 ms, of which the identity re-hash is 2.0 ms.
- **Memory.** Readers and the writer keep the shared evidence, schedule, hash
  state, counters, catalogues (only bars visible to retained decisions) and one
  record. No completed receipts or observations are kept.

### Scientific equivalence (real fixed window)

Schema 3 and schema 4 produced identical results for all of the following:

- the 25 QF-40 observation snapshots;
- the QF-32 analysis (25 predictions, mean 30m return
  0.000316990389268332576478386732061272, 17 positive);
- the header record counts;
- `schedule_id` and the QF-42 window identity.

Physical window and result IDs differ, as intended.

## Event and dense modes

Schema 4 is the event representation. It stores many small receipts and few
rich observations. A dense study, where nearly every timestamp becomes an
observation, pays a fixed envelope of about 354 bytes per decision over schema 3
in the scale fixture. It should use a future columnar representation instead.
Both modes share schedules, sources, feature and outcome definitions, validation
partitions and provenance.

No-prediction receipts are coverage evidence, not negative labels. Triggered
observations, winners and losers alike, keep their causal features and outcome
statuses for later conditional research.

## Out of scope

This change does not touch:

- canonical dataset loading (QF-65);
- QF-63 feature-series preparation;
- fsync or checkpoint batching;
- feature-manifest or study-envelope normalization;
- dense or columnar datasets, ML, or batch outcome execution;
- new rules or outcome types.
