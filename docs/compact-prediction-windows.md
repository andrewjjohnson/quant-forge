# Compact historical prediction windows (QF-55)

QF-56 adds explicit compact incremental execution, per-decision durable progress,
exact-prefix resume and QF-32/QF-39 production. See
[incremental prediction windows](incremental-prediction-windows.md). The v1
contracts below remain available; QF-57 owns downstream compact consumption.

QF-55 adds representation version `"2"` to the existing QF-42 historical-window
contract. Execution still returns v1 by default. QF-55 itself adds no scheduler, strategy, outcome calculator, or database.
QF-56 builds these same records incrementally.

## Existing v1 representation

`PredictionWindowResult` retains its `PredictionDecisionSchedule`, an immutable
`identity_snapshot`, and an ordered tuple of `PredictionWindowDecision` objects.
Each decision retains the original typed QF-11 result **and** a primitive snapshot:

```text
manifest
  component/schema_version="1"/engine_version/prediction_engine_version
  schedule, market_data, configuration, dataset_family_fingerprint
  context_environment, indicator_backend_environment
  window_id, window_result_id, schedule_id, record_counts
decisions[]
  decision_timestamp, prediction_study_id, context_id, status
  generated_signals
  prediction_study
    manifest: component, engine_version, configuration, market_data,
              prediction_context, feature_outcome_boundary, record_counts, study_id
    rows[]
```

The v1 window constructor already requires exact equality of every decision's
market data, study configuration, and prediction engine version with the window
identity. These fields are semantically shared, rather than merely equal in a
particular example. The v1 result ID hashes the complete embedded decisions;
counting and serialization materialize their snapshots.

## V2 shared boundary and records

`PredictionWindowEvidence` freezes the **complete existing scientific window
identity** once. This retains:

- Full `market_data`, including QF-52 `intraday_provenance`: canonical input and
  provenance IDs, source/request/session/family/feed identities, original
  retrieval time, causal cutoff, bounded content digest, local session evidence,
  and corporate-action/adjustment semantics. No evidence is removed or replaced.
- Full study `configuration`, including rule parameters, feature and context
  requirements, outcome/evaluator schemas, QF-46 temporal configuration and
  QF-49/QF-47 outcome-source references.
- The original schedule, its exact ordered timestamps and timeframe/session
  policy, both engines, dataset-family identity, context environment, and fixed
  indicator/backend environment.

The nested original identity retains its schema `"1"` because that describes the
existing scientific configuration. The **outer evidence and artifact schema are
explicitly `"2"`**. The schedule's schema and schedule ID remain unchanged.

`CompactPredictionWindowDecision` retains every original decision field and
every original study manifest/row field except the three shared study fields:
`market_data`, `configuration`, and `engine_version`. It adds `record_type`,
zero-based `sequence`, `shared_evidence_id`, and `decision_id`. Each record keeps
the original QF-11 `prediction_study_id` and study manifest `study_id`.

Timestamp, context ID, prediction/candidate dispositions, generated signals
(including unlabeled signals), features, rule evidence, labeled rows, all outcome
and evaluation values/IDs, temporal-resolution requests, diagnostic status,
feature/outcome boundary disclosure, and per-decision counts remain verbatim.

Context snapshots remain decision-specific, including requirements and indicator
manifests nested within them. Visible bar IDs, developing bars, availability,
ages, source timestamps, and selected datasets can differ. A skipped decision
may intentionally retain a rejected foreign context for audit. Its source cannot
be replaced with the window's successful-context assumptions. No smaller shared
groups are introduced: current QF-42 windows permit exactly one
market/configuration/engine scope. Different bounded views require different
windows. Context normalization beyond this boundary is deferred.

## Deterministic identities

All hashes use the existing ASCII-escaped, sorted-key, finite-value, compact JSON
and SHA-256 conventions from `quantforge.configuration`:

1. `evidence_id = configuration_identity(shared_evidence_record)`. The record
   includes representation version and the entire original window identity,
   binding its exact schedule, source ancestry, configuration and environments.
2. V2 `window_id` hashes component `quantforge_prediction_window`, schema `"2"`,
   and `shared_evidence_id`. It intentionally differs from v1.
3. `decision_id` hashes the complete normalized decision except its own ID,
   including sequence and the shared reference. Original QF-11/context/row IDs
   remain unchanged; separate decisions remain distinguishable even if they have
   identical skipped-study results.
4. `window_result_id` is exactly `configuration_identity({"window_id": ...,
   "decisions": [...]})`, with normalized records in exact schedule order.

`WindowResultIdentity` encodes that canonical array incrementally. It is not a
new hash-chain algorithm. Changing order, membership, shared evidence, decision
contents or outcome configuration changes the corresponding identities.

Original QF-11 study IDs are independently checked with `StudyIdentity`. It
pre-encodes and hashes the immutable canonical prefix through `market_data`,
copies the SHA-256 state, and adds the individual context plus study configuration.
Tests compare this directly with the existing QF-11 identity. Verification never
needs a collection of expanded results or repeated market serialization.

## Canonical serialization

`CompactPredictionWindowResult.serialize()` returns finalized UTF-8 JSON Lines.
Each line is canonical repository JSON followed by exactly one LF:

```text
{"component":"quantforge_prediction_window", ..., "record_type":"header", "schema_version":"2", ...}
{"record_type":"shared_evidence","schema_version":"2","window_identity":{...}}
{"decision_id":"...", ..., "record_type":"decision", "sequence":0,"shared_evidence_id":"...", ...}
{"decision_id":"...", ..., "record_type":"decision", "sequence":1,"shared_evidence_id":"...", ...}
```

The header has exactly `record_type`, `component`, `schema_version`, `window_id`,
`window_result_id`, `shared_evidence_id`, `schedule_id`, `decision_count`, and
`record_counts`. Shared market/configuration payloads physically occur once.
They are forbidden inside compact study manifests.

`iter_serialized_records()` yields the same finalized bytes one record at a time.
It serializes an already completed model; it does **not** implement append,
checkpoint, resume, or incremental decision execution. `serialize()` and
`to_primitive()` are convenient in-memory inspection APIs; use the iterator for
bounded output buffering. `to_primitive()` exposes a normalized inspection object
with `header`, `shared_evidence`, and `decisions`; it is not the disk encoding.

```python
from quantforge.prediction.window_compact import CompactPredictionWindowResult
from quantforge.prediction.window_reader import PredictionWindowReader

compact = CompactPredictionWindowResult.from_window(existing_window)
with artifact_path.open("xb") as output:
    output.writelines(compact.iter_serialized_records())

reader = PredictionWindowReader.open(artifact_path)
reader.verify_integrity()
for decision in reader.iterate_decisions():
    record = decision.to_primitive()
```

Conversion consumes already retained v1 results; it does not solve the execution
retention problem. QF-56 uses the same evidence/decision/serialization primitives
through its separate incremental execution and persistence APIs.

## Common reader and validation

`PredictionWindowReader.open(path)` recognizes current v1 JSON (including pretty
JSON) and canonical v2 JSONL. Unknown/corrupt versions, duplicate JSON keys,
non-finite numbers, mixed layouts, and noncanonical/truncated v2 lines reject.
It never rewrites files. Existing v1 validation remains available and explicitly
rejects non-v1 versions. No v2 artifact silently enters v1 interpretation.

- `schema_version`: the physical input representation.
- `header()`: detached version-specific header/manifest.
- `shared_evidence()`: immutable common scientific scope.
- `decision_count`: scheduled count without parsing any decision payload.
- `iterate_decisions()`: normalized records in exact order for both versions.
- `verify_integrity()`: exhaustively checks references, record identities, exact
  membership/order, totals, and result identity with bounded decision state.

Opening/inspecting metadata does not certify the unread body. Iteration checks
each record before yielding; whole-file counts and identity are checked at
exhaustion. Exhaust verification before accepting an artifact if only a prefix
will subsequently be read. Each v2 traversal checks the header/shared preamble
again and verifies the complete ordered body. No indexed-access API is needed.

`validate_prediction_window_reader` additionally verifies scientific evidence
against trusted `expected_identity`, schedule, outcome sessions, and strategy
parameters. These inputs must come from independently validated research inputs,
as with the existing v1 validator. It reuses QF-42's decision/signal/row/context
checks without constructing embedded QF-11 results. Source/feed/session/outcome
bindings are checked with existing data validators; no predictions, features,
indicators, outcomes, or metrics are recomputed.

Bounded QF-52 windows require independent `canonical_metadata`. The validator
checks local bounded provenance, then uses `validate_prediction_view_lineage`
to verify exact ancestral subset membership. Cutoff cannot follow the first
decision. The parent remains outside shared evidence and the evaluator. Opaque
ancestor hashes alone are not treated as proof of membership or authenticity.
Rehashing ancestry, visible evidence, cutoff, or adjustment claims cannot bypass
these checks.

V2 reading needs shared evidence plus schedule metadata, one decision, hash state,
and aggregate counters. Schedule storage grows with exact membership, as QF-42
already requires; completed decision payloads are not retained. V1 remains a
materialized document because its existing format has no record boundaries,
but the common iterator normalizes one v1 decision at a time. The v1 adapter does
not change old IDs or artifacts.

## Structural scale regression

`tests/performance/test_compact_prediction_window_shape.py` uses 5,760 QF-42
scheduled timestamps, existing skipped-decision context structure, and a
3,263,326-byte synthetic immutable market-evidence field. This is a **storage-shape
fixture**, not a claimed canonical market dataset or performance benchmark. The
separate integration fixtures use real validated synthetic intraday caches,
bounded views, membership certification, all four forward-return horizons, MFE/MAE,
and target/stop outcomes.

Measured deterministic shape (including LF delimiters):

| Item | Bytes |
| --- | ---: |
| Synthetic shared payload | 3,263,326 |
| Entire shared record, including schedule and configuration | 3,448,976 |
| Header | 639 |
| Per-decision record | 9,020–9,023 |
| Mean per-decision record | 9,022.8073 |
| Complete compact artifact, 5,760 decisions | 55,420,985 |
| Theoretical repeated payload alone, 5,760 decisions | 18,796,757,760 |

The test proves exact byte accounting:
`header + shared_record + sum(decision_record_sizes)`.
The marker appears once, adding a decision adds only its normalized line, and
counting constructs no decisions. Instrumented full iteration decodes the shared
identity once per traversal and forbids embedded-decision construction. No
18.8 GB legacy fixture is allocated and no wall-clock threshold is asserted.
Shared **market/provenance evidence** contributes O(1) storage per window;
membership and decision-specific evidence retain their necessary O(N) storage.

## Scope and follow-ups

QF-55 introduced this foundation in five production files: `window_compact.py`,
`window_encoding.py`, `window_reader.py`, `window_compact_validation.py`, and the
small shared-decision-validator extraction/version check in `window_validation.py`.
That foundation stays within the prediction package. Prediction/outcome
execution and numerical behavior are unchanged.

QF-56 owns incremental QF-42 execution, durable checkpoints, prefix resume, and
QF-32/QF-39 persistence. QF-57 owns QF-40 OOS/holdout, QF-9 manifest/integrity, and
QF-41 report integration. The QF-57 consumer migrations remain deferred. QF-45
strategy logic is untouched. No generic architectural blocker was found.
