# Compact prediction-window consumers (QF-57)

QF-40, QF-9 and QF-41 consume QF-56 finalized QF-55 JSONL windows directly.
The existing scheduling, typed outcomes, frozen selections and holdout ledger
remain authoritative. No compact record is expanded into a legacy QF-11 result.

## Reader boundary

`PredictionWindowReader` is the common semantic interface. `from_snapshot`
accepts an embedded v1 window; `from_reference(snapshot, root=...)` resolves the
QF-56 `{schema_version, path, header}` reference. Compact references require
schema `"2"`, the fixed `prediction-window.jsonl` filename, confinement within
the supplied directory and an exact header match. Unknown versions fail closed.

`manifest()` exposes shared scientific metadata plus the actual physical
schema/window/result identity. It is an inspection view, not a serialization
format. Existing `evidence`, `header`, `decision_count`, `iterate_decisions` and
`verify_integrity` implement shared-scope access, normalized iteration, exact
membership, canonical hashing and count checks. Scientific validation reuses
`validate_prediction_window_reader` with trusted configuration and ancestry.
The representation and incremental writer contracts are unchanged.

## OOS aggregation

`load_oos_source` validates each completed QF-39 test artifact against the frozen
selection, test partition, QF-48 schedule and QF-52 canonical parent. It retains
readers alongside fold references. Development and selection artifacts are never
included as OOS observations. Protected outcome boundaries remain checked.

`aggregate_prediction` streams stored normalized signals and rows through
`PredictionObservationAccumulator`. Eligibility, Decimal arithmetic, Wilson
intervals, baseline pairing, missingness, typed events and configuration turnover
retain their existing meanings. No prediction, outcome or selection is executed.

Counts use counters. Exact median/minimum/maximum/mean calculations retain only
numeric scalar samples for signed outcomes, MFE and MAE. Memory therefore grows
with numeric sample count and schedule membership, not repeated scientific
evidence or complete decision graphs. Pooled and current-window numeric buffers
can coexist. This is bounded retained evidence, not a claim of constant memory.

A compact-backed prediction aggregate uses schema `"2"` with `window_sources`
instead of duplicated `observations`. Each source binds fold, selection, physical
schema, window/result/shared-evidence/schedule identities and decision count.
Summary, stability and provenance semantics are unchanged. All-v1 aggregates keep
schema `"1"` and the original observations; both versions remain readable. Physical
artifact IDs truthfully differ across representations. No migration is performed.

## Final holdout

The existing ledger still owns reserve → reserved/unconsumed → explicit consume
→ permanently consumed. Reports only query it. Compatible retries reuse stored
results; incompatible selections and foreign partitions are rejected.

For an already finalized compact holdout, supply its path when preparing the
same frozen request:

```python
evaluation = HoldoutEvaluation.prepare(
    source,
    evaluator,
    selection_fold_id=selected_fold_id,
    finalized_prediction_window=finalized_path,
)
consumed = ledger.consume(evaluation, run_id="explicit-final-holdout")
```

This optional path is an operational input, not scientific request identity.
The evaluator's existing validator checks the exact original frozen parameters,
lineage, membership, outcome policy and canonical ancestry. Consumption does not
execute the evaluator in this mode. After durable exposure is recorded, QF-40
copies bytes in bounded chunks, verifies the copy, fsyncs it, publishes without
overwriting an existing final and validates the published artifact. The ordinary
first-evaluation path remains available and can produce compact results through
QF-56. Neither path resets a consumed marker after failure or corruption.

Externally supplied results must come from the authorized holdout lifecycle;
the ledger cannot retroactively establish that externally evaluated data was
previously unseen. The path option is for ingesting finalized evidence, not for
bypassing the research process or authorizing pre-consumption exploration.

## Artifact graph and integrity

QF-9 adds `jsonl` to the existing artifact format enum. One physical file has
three indexed metadata views: `/header` (prediction result), `/shared_evidence`
(source evidence), and `/shared_evidence/window_identity/configuration`
(configuration). All bind the same whole-file SHA-256 and exact header. Metadata
contains source/configuration identities rather than copied evidence bodies.
Window ownership and logical IDs are stable across standalone, parameter and
validation inspection, allowing existing manifest attachment logic to reuse
identical entries. Content identity does not confer a validation role: OOS/test/
holdout classification comes from checked fold/partition/manifest relationships.
Regression tests reuse each identical entry once across repeated attachments,
then reject rehashed foreign fold, role and validation-lineage wrappers while
the compact content and entry identities remain unchanged. Result→shared,
result→configuration, fold→result,
aggregate→fold and holdout→result relationships use the existing graph.

Initial producer inspection exhausts compact structural/scientific validation,
checks frozen scope and summary availability against stored observations, applies
metadata safety checks and detects changed input bytes before returning an index.
Summary integrity checks counts and references without recalculating metrics.
Bounded standalone/parameter inspection accepts `canonical_metadata=`; validation
inspection uses the trusted parent already supplied by the QF-8 plan.

Manifest replay authenticates the indexed whole-file hash and cached control
records once per file per verification pass. It need not deserialize decisions
again merely to display previously verified aggregates. Hashing still reads the
file bytes. A new producer inspection does perform full streaming validation.
This preserves the existing distinction between producer validation and indexed
immutable-artifact verification. Repeated independent inspections can verify the
same shared scope again; there is no global cache or one-time trust bypass.

Missing/altered shared evidence, invalid references/versions, modified/reordered/
duplicate/truncated records, count mismatches, foreign fold/plan/ancestry/freeze,
incorrect aggregate references, staging inputs and replacement races fail closed.

## Reports and offline use

QF-41 displays QF-40 summaries, QF-32 published parameter results, QF-9 provenance
and the authoritative holdout state. JSONL content retained for presentation is
limited to bound header/identity metadata. It does not iterate decisions for
aggregate sections or recompute research. Existing development, OOS, holdout and
warning sections are preserved. A saved reserved report consults the current
ledger and cannot make a consumed holdout appear pristine again.

The offline integration test builds synthetic immutable bounded caches, disables
network connections, evaluation and legacy expansion, consumes finalized evidence,
then reads aggregates, indexes/verifies artifacts and renders the consumed report.
It also disables compact decision iteration during rendering. No provider access,
API credentials or source re-download is required.

## Acceptance fixtures and limits

- Real QF-39/QF-56 v1/v2 comparison: 17 eligible predictions, identical summaries,
  candidate configuration and exact membership, plus equal OOS report content.
- Real typed outcomes: 10/30/60/120-minute returns, excursions and target/stop
  values and eligibility remain equivalent. Separate reduction tests preserve
  unavailable/incomplete and same-bar-ambiguity statuses.
- Synthetic scale: 5,760 normalized skipped decisions, 55,420,985 final bytes and
  one shared scientific verification per standalone inspection. The QF-40
  reducer, QF-9 index and QF-41 stored-summary presentation remain compact.
  Expansion helpers fail if called. A separate 5,760-observation reducer test
  exercises nonempty median buffers, exclusions and typed missingness.
- Smaller full integration covers actual OOS aggregation, parameter inspection,
  both holdout states, standalone/attached reports and offline bounded ancestry.

The scale fixture is a structural test, not the actual SPY EMA study, a benchmark
of nine parameter combinations, or a profitability claim. No hard RSS or timing
threshold is imposed. QF-45 remains separate work after this infrastructure merges.
See [ADR 0029](decisions/0029-consume-compact-window-evidence.md).

## Implementation inventory

Seventeen production files change, confined to the downstream boundaries:

| Package | Files |
| --- | --- |
| `prediction` | `window_reader.py` |
| `oos` | `models.py`, `source.py`, `prediction.py`, `common.py`, `holdout.py`, `holdout_evaluation.py` |
| `experiments` | `_compact_window.py`, `adapters.py`, `artifacts.py`, `validation.py`, `_aggregate_integrity.py`, `_holdout_integrity.py`, `_prediction_counts_integrity.py`, `_prediction_trial_integrity.py` |
| `reporting` | `_research_inputs.py`, `_research_sections.py` |

Six test files are added:

- `tests/integration/test_compact_window_consumers.py`
- `tests/integration/test_compact_window_consumer_integrity.py`
- `tests/integration/test_compact_window_outcome_consumers.py`
- `tests/integration/test_compact_window_offline.py`
- `tests/performance/test_compact_window_consumers.py`
- `tests/unit/oos/test_compact_accumulation.py`

QF-55 encoding/validation and QF-56 writer/checkpoint modules are reused without
modification. No QF-45 strategy configuration or execution is introduced, and
no missing generic upstream contract was found.

## Validation record

Local acceptance used uv 0.12.1, Python 3.13.14 and frozen dependencies:

| Check | Result |
| --- | --- |
| Frozen all-extras sync | Passed |
| Ruff format/check, pyright | Passed; no type errors or warnings |
| Pre-commit, all tracked files including additions | Passed |
| Full suite / CI-equivalent pytest | 6,840 passed, 3 skipped |
| Final QF-57 focused matrix | 39 passed |
| Targeted QF-55/56, QF-39, QF-49/47 and legacy-window regressions | 507 passed |

The three role/lineage regressions were added after full-suite collection and
passed in the final focused run. The full suite includes QF-40, QF-9 and QF-41.
Skipped tests require explicit live Alpha Vantage/Tiingo opt-in; warnings are
existing `exchange_calendars` NumPy timedelta deprecations. No live market-data
verification was required. Commands for the focused matrix are in
[development](development.md); the draft PR records exact acceptance commands.
