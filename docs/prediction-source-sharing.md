# Immutable prediction source sharing (QF-58)

QF-58 changes execution-object copying only. Scheduling, causal selection,
indicator calculations, outcomes, validation, compact formats, scientific hashes,
and checkpoint/resume contracts are unchanged.

## Object graph and copy boundary

Previously, `prediction/window.py:iter_prediction_window_decisions` called
`deepcopy(study)` once for its pristine template and `deepcopy(template)` for every
scheduled decision. The same iterator serves both legacy materialized execution
and QF-56 incremental execution (`window_execution.py`).
At base commit `580b553`, these calls were lines 501 and 512 respectively.

The study graph contains:

- strategy, outcome labeler and evaluator instances, including their parameters,
  indicators, mutable counters, caches, diagnostics and custom attributes;
- an immutable feature-configuration snapshot and schema string;
- `outcome_source`, a `TimeframeBarSeries` with the complete source bar tuple,
  dataset/family reference, timeframe/session policy, manifest identity and,
  for canonical source series, developing-source coverage evidence;
- any fields added by custom `PredictionStudy` subclasses.

Intraday source bars carry immutable Decimal OHLCV, observation/session timestamps,
provider/retrieval/request/snapshot provenance, feed scope and adjustment basis.
Session bars carry completed periods, session dates, ordered constituent IDs and
source identity. They are frozen/slotted records and tuples, not DataFrames or
arrays. Runtime subclasses can nevertheless carry mutable attributes. Calendar
`pandas.Timestamp` values also have writable instance dictionaries.

The separate `PredictionStudyDatasetSession` (QF-3/QF-52 daily metadata), context
provider, exact schedule, validation plan and optional indicator-output cache are
arguments outside this study graph. They were not copied by these two calls and
QF-58 does not change them. QF-11 retains its existing detached dataset/session
checks. The context provider creates a new view for each exact decision; a shared
outcome source never becomes the rule's input.

`prediction/source_sharing.py:prediction_source_copy_memo` checks the source graph
once per iterator. Only exact reviewed types qualify. Actual fields are checked,
so a frozen annotation cannot conceal a list, dictionary, array, custom subclass,
mutable timezone or cyclic record. Unsupported graphs retain ordinary deepcopy.
The check supplements, and never replaces, existing scientific source validation.

Known calendar timestamps without attached state are normalized once to standard
`datetime` only if both equality and ISO serialization remain exact. Sub-microsecond
values fall back without rounding. Only enclosing tuples/records that need these
replacements are rebuilt. All unchanged immutable leaves retain their references.
An already deeply immutable source is reused directly.

Python object identity does not participate in scientific hashing. The helper's
`id(...)` values are temporary deepcopy-memo keys; QF-11's existing labeler IDs
likewise scope an operational validation memo. Scientific identities continue
to derive from the unchanged canonical values and serialized evidence.

Both deepcopy calls receive a **fresh** memo containing only the original and,
when different, normalized source roots (at most two entries). The normalized
source is shared by every decision; no per-decision graph walk is needed. The
mutable template and each decision remain independent, including study extensions,
rule state, labeler/evaluator state and nested containers. A populated deepcopy
memo is never reused across decisions.

QF-11's `study.py` bounded future-label source and resolution copies remain
untouched. A labeler still receives detached data restricted to its declared
same-session reach; its mutations still trigger the existing checks. There is no
custom global `__deepcopy__`, cache of scientific validation, index optimization,
new callback protocol, or modified source serialization.

## Exact regression evidence

`test_prediction_source_sharing.py` uses a deliberately mutable generic rule and
study extension, real canonical intraday fixtures, QF-52 bounded metadata,
completed daily context, normalized TA-Lib SMA output, candidate/no-candidate
branches, all four forward-return horizons, MFE/MAE and target/stop outcomes.

For each outcome configuration, old semantics are obtained by replacing only the
new copy-memo helper with an empty memo. Both paths use the same unchanged engine.
The tests require byte-for-byte equality of legacy results and finalized compact
JSONL, covering timestamps, candidate/rule output, features/indicator values,
context and prediction IDs, outcome values/statuses, rows, compact decision
hashes and final window identity. Context value snapshots are compared exactly.
Exact QF-48 membership is captured from the same canonical source and schedule.

Adjacent decisions expose 240, 241 and 242 completed primary bars. All visible
ends are at/before their own cutoff; daily context remains January 2 during
January 3 decisions. The canonical input includes the future January 3 daily
close, while bounded metadata and rule contexts exclude it. Rule/study mutable
state starts pristine at every decision and remains unchanged on the caller.

For three decisions, instrumentation observes four full-source deep copies under
old semantics (template + three decisions) and **zero** under sharing. The two
candidate decisions still produce four small detached labeler-source copies.
Those are intentional safeguards, not copies of the large backing source.

A checkpoint produced with old-copy execution commits two decisions. Optimized
execution resumes at sequence 2, recomputes zero completed decisions, preserves
the saved prefix bytes, and finalizes exactly the uninterrupted baseline's bytes
and result identity. A finalized rerun performs no further context evaluation.

Unit tests additionally exercise ordinary field/tuple mutation, detached serializers,
mutable source/bar/provenance subclasses, malformed nested containers, unsupported
timezones, timestamp annotations and nanosecond precision. Intraday, daily and
weekly artifact types are covered. Frozen record bypasses such as
`object.__setattr__` are not an allowed source mutation API; QF-11's existing
component-facing defensive copies remain unchanged.

## Real QF-45 profile

The unchanged QF-45 example from commit `9534723` ran against this branch's
generic engine using its existing local immutable cache: 48,480 two-minute bars
and 250 daily bars. Fixed EMA parameters remained 8/48/50. Two independent copies
of `reports/qf45-compact/fixed` supplied the same 76-decision checkpoint. Only
`adapter.select(config, 0, destination)` was invoked; neither the full study nor
holdout evaluation/consumption ran. Execution stopped immediately after a bounded
number of durable appends.

The uninstrumented pass appended **13 decisions**, sequences 76–88, from
2025-06-27 16:04 through 16:28 UTC (12:04–12:28 America/New_York). The **12
append-to-append intervals** averaged **1.488789 seconds/decision**, with a range
of **1.340562–1.634873 seconds**. This is approximately **39% less elapsed time**
and **1.63× throughput** versus the prior 2.423–2.440 seconds/decision. The first
append took 101.928 seconds, including selection startup; it is excluded from
the steady-state mean. Total bounded selection took approximately 119.8 seconds.
Frozen-input loading was separate and took 208.036 seconds. The uninstrumented
pass used only append-boundary timestamps, with no profiler or copy-count hooks.

These are bounded operational timings on one host, not a speed threshold or a
full-study extrapolation. All measured QF-45 decisions were no-candidate decisions;
candidate/outcome equivalence is covered separately by the deterministic
regression fixture. The architectural acceptance criterion is zero recursive
full-source copies per historical decision.

### Instrumented decision costs

The independent cProfile pass appended **7 decisions**, sequences 76–82.
Six steady intervals averaged 1.743420 seconds, with profiler overhead. Cumulative
function time divided by seven supplies the table below, except study copies,
which use the eight observed template/decision copies. Nested costs overlap and
must not be added together.

| Operation | Prior baseline | QF-58 |
| --- | ---: | ---: |
| QF-42 study copy | 2.707 s/copy | 0.000585 s/mutable-shell copy |
| Per-decision recursive large-source copy | 1 per decision | **0 calls / 0 s** |
| QF-11 execution | 1.842 s/decision | 1.704550 s/decision |
| Context preparation | 1.626 | 1.481669 |
| Context selection (two timeframes) | 1.459 | 1.316017 |
| Plan hashing from context selection | 1.001 | 1.005572 |
| Outcome-source validation | 0.174 | 0.178892 |
| Multi-timeframe alignment | 0.00823 | 0.008470 |
| TA-Lib computation | 0.000741 | 0.000790 |
| Compact conversion | 0.00694 | 0.007019 |
| New-decision compact validation/hashing | 0.00256 | 0.002413 |
| QF-56 append/persistence | 0.00359 | 0.003532 |
| Journal/checkpoint/directory fsync | 0.000397 | 0.000363 |

The one-time graph verification and timestamp normalization cost **1.571044
seconds**, producing one shared 48,480-bar backing and a two-entry memo. All seven
executions used that same backing; eight study copies cost 0.004680 seconds in
total by direct elapsed-time instrumentation. Source reduction instrumentation
observed **zero** large-source copies during the iterator. One earlier factory
copy remains in `walk_forward/prediction.py:_grid`; that unrelated startup path
was not optimized (its two deepcopy calls, including factory and analyzer,
totaled 2.864 seconds). No global source-copy override was installed.

Plan hashing had 16 total calls: 14 from the two context timeframes across seven
decisions (7.039003 seconds), plus two startup calls. Only the 14 decision calls
are in the table. `_accept` likewise had 76 prefix-validation calls and seven
new-append calls; only the latter contribute to the per-decision figure. Prefix
validation is required integrity work, not recomputation of completed decisions.

### Real checkpoint compatibility

Both copied checkpoints recognized all **76 completed decisions** and resumed
at **sequence 76, 2025-06-27T16:04:00+00:00**, recomputing **zero** completed
decisions. All **4,376,906 original journal bytes** remained unchanged, with SHA-256
`f15a7eb361dca58b448f8d33f847706f1afb196758a6e9e29ef3058b9fbfb771`.
The copies reached 89 and 83 durable decisions respectively; all 83 overlapping
records were byte-identical. All five files in the original checkpoint tree were
byte-identical afterward, and the original remained at 76 decisions.

The following identities were preserved across each resume:

| Identity | Value |
| --- | --- |
| Window | `1b387086f589180ad8ab973a76e3d576fae3190129a419d8ac1b336e98ef92c9` |
| Schedule | `2a460b8db97e538add1d61d1e3cc08a554f6f72ba2731a2cc0df478572e39ec1` |
| Shared evidence | `bf845617bff43e2916b8e7304b6958d3ddf3f737bde09f539d0ba11e27463946` |
| First appended decision | `28f726f8037c0386c4d49143af47acb04cd2bd45bead339900e109543c4b6bf8` |

The real window was deliberately not finalized. Exact final-window identity
parity is established by the completed deterministic old/new fixture instead.
No QF-45 strategy, research parameter, holdout ledger or holdout outcome changed.

### Startup cost and measured invocation scope

The instrumented selection's first append took **189.874 seconds**. Existing
`bounded_prediction_view` still ran **twice for 111.660 seconds** (prior 110.953).
One call came from `prediction_metadata_prefix` (55.783 seconds), the other from
`validate_prediction_view_lineage` (55.877 seconds). The lineage call totaled
**62.422 seconds** (prior 62.042), overlapping the second projection. Selection
partition construction totaled 105.819 seconds, including prefix projection and
independent dataset/provenance validation. These startup times are not additive.

A separate synthetic diagnostic used **two folds and two trials per fold**, then
resumed the completed first fold. It traced both projection callers and lineage
validator call stacks through the unchanged generic compact path:

| Operation | Fold 0 initial selection | Fold 1 initial selection | Fold 0 completed resume |
| --- | ---: | ---: | ---: |
| Partition prefix projection | 1 | 1 | 1 |
| Lineage validation at incremental-window entry | 2 | 2 | 0 |
| Lineage validation while reading completed trial artifacts | 2 | 2 | 4 |
| Projection nested inside lineage validation | 4 | 4 | 4 |

Each initial trial caused one entry validator and one completed-artifact
validator. Completed resume validated each existing artifact during record loading
and again while assembling the result. The two initial folds had distinct cutoffs
(July 8 and July 9 at 16:00 UTC); every projection within a fold reused that fold's
cutoff, yet reconstruction still repeated. Thus this work is **not once per
entire run or once per unique source window**: partition projection repeats per
selection invocation/fold, and lineage reconstruction repeats per trial validator,
including artifact consumption and resume. The actual interrupted QF-45 segment
only reached one trial-entry validator, explaining its two total projections.
No synthetic holdout evaluation was invoked by this diagnostic.

### Follow-up recommendation and reproduction

**QF-59 is warranted.** The smallest next steady-state scope is a validated,
immutable context-selection preparation object: retain the unchanged plan's
scientific ID once and precompute per-source observation keys/indexes, then select
exact visible/warm-up ranges for each cutoff. Preserve exact membership, validation
failures and canonical hash composition. Likely files are `validation/context.py`,
`validation/models.py`, `validation/partitioning.py` and
`walk_forward/prediction.py`. Context selection still allocates and validates all
source timestamp keys and reserializes the large unchanged plan twice per decision;
the 1.005572-second hash cost dominates its 1.316017-second total.

Startup projection/lineage reuse should be separately bounded within that follow-up
or a later story. The narrow candidate is a verified immutable ancestry/projection
record keyed by canonical source identity, cutoff and start, reused only inside
the same trusted execution. Relevant files are `data/prediction_views.py`,
`data/prediction_inputs.py`, `walk_forward/partitions.py` and
`prediction/window_compact_validation.py`. Do not weaken independent ancestry or
persist unverifiable object-identity caches. No such optimization is part of QF-58.

Local diagnostic evidence is retained under ignored `reports/qf58-source-sharing/`:
`profile_resume.py`, `uninstrumented.json`, `instrumented.json`, `resume.prof`,
`profile.txt`, `verification.json`, `startup_scope.py` and `startup-scope.json`.
To reproduce, extract the unchanged QF-45 examples from `9534723` into a temporary
package path, load the existing frozen cache, copy the original checkpoint to
fresh destinations, prepare its unchanged fixed configuration, validate the
adapter, and call only first-fold `select`. Count durable appends and interrupt
after 13/7 for the timing/profile passes. Compare original prefix bytes and
identities before/after; never point the harness at the original checkpoint for
writes. Profile/copy-count hooks exist only in the diagnostic process and tests.

## Validation commands

All commands used uv **0.12.1**, Python **3.13.14**, and the frozen lockfile.
On the measurement host, `uv` below was
`/Users/ajohnson/.local/bin/uv`, with
`UV_CACHE_DIR=/private/tmp/quantforge-uv-cache`. The Homebrew uv 0.12.2 was not used.

```bash
uv sync --all-extras --frozen
uv run --frozen ruff format --check .
uv run --frozen ruff check .
uv run --frozen pyright
uv run --frozen pytest -n 0 tests/unit/prediction/test_source_sharing.py \
  tests/integration/test_prediction_source_sharing.py
uv run --frozen pytest tests/unit/prediction \
  tests/unit/validation/test_timestamp_prediction_membership.py \
  tests/unit/walk_forward \
  tests/integration/test_bounded_prediction_views.py \
  tests/integration/test_bounded_prediction_workflow.py \
  tests/integration/test_bounded_session_integrity.py \
  tests/integration/test_compact_prediction_provenance.py \
  tests/integration/test_prediction_source_sharing.py \
  --junitxml=reports/tests/qf58-focused.xml
uv run --frozen pytest --junitxml=reports/tests/junit.xml
uv run --frozen pre-commit run --all-files
```

Results: frozen sync, formatting, lint, strict typing and all pre-commit hooks
passed. Broad focused regression: **1,010 passed**, including QF-42, QF-11,
QF-52, QF-48, QF-55, QF-56 and QF-39 prediction tests. Full CI-equivalent local
suite: **6,867 passed, 3 skipped**, in 469.34 seconds. The skipped cases are
explicit live-provider opt-ins. Two existing exchange-calendar/NumPy deprecation
warnings remain. Two additional QF-58 safety tests were added afterward;
the final focused QF-58 run passed **19 tests**, including those additions.

The QF-45 examples and their tests are on preserved commit `9534723`, outside
merged main. They were extracted unchanged to `/private/tmp/qf58-reference`,
with only that examples package added to `quantforge.__path__` in a separate
Python process. The following two original test files ran against the QF-58
engine via `pytest.main(['-n', '0', ...])`: **20 passed**.

- `tests/unit/test_spy_ema_smoke.py`
- `tests/integration/test_spy_ema_compact.py`

No QF-45 files were copied into this branch or changed. Local JUnit evidence is
under ignored `reports/tests/` (`qf58-focused.xml`, `qf58-qf45.xml`, `junit.xml`).
