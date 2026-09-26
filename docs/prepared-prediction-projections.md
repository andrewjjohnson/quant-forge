# Prepared bounded prediction projections (QF-60)

QF-60 reuses QF-52 projection and immutable expected-lineage preparation during
one sequential execution. Scientific verification still runs at every artifact
boundary. No preparation enters persisted identities, compact records or resumes.

## Previous call paths

For exact-timestamp selection the path was
`PredictionEvaluator.select -> _partition -> partition ->
prediction_metadata_prefix -> bounded_prediction_view`. Session-based prediction
partitions instead use `partition -> project_dataset -> bounded_prediction_view`.
Test execution and test-artifact validation independently call `_partition`.
`PredictionEvaluator.membership -> _adapters.membership` also constructs the
fold's development, selection and test projections. Each role remains separate.

For each compact trial the writer-entry path was
`PredictionGridStudy._execute -> run_incremental_prediction_window_in_session ->
PredictionWindowDecisionValidator -> validate_prediction_view_lineage`.
Completed validation uses `PredictionGridStudy._result -> _validate_record ->
_PredictionGridStore.validate_artifact -> validate_prediction_window_reader ->
PredictionWindowDecisionValidator -> validate_prediction_view_lineage`.
Resume also invokes `_validate_record` while loading each completed trial.

The old lineage function validates the supplied bounded record, reconstructs
canonical daily bars from the independently retained plan's session evidence,
then calls `bounded_prediction_view` again. That factory validates the full
canonical input, selects bounded evidence and validates the result. The same
expected view was consequently reconstructed twice per completed trial, even
when only rule parameters differed.

## Compatibility and verification

`data.prepared_prediction_views.PreparedProjectionRegistry` keys preparation by:

- the complete current canonical metadata fingerprint, including canonical IDs,
  source ancestry, family/timeframe/session/feed identities, adjustment and
  corporate-action policy, retrieval metadata, and all retained evidence;
- a scope snapshot containing QF-8 plan ID, fold ID and the complete validation
  window, including role, evaluation bounds and warm-up declarations;
- exact aware UTC cutoff, optional observed start, and QF-52 adapter version.

Canonical evidence snapshots are hashed directly from their immutable canonical
bytes. They are not decoded into a second full intraday source collection.
Every lookup fingerprints current metadata, so unchanged declared IDs cannot
hide changed evidence. Projection requests additionally compare the actual daily
bar digest. Only exact, reviewed immutable carrier types are admitted; mutable
containers and subclasses cannot alias a previously verified entry merely by
serializing identically. Invalid input fails closed.

The returned QF-52 view retains its original ID and full bounded provenance.
The initial projection also supplies the immutable expected lineage; it is not
reconstructed a second time. `verify_lineage` still calls
`validate_bounded_prediction_record` for **every** supplied record and compares
its complete provenance with the expected result. The enclosing compact validator
still verifies expected window identity, exact schedule, every decision, context,
outcome source and strategy parameters. The preparation key never substitutes
for those authoritative checks.

Lineage compatibility uses the same source/scope/cutoff/start/policy key and binds
the validated record's QF-52 view identity and full lineage to that expected view.
An omitted start may match an explicitly requested start only when it equals the
prepared prefix's actual first observed session. A narrower explicit subrange
cannot satisfy an omitted-start request. Different cutoff instants remain
separate even when they select the same completed daily bars.

Rule parameters are absent from projection identity. Within QF-39 the fixed plan
already declares permitted source/context requirements. Standalone QF-32 grids
use their complete context environment as scope. Changing the source, partition,
role, plan, policy or material context environment requires separate preparation;
trial identities, ranking and tie-breaks are unchanged.

## Lifecycle and memory

`WalkForwardStudy.run/resume` opens `PredictionEvaluator.preparation_scope` and
closes it in `finally`, including interruption. Nested selection, evaluation and
validation borrow that registry. Direct evaluator calls create their own scope;
callers intentionally coordinating compatible calls can use an explicit
`with adapter.preparation_scope():` block. No state survives that block.

Each standalone grid `run`, `resume` or `load_result` owns one registry unless it
borrows the enclosing walk-forward registry. Standalone incremental execution
and offline readers without supplied preparation keep the original independent
lineage path. Grid construction/universe enumeration does not prepare lineage.

The registry retains at most **two** views using least-recently-used eviction.
This matches sequential selection/test reuse while bounding retention across
many folds. Each entry holds one ordinary bounded daily carrier, its immutable
bounded session-evidence snapshot, a source digest and small key/scope metadata.
It retains no parent dataset, canonical intraday bar tuple, decisions, results,
mutable study state or copied canonical evidence. Eviction only causes a correct
rebuild; it cannot change research identity. The existing QF-58 source backing
and QF-59 context indexes are unchanged. Registries are local to sequential
orchestration, not shared between threads/processes.

A restarted compact resume reconstructs preparation once for its partition and
continues at the verified incomplete suffix. Completed decisions are not executed
again. Existing QF-56 checkpoint/version IDs and final canonical bytes remain
unchanged. No opaque handles or cache state are persisted.

## Offline and holdout boundaries

The original `validate_prediction_view_lineage` remains unchanged. QF-9,
QF-40 and QF-41 readers reconstruct and verify persisted scientific evidence with
no registry, provider credentials or network. Tests disable registry construction
and network access during independent inspection, aggregation and reporting.

QF-39 still rejects final-holdout partition requests before any projection.
Generic QF-40 evaluation remains behind its existing explicit reserve/consume
contracts, with a different role/window scope. The real diagnostic calls only
fold-zero selection on disposable copies; it never creates a holdout projection
or evaluates/consumes holdout.

## Deterministic evidence

The two-fold/two-trial fixture uses actual canonical inputs, QF-11 execution,
QF-32 ranking and QF-55/QF-56 persistence. Reference mode invokes the unchanged
preparation and lineage functions without reuse; optimized mode uses the registry.

| Work | Reference initial | Prepared initial | Same-scope completed fold-0 resume | Fresh fold-0 resume |
| --- | ---: | ---: | ---: | ---: |
| Projection builds | 10 | 2 | 0 | 1 |
| Independent lineage reconstructions | 8 | 0 | 0 | 0 |
| Prepared expected-view registrations | N/A | 2 | 0 | 1 |
| Lineage verifications | 8 | 8 | 4 | 4 |
| Trial executions | 4 | 4 | 0 | 0 |

Each initial partition had one direct projection plus four projections inside
lineage validation. Two different folds correctly retain two different views.
The eight initial lineage lookups reuse those two preparations. Same-scope
completed resume adds one projection hit and four lineage hits. A fresh resume
builds its partition once and reuses it for all four lineage checks.

Selection evidence, winning combination/trial IDs and compact JSONL bytes match
exactly across modes. This covers ranking/tie-breaks, exact membership, source,
bounded-view, lineage, context, prediction, outcome and compact hashes. Interrupted
resume retains the original two prefix records byte-for-byte in the finalized
artifact and executes only the remaining four decisions across two trials.
Finalized reruns execute zero decisions. Standalone grids reconstruct expected
lineage once per run/resume/load invocation while verifying every trial.

## Performance measurements and verification

Diagnostic scripts and raw output are ignored under
`reports/qf60-prepared-projections/`; no provider data is committed. Timings are
descriptive measurements on one host, not test thresholds. Projection totals
include canonical validation and bounded-result validation. Lineage timings
overlap projection work in the reference implementation and must not be added to
those totals.

### Synthetic preparation, verification and completed resume

| Work | Before | After |
| --- | ---: | ---: |
| Two-fold initial projection builds | 10 / 7.469219 s | 2 / 1.532333 s |
| Independent lineage reconstruction/verification | 8 / 6.785752 s | 0 reconstructions; 8 verifications / 0.798861 s |
| Fold-0 completed resume projection builds | 5 / 3.716950 s | 0 same scope; 1 / 0.779672 s fresh scope |
| Fold-0 completed resume lineage checks | 4 / 3.340221 s | 4 / 0.350638 s same scope; 4 / 0.348757 s fresh scope |

Initial fold-0 selection, including both trials and verification, took 3.113089 s;
its one projection took 0.753878 s and its four lineage checks 0.350337 s.
Fold-1 selection took 3.368365 s, with projection 0.778455 s and lineage checks
0.448524 s. Registering each already-built expected view took 6–8 microseconds;
this is **not** a claim that projection preparation itself is that cheap.
Successful dictionary lookups took approximately 6–11 microseconds, excluding
source fingerprinting and authoritative verification.

The entire completed fold-0 resume took **1.819535 s** with retained preparation
and **2.568053 s** after a fresh invocation rebuilt it once. Both executed zero
trials/decisions and returned unchanged selection evidence. The small fixture
measures completed resume; the real diagnostic intentionally remains incomplete.

### Disposable frozen QF-45 diagnostic

The unchanged QF-45 examples from commit `9534723` ran against this engine and
the same frozen Massive cache: 48,480 two-minute bars, 250 daily bars and fixed
EMA 8/48/50. Only fold-zero selection ran. Three independent copies of the
original 76-decision checkpoint received 49, 5 and 13 durable appends. The second
selection borrowed the first selection's preparation; the third explicitly
started with an empty registry under cProfile. No other checks overlapped the
timing samples. No complete real study or real holdout projection was run.

| Measurement | QF-59 baseline | QF-60 first preparation | QF-60 compatible repeat |
| --- | ---: | ---: | ---: |
| Frozen input loading, separate from selection | 207.977421 s | 206.349341 s | same loaded inputs |
| Projection builds, uninstrumented | 2 (time not separately recorded) | 1 / 29.777682 s | 0 |
| Complete projection request, including fingerprint | not recorded | 29.793960 s | 0.016632 s |
| Expected-lineage registrations | independent reconstruction | 1, supplied by the projection | 0 |
| Additional independent lineage reconstructions | 1 | 0 | 0 |
| Expected-lineage lookup, including fingerprint | not recorded | 0.015529 s | 0.015732 s |
| Authoritative lineage verifications | 1 | 1 / 2.945240 s | 1 / 2.982642 s |
| Preparation reuse hits | 0 | 1 lineage hit | 1 projection + 1 lineage hit |
| Time to first new decision, including selection startup | 103.302827 s | **70.887229 s** | **41.589353 s** |
| Entire bounded selection sample | 112.660529 s (49 appends) | 79.873482 s (49 appends) | 42.345470 s (5 appends) |
| Uninstrumented steady append interval | 0.194204 s | **0.186539 s** | 0.180950 s (4 intervals) |

The initial sample's 48 intervals ranged from 0.178638 to 0.194782 s. Its first
append improved by 31.38%; the steady rate was 3.95% lower, showing no material
regression. The five-append repeat demonstrates compatible reuse, not a complete
second trial. Configuration/adapter setup outside `select` remains excluded from
the selection timers, as in QF-59.

Like-for-like profiled startup reduced **two projections / 112.535333 s** to
**one / 54.745084 s** (51.35% less projection time). The old independent lineage
construction/verification took 62.124353 s, including its 55.619197 s projection.
The new first projection already prepares expected lineage; its extra registration
took 0.000018 s and its one mandatory lineage verification took **6.299611 s**,
including a 0.015836 s authenticated lookup. There is no separate reconstruction.
The first profiled append fell from 192.203632 to 132.989111 s. The complete
13-append profiled sample took 136.860642 s; its 12 intervals averaged 0.319910 s.
Do not compare profiled component times directly with uninstrumented wall time.

| Steady component, cumulative profiled seconds / 13 new decisions | QF-59 | QF-60 |
| --- | ---: | ---: |
| Indexed context selection, both timeframes | 0.008261 | **0.008309** |
| Indexed bounds lookup, both timeframes | 0.00001587 | 0.00001718 |
| Outcome-source validation | 0.173905 | **0.173073** |
| Full-plan hashing inside decisions | 0 calls | **0 calls** |
| Projection/lineage preparation inside decisions | 0 calls | **0 calls** |

The profile records 26 indexed lookups for 13 decisions across two timeframes.
Four plan hashes occur at startup only: purge, environment capture, QF-59 prepared
context and the new partition preparation scope. No reference context selector
or full-source membership scan runs inside prepared decisions. Outcome-source
validation is unchanged. The sampled real decisions emitted no candidates;
generic integration tests separately cover actual prediction/outcome calculations.

The retained real view contains 120 daily bars, 3,256,805 bytes of immutable
bounded session evidence, 4,462 bytes of family manifest, and small metadata.
It retains **zero canonical intraday bars**. At most two such views are retained;
later ranges can have larger bounded evidence, but retention cannot grow with
the number of trials/folds. These are serialized payload sizes, not measured
process RSS. No cache files or additional scientific storage are introduced.

All 125 initial-sample canonical records are byte-identical to the QF-59 sample;
the 81/89 records of the other copies match the overlapping prefix. Each resumed
at sequence 76, `2025-06-27T16:04:00+00:00`, with **zero completed decisions
recomputed**. All five original checkpoint files remain byte-identical, and the
original still contains 76 decisions. Its 4,376,906 journal bytes retain SHA-256
`f15a7eb361dca58b448f8d33f847706f1afb196758a6e9e29ef3058b9fbfb771`.

| Preserved real identity | Value |
| --- | --- |
| Window | `1b387086f589180ad8ab973a76e3d576fae3190129a419d8ac1b336e98ef92c9` |
| Schedule | `2a460b8db97e538add1d61d1e3cc08a554f6f72ba2731a2cc0df478572e39ec1` |
| Shared evidence | `bf845617bff43e2916b8e7304b6958d3ddf3f737bde09f539d0ba11e27463946` |
| First appended decision | `28f726f8037c0386c4d49143af47acb04cd2bd45bead339900e109543c4b6bf8` |

### Remaining cost and QF-61 recommendation

**QF-61 recommended: YES.** Exact non-holdout membership contains 5,760, 5,850
and 5,850 selection decisions, plus 3,900 OOS test decisions per fold. At the
observed 0.186539 s/decision, a fixed single combination implies 29,160 decisions
and approximately **1.51 hours of decision work**. Nine selection combinations
plus the three OOS tests imply 168,840 decisions and approximately **8.75 hours of
decision work**. These are extrapolations, not measured full-study completion:
input loading, adapter/partition startup, completed-artifact verification,
aggregation and reporting add time; candidate-bearing decisions may cost more.
The real holdout is excluded from these counts and was never inspected.

Outcome-source validation remains the dominant measured steady component and
therefore the next likely total-workflow cost. A narrow QF-61 could prepare and
reuse immutable outcome-source invariant evidence within an execution, while
retaining per-request causal/horizon checks, authoritative source/ancestry
verification, independent offline verification and exact outcome/checkpoint IDs.
It must not change label calculations, barriers, trading rules or QF-45 parameters.
QF-60 implements none of this and does not create QF-61.

### Local reproduction and checks

All checks use uv **0.12.1**, Python **3.13.14**, frozen dependencies and
`UV_CACHE_DIR=/private/tmp/quantforge-uv-cache`; the executable on this host is
`/Users/ajohnson/.local/bin/uv`. The before/after startup harnesses are
`baseline.py` and `startup_scope.py`; `profile_resume.py` creates disposable real
checkpoint copies and stops only after the requested number of durable appends.
Use new destinations when repeating; never resume the authoritative checkpoint.

```bash
uv sync --all-extras --frozen
uv run --frozen ruff format --check .
uv run --frozen ruff check .
uv run --frozen pyright
uv run --frozen pytest tests/integration/test_prepared_prediction_views.py \
  --junitxml=reports/tests/qf60-compatibility.xml
uv run --frozen pytest tests/unit/prediction tests/unit/validation \
  tests/unit/walk_forward tests/integration/test_bounded_prediction_views.py \
  tests/integration/test_bounded_prediction_workflow.py \
  tests/integration/test_bounded_session_integrity.py \
  tests/integration/test_compact_prediction_provenance.py \
  tests/integration/test_incremental_prediction_provenance.py \
  tests/integration/test_prediction_source_sharing.py \
  tests/integration/test_prepared_prediction_execution.py \
  tests/integration/test_prepared_prediction_views.py \
  tests/integration/test_prepared_projection_execution.py \
  tests/integration/test_compact_window_consumers.py \
  tests/integration/test_compact_window_consumer_integrity.py \
  tests/integration/test_compact_window_outcome_consumers.py \
  tests/integration/test_compact_window_offline.py \
  tests/performance/test_prediction_context_index.py \
  --junitxml=reports/tests/qf60-focused.xml
PYTHONPATH=. uv run --frozen python reports/qf60-prepared-projections/check_qf45.py
uv run --frozen pytest --junitxml=reports/tests/junit.xml
uv run --frozen pre-commit run --all-files
git diff --check
```

The QF-45 checker verifies all six extracted example/test files against commit
`9534723`, then runs the unchanged `test_spy_ema_smoke.py` and
`test_spy_ema_compact.py` with `pytest.main(['-n', '0', ...])`: **20 passed**.
No QF-45 files were copied into or modified on this branch. The focused regression
suite passed **1,350 tests** (4 warnings) before the last compatibility edge cases
were added; the final compatibility run passed **22 tests**. The final full suite,
including those cases, passed **6,920 tests**, with **3 optional live-API tests
skipped** and **2 warnings**, in 497.86 s. Frozen sync, formatting (561 files),
linting, typing (zero errors/warnings), all pre-commit hooks, and both working-tree
and staged diff checks passed. GitHub CI is deliberately left for the owner to
report and is not waited on or polled.
