# Indexed historical prediction context (QF-59)

QF-59 removes repeated invariant work from the generic permitted-context path.
It changes no scientific schema, schedule, numerical method, outcome, strategy,
compact format or checkpoint identity. See [ADR 0031](decisions/0031-prepare-indexed-prediction-context.md).

## Previous path

`walk_forward/prediction.py:_PermittedContextProvider.get_context_at` called
`validation/context.py:select_prediction_context_observations` once per required
source. That selector rebuilt `TimestampBoundary` objects for every bar,
`validate_source_observations` rebuilt/checked the full chronology, and two
comprehensions found history and visible bars. Its final `plan.plan_id` serialized
and hashed the complete environment, folds and QF-48 schedule. The provider then
built a membership set and scanned all source bars again to find selected rows.
Two QF-45 timeframes therefore caused two plan hashes and two full source
validations per decision, plus the final retrieval scans.

## Prepared boundary

`PreparedValidationPlan.capture(plan)` captures the existing authoritative ID.
It is an explicit immutable value, not a cache attached to the caller's plan.
`PreparedPredictionContext.capture(plan, window, series=..., input_identity=...)`
captures the small operational selection fields and verifies source immutability
through QF-58. No live plan is consulted during prepared decisions. The full plan
and its potentially large source-evidence serialization are not copied into the
prepared object. A changed or mutable live plan cannot poison a memo: preparing
again reads its current ID, and explicit cross-scope reuse must call
`validate_compatible` at the boundary.

Each `PredictionSourceIndex` retains one immutable source reference, ordered UTC
bar-end keys/timestamps, a window-start integer, timeframe ID and warm-up count.
`bisect_left` resolves the fixed window boundary once; `bisect_right` resolves each
inclusive causal cutoff in O(log n). The original warm-up and visible intervals
are sliced without rediscovering membership by scanning. Daily and higher
context use the same mechanism. At the actual session close, that day's completed
daily bar becomes eligible exactly as in the reference path; before close it
remains hidden. No 390-bar/session or adjacent-decision assumption is used.

The preparation compatibility key is the tuple of authoritative plan ID, exact
window (including role and warm-up), bounded-input ID, and exact source artifacts
(reference/ancestry, family manifest, timeframe/session semantics and contents).
The QF-52 view ID already binds canonical ancestry and cutoff. Equal prices under
different ancestry do not authorize reuse. The provider additionally enforces
its restricted QF-42 schedule with a frozenset. That schedule may be narrower than
the plan's exact QF-48 membership after purging; indexes never redefine it.

Prepared indexes are independent of indicator parameter values because source
selection uses the plan's declared per-timeframe warm-up. Actual requirements,
backend/configuration, completed/developing policy, freshness and alignment are
still checked by QF-20/QF-28/QF-11. The outcome-source configuration does not
participate in context lookup independently of the unchanged plan environment.

The permitted-context provider prepares once before grid/window iteration. Both
legacy and compact QF-42 paths consume its normal QF-11 interface. Each fold/role
gets its own scope. Direct/small callers can keep using the unchanged reference
selector. Mutable/custom source graphs and invalid preparations use that path,
preserving per-decision error handling. No preparation state is persisted.

Selection still constructs and validates the **selected subset** for each decision.
`WindowObservationSelection` checks subset chronology, the causal wrapper checks
its cutoff, and `TimeframeBarSeries` validates/sorts the bounded bars before
QF-20 alignment. These costs are O(selected bars), not repeated full-source
membership discovery. Indicator evaluation still consumes exactly the original
history; this story introduces no rolling-window truncation or precomputation.

## Exact regression evidence

The focused tests compare complete selection primitives (including IDs), selected
bar values/order, warm-up, provenance and source membership against the unchanged
reference selector. They cover adjacent, irregular and reverse-order requests;
normal/early-close sessions; first/last completed bars; July 4's holiday gap;
cross-session changes; missing bars; daily anchors before/at close; preceding
weekly anchors; insufficient warm-up; exact timestamp rejection; incompatible
plans, roles, views, source ancestry, timeframe and session policy.
Holiday, out-of-window and off-hours requests also preserve the reference
selector's rejection type and exact message.

Canonical QF-52 fixtures execute actual QF-11 forward-return, excursion and
target/stop outcomes through both providers. Complete serialized prediction
windows and QF-55 bytes match exactly, including context, prediction and outcome
IDs. A reference checkpoint with two durable decisions resumes at sequence two,
executes zero completed decisions, preserves prefix bytes, and finalizes exactly
to the uninterrupted reference. A finalized rerun executes no contexts.

The structural test observes 14 plan hashes/14 full source validations for seven
two-timeframe reference decisions. For 1,000 prepared requests it observes one
plan hash, two index builds/full source validations and 2,000 indexed lookups.
There are zero full-source membership/retrieval scans per prepared decision.

## Real bounded QF-45 reprofile

The unchanged QF-45 example at commit `9534723` ran against the QF-59 engine and
existing immutable Massive cache: 48,480 two-minute bars, 250 daily bars, fixed
EMA 8/48/50. The harness copied `reports/qf45-compact/fixed` twice and called only
`adapter.select(config, 0, destination)`. It stopped after bounded durable appends;
it never ran the complete study or evaluated/consumed holdout.

The uninstrumented pass appended **49 decisions**, sequences 76–124, June 27,
2025 16:04–17:40 UTC (12:04–13:40 America/New_York). Its 48 append-to-append
intervals averaged **0.194204 seconds/decision**, range **0.184479–0.202260**.
Compared with QF-58's **1.488789**, this is **86.96% less elapsed time**, or **7.67×
throughput**. Timings are descriptive measurements on one host, not thresholds.
A brief local type check overlapped part of the uninstrumented segment; the full
interval range is reported rather than claiming an isolated benchmark host.

The independent cProfile pass appended **13 decisions**, sequences 76–88.
Its 12 steady intervals averaged 0.321172 seconds with profiling overhead.
Cumulative times below divide by 13. Nested costs overlap and must not be added.
All real sampled decisions emitted no candidates; the generic integration tests
separately exercise actual candidate/outcome calculations.

| Operation | Before QF-59, s/decision | After QF-59, s/decision |
| --- | ---: | ---: |
| Uninstrumented append interval | 1.488789 | **0.194204** |
| QF-11 execution | 1.704550 | 0.297183 |
| QF-11 context preparation | 1.481669 | 0.081937 |
| Context selection, two timeframes | 1.316017 | 0.008261 |
| Full validation-plan hashing inside decisions | 1.005572 | **0; no calls** |
| Indexed bound lookup, two timeframes | full scans | 0.00001587 |
| Outcome-source validation | 0.178892 | 0.173905 |
| Multi-timeframe alignment | 0.008470 | 0.008264 |
| TA-Lib compute | 0.000790 | 0.000771 |
| Compact conversion | 0.007019 | 0.006932 |
| New-decision compact validation/hashing | 0.002413 | 0.002287 |
| QF-56 append/persistence | 0.003532 | 0.003334 |

The new selection timing includes constructing its bounded `TimeframeBarSeries`;
the former reference selector's timing excluded the provider's subsequent
full-source retrieval scan. The new bound lookup is about **7.93 microseconds per
timeframe**; the larger selection cost includes required subset validation.
Outcome-source validation is now the largest measured steady-state component.
No outcome optimization was implemented.

### One-time preparation and startup

| Work | Calls | Total profiled seconds |
| --- | ---: | ---: |
| Authoritative plan preparation/hash | 1 | 0.508891 |
| Source timestamp index construction/chronology validation | 2 | 0.184891 |
| QF-58 immutable backing verification/normalization for context sources | 2 | 1.591561 |
| Source indexes including their immutable backing checks | 2 | **1.776452** |
| Complete prepared context capture, including the above | 1 | **2.336378** |
| Existing QF-58 outcome-source backing preparation | 1 | 1.518249 |
| QF-52 bounded projection | 2 | **112.535333** |
| Independent QF-52 lineage validation, overlapping projection | 1 | **62.124353** |

Plan hashing had **three total startup calls**: one preparation, one existing
purge call and one existing environment capture. There were **zero** decision
calls, versus 14 decision calls in QF-58's seven-decision profile (two per
decision). Source validation had two index-build calls and two existing partition
startup calls; none came from a prepared decision. Exactly 26 indexed lookups
served 13 decisions across two sources. The unchanged reference selection
function had zero calls during those decisions.

QF-52 projection split into 56.916137 seconds from `prediction_metadata_prefix`
and 55.619197 seconds from independent lineage reconstruction. These are the
same two invocation boundaries as before. The first uninstrumented append took
103.302827 seconds including selection startup; total bounded selection took
112.660529 seconds. Instrumented first append took 192.203632 seconds, and total
selection 196.090847 seconds. Frozen-input loading separately took 207.977421
seconds. Configuration/adapter setup outside the selection call is excluded from
these selection timers; no full-workflow speedup is inferred from warmed rates.

### Real checkpoint compatibility

Both disposable copies recognized all 76 durable decisions and started at
**sequence 76, 2025-06-27T16:04:00+00:00**. Completed decisions recomputed: **zero**.
All **4,376,906 original journal bytes** remained identical, SHA-256
`f15a7eb361dca58b448f8d33f847706f1afb196758a6e9e29ef3058b9fbfb771`.
The copies reached 125 and 89 durable decisions. All 89 overlapping canonical
records were byte-identical to each other **and to the pre-QF-59 QF-58 output**.
All five original checkpoint-tree files were unchanged; the original still has
76 decisions. Holdout was neither evaluated nor consumed.

| Preserved identity | Value |
| --- | --- |
| Window | `1b387086f589180ad8ab973a76e3d576fae3190129a419d8ac1b336e98ef92c9` |
| Schedule | `2a460b8db97e538add1d61d1e3cc08a554f6f72ba2731a2cc0df478572e39ec1` |
| Shared evidence | `bf845617bff43e2916b8e7304b6958d3ddf3f737bde09f539d0ba11e27463946` |
| First appended decision | `28f726f8037c0386c4d49143af47acb04cd2bd45bead339900e109543c4b6bf8` |

The real study was deliberately not finalized. Exact final-window parity is
established by the completed deterministic reference/indexed resume fixture.

## QF-60 recommendation

**QF-60 recommended: YES.** The measured 112.535-second projection startup remains
material after the steady-state improvement. Its 62.124-second lineage component
overlaps projection and is not an additional independent total.
A repeated diagnostic with two synthetic folds and two trials each found:

| Operation | Fold 0 initial | Fold 1 initial | Fold 0 completed resume |
| --- | ---: | ---: | ---: |
| Partition prefix projection | 1 | 1 | 1 |
| Lineage validator invocations | 4 | 4 | 4 |
| Projection inside lineage validation | 4 | 4 | 4 |

For each initial trial, one lineage check occurs at writer entry and one during
completed-artifact validation. Completed resume checks each artifact during
record loading and final result assembly. The real interrupted profile reaches
only the first writer-entry validation, explaining its two total projections.

The smallest separate scope is execution-local reuse of already validated bounded
projections and independently verified lineage keyed by canonical ancestry,
cutoff/start, view content and complete scientific compatibility. Preserve
independent offline validation and fail closed on incompatible evidence. Likely
files: `data/prediction_views.py`, `data/prediction_inputs.py`,
`walk_forward/partitions.py`, `prediction/window_compact_validation.py` and the
narrow grid validation caller if needed. This branch implements **none** of that
reuse and changes no QF-45 strategy, parameters or downstream QF-40/QF-9/QF-41 code.

## Reproduction and local verification

Ignored diagnostic evidence lives in `reports/qf59-indexed-context/`:
`profile_resume.py`, `uninstrumented.json`, `instrumented.json`, `resume.prof`,
`profile.txt`, `metrics.json`, `verification.json`, `startup_scope.py` and
`startup-scope.json`. It uses unchanged QF-45 examples/tests extracted from
`9534723` into `/private/tmp/qf58-reference`, with that examples package appended
to `quantforge.__path__`. Reproduce with fresh checkpoint-copy destinations;
never write to the original checkpoint. The harness raises `KeyboardInterrupt`
only after its requested number of durable appends.

All checks use uv **0.12.1**, Python **3.13.14**, frozen dependencies and
`UV_CACHE_DIR=/private/tmp/quantforge-uv-cache`; the executable on this host is
`/Users/ajohnson/.local/bin/uv`.

```bash
uv sync --all-extras --frozen
uv run --frozen ruff format --check .
uv run --frozen ruff check .
uv run --frozen pyright
uv run --frozen pytest tests/unit/validation/test_prepared_prediction_context.py \
  tests/integration/test_prepared_prediction_execution.py \
  tests/performance/test_prediction_context_index.py
uv run --frozen pytest tests/unit/prediction tests/unit/validation \
  tests/unit/walk_forward tests/integration/test_bounded_prediction_views.py \
  tests/integration/test_bounded_prediction_workflow.py \
  tests/integration/test_bounded_session_integrity.py \
  tests/integration/test_compact_prediction_provenance.py \
  tests/integration/test_incremental_prediction_provenance.py \
  tests/integration/test_prediction_source_sharing.py \
  tests/integration/test_prepared_prediction_execution.py \
  tests/performance/test_prediction_context_index.py \
  --junitxml=reports/tests/qf59-focused.xml
uv run --frozen pytest --junitxml=reports/tests/junit.xml
uv run --frozen pre-commit run --all-files
git diff --check
```

The unchanged QF-45 `test_spy_ema_smoke.py` and `test_spy_ema_compact.py` ran via
`pytest.main(['-n', '0', ...])` against this engine: **20 passed**. No QF-45 files
were copied into or modified on this branch. QF-59's final focused suite passed
**25 tests**; the broader focused regression run passed **1,287 tests**. The final
full suite passed **6,894 tests**, with **3 optional live-API tests skipped** and
2 warnings, in 468.08 seconds. Frozen sync, formatting, linting, typing,
pre-commit and repository diff checks all passed. GitHub CI is deliberately left
for the owner to report; this task does not wait for or poll it.
