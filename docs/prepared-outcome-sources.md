# Prepared outcome sources (QF-61)

QF-61 separates immutable source admission from observation-specific outcome
requests. It preserves every scientific engine/schema/configuration version.
See [ADR 0033](decisions/0033-prepare-outcome-sources.md).

## Previous execution and measured cause

`run_prediction_study_in_session` fixed the rule's signals, then unconditionally
called `validate_prediction_source`, even for an empty signal tuple. That calls
`data.prediction_inputs._validate_source`: verify the reference against the
canonical/bounded ancestry, hash the session policy, construct adjustment basis,
then visit **every** source bar to check symbol, adjustment basis, provider/request/
snapshot ancestry and hash its feed scope. It finally compares family manifests.
It does not download data or reconstruct the canonical dataset itself. Canonical
schema, prices, ordering and artifact fingerprints were already validated by the
`TimeframeBarSeries.from_*_dataset` constructors.

QF-60 measured 0.173073 profiled seconds per decision in that source validation,
against 0.186539 uninstrumented seconds per decision overall. These two timing
methods are not additive; they nevertheless identified the repeated scan as the
dominant cost. The real sample emitted no candidates. Labeler `validate_dataset`
for QF-47/QF-49 is a metadata-only no-op, so it was not the expensive validation.

For actual candidates, `bounded_outcome_source` also scanned the complete source
to select one session, then scanned that session for the preceding anchor and
future reach. `resolve_future_observation` independently scanned the full source
for the expected endpoint/developing interval. QF-49 then checked the bounded
reference/future price basis. QF-47 checked the bounded path and expected calendar
intervals. Both use the same QF-11 source validation and dispatch; they did not
have independent full-source validators. Multiple configured feature outcomes
reuse the existing dataset session and now share its preparation registry.

## Prepared capability and lifecycle

QF-63 note: the registry now retains the QF-58 normalized backing per original
source (`copy_memo`). A later QF-42 iterator (QF-32 trial 2..k) or QF-7
configured outcome in the same session therefore reuses the exact admitted
backing. Previously each normalized its own copy, missed the exact-backing check
and re-authenticated the full source on every decision. See
[prepared historical features](prepared-historical-features.md).

`prediction.prepared_outcomes.PreparedOutcomeSources` belongs to one
`PredictionStudyDatasetSession`. It admits the finite sources used by that
execution; there is no process-global cache. QF-42's per-decision dataset-session
shells retain the same registry while keeping labelers and other mutable
components isolated. QF-7 fixed-candidate replay shares it across configured
outcomes. New independent dataset sessions start with empty preparation.

Preparation remains after causal signal generation, exactly where source
validation previously ran. The first decision can prepare once even if empty.
Subsequent empty decisions perform only the immutable capability/metadata binding
check: no observation request, resolution, horizon/path traversal, label callback,
or request validation. Candidate decisions construct the unchanged explicit
`OutcomeEvaluationRequest`, select indexed bounds, and execute normal QF-11
detached label callbacks and evaluator/mutation guards.

`PreparedOutcomeSource` retains one immutable `TimeframeBarSeries` backing,
source/input compatibility ID, dataset ID/fingerprint, sorted bar-end timestamps
and read-only session-to-position ranges. Binary searches identify the bounded
same-session path and at most two endpoint candidates (the exact end and a
possible developing predecessor). The original resolver still decides status
using the actual full-source final coverage boundary and canonical calendar.
Observed gaps never redefine expected alignment.

QF-58's recursive exact-type admission is reused for sources. Its supported
calendar `Timestamp` normalization preserves exact values and rejects attached
state, subclasses or lost precision. QF-52 cutoff normalization uses the same
rule; QF-60's reviewed immutable metadata check is reused. Unsupported mutable
graphs retain the original validation/scan path. They cannot poison preparation.

QF-59's context index binds warm-up/window/role selection; it is intentionally not
passed to the future-label layer. QF-61 uses the same binary-search approach with
outcome-specific session ranges and coverage semantics. QF-60 projection reuse
and independent lineage verification are unchanged.

## Compatibility identity

The operational key hashes the complete prediction market-data/provenance
snapshot, source family reference (including canonical ancestor and feed), family
manifest ID, full timeframe/session/calendar policy, streamed serialized contents
of every bar, bar count, developing-source coverage evidence and preparation
version `1`. Bar content includes symbol, adjustment/corporate-action basis,
schema, provider/request/snapshot ancestry, prices and boundaries. The prediction
input binds the canonical or QF-52 bounded role/cutoff evidence already supplied
by the caller. Different bounded inputs cannot reuse one registry.

An independently reconstructed source must pass recursive immutability checking
and complete content authentication before its key can match. Equal prices under
different provenance are not compatible. A previously admitted exact immutable
backing can use a constant-size capability binding check without recomputing its
content digest. This identity proof is not a cache keyed by Python object IDs or
paths; the only object-ID memo is the existing QF-58 normalization/copy mechanism.
No strategy, prediction candidate or direction enters the source key.

## Validation split

| Work | Preparation | Each requested observation |
| --- | --- | --- |
| Canonical schema/OHLC/price/artifact validity | Existing artifact constructors remain authoritative | No repeated artifact load |
| Full symbol, adjustment, feed/request/snapshot ancestry and family validation | Once per compatible source/input | Source/input/configuration binding checked |
| Immutable graph and content/provenance fingerprint | Once for admitted backing; independent new handles are authenticated | Exact admitted backing retained |
| Ordering, duplicate/overlap and timeframe consistency | Once with index construction | No full-source scan |
| Timestamp positions and session ranges | Once | Binary searches, bounded slice |
| Observation timestamp, session and exact completed reference | No cached label assumptions | Checked by original anchor/resolver/labeler contracts |
| Horizon, source resolution, ceiling alignment and temporal reach | No cached outcome configuration | Original QF-46 checks |
| Session overflow, early close, missing/developing data, dataset end | Retain coverage evidence | Original status classification |
| MFE/MAE path completeness and target/stop ambiguity | Locate bounds only | Original QF-47 traversal/arithmetic/evaluation |
| Per-label configuration and mutation isolation | Not cached | Original QF-11 guards and detached bounded copies |

Expected calendar-window enumeration and `OutcomeResolution` invariant checks
remain in the existing QF-46/QF-47 request code. This change removes complete
historical-source scans, not the checks of expected intervals for a requested path.

## Explicit anchors and future dense callers

The prepared source accepts existing observation requests; it does not require a
prediction rule or candidate. A caller can prepare once and reuse the capability
for thousands of explicit anchors, in any order, across sessions and horizons:

```python
from copy import deepcopy
from quantforge.prediction.prepared_outcomes import PreparedOutcomeSources
from quantforge.prediction.timestamp_execution import bounded_outcome_source
from quantforge.prediction import evaluate_outcome_request

# dataset and source have passed the normal canonical input/artifact boundary.
registry = PreparedOutcomeSources()
prepared = registry.prepare(dataset, source)
assert prepared is not None
labeler.validate_dataset(dataset)
for request in explicit_observation_requests:
    bounded, resolution = bounded_outcome_source(
        prepared.source, request, prepared=prepared
    )
    label = evaluate_outcome_request(
        labeler,
        dataset,
        request,
        source=deepcopy(bounded),
        resolution=deepcopy(resolution),
    )
```

The existing generic dispatch checks each request's label configuration and
dataset identity. Production custom-component orchestration should preserve
QF-11's dataset/configuration/value snapshots and mutation guards as well as the
bounded copies shown here. This is source preparation, not a new dataset engine.
Future triggered, all-bars and deterministic-sampling selectors can all supply
the same anchor contract. No observation-mode strings, ML model, feature engine,
parallel execution or dense dataset product are introduced.

## Memory, resume and offline boundaries

Storage is O(source bars + source sessions) references/positions, not copied price
arrays. Content hashing streams one serialized bar at a time; no second full
serialized dataset is retained. QF-58 may normalize timestamp shells once for
otherwise immutable backing. QF-11 continues copying only requested bounded
callback inputs. The registry retains no predictions, labels or completed rows.

Operational state is discarded with the dataset session and rebuilt once after
restart. Completed compact prefixes are verified by the original QF-56 path;
they are not executed again. No handles/indexes/compatibility keys enter result,
request, prediction, outcome, context, checkpoint or compact scientific identity.
Existing engine/schema versions and independent offline validators are unchanged.

The future-bearing capability is never attached to `PredictionRuleContext`,
`MultiTimeframeContext`, normalized features or user rule logic. Real QF-45
diagnostics use only copied fold-zero selection checkpoints and the frozen source;
the authoritative checkpoint, strategy parameters and real holdout remain intact.

## Evidence and local checks

The focused tests compare the unchanged reference path (preparation disabled)
against indexed execution. They cover 10/30/31/60/120-minute endpoints, normal and
early-close sessions, holiday gaps, exact-close and overflow behavior, missing/
developing/end-of-data statuses, exact anchor rejection, MFE/MAE, target-first,
stop-first, neither and same-bar ambiguity. Full primitive/serialized equality
also checks IDs, provenance, context and compact hashes.

Invocation fixtures cover 30 empty decisions (one validation/index build, zero
requests), 30 mixed decisions (10 candidates/requests), 30 candidate decisions
(30 requests), and three configured outcomes sharing one preparation. The dense
fixture uses 1,000 distinct explicit anchors with one full validation, one index
build, 1,000 request checks and 1,000 equal reference labels. A reference-produced
two-decision prefix resumes in a fresh session, executes only the third decision,
preserves journal prefix bytes and produces byte-identical final compact output.
Finalized retry executes no decisions and needs no prepared outcome state.
The QF-57 offline fixture disables preparation and network access before QF-40,
QF-9 and QF-41 consumption/verification/reporting.

### Deterministic scale fixture

The isolated `-n 0` QF-61 test run records measurements without timing thresholds.
The 1,000-anchor fixture uses seven synthetic sessions and a 30-minute explicit
forward-return request, without any prediction rule. It measured preparation at
0.042656 s, including full validation at 0.002197 s and index construction at
0.000196 s. All 1,000 labels matched the reference; indexed resolution plus full
label dispatch took 21.820215 s (0.021820 s/request). This includes the preserved
calendar/status/request checks, not only the binary searches.

| Thirty-decision synthetic fixture | Validation/index builds | Requests/labels | Seconds/decision | Label dispatch seconds/request |
| --- | ---: | ---: | ---: | ---: |
| Empty | 1 / 1 | 0 / 0 | 0.039140 | no calls |
| Mixed | 1 / 1 | 10 / 10 | 0.047073 | 0.000297 |
| Forward return | 1 / 1 | 30 / 30 | 0.064945 | 0.000276 |
| MFE/MAE | 1 / 1 | 30 / 30 | 0.078649 | 0.012967 |
| Target/stop | 1 / 1 | 30 / 30 | 0.079609 | 0.012909 |

Decision measurements include ten three-decision window setups and the cold source
preparation; they are not warmed steady throughput. Label dispatch excludes source
bounds/resolution and includes the unchanged concrete labeler/evaluator work.
These lightweight fixture timings are distinct from the real-source profile below.

### Real QF-45 disposable resume

The diagnostic used the same frozen 48,480 two-minute bars and 250 daily bars as
QF-60, with the unchanged QF-45 8/48/50 configuration from commit `9534723`.
Three independent copies of the authoritative 76-decision fold-zero selection
checkpoint stopped after 257, 5 and 65 additional durable appends respectively.
The first two were uninstrumented apart from narrow counters/timers; the last
used cProfile and an empty projection registry. No test workloads overlapped
these measurements. Adapter construction is outside the selection timers, as in
QF-60. The real study was not completed and no holdout projection was built.

| Uninstrumented measurement | QF-60 reference | QF-61 |
| --- | ---: | ---: |
| Frozen input loading | 206.349341 s | 205.218223 s |
| First projection build | 1 / 29.777682 s | 1 / 29.612592 s |
| Reused projection request | 0.016632 s | 0.015231 s / zero builds |
| Time to first new durable append, fresh projection | 70.887229 s | 72.020635 s |
| Time to first new durable append, reused projection | 41.589353 s | 42.265536 s |
| Outcome preparation, including normalization/content identity | per-decision validation | 1 / 1.111216 s |
| Full outcome-source validation | every decision | 1 / 0.081909 s |
| Timestamp/session index construction | repeated source scans | 1 / 0.007926 s |
| Complete source compatibility digest | not separately retained | 1 / 0.731551 s |
| Matching first 48 steady intervals, no candidates | 0.186539 s/decision | **0.106408 s/decision** |
| Extended 256 intervals, including one candidate | not sampled | **0.132651 s/decision** |
| Extended 255 no-candidate intervals | not sampled | 0.132426 s/decision |
| One actual candidate's complete append interval | not sampled | 0.190032 s |
| Actual candidate request binding check | not sampled | 0.000001834 s |
| Actual candidate label dispatch | not sampled | 0.000310 s |

The like-for-like steady sample improved **42.96%**. The extended sample spans a
session transition and larger contexts; it is not a like-for-like comparison
with the old short sample. Startup adds the expected one-time source preparation;
QF-60 projection time did not regress. Fresh dataset sessions rebuilt outcome
preparation once even when the existing QF-60 projection registry was reusable.
The real initial sample emitted one candidate: 257 decisions, one validation,
one index build, one request check and one label dispatch. No-candidate decisions
performed zero request checks and zero labels.

| Component in separate 65-decision cProfile sample | Profiled seconds/decision |
| --- | ---: |
| Indexed context selection, both timeframes | 0.009094 |
| Complete rule-context preparation, including selection | 0.090419 |
| Context value serialization, three calls/decision | 0.040365 |
| Outcome registry reuse after subtracting its one cold preparation | approximately 0.000101 |
| TA-Lib adapter computation, three calls/decision | 0.000806 |
| Compact conversion | 0.006623 |
| Durable append, including validation/write/checkpoint | 0.003444 |
| Journal write / checkpoint publication (within append) | 0.000413 / 0.000226 |

These inclusive profile measurements overlap and must not be added to each
other or subtracted from uninstrumented timings. The profiled source validation
cost was 0.175640 s **once**, comparable to the old 0.173073 s **each decision**.
The remaining measured work is primarily context construction, immutable value/
manifest serialization and existing isolation checks. The bounded sample does
not demonstrate a new pathological generic blocker. **Another performance ticket
is not required; resume QF-45 after QF-61 review.** Continue fixed run, small grid,
walk-forward/OOS, initial report and manual audit, preserving the existing explicit
holdout gate. This recommendation does not authorize consuming holdout here.

All 125 overlapping QF-60 canonical records are byte-identical. The three copies
retain window, schedule and shared-evidence identities; each begins at sequence
76, `2025-06-27T16:04:00+00:00`, with zero completed decisions recomputed. All five
authoritative checkpoint files are byte-identical and still contain 76 decisions.
The preserved 4,376,906-byte journal prefix has SHA-256
`f15a7eb361dca58b448f8d33f847706f1afb196758a6e9e29ef3058b9fbfb771`.
The synthetic resume fixture, unlike the intentionally interrupted real sample,
also proves byte-identical finalized output against the reference engine.

| Preserved identity | Value |
| --- | --- |
| Window | `1b387086f589180ad8ab973a76e3d576fae3190129a419d8ac1b336e98ef92c9` |
| Schedule | `2a460b8db97e538add1d61d1e3cc08a554f6f72ba2731a2cc0df478572e39ec1` |
| Shared evidence | `bf845617bff43e2916b8e7304b6958d3ddf3f737bde09f539d0ba11e27463946` |
| First resumed decision | `28f726f8037c0386c4d49143af47acb04cd2bd45bead339900e109543c4b6bf8` |

The retained source has 48,480 timestamp references and 250 session ranges plus
one immutable backing. No second price store, new cache artifact or scientific
storage is added. These counts describe retained structure, not measured RSS.
The disposable script, JSON measurements, profile and copies are ignored local
diagnostics under `reports/qf61-prepared-outcomes/`; no market data is committed.

### Scope and local verification commands

Four production files change: the new `prediction/prepared_outcomes.py`,
`prediction/study.py`, `prediction/timestamp_execution.py`, and
`prediction/outcome_resolution.py`. Four test files change: the new unit and
performance `test_prepared_outcomes.py` files, the new integration
`test_prepared_outcome_execution.py`, and the existing integration
`test_compact_window_offline.py`. Architecture/development guidance and ADR 0033
document the boundary. No dependencies, QF-45 strategy files, scientific schemas,
execution assumptions or holdout gates change. No ML or dense dataset feature is
implemented.

Commands use uv **0.12.1**, Python **3.13.14**, frozen dependencies and
`UV_CACHE_DIR=/private/tmp/quantforge-uv-cache`. Pre-commit uses the existing
`PRE_COMMIT_HOME=/private/tmp/quantforge-pre-commit-cache` environments.

```bash
uv sync --all-extras --frozen
uv run --frozen ruff format --check .
uv run --frozen ruff check .
uv run --frozen pyright
uv run --frozen pytest -n 0 tests/unit/prediction/test_prepared_outcomes.py \
  tests/integration/test_prepared_outcome_execution.py \
  tests/performance/test_prepared_outcomes.py \
  --junitxml=reports/tests/qf61-focused.xml
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
  tests/integration/test_prepared_outcome_execution.py \
  tests/integration/test_compact_window_consumers.py \
  tests/integration/test_compact_window_consumer_integrity.py \
  tests/integration/test_compact_window_outcome_consumers.py \
  tests/integration/test_compact_window_offline.py \
  tests/performance/test_prediction_context_index.py \
  --junitxml=reports/tests/qf61-regressions.xml
PYTHONPATH=. uv run --frozen python reports/qf61-prepared-outcomes/check_qf45.py
uv run --frozen pytest --junitxml=reports/tests/junit.xml
uv run --frozen pre-commit run --all-files
git diff --check
git diff --cached --check
```

The QF-45 checker verifies all six extracted example/test files against commit
`9534723` before running the unchanged smoke and compact tests with `-n 0`.
The regression selection covers QF-11/42/47/49/52/55/56/57/58/59/60 and the
QF-32/QF-39 parameter/walk-forward integrations; the full suite covers the remaining
component boundaries. GitHub CI is left to the owner and is not waited on or polled.

Final local results: frozen sync, formatting (**567 files**), linting, strict
typing (**zero errors/warnings**), all seven pre-commit hooks, working-tree and
staged diff checks passed. QF-61 focused tests passed **67 tests** in 90.24 s;
their six warnings concern recording benchmark properties in xunit2 JUnit output.
The broader regression selection passed **1,421 tests** (four existing calendar
dependency deprecation warnings) in 286.10 s. Unchanged QF-45 focused tests passed
**20 tests** in 7.11 s. The final full suite passed **6,987 tests**, with **three
optional live-provider tests skipped** and **two existing calendar dependency
deprecation warnings**, in 513.59 s. No failed tests remain. These are local
results; GitHub CI was not monitored.
