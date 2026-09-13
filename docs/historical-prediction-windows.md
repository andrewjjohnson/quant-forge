# Historical prediction windows

QF-42 evaluates one QF-11 multi-timeframe `PredictionStudy` at every scheduled
decision in an explicit historical interval. It returns an ordered collection
of the original results. Fold selection, frozen configurations, walk-forward
orchestration, and OOS aggregation remain QF-39/QF-40 responsibilities.

## Decision schedule

`PredictionDecisionSchedule(primary_timeframe, start_timestamp, end_timestamp)`
uses a **closed interval** of timezone-aware timestamps normalized to UTC.
Every expected completed primary bar whose **end** lies in that interval is
scheduled, including a bar that started before the requested boundary. Earlier
context remains available under QF-28 declarations. Callers requiring purged
decision boundaries must supply them; this API does not perform QF-8 partitioning.

The schedule reuses QF-18's `intraday_session_windows()` and the existing exchange
calendar. It respects actual session opens/closes, holidays, early closes, DST,
session-open or clock anchoring, and explicit extended-hours scope. For regular
sessions, `expected_sessions(..., include_overnight=True)` uses the calendar's
local open/close day offsets to include adjacent trade-date labels before
filtering bar ends against the exact closed UTC interval. Evening bars belonging
to the next trade date remain scheduled, including Sunday openings. Candidate
labels are bounded by the available calendar; requests beyond its supported
local dates fail explicitly. Explicit extended hours retain their same-day
session-label policy. Completed leading and terminal partial-duration bars are
eligible at their actual ends.
Labels never advance availability. The primary must be intraday and
completed-only, with cross-session bars prohibited, matching the supported QF-28
contract. Contextual developing bars remain an explicit QF-21/QF-28 policy.

Calendars with an intraday recess anywhere in their available schedule, such as
XHKG's lunch break, are rejected with `InvalidPredictionConfigurationError`
before decisions are constructed, under both regular and extended-hours scope.
QF-18's shared window contract assumes a continuous session; supporting recesses
requires a separate aggregation/window policy. Recesses must not be represented
as missing observations or skipped decisions.

Scheduling consults the calendar, **not observed market rows**. A missing primary
bar therefore does not remove a decision. Its context must contain a completed
primary bar ending at exactly the scheduled timestamp; an earlier bar cannot
substitute for it. Missing, stale, incompatible, or incorrectly timestamped
contexts follow the existing `FAIL` or `SKIP` policy. `FAIL` returns no partial
window; `SKIP` retains the QF-11 skipped result. When the historical adapter
rejects a returned context, its exact snapshot and context ID remain in the
skip evidence, including an incorrect source `as_of` alongside the requested
decision timestamp. Rejected contexts never reach indicator or rule execution.
Failures before a context is resolved retain `source_context: null`.
An interval containing no primary bar ends produces a valid empty collection.

`decision_timestamps` is an ordered immutable tuple. The parallel
`decision_sessions` tuple derives each exchange trade-date label in the same
calendar-window traversal. It does not infer a session by taking the UTC or local
date of a timestamp: an evening opening can belong to the next trade date, and
a midnight close can belong to the previous one. These derived labels require
no change to the serialized schedule or its identity. Boundaries, the full primary
timeframe/session configuration, schedule policy/schema, and the UTC timestamp
sequence participate in `schedule_id`.

## Execution and provider contract

`PredictionWindowContextProvider.get_context_at(requirements, *, as_of)` is an
additive timestamp-aware provider contract. A local provider can compose already
validated `TimeframeBarSeries` artifacts:

```python
from dataclasses import dataclass
from datetime import datetime

from quantforge.data import (
    MultiTimeframeContext,
    TimeframeBarSeries,
    build_multi_timeframe_context,
)
from quantforge.prediction import PredictionContextRequirements


@dataclass(frozen=True)
class LocalWindowProvider:
    series: tuple[TimeframeBarSeries, ...]

    def get_context_at(
        self, requirements: PredictionContextRequirements, *, as_of: datetime
    ) -> MultiTimeframeContext:
        return build_multi_timeframe_context(
            as_of=as_of,
            primary_timeframe=requirements.primary.timeframe,
            required_timeframes=requirements.context_timeframe_requirements(),
            completion_policy=requirements.context_completion_policy,
            series=self.series,
        )
```

`run_prediction_window()` validates the QF-3 outcome dataset once, then delegates
each decision to the existing `run_prediction_study_in_session()` with a
provider bound to that UTC timestamp. Context family identity must match the
declared `dataset_family_fingerprint`. The caller supplies immutable artifacts
and provider-environment provenance; rules do not download data.

Each decision receives an independent deep copy of one pristine study template.
Rule, labeler, and evaluator objects must support independent copying while
retaining their scientific configuration. Future-bearing callback state cannot
carry into the next decision. QF-11's labeler-validation memo is fresh for each
component lifetime. QF-11 still owns context validation, normalized indicators,
causal predictions, outcome labeling, evaluation, warm-up and mutation checks,
and every per-decision identity.

```python
from quantforge.prediction import PredictionDecisionSchedule, run_prediction_window

schedule = PredictionDecisionSchedule(primary_timeframe, permitted_start, permitted_end)
window = run_prediction_window(
    prediction_dataset,
    prediction_study,
    schedule=schedule,
    context_provider=LocalWindowProvider(validated_series),
    dataset_family_fingerprint=family.family_id,
    context_environment={
        "provider_id": "local_immutable_artifacts",
        "provider_version": "1",
        "family_manifest_id": family.manifest_id,
    },
)
```

## Result and provenance

`PredictionWindowResult` contains its schedule and ordered
`PredictionWindowDecision` values. Each retains the timestamp, original typed
`PredictionStudyResult`, original context ID when available, and an immutable
primitive snapshot. `results` exposes the underlying ordered tuple without
merging rows or assigning a synthetic QF-11 study ID. Intraday decisions in one
session remain separate observations even when their session-based label is
identical.

Each serialized decision embeds the unchanged QF-11 result and separately
preserves all fixed `generated_signals`, including end-of-data signals omitted
from QF-11's labeled rows. QF-7 accepted, rejected, blocked, and overlapping
dispositions remain intact. QF-31 `NO_PREDICTION` remains its original rejected
candidate with explicit reason and condition evidence. The window's
`no_prediction_decisions` count means a valid context emitted **zero signals**;
it does not erase rejected candidates. Signals without a disposition contract
are counted as `unclassified`.

`counts_primitive()` reports scheduled, valid, skipped, empty/no-prediction,
generated-signal, unavailable-outcome, and supported signal-disposition counts.
These describe decisions/signals, not independent statistical samples.

`window_id` binds the schedule, QF-11 configuration and engine, QF-3 dataset and
fingerprint, context family/provider environment, outcome/evaluator/feature
configuration, and indicator/backend provenance. Optional
`indicator_backend_environment` binds additional fixed settings; QF-32 supplies
its existing backend environment automatically. `window_result_id` additionally
hashes all ordered decision snapshots. Serialization uses the repository's
canonical JSON and SHA-256 conventions.

Window engine version 4 rejects calendars with recesses and validates historical
provider return types before the grid cache. It includes overnight trade-date
scheduling and preserves contexts rejected by either the historical adapter or
scheduled grid's family
check in QF-11 skip manifests and result identities. Window schema version 1 is
unchanged. Scheduled grid identities also bind the window engine version, so
artifacts from the earlier engine cannot be reused through resume.

Future bars within the **same immutable provenance** cannot change earlier
contexts or decisions. A newly persisted source snapshot, family, or outcome
dataset intentionally changes identity even when its historical numerical
prefix is equal. Existing QF-11 single-decision identities and schemas are unchanged.

## QF-32 analysis, ranking, and resume

Pass `decision_schedule=schedule` to the existing `PredictionGridStudy`, a
timestamp-aware context provider, and a `PredictionWindowAnalyzer` implementing
`analyze_window(window)`. Analyzer metadata and returned `PredictionTrialAnalysis`
use the existing contracts. Single-decision `analyze(result)` callers require no
changes when the schedule is omitted.

Domain analyzers reuse their observation calculations over `window.decisions`
and the original result rows, retaining the enclosing UTC timestamp and source
result/context/row IDs. They analyze the full collection with declared baseline
and period/weekday semantics. Arbitrary per-decision metrics cannot be averaged
generically, so the grid does not invent that aggregation. Existing candidate
enumeration, parameter IDs, backend validation, eligibility constraints, ranking,
tie breaking, and neighborhood stability are reused. Candidates with a different
primary timeframe are excluded as incompatible with the fixed schedule.

The grid additionally preserves all decision timestamps, result IDs, and context
IDs under the reserved analysis-artifact key `prediction_window_sources`.
Successful artifacts use `artifacts/<trial-id>/prediction-window.json`, embedding
the full window and analyzer evidence. Single-decision artifacts retain their
original shape and `prediction-study.json` path. Context-cache keys add the UTC
decision timestamp; indicator caches retain their existing full context,
configuration, and backend keys. Native configurations remain native.
The scheduled grid retains an incompatible returned family's snapshot as skip
evidence before rejecting it from the cache. Neither indicators nor rules run
against that context; the declared `FAIL` policy still fails the candidate.
Malformed timestamp-provider returns are rejected before cache field access.
They follow the same policy as standalone windows: `SKIP` records each scheduled
decision with `source_context: null` and `FAIL` fails the candidate with a context
data error. Neither invalid objects nor their absence become cached contexts.

Persistence is **incremental per candidate**, using QF-32's atomic writes and
terminal states. A window succeeds only after every scheduled decision and the
collection analyzer finish. Resume validates the manifest, candidate definition,
artifact SHA-256, schedule, ordered decision coverage, result identity, and
analysis before skipping completed candidates. Changed or truncated artifacts
are rejected. An interrupted candidate remains pending/running and reruns its
entire window; partial results cannot be ranked. Failed candidates retain
sanitized diagnostics and obey the existing fixed `retry_failed` policy.

Load and resume recompute `window_id` from the persisted scientific identity
fields and compare it with the identity captured from the current candidate,
validated outcome dataset, and provider/backend environments. Each decision's
configuration, dataset, engine, QF-11 study ID, and exact source-context ID must
agree; row and reserved analyzer references must point to that same evidence.
Each row's QF-11 outcome, evaluation, and row IDs are recomputed from their
serialized payloads. Outcomes and evaluations must retain the configured
component provenance and correct dataset/signal/outcome references. Every row's
prediction and features must match a distinct entry in `generated_signals`;
decision counts must match the signal and row collections. An outcome must occur
exactly `required_future_sessions` positions after its signal in the validated
outcome dataset. A generated signal may lack a labeled row only when that exact
future session is beyond the dataset boundary.
Window-level `record_counts` are also recomputed from the validated decisions
through the same helper used by `PredictionWindowResult.counts_primitive()`.
All decision, signal, unavailable-outcome, and disposition totals must agree.
Every generated signal, including those without rows, must retain the candidate's
strategy identity and exact parameter payload, match the dataset symbol and
scheduled exchange session, occur in the validated outcome dataset after warm-up,
and obey QF-11 ordering and one-signal-per-session rules.
The available context's `decision_session` must match that schedule-derived
session independently, including decisions that produce no predictions.
Available rule contexts must match the outcome dataset's identity, symbol, and
complete adjustment basis. Source and rule snapshots must contain exactly the
ordered primary and contextual entries declared by the candidate, with matching
requirements, completion policies, dataset references, and selected bar IDs.
Every required timeframe must be available, retain permitted bars, and respect
its age limit. Bar timestamps cannot exceed the scheduled decision. The primary
must have a completed latest bar ending at that exact timestamp with zero age.
Developing contextual bars retain their content-derived identity and causal
observation boundaries; their expected completion boundary remains in the future.
Every rule-facing indicator manifest must match its declared alias, backend,
source fields, output schema, warm-up, completion policy, selected bars, and
dataset/feed reference. Its bound configuration identity is recomputed from that
fixed declaration and validated source metadata without indicator execution.

Returned source snapshots retained by `SKIP` still satisfy QF-20's internal
contract: canonical timeframe definitions, ordered unique requirements and exact
coverage, completion policy, common-family/source evidence, and coherent
missing/stale/available metadata. Their bar timestamps and ages are checked against
their own source `as_of`. A source can therefore retain a legitimate wrong
timestamp, family, or requirement relative to the requested decision, or missing
and stale observations, while malformed audit evidence is rejected. Failures
without a returned source still retain `source_context: null`. Regenerating
artifact or trial checksums cannot bypass these checks.

Recovered succeeded trials also pass the same analyzer invariants used during
execution before they can resume or enter ranking. Nonempty analyses must retain
period, weekday, and matched-baseline comparisons; every retained matched
comparison must name the configured baseline. Empty analyses may omit comparisons.
These shared checks apply to both historical windows and single-decision grids.

Validation does not call the context provider or rerun predictions or analysis.
Schemas and generated IDs are unchanged; compatible existing artifacts still
resume. These checks establish provenance consistency, not provider authenticity
or independent reproduction of numerical results.

There are no per-decision checkpoints or typed component deserializers in this
story. Standalone windows expose `to_primitive()` and `serialize()`; QF-32 owns
persistence/resume. Work and retained results grow with the decision count.
Session-based labels can overlap across intraday decisions; this API does not
deduplicate them or claim independent samples or validated profitability.

The deterministic SPY fixture in
`tests/unit/prediction/test_prediction_window.py` exercises four decisions per
candidate, real QF-20/QF-11 execution, complete-collection rankings, abstention,
missing context, identity safety, and interrupted resume.
