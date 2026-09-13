# ADR 0015: Compose historical prediction windows from single decisions

- Status: Accepted
- Date: 2026-09-13
- Jira: [QF-42](https://frostfiredigital-37308542.atlassian.net/browse/QF-42)

## Context

QF-28 resolves one multi-timeframe decision and QF-32 executes one QF-11 study
per candidate. Historical analysis needs an explicit sequence retaining every
source identity. QF-39 will consume this sequence; fold selection is separate.

## Decision

Add a calendar-derived `PredictionDecisionSchedule` and timestamp-aware context
provider. Reuse QF-18 session windows to schedule completed primary bar ends in
a closed UTC interval. Missing observations follow the existing QF-28 policy.

Execute independent copies of one configured QF-11 study through its unchanged
runner. Collect original results in `PredictionWindowResult` with immutable
decision snapshots and explicit timestamps. Preserve unlabeled signals alongside
the unchanged QF-11 serialization.

Extend QF-32 with an optional schedule and `analyze_window()` contract. Reuse
candidate search, backend validation, ranking, stability, caches, and atomic
candidate persistence. Completed windows resume; interrupted windows rerun in
full. Do not fabricate a QF-11 identity or average arbitrary decision metrics.

## Consequences

Single-decision public contracts and identities remain compatible. Window
identity includes scheduling and complete scientific provenance. Component
copies isolate future-bearing callback state. Work and memory grow with the
decision count; per-decision persistence is deferred. Domain analyzers retain
responsibility for overlapping labels and matched baselines. Rolling/expanding
folds, selection, OOS aggregation, holdout consumption, and backtest boundaries
remain separate work.
