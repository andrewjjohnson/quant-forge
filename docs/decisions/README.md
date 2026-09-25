# Architecture Decision Records

Use Architecture Decision Records (ADRs) for durable decisions that are expensive, risky, or confusing to reverse.

## When to create an ADR

Create an ADR for decisions such as:

- choosing the authoritative backtesting engine;
- defining timestamp and execution conventions;
- selecting canonical market-data schemas;
- selecting a persistence or experiment-tracking backend;
- choosing adjusted-price and corporate-action policies;
- defining the paper/live broker abstraction;
- adopting an options-data model;
- choosing deterministic parallel-execution behavior.

Do not create ADRs for routine implementation details that are easy to change.

## Accepted records

- [ADR 0001: Use chronological next-open backtesting](0001-use-chronological-next-open-backtesting.md)
- [ADR 0002: Use parent-owned deterministic grid execution](0002-use-parent-owned-deterministic-grid-execution.md)
- [ADR 0003: Use raw prices with explicit corporate actions](0003-use-raw-prices-with-explicit-corporate-actions.md)
- [ADR 0004: Use exchange-session timeframe semantics](0004-use-exchange-session-timeframe-semantics.md)
- [ADR 0005: Require common-source dataset families](0005-require-common-source-dataset-families.md)
- [ADR 0006: Use provider-neutral intraday request and bar contracts](0006-use-provider-neutral-intraday-contracts.md)
- [ADR 0007: Cache intraday source snapshots separately](0007-cache-intraday-source-snapshots-separately.md)
- [ADR 0008: Bind intraday quality reports to datasets](0008-bind-intraday-quality-reports-to-datasets.md)
- [ADR 0009: Derive intraday bars from verified session windows](0009-derive-intraday-bars-from-verified-session-windows.md)
- [ADR 0010: Derive daily and weekly bars from exchange sessions](0010-derive-daily-weekly-bars-from-exchange-sessions.md)
- [ADR 0011: Align only completed multi-timeframe bars by default](0011-align-only-completed-multi-timeframe-bars.md)
- [ADR 0012: Reconstruct developing bars only from causal source intervals](0012-reconstruct-developing-bars-from-causal-source-intervals.md)
- [ADR 0013: Keep standard indicator definitions backend-neutral](0013-keep-standard-indicators-backend-neutral.md)
- [ADR 0014: Use fixed validation plans with boundary purging](0014-use-fixed-validation-plans-with-boundary-purging.md)
- [ADR 0015: Compose historical prediction windows from single decisions](0015-compose-historical-prediction-windows.md)
- [ADR 0016: Isolate backtest evaluation accounts from historical context](0016-isolate-backtest-evaluation-accounts.md)

- [ADR 0017: Compose walk-forward studies](0017-compose-walk-forward-studies.md)
- [ADR 0018: Persist holdout exposure before evaluation](0018-persist-holdout-exposure-before-evaluation.md)
- [ADR 0019: Index existing research artifacts](0019-index-existing-research-artifacts.md)
- [ADR 0021: Define outcome time contracts before timestamp membership](0021-define-outcome-time-contracts-before-timestamp-membership.md)

- [ADR 0022: Bind prediction membership to exact decision schedules](0022-bind-prediction-membership-to-exact-decision-schedules.md)

## File naming

```text
NNNN-short-kebab-case-title.md
```

Example:

```text
0001-use-next-bar-execution-for-close-derived-signals.md
```

## Suggested template

```md
# ADR NNNN: Decision title

- Status: Proposed | Accepted | Superseded | Deprecated
- Date: YYYY-MM-DD
- Jira: QF-###

## Context

What problem or constraint requires a decision?

## Decision

What was decided?

## Consequences

What benefits, costs, risks, and follow-up work result?

## Alternatives considered

What reasonable alternatives were evaluated, and why were they not selected?

## Validation

How will the decision be tested or revisited?
```

## Rules

- Link the relevant Jira issue and pull request.
- Describe statistical and research-integrity consequences.
- Do not rewrite accepted ADR history to make old decisions appear different.
- Supersede an old ADR with a new one when the decision changes.

- [0023: Preserve intraday prediction-input provenance](0023-preserve-intraday-prediction-input-provenance.md)
- [0024: Separate causal prediction views from canonical source evidence](0024-separate-bounded-prediction-views.md)

- [0025: Plan Tiingo history by exchange session](0025-plan-tiingo-history-by-exchange-session.md)
- [0026: Retain Massive pages in logical acquisition snapshots](0026-retain-massive-pages-in-logical-snapshots.md)
- [0027: Normalize historical-window evidence](0027-normalize-historical-window-evidence.md)

- [0028: Checkpoint compact prediction journals](0028-checkpoint-compact-prediction-journals.md)

- [0029: Consume compact window evidence downstream](0029-consume-compact-window-evidence.md)
- [0030: Share immutable prediction source backing](0030-share-immutable-prediction-source.md)
