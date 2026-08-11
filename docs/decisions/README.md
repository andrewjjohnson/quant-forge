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
