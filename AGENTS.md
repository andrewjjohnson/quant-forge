# QuantForge Agent Instructions

This file is the authoritative set of repository instructions for AI coding agents and human contributors. More specific `AGENTS.md` files may be added in subdirectories later; when present, the closest applicable file takes precedence.

## Project purpose

QuantForge is a quantitative research and systematic-trading platform for:

- ingesting and validating market data;
- defining reusable indicators and strategies;
- running reproducible backtests;
- exploring strategy parameters programmatically;
- analyzing signal-level features;
- performing walk-forward and holdout validation;
- supporting paper trading and, later, live execution.

Correctness, reproducibility, and research integrity are more important than implementation speed.

## Current scope

The first milestone is Jira epic **QF-1: MVP Quantitative Research Foundation**.

Unless the active Jira ticket says otherwise, do not implement:

- live brokerage execution;
- production paper-trading infrastructure;
- options-chain simulation;
- machine-learning trade decisions;
- a user-facing dashboard.

## Before changing code

1. Read the active Jira ticket and its acceptance criteria.
2. Read this file and the relevant documentation under `docs/`.
3. Inspect the existing implementation and tests before proposing an architecture.
4. Identify time-series, data-quality, and reproducibility risks.
5. Keep work limited to one Jira ticket unless explicitly instructed otherwise.

Do not silently broaden scope.

## Repository architecture

Keep these responsibilities separate:

1. **Data acquisition** — retrieves provider data.
2. **Data normalization and validation** — converts provider data to canonical schemas.
3. **Indicators and features** — computes values using information available at that timestamp.
4. **Strategy logic** — produces signals; it does not download data or simulate fills.
5. **Execution simulation** — converts eligible signals into orders and fills.
6. **Portfolio accounting** — owns cash, positions, realized/unrealized P&L, and equity.
7. **Metrics and validation** — evaluates results without changing them.
8. **Optimization** — selects parameter sets but does not alter backtest semantics.
9. **Reporting** — renders immutable results and manifests.
10. **Broker adapters** — reserved for later paper/live execution.

Avoid circular dependencies. Domain logic must not depend on CLI, notebooks, dashboards, or report-rendering code.

See `docs/architecture.md`.

## Python standards

- Use the Python version declared by the repository configuration.
- Add type annotations to public functions, methods, classes, and data structures.
- Prefer small, explicit modules over large general-purpose files.
- Prefer immutable dataclasses or validated models for configuration and result records.
- Use descriptive domain names such as `signal_timestamp`, `execution_timestamp`, and `adjustment_type`.
- Do not use ambiguous names such as `date`, `value`, or `data` where a precise name is available.
- Avoid hidden global state.
- Do not mutate caller-owned DataFrames or arrays unless the API explicitly documents mutation.
- Prefer pure functions for indicators, labels, metrics, and transformations.
- Make randomness explicit and seedable.
- Do not catch broad exceptions unless re-raising with useful context.
- Never log credentials, API keys, account identifiers, or broker secrets.
- Do not introduce a dependency when a clear standard-library implementation is sufficient.
- Document non-obvious numerical assumptions and units.

## Time-series correctness

These rules are non-negotiable:

- Never use future information to compute a feature, signal, filter, order, or fill.
- State whether timestamps represent bar open, bar close, exchange time, or UTC.
- A signal derived from a completed bar may not fill before the next permitted execution point.
- Do not use centered rolling windows.
- Do not backfill indicator values from the future.
- Do not silently forward-fill market prices or corporate-action data.
- Sort time-series data explicitly and reject duplicate keys unless deterministic reconciliation is documented.
- Make warm-up periods explicit.
- Keep contemporaneous features separate from forward-looking outcome labels.
- Tests must fail if rows are shifted in a way that introduces leakage.

Read and follow `docs/research-integrity.md`.

## Market-data rules

- Preserve raw provider responses or immutable raw extracts when practical.
- Normalize into a documented canonical schema.
- Record provider, symbol, retrieval timestamp, timezone, adjustment policy, and date range.
- Fingerprint datasets used by experiments.
- Distinguish missing market sessions from missing observations.
- Validate OHLC relationships, duplicate bars, ordering, non-finite values, and impossible values.
- Never combine adjusted and unadjusted prices without an explicit conversion policy.
- Never assume the latest constituent list represents a historical index universe.

## Backtesting rules

Every reportable backtest must explicitly define:

- initial capital;
- signal timing;
- order timing;
- fill-price rule;
- commissions and fees;
- slippage model;
- position-sizing rule;
- handling of rejected orders;
- data-adjustment policy;
- benchmark;
- date range and universe;
- random seed, when applicable.

A zero-cost or perfect-fill run may be used for debugging, but it must not be presented as realistic performance.

Every trade must be traceable to:

- strategy version;
- parameter set;
- signal timestamp;
- order and fill records;
- data fingerprint;
- run identifier.

## Optimization and statistical rules

- Do not optimize solely for maximum return.
- Enforce minimum sample-size and risk constraints where the ticket requires them.
- Preserve failed and pruned trials with status and error context.
- Persist results incrementally for resumability.
- Prefer broad stable parameter regions over isolated optima.
- Keep development, validation, walk-forward, and final holdout periods distinct.
- Never use test or holdout outcomes to tune parameters.
- Treat relationships found through feature analysis as hypotheses until they pass untouched out-of-sample validation.
- Clearly label in-sample-only results.
- Account for multiple comparisons when testing many indicators, ranges, symbols, or horizons.

## Testing expectations

Every behavior change requires tests appropriate to its risk.

Use:

- unit tests for pure calculations and validation rules;
- property-based tests for invariants and edge cases;
- fixture or golden tests for orders, fills, trades, and metrics;
- integration tests for component boundaries;
- regression tests for corrected defects;
- performance tests for code expected to scale across large parameter grids.

High-priority invariants include:

- no future-data access;
- deterministic repeated runs;
- conserved cash and position accounting;
- correct fee/slippage application;
- no overlapping train/test windows;
- expected behavior at missing-data and warm-up boundaries;
- stable serialization schemas.

Tests must not depend on live external APIs unless explicitly marked as optional integration tests.

## Commands

Use the commands documented in `docs/development.md` and the repository configuration. The intended initial toolchain is:

```bash
uv sync --all-extras
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest
```

If the repository implements wrapper commands such as `make check`, prefer those.

Before declaring work complete:

1. format the changed code;
2. run linting;
3. run type checking;
4. run relevant tests;
5. run the full test suite when practical;
6. update documentation affected by the change.

Never claim a command passed unless it was actually executed successfully.

## Git and Jira workflow

- Every implementation must correspond to a Jira issue using the `QF-###` format.
- Start from the current default branch.
- Use one feature branch per Jira issue.
- Preferred branch format:

  `qf/<jira-key>-<short-kebab-case-description>`

  Example: `qf/QF-3-market-data-ingestion`

- Do not commit directly to the default branch.
- Do not combine unrelated Jira issues in one branch or pull request.
- Do not rewrite shared branch history.
- Leave the working tree clean.
- Do not merge pull requests unless explicitly instructed by the repository owner.

## Commit messages

Use Conventional Commits and include the Jira key:

```text
<type>(<scope>): <description> [QF-###]
```

Examples:

```text
chore(repo): initialize Python tooling [QF-2]
feat(data): add adjusted OHLCV provider [QF-3]
test(backtest): verify next-bar execution [QF-5]
```

Preferred types:

- `feat`
- `fix`
- `test`
- `refactor`
- `docs`
- `chore`
- `ci`
- `perf`

Commit cohesive units of work. Do not hide unrelated changes in a single commit.

## Pull requests

PR titles must use:

```text
QF-###: Clear imperative summary
```

Open pull requests as drafts unless explicitly instructed otherwise.

Every PR description must include:

- Jira ticket;
- summary;
- implementation notes;
- tests executed and results;
- research-integrity considerations;
- schema or compatibility impact;
- risks and limitations;
- follow-up work.

Complete `.github/pull_request_template.md`.

## Documentation and decisions

Update documentation whenever a change affects:

- architecture;
- developer commands;
- public interfaces;
- data schemas;
- timestamp semantics;
- fill assumptions;
- numerical methods;
- persistence formats;
- reproducibility.

Record durable or difficult-to-reverse architectural choices as ADRs under `docs/decisions/`.

## Definition of done

Work is complete only when:

- Jira acceptance criteria are met;
- implementation is scoped to the ticket;
- tests cover the behavior and important failure modes;
- formatting, linting, type checking, and tests pass;
- documentation is updated;
- no secrets or large generated datasets are committed;
- the PR explains assumptions and research-integrity risks;
- the branch is ready for human review.

## Agent conduct

- Be explicit about assumptions.
- Ask for clarification only when a decision is genuinely blocking and cannot be inferred safely.
- Prefer a small correct implementation over speculative abstraction.
- Do not delete or weaken tests to make checks pass.
- Do not change statistical, execution, or accounting assumptions silently.
- Do not present experimental profitability as financial advice or as evidence of future returns.
