# QuantForge Development Guide

This document defines the intended local workflow. Update commands when repository tooling changes.

## Prerequisites

- Git
- The Python version declared in `pyproject.toml`
- `uv`
- Access to any configured market-data provider for optional integration tests

Do not store provider or broker credentials in the repository.

## Initial setup

```bash
git clone <repository-url>
cd QuantForge
uv sync --all-extras
```

Copy the example environment file when present:

```bash
cp .env.example .env
```

Use placeholder or paper credentials only during development.

## Common commands

Format:

```bash
uv run ruff format .
```

Check formatting:

```bash
uv run ruff format --check .
```

Lint:

```bash
uv run ruff check .
```

Apply safe automatic lint fixes:

```bash
uv run ruff check --fix .
```

Type check:

```bash
uv run pyright
```

Run tests:

```bash
uv run pytest
```

Run a focused test:

```bash
uv run pytest tests/path/test_module.py -q
```

Run tests with coverage when configured:

```bash
uv run pytest --cov=quantforge --cov-report=term-missing
```

If wrapper commands such as `make check` or `just check` are added, use those as the stable contributor interface.

## Branch workflow

Start from the current default branch:

```bash
git switch main
git pull --ff-only
git switch -c qf/QF-###-short-description
```

Use one Jira issue per branch.

Commit format:

```text
<type>(<scope>): <description> [QF-###]
```

Example:

```bash
git commit -m "feat(data): add adjusted OHLCV validation [QF-3]"
```

Push and open a draft PR:

```bash
git push -u origin HEAD
```

The PR title should be:

```text
QF-###: Clear imperative summary
```

## Adding a module

Before adding a module:

1. identify its architectural responsibility;
2. verify that an existing module does not already own that responsibility;
3. define public interfaces and domain terminology;
4. avoid introducing dependencies from domain code to infrastructure;
5. add tests near the appropriate test layer;
6. update architecture documentation when boundaries change.

## Adding an indicator

Use the public QF-4 contracts and examples in
[`strategy-contracts.md`](strategy-contracts.md).

An indicator should:

- accept canonical aligned inputs;
- document required columns, units, lookback, and warm-up;
- use only current and historical observations;
- return output aligned to the input index;
- not mutate its inputs;
- handle insufficient history deterministically;
- include tests against hand-calculated or trusted fixture values;
- include a test that protects against accidental forward shifts.

## Adding a strategy

Use the public QF-4 contracts, decision schema, and examples in
[`strategy-contracts.md`](strategy-contracts.md).

A strategy should:

- declare its parameters;
- declare required market fields and indicators;
- define the exact timestamp at which a signal is known;
- emit signals without assuming execution;
- document long/short and position constraints;
- avoid direct data-provider, storage, or broker dependencies;
- include tests for warm-up, entry, exit, and no-signal cases.

## Adding a backtest feature

Use the public contracts, numerical policies, and execution sequence in
[`backtesting.md`](backtesting.md). QF-5's chronological next-open behavior is
recorded in ADR 0001; changes to that behavior require explicit compatibility
and research-integrity review.

Changes to execution or accounting require:

- explicit semantics in documentation;
- deterministic fixture tests;
- tests for fees and slippage;
- tests for cash and position invariants;
- checks at timestamps and session boundaries;
- comparison with existing behavior to identify breaking changes.

Do not alter fill rules solely to improve historical results.

## Adding feature analysis

Keep contemporaneous features separate from forward-looking labels.

Every signal-feature row should identify:

- symbol;
- signal timestamp;
- strategy version;
- parameter set;
- data fingerprint;
- feature-schema version.

Outcome labels may include:

- forward returns by horizon;
- maximum favorable excursion;
- maximum adverse excursion;
- target-before-stop outcome;
- bars to event.

Outcome-label code must never be imported into live signal-generation paths.

## Test organization

Suggested layers:

```text
tests/
  unit/
  property/
  integration/
  regression/
  performance/
```

Use deterministic fixtures. Avoid network access in ordinary unit tests.

Mark optional external tests clearly, for example:

```bash
uv run pytest -m integration
```

## Data during development

- Do not commit proprietary or licensed market data.
- Do not commit large generated datasets or reports unless intentionally retained as small fixtures.
- Keep raw data immutable.
- Use tiny synthetic or redistributable fixtures for tests.
- Record timezone, adjustment policy, symbol, and units in fixtures.
- Avoid fixtures copied from production accounts.

## Secrets

Use environment variables or an approved secret manager.

Never commit:

- market-data API keys;
- broker credentials;
- account IDs;
- webhook secrets;
- private certificates;
- `.env` files containing secrets.

Provide `.env.example` with names and safe placeholders only.

## Pull-request readiness

Before opening or updating a PR:

```bash
uv sync --all-extras --frozen
uv run ruff format --check .
uv run ruff check .
uv run pyright
uv run pytest
uv run pre-commit run --all-files
```

Then complete the PR template honestly. Document unavailable checks instead of claiming success.

## Debugging research discrepancies

When two runs differ unexpectedly, compare:

1. code commit and dirty state;
2. dependency lock;
3. input dataset fingerprint;
4. timezone and calendar;
5. adjustment policy;
6. strategy parameters;
7. cost and fill model;
8. random seeds;
9. numerical precision;
10. parallel scheduling or reduction order.

Do not “fix” discrepancies by rounding or dropping rows without identifying the cause.
