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

Run the deterministic offline Tiingo/provider and corporate-action tests:

```bash
uv run pytest tests/unit/data/test_tiingo_provider.py \
  tests/unit/backtesting/test_runner.py
```

Run the deterministic offline Tiingo intraday adapter and cache tests:

```bash
uv run pytest tests/unit/data/test_tiingo_intraday.py
```

Run the focused provider-neutral intraday coverage validation tests:

```bash
uv run pytest tests/unit/data/test_intraday_validation.py
```

Run deterministic session-aware intraday aggregation and derived-cache tests:

```bash
uv run pytest tests/unit/data/test_intraday_aggregation.py
```

Run deterministic exchange-session daily/weekly aggregation and immutable-cache
tests:

```bash
uv run pytest tests/unit/data/test_session_aggregation.py
```

Run completed-bar multi-timeframe alignment, causality, family, holiday, and
early-close tests:

```bash
uv run pytest tests/unit/data/test_multi_timeframe.py
```

Run the timeframe-neutral indicator compatibility, identity, causality, and
developing-bar tests:

```bash
uv run pytest tests/unit/indicators
```

Run the focused backend-neutral TA-Lib MACD and stochastic contract tests:

```bash
uv run pytest tests/unit/indicators/test_macd.py \
  tests/unit/indicators/test_stochastic.py \
  tests/unit/indicators/test_timeframe_evaluation.py
```

Run the opt-in live Tiingo integration only with both the key and explicit flag:

```bash
TIINGO_API_KEY=... QUANTFORGE_RUN_LIVE_TIINGO=1 \
  uv run pytest -m integration tests/integration/test_tiingo_market_data.py
```

Run the fixed opt-in Tiingo SPY intraday verification. Consolidated data is the
default; set `QUANTFORGE_TIINGO_INTRADAY_FEED=iex` to request the explicitly
IEX-only path:

```bash
TIINGO_API_KEY=... QUANTFORGE_RUN_LIVE_TIINGO_INTRADAY=1 \
  uv run pytest -m integration \
  tests/integration/test_tiingo_intraday_market_data.py
```

Run the fixed 2020-2025 real SPY example, or its explicitly synthetic offline
mode. The maintained example explicitly selects price-return-only dividend
treatment and prints the corresponding exclusions:

```bash
TIINGO_API_KEY=... uv run python scripts/run_spy_backtest.py
uv run python scripts/run_spy_backtest.py --fixture
```

Run the provider-neutral QF-11 overnight-gap analysis against the fixed
2020-2025 Tiingo SPY request, or reproduce it from an existing cache entry:

```bash
TIINGO_API_KEY=... uv run python scripts/run_spy_gap_prediction.py
uv run python scripts/run_spy_gap_prediction.py --dataset-id <dataset-id>
```

The command prints direction accuracy and average gap sizes; it creates no
orders or fills. Use `--refresh` only to intentionally retrieve a new immutable
Tiingo snapshot. A cached ID must come from the exact raw, unadjusted Tiingo SPY
request for 2020-01-01 through 2025-12-31, use XNYS, include the expected first
and last sessions, and contain no missing sessions. Use the provider-neutral
prediction API for intentionally different datasets.

Run the QF-11 exploratory comparison study on the same request or a cached
dataset. This preserves the original strategy and evaluates the focused,
RSI-only, and always-UP configurations separately:

```bash
TIINGO_API_KEY=... uv run python scripts/analyze_spy_gap_predictions.py
uv run python scripts/analyze_spy_gap_predictions.py --dataset-id <dataset-id>
```

The cached ID accepted by this maintained script must come from the exact raw,
unadjusted Tiingo SPY request for 2020-01-01 through 2025-12-31 with complete
XNYS session coverage. Use the public comparison API for intentionally different
providers, price bases, symbols, calendars, requested ranges, or incomplete
samples.

The comparison creates no orders or fills. Its 2020–2025 periods have already
been inspected and are not untouched holdout results.

Build/resume the QF-7 signal-feature dataset from an existing immutable QF-3
cache entry and run the documented three-feature exploratory comparison:

```bash
uv run python scripts/analyze_signal_features.py --dataset-id <dataset-id>
```

This command performs no provider retrieval. It writes the analytics dataset to
ignored `reports/features/` and the deterministic descriptive analysis to
ignored `reports/feature-analysis/`. See
[`signal-feature-datasets.md`](signal-feature-datasets.md) for schemas, formulas,
resume behavior, and research limitations.

Run the focused QF-29 causal capture, provenance, backend-identity, Parquet, and
resume tests with:

```bash
uv run pytest tests/unit/prediction/test_multi_timeframe_feature_dataset.py
```

Downloaded provider responses remain under ignored `data/`; structured results
remain under ignored `reports/`. Use `--refresh` only when intentionally
retrieving a new immutable provider revision. Never stage either directory.

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

## Adding an optimization feature

Use the QF-6 contracts and deterministic policies in
[`optimization.md`](optimization.md). Optimization code coordinates generic
QF-4 factories and QF-5 results; it must not branch on strategy names or
reimplement metrics, execution, costs, accounting, or market-data validation.

Changes to search, ranking, stability, persistence, or parallel execution
require:

- stable primitive configuration and schema/version review;
- identity-change tests for scientific inputs;
- exact combination-order and exclusion tests;
- sequential/process equivalence when execution is affected;
- resume call-count tests when state transitions are affected;
- synthetic objective surfaces when stability rules are affected; and
- explicit in-sample, overfitting, and multiple-comparison limitations.

Keep normal tests network-free. A documented SPY example must consume an
already-loaded/cache-validated QF-3 dataset and must not retrieve provider data
inside any trial.

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

For prediction studies, use the generic `PredictionStudy` composition and
contracts in [`prediction-analysis.md`](prediction-analysis.md). Implement a
causal prediction rule, a typed outcome labeler with an explicit future-session
horizon and required market fields, and a typed evaluator. Keep each component's
configuration and result schema versioned, and test that they all participate in
study identity. Include a non-gap contract fixture and append-future tests
proving historical predictions do not change.

For the concrete next-session gap study, require the immediate calendar
successor and keep `run_prediction_analysis` and its legacy gap export schema
backward compatible. Add new hypotheses as prediction strategies or new study
compositions; do not change the original QF-11 baseline logic.

For QF-7 datasets, use `SignalFeatureCandidate` as the QF-11 prediction record,
register causal `ContextualFeature` implementations, and wrap each typed QF-11
labeler/evaluator pair with `PredictionStudyOutcome`. Do not call outcome code
from a candidate rule. The builder supplies context features only a prefix
through the signal session, checkpoints complete rows atomically, and validates
all persisted scientific configuration on resume. Add append-future, explicit
unavailable, deterministic identity, and interrupted/resumed equivalence tests.

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
