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

Run the full test suite (four worker processes by default):

```bash
uv run pytest
```

Tests are distributed by file (`--dist=loadfile`) so module-scoped fixtures stay
on one worker. The fixed four-worker default applies locally and in CI. Some
tests also create their own execution workers; reduce the pytest worker count
on machines with limited CPU or memory. All tests are collected; parallel
execution does not exclude integration or integrity coverage. Live-provider
tests keep their existing explicit opt-in requirements.

Run serially for debugging or an uncontended timing baseline:

```bash
uv run pytest -n 0
```

Use two workers on a constrained machine or to compare against the previous
parallel configuration:

```bash
uv run pytest -n 2
```

The default output includes the 30 slowest setup/call/teardown durations. Save
per-test timings and results using the same command as CI:

```bash
uv run pytest --junitxml=reports/tests/junit.xml
```

CI uploads this file as the `test-results` artifact, including test failures,
with 14-day retention. The generated report stays ignored by Git. Use the same
machine, worker count, and test selection when comparing timings; JUnit test
durations sum worker time, which differs from parallel wall time.

Selected experiment-integrity tests generate real backtest and prediction
studies once per module. Each corruption case copies the producer files into
its own temporary directory, reloads and checks their original identities, and
deep-copies mutable aggregate records. Never share writable study directories
between tests or cache the results of integrity checks across mutations.
Keep engine, execution, resume, and temporal-safety tests exercising producers.

Run a focused test:

```bash
uv run pytest tests/path/test_module.py -q
```

Run the deterministic offline Tiingo/provider and corporate-action tests:

```bash
uv run pytest tests/unit/data/test_tiingo_provider.py \
  tests/unit/backtesting/test_runner.py
```

Run QF-43 context/evaluation isolation and QF-6 identity/resume regressions:

```bash
uv run pytest tests/unit/backtesting/test_evaluation.py \
  tests/unit/optimization/test_evaluation.py
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

Build or verify the deterministic, indicator-free QF-30 SPY multi-timeframe
example entirely from the committed local fixture and immutable cache:

```bash
uv run python scripts/export_spy_multi_timeframe_context.py
```

The script exports canonical 5-minute, derived 4-hour/daily/weekly, completed-
only, and separately developing-as-of artifacts. It makes no provider request
and requires no credential. See
[`spy-multi-timeframe-example.md`](spy-multi-timeframe-example.md).

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

Run the QF-31 typed condition, accepted/rejected outcome, causality, identity,
reference-rule, and QF-7/QF-29 integration tests with:

```bash
uv run pytest tests/unit/prediction/test_technical_confluence.py
```

The fixed rule specification and exact comparison boundaries are documented in
[`technical-confluence-prediction.md`](technical-confluence-prediction.md).

Run the QF-33 scanner, historical/current parity, alert payload, sink,
deduplication, developing-bar, stale-data, backend-mismatch, and temporal-safety
tests with:

```bash
uv run pytest tests/unit/prediction/test_prediction_scanner.py
```

Run the QF-8 validation-plan identity, chronological-boundary, prediction-label
purging, embargo, warm-up, native-backend compatibility, and study-neutral
backtest fixture tests with:

```bash
uv run pytest tests/unit/validation
```

See [`validation-plans.md`](validation-plans.md) for the public contracts,
closed interval semantics, fixed research provenance, and cache-validation
rules.

Run the deterministic cache-only SPY scanner example with:

```bash
uv run python scripts/scan_spy_predictions.py
```

The example constructs no provider client, reads no credentials, and submits no
orders. It rebuilds all declared derived timeframes from the QF-30 immutable
cache fixture and writes alerts under ignored `reports/qf33-spy-alerts/`. See
[`current-data-prediction-scanner.md`](current-data-prediction-scanner.md) for
the historical-study parity guard, alert schema, and deduplication policies.

Render the matching fixed SPY decision as a standalone synchronized study
inspection artifact with no network access or running server:

```bash
uv run python scripts/render_spy_study_inspection.py
```

See [`static-study-inspection.md`](static-study-inspection.md) for exact bar,
indicator, provenance, developing-bar, future-outcome, and immutable-export
semantics.

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

Run QF-46 anchor/horizon, endpoint-resolution, metadata, identity, QF-42 dispatch,
and QF-9 integrity fixtures:

```bash
uv run pytest tests/unit/prediction/test_outcome_temporal.py \
  tests/unit/prediction/test_outcome_resolution.py \
  tests/unit/prediction/test_outcome_temporal_integration.py
```

See [`outcome-temporal-contracts.md`](outcome-temporal-contracts.md). These fixtures
calculate no concrete intraday outcomes and implement no timestamp membership.

Run QF-41 presentation-only study-family, warning, integrity, holdout-state,
security and no-recomputation fixtures (including QF-34 compatibility):

```bash
uv run pytest tests/unit/reporting
```

See [`static-research-reports.md`](static-research-reports.md) for the static
HTML API and source-of-truth rules.

Run QF-9 observational manifest/index, identity, immutable persistence,
credential, hash/tamper and permanent holdout-state fixtures:

```bash
uv run pytest tests/unit/experiments
```

See [`experiment-manifests.md`](experiment-manifests.md) for indexing existing
prediction/backtest exports and QF-8/QF-39/QF-40 validation artifacts.

Run QF-40 OOS-only aggregation, reset-account stitching, and permanent holdout
consumption/failure/retry fixtures, together with QF-39 compatibility regressions:

```bash
uv run pytest tests/unit/oos tests/unit/walk_forward
```

The offline prediction, backtest, and final-holdout examples and public contracts
are documented in [`oos-holdout-aggregation.md`](oos-holdout-aggregation.md).

Run the deterministic QF-42 historical schedule, multi-decision, provenance,
window-analysis, and resume regression tests:

```bash
uv run pytest tests/unit/prediction/test_prediction_window.py tests/unit/prediction/test_prediction_window_context_validation.py
```

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

## Timestamp prediction acceptance tests (QF-48)

For concrete QF-49 endpoint arithmetic, replay/export, resume, and QF-39/QF-40/QF-9
integration (deterministic offline fixtures):

```bash
uv run --frozen pytest tests/unit/prediction/test_intraday_forward_return.py \
  tests/unit/prediction/test_intraday_forward_return_integration.py
```

See [intraday forward returns](intraday-forward-returns.md).

```bash
uv run --frozen pytest tests/unit/validation/test_timestamp_prediction_membership.py \
  tests/unit/walk_forward/test_timestamp_membership.py \
  tests/unit/prediction/test_timestamp_replay.py
```

These fixtures execute metadata-only elapsed labels through the real QF-8/QF-39
and QF-40 paths, preserve replay anchors, and reject incompatible resume artifacts.
See [timestamp prediction validation](timestamp-prediction-validation.md).

## Walk-forward acceptance tests (QF-39)

Run the deterministic two-fold prediction/backtest fixtures, exact QF-8
membership/leakage checks, frozen-selection and interruption/resume cases:

```bash
uv run pytest tests/unit/walk_forward
```

These fixtures use existing QF-42/QF-43 engines and local synthetic artifacts.
They require no provider credentials. See
[`walk-forward-studies.md`](walk-forward-studies.md).

## Intraday path acceptance tests (QF-47)

Use required uv 0.12.1 and the frozen dependency workflow.

```bash
uv run --frozen pytest tests/unit/prediction/test_intraday_path.py \
  tests/unit/prediction/test_intraday_path_integration.py \
  tests/unit/prediction/test_feature_outcomes.py \
  tests/unit/prediction/test_intraday_forward_return.py \
  tests/unit/prediction/test_intraday_forward_return_integration.py
```

See [intraday path outcomes](intraday-path-outcomes.md) for exact intervals,
reference price, ratio conventions, ambiguity, completeness, and provenance.
The fixtures cover real QF-7/QF-29/QF-39 exports, QF-48 timestamp validation,
QF-40 aggregation without path recomputation, safe resume, and QF-9 inspection.

## Intraday prediction provenance acceptance tests (QF-51)

```bash
uv run --frozen pytest tests/integration/test_intraday_prediction_provenance.py \
  tests/integration/test_intraday_prediction_manifest_integrity.py \
  tests/integration/test_intraday_prediction_feed_integrity.py
```

The synthetic cache-only fixture exercises canonical SPY one-minute input,
derived two-minute/daily bars, truthful unavailable events, context construction,
QF-49/QF-47 outcomes, QF-9 integrity, and compatible/incompatible resume. The reader
regressions reject rehashed source-lineage, feed-scope, outcome-timeframe, and
session-policy mismatches in prediction/feature artifacts without executing
research. Projection tests also reject source metadata, bars, and raw extracts that differ from the
immutable intraday cache. See
[intraday prediction provenance](intraday-prediction-provenance.md).
