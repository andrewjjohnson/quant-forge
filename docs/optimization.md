# Deterministic parameter-grid optimization

QF-6 implements in-sample Cartesian parameter studies through
`quantforge.optimization`. It coordinates existing components without changing
their semantics:

```text
QF-3 immutable MarketDataset
        |
        v
typed QF-6 search space and pre-execution constraints
        |
        v
generic QF-4 StrategyFactory -> real Strategy parameter contract
        |
        v
unchanged QF-5 run_backtest
        |
        v
incremental trial records -> eligibility/ranking -> stability -> export
```

QF-6 never downloads data during a trial, generates strategy signals, simulates
fills, owns portfolio accounting, or recalculates performance metrics. A study
must receive an already-loaded immutable QF-3 dataset. Every successful trial
retains QF-5's run ID, complete performance schema, strategy configuration ID,
validated-bar fingerprint, costs, and immutable artifact location.

## Public API and SPY moving-average example

The following modest study assumes `dataset` is an already-loaded or cached,
schema-version-4, unadjusted SPY daily `MarketDataset` with complete corporate
actions and raw OHLCV compatible with QF-5. It
does not retrieve provider data.

```python
from decimal import Decimal
from pathlib import Path

from quantforge.backtesting import (
    BacktestConfig,
    BasisPointSlippage,
    DividendPolicy,
    ExplicitZeroFees,
    PerShareCommission,
)
from quantforge.optimization import (
    ExecutionConfig,
    FilePersistenceConfig,
    GridSearchConfig,
    GridSearchStudy,
    IntegerValues,
    MaximumDrawdown,
    MetricName,
    MinimumTrades,
    MovingAverageCrossoverFactory,
    ParameterLessThan,
    ParameterSearchSpace,
    PositiveReturn,
    RankingConfig,
    StabilityConfig,
)

study = GridSearchStudy(
    dataset=dataset,
    strategy_factory=MovingAverageCrossoverFactory(),
    config=GridSearchConfig(
        label="SPY daily moving-average grid",
        search_space=ParameterSearchSpace(
            {
                "fast_window": IntegerValues([5, 10, 15, 20]),
                "slow_window": IntegerValues([30, 50, 100, 200]),
            }
        ),
        parameter_constraints=(ParameterLessThan("fast_window", "slow_window"),),
        backtest=BacktestConfig(
            initial_capital=Decimal("100000"),
            commission=PerShareCommission(
                amount_per_share=Decimal("0.005"),
                minimum=Decimal("1"),
            ),
            fees=ExplicitZeroFees(),
            slippage=BasisPointSlippage(Decimal("5")),
            dividend_policy=DividendPolicy.PRICE_RETURN_ONLY,
            annual_risk_free_rate=Decimal("0.03"),
        ),
        execution=ExecutionConfig(),
        ranking=RankingConfig(
            objective=MetricName.SHARPE_RATIO,
            hard_constraints=(
                MinimumTrades(5),
                MaximumDrawdown(Decimal("0.30")),
                PositiveReturn(),
            ),
        ),
        stability=StabilityConfig(),
        persistence=FilePersistenceConfig(Path("reports/optimization")),
    ),
)

result = study.run()
# After an interruption, construct the identical study and call:
resumed = study.resume()
# Rebuild rankings and exports without invoking QF-5:
loaded = study.load_result()
study.export(loaded)
```

`MovingAverageCrossoverFactory` is a generic `StrategyFactory` implementation,
not a branch in the trial runner. Other strategies extend optimization by
supplying a serializable factory with strategy name/version, parameter contract
order, required parameter names, primitive configuration, and `build()`. The
factory must construct the real QF-4 parameter model and return a QF-4
`Strategy`. The optimizer verifies the produced configuration identity before
the combination becomes executable. Factory configuration records every
behavioral default used when an optional parameter is absent; the moving-average
factory identity includes its default source field and target-long weight. A
default change therefore changes combination, study, and trial identities even
when that parameter is not searched.

## Search spaces and canonical order

`ParameterSearchSpace` accepts typed finite value models:

- `IntegerValues` accepts explicit true integers and provides an inclusive
  positive-step range helper. Booleans are rejected.
- `FloatValues` accepts explicit numeric values and provides an inclusive range
  helper. Values are normalized through exact `Decimal` text. The range helper
  adds `Decimal` steps under a fixed 34-digit context, so a `0.1, 0.2, 0.3`
  grid does not accumulate binary artifacts such as
  `0.30000000000000004`. Use decimal strings or `Decimal` values when the input
  itself must be exact.
- `CategoricalValues` accepts stable string/integer primitives and `StrEnum`
  values serialized through their stable value. It rejects booleans and
  arbitrary objects.
- `BooleanValues` accepts only actual `True` and `False` values; integers `0`
  and `1` are not substituted.

Candidate collections must be nonempty. Duplicate raw integer, categorical, or
boolean values and duplicate normalized float values are rejected. This policy
prevents silent deduplication from hiding mistakes.

Parameter order comes from the strategy factory's declared QF-4 parameter
contract order. A search over only a subset retains the relative contract
order. Input dictionary construction order therefore cannot change Cartesian
order, combination IDs, trial IDs, or exports. The rightmost searched parameter
changes fastest. `iter_combination_candidates()` exposes lazy generation, while
`combination_count()` and `count_expression()` expose grid size before any
strategy or backtest runs.

## Parameter constraints and exclusions

Constraints execute before QF-5. `ParameterComparison` requires an explicit
`ComparisonOperator` member and supports comparisons against another searched
parameter or a constant; raw or unsupported operator strings fail during
construction. Convenience constraints include `ParameterLessThan` and
`ParameterAtMost`. A named, versioned
`CustomParameterConstraint` is available for uncommon deterministic predicates;
anonymous lambdas, nested local functions, and entry-point or notebook
`__main__` functions are rejected because they have no durable import identity.

Unknown parameter names and malformed constraints fail study construction.
Every assignment that passes declarative constraints is then passed through the
factory's actual QF-4 parameter constructor. Parameter-model failures become
structured `EXCLUDED` records with exact searched parameters and a concise
reason. Factory identity, version, configuration, and unexpected implementation
failures abort study construction rather than being mislabeled as parameter
exclusions. Exclusions are not failed backtests and never invoke QF-5.

## Deterministic identities

Canonical JSON with sorted object keys and no nonfinite numbers is SHA-256
hashed for every identity.

- A combination ID covers the factory identity and normalized searched
  parameters. It excludes status and timestamps.
- A study ID covers the stable label, complete QF-3 dataset identity and
  metadata, strategy/factory identity and version, search candidates, parameter
  constraints, QF-5 backtest configuration, ranking and stability
  configurations, and optimization/QF-5 schema and engine versions.
- A trial ID covers the study and combination IDs, QF-3 dataset ID, complete
  strategy parameters and configuration identity, QF-5 configuration, and
  relevant engine/schema versions.

The factory's declared strategy name and implementation version are hashed as
explicit identity fields, independently of its serialized factory
configuration. A custom factory therefore cannot alias another strategy by
omitting those fields or providing stale values in `configuration()`.

Study construction runs the authoritative QF-3 dataset validator before
calculating an identity or accessing persisted trials. Resume and result loading
therefore cannot trust old successes when the supplied bars no longer reproduce
their dataset metadata and content identity. Before exporting a successful QF-5
artifact, the parent also verifies the returned QF-3 provenance, independently
fingerprinted bars, and complete QF-5 backtest configuration against the study
inputs, then verifies the executed strategy name, implementation version,
configuration ID, and complete parameter mapping against the candidate encoded
by that trial ID. A mismatch becomes a failed trial and cannot occupy the
candidate's artifact path.

Execution mode, worker count, persistence path, retry policy, and diagnostic
timestamps are operational rather than scientific inputs and do not change the
study/trial identity. They remain explicit in the manifest, and an existing
manifest can resume only under the exact same operational configuration. This
allows sequential and process execution in separate stores to prove equivalent
scientific outputs without treating scheduling as a different experiment.

Changing dataset content/identity, strategy version, parameter candidates,
constraints, initial capital, costs, objective, hard constraints, or stability
thresholds changes the study ID. A changed study cannot silently resume an old
directory.

## Execution and failure policy

`ExecutionMode.SEQUENTIAL` is the reference mode. It visits combinations in
canonical order and persists `RUNNING`, then `SUCCEEDED` or `FAILED`, for every
attempt. One trial-domain failure does not stop later combinations unless
`fail_fast=True`.

`ExecutionMode.PROCESS` uses a bounded standard-library `ProcessPoolExecutor`.
The immutable dataset, generic factory, and QF-5 configuration are initialized
once per worker rather than submitted with every task. At most
`maximum_workers` trials are outstanding. The parent process alone owns trial
persistence and QF-5 artifact export. It handles completed futures in canonical
combination order across the entire completed batch before deciding whether to
submit replacement work. This preserves `fail_fast` when successes and failures
complete together. Final results are always sorted canonically, so worker
completion order cannot affect ranking or export.
A broken process pool halts new scheduling: active trials receive
`worker_failure` records while combinations not yet submitted remain `PENDING`
for an explicit resume on a fresh pool. See ADR 0002.

Failures are stored with one of the categories `strategy_failure`,
`backtest_domain_failure`, `worker_failure`, `persistence_failure`, or
`unexpected_implementation_failure`, plus the exception class and a sanitized,
single-line message limited to 500 characters. Metrics stay undefined; they are
never replaced with zeros. Full tracebacks, environment dumps, and credentials
are not persisted.

The default `retry_failed=False` means resume does not silently rerun failures.
Set it before the initial study to make every later resume retry failed trials.
Before a failed trial advances back to `RUNNING`, its category, exception type,
sanitized message, and attempt timestamps are atomically appended to the trial's
`failed_attempts` history. Later success, failure, or interruption therefore
cannot erase prior attempt context, and the history is included in `trials.csv`.
Persisted stale `RUNNING` trials are always retried because they have no complete
result. State transitions are validated; successful and excluded records are
immutable.

## Incremental persistence and resume

The manifest is written atomically before trial records. Parent-owned writes use
a same-directory temporary file, `fsync`, and atomic replacement. Corrupt JSON,
wrong identities, incompatible manifests, and invalid state transitions fail
clearly. A non-empty study directory without `manifest.json` is treated as an
orphaned or corrupt store and is rejected for both fresh runs and resume; it is
never initialized over existing trial files. Existing successful QF-5 artifacts
are accepted only when every file in QF-5's authoritative artifact set is
present, their manifest carries the expected immutable run ID, and every CSV
matches its manifest-bound SHA-256 digest. This includes the strategy and
benchmark dividend-cashflow and split-adjustment ledgers.
Before scheduling, QF-6 rejects any persisted trial ID outside the study's exact
candidate set. Before ranking, result loading, or export, the persisted IDs must
also cover that complete candidate set, so extra files cannot inflate results or
reach stability analysis. Every loaded record must also match its candidate's
combination index, parameters, strategy configuration, dataset, and backtest
configuration. Successful compact metrics and run provenance are cross-checked
against the record's exact immutable QF-5 artifact manifest before they can be
skipped on resume or used by ranking and stability analysis.

```text
reports/optimization/<study-id>/
  manifest.json
  trials/<trial-id>.json
  backtests/<trial-id>/<qf5-run-id>/...
  ranking.json
  stability.json
  trials.csv
  eligible_rankings.csv
  ineligible_trials.csv
  failures.csv
  exclusions.csv
  stability.csv
  parameter_summary.csv
  summary.json
```

On resume, QF-6 verifies the exact manifest, skips immutable successes and
exclusions, applies the configured failed-trial policy, retries stale running
records, executes only pending work, and reconstructs ranking/stability from
all records. Resume after a complete study invokes zero backtests. `load_result`
and `export` reconstruct derived outputs without retaining or rerunning full
QF-5 result objects; each success links to its immutable QF-5 artifact.

## Ranking and eligibility

`MetricName` contains only fields from QF-5 `PerformanceSummary`. QF-6 reads
those values; it does not implement alternate formulas. The primary objective
has an explicit maximize/minimize direction. An undefined objective is
ineligible and remains visible.

Hard constraints execute before ranking:

- `MinimumTrades(n)` uses inclusive `trade_count >= n`;
- `MaximumDrawdown(magnitude)` uses QF-5's negative-decimal convention and
  requires `maximum_drawdown >= -magnitude` inclusively; and
- `PositiveReturn()` requires strictly `total_return > 0`.

`MetricThreshold` supports other typed QF-5 metric thresholds. Every hard
constraint must expose a typed QF-5 `MetricName`, a typed `ThresholdOperator`, a
finite numeric threshold, and stable primitive serialization; raw or malformed
deserialized constraints fail configuration instead of falling through to
equality. An undefined required metric makes the successful trial ineligible.
Failed and excluded trials are never eligible. `minimum_successful_trials` can
suppress ranking when a study has too few successful observations.

Ranking uses the objective first, configured metric tie breakers next, and the
combination ID ascending as the final fallback. The defaults prefer the less
negative maximum drawdown, then greater completed-trade count. Ranking never
uses completion order. Objectives, directions, and every tie-breaker metric and
direction must be their typed enum members; raw or unsupported deserialized
strings fail configuration instead of falling through to alternate ordering.
Full results remain exported even when a caller uses a top-N slice of
`StudyResult.rankings`.

## Stability and isolated peaks

Each assignment is represented by candidate-list coordinates. Integer and
float neighbors differ by one candidate-list position in exactly one dimension;
raw numeric spacing is irrelevant, so `[5, 10, 100]` treats 10 and 100 as
adjacent. Categorical and boolean dimensions treat every alternate value as
adjacent while all other dimensions match. A valid neighbor excludes parameter
assignments rejected by declarative constraints or the real strategy model.

For every eligible center, QF-6 reports:

- valid and excluded immediate-neighbor counts;
- successful eligible neighbor count and raw objective values;
- neighbor mean, median, direction-aware worst value, and population standard
  deviation;
- the fraction of valid neighbors satisfying ranking constraints;
- direction-aware center minus neighbor-median difference and relative
  difference;
- whether the center is on a numeric search boundary;
- a separate stability score and classification; and
- isolated-peak status with its supporting reason.

Missing, failed, pending, and ineligible neighbors are not treated as zero.
The stability score is `constraint_pass_fraction / (1 + relative_dispersion)`.
A private 34-digit decimal context makes neighborhood reductions independent of
the caller's ambient decimal precision.
A `stable` classification also requires the configured minimum eligible-neighbor
count, constraint-pass fraction, and maximum relative dispersion. Objective and
stability ranks remain separate.

An isolated peak must be within the configured top objective fraction, have
enough eligible neighbors, exceed their median by both configured absolute and
relative drops, and have a neighbor constraint-pass fraction no greater than
the configured maximum. Boundary points still need the same minimum neighbor
evidence and are labeled in the reason.

`recommended_robust` is the highest stability-ranked, non-isolated `stable`
trial inside the configured top objective fraction. It may be undefined. It is
an in-sample descriptive recommendation only, not approval for paper or live
trading. `best_objective` and `best_stability` remain independently visible.
Setting either top-fraction threshold to zero selects no trials for that
isolated-peak or robust-recommendation step.

Parameter summaries preserve contract/candidate order and report, for every
candidate value, successful count, eligible count, constraint-pass fraction,
eligible objective mean/median, and direction-aware best. They are descriptive
and do not establish causal parameter importance.

## Scale and research-integrity limits

Cartesian grids grow multiplicatively. Before combination expansion, QF-6 can
explain a grid such as `4 x 10 x 20 x 2 = 1,600`. The default safeguard is
10,000 combinations. A larger study fails before execution unless
`allow_large_grid=True` is explicit. Worker count is always bounded and nested
parallelism is not introduced.

Every grid result is in-sample and vulnerable to overfitting, repeated human
inspection, and multiple comparisons. Stability shows local sensitivity but is
not out-of-sample validation. QF-6 does not implement random/Bayesian search,
cross-validation, walk-forward optimization, holdout evaluation, Monte Carlo,
distributed execution, deployment, or live trading. Preserve an untouched
validation design in later work and never present the highest historical rank
as evidence of future profitability.
