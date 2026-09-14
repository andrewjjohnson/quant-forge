# OOS aggregation and final holdout consumption

QF-40 adds `quantforge.oos`. It reads QF-39's immutable **test-window** artifacts,
summarizes their existing results, and separately manages explicit final-holdout
consumption. It does not run walk-forward selection, grids, indicator generation,
outcome labeling, or either execution engine during aggregation.

## Read-only aggregation

```python
from pathlib import Path
from quantforge.oos import (
    aggregate_prediction,
    aggregate_backtest,
    export_oos_aggregate,
    load_oos_source,
)

# plan is the exact existing QF-8 ValidationPlan; study_path is the QF-39 store.
source = load_oos_source(plan, study_path)
prediction_summary = aggregate_prediction(source)  # For a prediction source.
path = export_oos_aggregate(prediction_summary, Path("reports/oos"))
# For a trading source instead:
# trading_summary = aggregate_backtest(source)
```

`load_oos_source` needs the existing plan and files, with no evaluator, factory,
market-price input, or provider. It returns `OOSSource` in QF-8 fold order. It
verifies the QF-39 study envelope, exact plan, candidate universe, fold state,
frozen selection, role-specific membership/purge identities, eligible sessions,
and OOS fingerprint. Unexpected/duplicate fold directories, incompatible plans,
wrong families, training artifacts, and corrupted completed records fail closed.
Only completed folds contribute observations. Missing, pending, interrupted, and
failed folds retain their status and failure history; an old `oos.json` does not
override a failed state. Corruption is an error, not a missing-window fallback.

Prediction verification reuses QF-42's offline snapshot validator, binding its
schedule, signal/row identities, backend, context, outcome, evaluator, and bounded
dataset to the frozen selection. Test outcomes cannot reach the next protected
test or final holdout. Backtest verification uses QF-5's immutable export checks,
compares every exported table to the snapshot, and requires equity and benchmark
sessions to equal test membership and the exact QF-43 evaluation configuration.

Both aggregates expose a completeness record, including expected/completed
counts and missing/failed/incomplete fold IDs. A partial aggregate describes only
the observed successful windows. Missing windows are never inserted as zero
returns or successful predictions. Ordinary aggregation neither consults nor
modifies the holdout ledger and never claims that the holdout is pristine. Later
consumers must obtain current holdout state explicitly from the ledger.

## Prediction semantics

`PredictionOOSAggregate` contains the summary, per-window summaries, stability,
timestamp-level observations, and complete source references. Original generated
signals, labeled rows, decision/study/context IDs, and QF-39 selection/result IDs
remain attached to each observation. Rejected, blocked, and overlapping signals
are preserved but excluded from the prediction population. Accepted signals and
ordinary unclassified UP/DOWN signals are eligible. Signals without labels count
as predictions, with unavailable outcomes. Skipped/empty scheduled decisions
remain in the frequency denominator.

The prediction frequency is eligible predictions divided by scheduled decisions
in completed windows. Accuracy uses only rows with a stored boolean correctness
value, pooling counts rather than averaging fold accuracies. The existing QF-11
95% **Wilson score** implementation is reused unchanged. Its denominator and
missing count are explicit. Intraday decisions may share the same future outcome;
the Wilson interval is descriptive binomial uncertainty, not an independence
claim, a multiple-comparison correction, or evidence of profitability.

`PredictionMetricFields` maps fields from existing `evaluation.values`:

| Statistic | Default stored field |
| --- | --- |
| Accuracy | `direction_correct` |
| Signed outcome | `signed_prediction_return` |
| MFE / MAE | `mfe_percentage` / `mae_percentage` |
| Event rates | `label` |
| Matched baseline | Explicit `baseline_correct` field and `baseline_name` |

Means, medians, extrema, sample counts and unavailable counts are calculated from
stored numeric ratios. Missing fields are unavailable; no market data or other
study is used to substitute a metric. Event labels retain every observed category,
including ambiguous same-session outcomes, with counts/rates on available events.
Matched-baseline accuracy and difference use only rows where both stored
correctness fields exist. QF-39's selection analyzer evidence is **in-sample** and
cannot supply an OOS baseline. Most current QF-39 gap artifacts therefore report
the matched baseline as unavailable. A custom evaluator can persist a matched
baseline on each row and bind its explicit field through `PredictionMetricFields`.

Per-window accuracy and signed-outcome means have descriptive mean/median/range
summaries with missing windows disclosed. Different outcome/evaluator definitions
cannot be pooled under one statistical interpretation; aggregation rejects such
mixed semantics rather than silently mixing horizons or incompatible labels.
Custom stored fields must use the documented ratio units. No equity, trade P&L,
or transaction-cost result is manufactured for predictions.

## Backtest semantics

`BacktestOOSAggregate` preserves each complete native QF-5 result, including
equity, benchmark, completed/open trades, costs, warnings, dividends, splits, and
per-window performance/drawdown. QF-5 metric definitions are unchanged. Completed
trade/winner counts and positive/negative total-economic P&L are summed from those
native summaries to calculate pooled win rate and profit factor. No losing trades
means undefined profit factor (`null`); no completed trades means undefined win
rate. Open positions stay separate from completed-trade statistics.

For fold `k`, starting capital `C[k]`, and native marked equity `E[k,t]`, the
dimensionless reporting index is:

```text
B[1] = 1
I[k,t] = B[k] * E[k,t] / C[k]
B[k+1] = I[k,last]
combined return = I[last,last] - 1
```

The benchmark follows the same rule independently, including its first-open
entry costs through the first close. Each row preserves its source fold/run,
exchange-close session, window-start flag, and starting index. Drawdown includes
the initial index of one and uses QF-5's negative-ratio convention:
`I / running_peak - 1`. No training, selection, warm-up, or gap dates enter either
series. A missing fold has no invented observations; the entire result remains
explicitly partial.

This index chains **independent reset-account segment returns**, not executable
capital carryover. Every native fold still begins with its configured capital
and no positions. Terminal open positions retain their existing marks; there is
no invented liquidation fill, exit cost, or carried position in the next fold.
Fixed costs and whole-share sizing were incurred on native fold capital; this
index does not resimulate them at compounded capital. Native commissions, fees,
slippage costs, and dividend income are reported in their original currency
amounts across separate accounts. Exposure uses the fraction of actual OOS
sessions marked exposed. Profitable-window fraction uses completed windows as its
denominator, alongside explicit completeness. Empty studies have unavailable
return/drawdown/exposure rather than a zero-performance conclusion.

## Configuration stability

`ConfigurationStabilitySummary` retains each fold's exact candidate ID, searched
parameters, complete executable definition, selection ID, status, and the already
stored QF-6/QF-32 neighborhood evidence for that selected trial. Rankings are not
used as OOS observations. Adjacent known selections determine configuration change
count/frequency, consecutive repeat-selection count/frequency, and per-parameter
change counts. A missing selection breaks adjacency; a failed test with a known
frozen selection still supplies selection evidence. Selection counts by candidate
and parameters by fold remain available. Turnover is descriptive, with no
automatic good/bad classification and no new optimization.

## Explicit holdout workflow

```python
from quantforge.oos import HoldoutEvaluation, HoldoutLedger

# Initialize ONCE for the research workspace; retain this store permanently.
ledger = HoldoutLedger.create(Path("research-state/holdouts"))
# On subsequent runs: ledger = HoldoutLedger(Path("research-state/holdouts"))
reserved = ledger.reserve(source)

# evaluator is the original QF-39 PredictionEvaluator or BacktestEvaluator.
# Choose the source freeze explicitly, without looking at holdout results.
evaluation = HoldoutEvaluation.prepare(
    source,
    evaluator,
    selection_fold_id=source.folds[-1].fold_id,
)
consumed = ledger.consume(evaluation, run_id="final-evaluation-001")
result = ledger.result(evaluation)

# Cached exact request: no evaluation callback. Explicit reproducibility rerun:
reproduced = ledger.consume(evaluation, run_id="repeat-001", reproduce=True)
assert reproduced == consumed
```

Reservation and evaluation are separate explicit operations. The initial durable
state is `RESERVED` (`reserved_unconsumed`). The first evaluation attempt makes a
one-way transition to `CONSUMED` **before** calling the evaluator. This deliberately
conservative transition may consume an interval even if evaluation subsequently
fails. Unknown exposure is never called pristine.

QF-40 validates the original adapter, plan, source data, backend, full candidate
universe and selected executable snapshot before exposure. The source selection
must already exist in the verified QF-39 source. Its exact freeze is immutable;
there is no selection callback, ensemble construction, or post-holdout reselection.
The implementation supports QF-39's existing one-selected-configuration policy.
Switching to a different freeze after consumption is rejected, including another
fold's freeze of the same logical parameters.

QF-8 supplies holdout membership and historical warm-up. A small additive adapter
entry point accepts that `HoldoutPartition` through `EvaluationPartition`; existing QF-39 test methods delegate
through the same entry point with unchanged membership/results. Prediction calls
the original QF-42/QF-11 path; backtests call QF-43/QF-5 with a fresh account. The
holdout's final maximum outcome-horizon sessions supply outcomes only, so every
prediction label remains inside the reserved interval. After excluding that tail,
the retained observation count must meet the source study's
`minimum_test_observations`; warm-up and outcome-only sessions do not count.
Insufficient holdout length fails during preparation and consumption validation,
before exposure. QF-40 does not ask QF-8 to purge the final holdout
against an invented later protected window; its serialized `purge` is explicitly
`null` and the holdout tail policy is recorded separately.

## Research lineage and prior exposure

Lineage is canonical JSON/SHA-256 over the fixed research environment (both source
identities, timeframes, sessions, aggregation, indicators/backends, rule, outcomes,
and execution), fold intervals and warm-up, purge/embargo, training mode, candidate
universe/grid/scientific policies, adapter/engine versions, selection policy, and
minimum counts. Run names, window names, reservation prose, and retry/continue
policies do not themselves create a new scientific lineage. Full original plan,
study, and frozen-selection identities remain attached to consumption requests.

Material changes yield different lineage IDs. They do not automatically restore
research ignorance. A separate conservative exposure registry rejects a pristine
claim for overlapping exchange-session date intervals on the same canonical symbol
that were consumed by another lineage. This guard spans provider/data revisions,
study families and backend/parameter changes, and catches trivially shifted
overlapping intervals. Distinct, nonoverlapping intervals may be reserved. It is
a local research safeguard, not a statistical independence guarantee. Timestamp
intervals resolve the dataset calendar's session-close keys to their session date
labels, matching QF-39 observation membership even when a close crosses UTC
midnight. The stored exposure scope declares `exchange_session_labels_v1`.
Earlier scopes without this date basis are rejected conservatively: preserve their
consumed evidence rather than creating a fresh ledger or assuming UTC dates prove
nonoverlap. Automatic migration of ambiguous exposure records is not provided.

Use one permanent ledger for the entire workspace. Moving to a new empty store,
deleting all evidence, restoring an old backup, or inspecting prices/results
outside these APIs cannot be detected automatically. Existing unrelated source
files are not a global exposure registry. Trusted user callbacks retain the
repository's existing responsibility not to inspect external holdout information.
The ledger prevents automatic reselection through its own consumption workflow;
it cannot sandbox arbitrary Python code or prevent manual research elsewhere.

## Persistence, crashes, and retries

```text
<ledger>/store.json
<ledger>/.lock
<ledger>/lineages/<lineage-id>/reservation.json
<ledger>/exposures/<lineage-id>.json       # Immutable CONSUMED marker + exact request
<ledger>/lineages/<lineage-id>/evaluation/...  # Existing QF-5 export when applicable
<ledger>/lineages/<lineage-id>/result.json
```

The ledger uses the repository's canonical JSON/SHA-256 envelopes and atomic
temporary-file replacement, with file **and directory** fsync. A POSIX advisory
lock covers validation, the permanent exposure marker, execution, and result
persistence. A concurrent operation fails explicitly; process exit releases the
OS lock. This is a local POSIX filesystem store, not a distributed/network store.

The immutable consumed marker includes the full request/provenance, exact freeze
and parameter snapshot, validation plan/study/lineage and holdout identities,
bounded membership, original consumption run ID and UTC timestamp. Result and
artifact hashes/references are attached only after durable result persistence.
For backtests, the exported run directory and its parent `evaluation` directory
are fsynced before publishing `result.json`, preserving both artifact file entries
and the run-directory rename. A failure at either sync leaves the holdout consumed
without a published result; an exact retry validates the export and repeats the sync.
The current typed `HoldoutConsumptionRecord` always consults this marker; initial
reservation metadata never overrides it.

| Failure/retry | Behavior |
| --- | --- |
| Marker write fails before exposure | No evaluator call; if no marker exists, reservation remains unconsumed. |
| Marker is written but fsync/evaluation/artifact write fails | Consumed, result unavailable; never reset. |
| Process exits after evaluation but before final result write | Marker remains consumed; exact recovery may rerun the original freeze. |
| Exact successful request repeated | Verify and return the stored result reference without reevaluating. |
| Explicit reproduction | Use original request/run/timestamp; artifact and result must match existing immutable bytes. |
| Result/evaluation artifacts without compatible ledger/reservation | Reject; do not initialize pristine state over orphaned evidence. |
| Result removed but consumed marker retained | Remain consumed; exact recovery can reproduce it. |
| Conflicting request or overlapping consumed lineage | Reject before evaluation; no automatic reselection. |

Aggregate artifacts are named by their deterministic content ID. Exact exports
verify existing content; `load_oos_aggregate` verifies schema and identity. Source
fold order and exact source hashes enter aggregate identity, alongside scientific
configuration and metric bindings. Neither wall-clock export time nor future
unrelated market-data appends changes an aggregate from the same immutable source
artifacts. New source revisions intentionally require new plan/study lineage.

## Structured outputs and scope

Public contracts are `OOSSource`, `PredictionOOSAggregate`, `BacktestOOSAggregate`,
`PredictionMetricFields`, `MetricSummary`, `ConfigurationStabilitySummary`,
`HoldoutEvaluation`, `HoldoutState`, `HoldoutConsumptionRecord`, and
`HoldoutLedger`. The aggregates, stability and consumption records expose
`to_primitive()`; aggregates expose content IDs. Stored holdout results have the
separate `final_holdout_result` kind and can never enter QF-39 fold aggregation.

QF-9 can consume these structured artifacts and hashes for its future generic
manifest/index work. QF-41 can consume them and query current holdout state for
reporting. No generic manifest infrastructure, HTML/static reporting, optimizer,
new execution engine, ML, specialized edge strategy, allocation, or brokerage
functionality is included.

## Offline acceptance examples

The tests run actual QF-39 selection/evaluation followed by QF-40 aggregation:

```bash
uv run pytest tests/unit/oos/test_prediction.py::test_real_prediction_oos_example_and_roundtrip
uv run pytest tests/unit/oos/test_backtest.py::test_real_backtest_equity_native_metrics_and_stitching
uv run pytest tests/unit/oos/test_holdout.py::test_explicit_first_consumption_and_exact_reproduction
uv run pytest tests/unit/oos tests/unit/walk_forward
```

The prediction example yields eight predictions across two OOS windows. The
backtest example includes a winning and a losing trade with real configured
commissions/slippage and independent starting-capital resets. Both final-holdout
fixtures demonstrate durable first consumption and exact reproduction. All data
is synthetic and all tests are offline; none is a profitability claim.
