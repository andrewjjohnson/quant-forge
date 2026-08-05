# Deterministic daily-bar backtesting

QF-5 implements QuantForge's first complete backtest through
`quantforge.backtesting`. It consumes one validated immutable QF-3
`MarketDataset` and one QF-4 `Strategy`. It never retrieves provider data and it
does not contain strategy-specific branches.

```text
QF-3 MarketDataset
        |
        v
QF-4 StrategyOutput and StrategyDecision
        |
        v
QF-5 orders -> fills -> cash/position state -> trades/equity
        |                                      |
        +-------------------> metrics <---------+
                               |
                               v
                     result and stable export
```

## Public API

```python
from decimal import Decimal
from pathlib import Path

from quantforge.backtesting import (
    BacktestConfig,
    BasisPointSlippage,
    ExplicitZeroFees,
    PerShareCommission,
    export_backtest_result,
    run_backtest,
)
from quantforge.strategies import (
    MovingAverageCrossoverParameters,
    MovingAverageCrossoverStrategy,
)

strategy = MovingAverageCrossoverStrategy(
    MovingAverageCrossoverParameters(
        fast_window=20,
        slow_window=50,
        target_long_weight=Decimal("0.75"),
    )
)
config = BacktestConfig(
    initial_capital=Decimal("100000"),
    commission=PerShareCommission(
        amount_per_share=Decimal("0.005"),
        minimum=Decimal("1"),
    ),
    fees=ExplicitZeroFees(),
    slippage=BasisPointSlippage(Decimal("5")),
    annual_risk_free_rate=Decimal("0.03"),
)
result = run_backtest(dataset, strategy, config)
artifact_path = export_backtest_result(result, Path("reports/backtests"))
```

Commission, additional transaction-fee, and slippage models are required
constructor arguments. `ExplicitZeroFees` records a deliberate zero-fee policy;
zero commission is likewise expressed with a zero-valued commission model. No
cost category has an implicit default. The configuration is frozen and includes
the execution, sizing, risk-free-rate, annualization, long-only,
forced-liquidation, engine, and result-schema assumptions.
Engine and result-schema versions are implementation-owned constants rather than
caller-supplied configuration; QF-5 serializes them but rejects constructor
overrides that could mislabel the executed implementation or exported schema.

Every commission, fee, and slippage model declares a nonempty
`implementation_version`, which is serialized beside its model name and
parameters. Custom model authors must increment that version whenever a
calculation change could alter fills or results, even when user-facing
parameters are unchanged. Each model also declares a structural `cost_category`
of `commission`, `transaction_fee`, or `slippage`; configuration construction
rejects a model placed in the wrong slot before any run can be serialized or
executed. QF-5 invokes custom configuration, commission, fee, and slippage
callbacks only through its private 34-digit `ROUND_HALF_EVEN` arithmetic
boundary. The strategy engine and benchmark share that evaluator, so the
caller's ambient Decimal context cannot change costs under a fixed run ID.

Commission and fee models must also declare and serialize
`buy_cost_is_non_decreasing_by_quantity=true`. This is a semantic contract that,
for a fixed positive fill price, the model's buy-side cost never decreases as
whole-share quantity increases. Custom schedules with rebates or discontinuous
discounts that violate this guarantee are rejected during configuration.

## Chronological convention

QF-5 uses `NextSessionOpenExecution`, the only supported MVP convention:

1. A bar for exchange session `t` becomes observable after its close.
2. QF-4 may emit a decision with `signal_session=t`.
3. The order decision is associated with `t`, but cannot fill at `t`.
4. QF-4 and QF-5 require the first exchange-calendar successor as the earliest
   execution session. Weekends and holidays are skipped by QF-3's calendar.
5. At that later session, the order references only the session's `open`.
6. Adverse slippage, commission, and separately auditable transaction fees are
   applied, then cash and whole shares are updated.
7. The position is marked to the same session's `close`; end-of-session cash,
   holdings, equity, return, peak, drawdown, and exposure are recorded.

The strategy equity curve's first daily return is exactly zero because a
first-session close decision cannot execute until a later session. Later daily
returns are arithmetic end-of-session equity returns while prior equity is
nonzero. The buy-and-hold benchmark is already invested during the first
session, so its first return measures initial capital to the first close and is
included in volatility, Sharpe, and Sortino inputs. After complete equity
depletion, the next return is undefined and stored as `null`, not as an
invented zero or infinity. Undefined observations are omitted from risk-metric
inputs. The initial-capital value remains the first running-peak candidate, so
entry costs can create an immediate drawdown.

A calendar-resolved signal whose execution session is beyond the dataset is
retained with an `unexecuted_end_of_data` order. An unresolved calendar date or
a missing execution bar is retained as a rejected order with a reason. No later
bar is substituted. Invalid or nonpositive OHLCV is rejected before the
strategy runs.

Before QF-5 applies its gap policy, QF-3's complete-dataset validator recomputes
every expected session across the requested range from the declared calendar,
rejects bars on non-session dates, and requires the recomputed gaps to equal the
immutable `missing_sessions` tuple. QF-5 then rejects a recomputed gap between
`actual_first_session` and `actual_last_session`. Otherwise an equity change
across several exchange sessions would be recorded and annualized as one daily
return. Missing sessions before the first observed bar or after the last
observed bar remain preserved in provenance and do not invalidate the observed
return series.

## Signals, orders, and fills

`SignalRecord` adds a stable signal ID to the original immutable QF-4
`StrategyDecision`; the decision remains the source of strategy identity,
parameter snapshot, indicator observations, target state, target weight, and
timing.

On flat-to-long transitions, target weight defines the fraction of available
cash used as the affordability budget. QF-5 finds the maximum whole-share
quantity whose slipped notional plus commission and fees fits that budget. Cash
may not become negative. The engine and benchmark use binary search only because
the required nondecreasing buy-cost contracts make affordability monotonic. On
long-to-flat transitions, the engine requests and sells the entire current
quantity. It does not rebalance an existing long position to its target weight.
Repeated already-satisfied targets become rejected audit orders with an explicit
no-op reason. An unaffordable entry requests zero shares and is rejected;
fractional shares are never created.

`OrderRecord` includes the run, signal, symbol, side, requested quantity,
decision and eligibility sessions, target, strategy identities, final status,
and reason. `FillRecord` links to the order and signal and separately records
reference open, final fill price, effective slippage per share and basis points,
notional, commission, additional fees, and signed cash effect. A buy cash effect
is `-(notional + commission + fees)`; a sell cash effect is
`notional - commission - fees`. Every MVP market order has zero or one full
fill.

Supported commission models are:

- `FixedCommission(amount)` per fill;
- `PerShareCommission(amount_per_share, minimum)`; and
- `BasisPointCommission(basis_points)` on final fill notional.

Supported additional-fee models are:

- `ExplicitZeroFees()` for a serialized, deliberate zero-fee policy; and
- `BasisPointFees(basis_points)` on final fill notional. The order side is
  supplied to every fee model so future regulatory or exchange policies can be
  side-specific without changing the execution contract.

`BasisPointSlippage` implements:

```text
buy fill  = reference open * (1 + basis_points / 10,000)
sell fill = reference open * (1 - basis_points / 10,000)
```

All model parameters and calculated costs must be finite and nonnegative.
Slippage below 10,000 basis points is required so sell fills remain positive.

## Portfolio and trades

`PositionRecord` is emitted for every session with shares, commission- and
fee-inclusive entry cost basis, average entry cost, market value, cumulative
realized P&L, and unrealized P&L. `DailyPortfolioRecord` contains the
corresponding cash, close mark, equity, nullable daily return, running peak,
negative-decimal drawdown, exposure state and weight, and associated order/fill
IDs. The accounting identity is always:

```text
equity = cash + shares * session close
```

A completed `TradeRecord` traces entry and exit through signal, order, and fill
IDs and preserves the strategy configuration identity. For quantity `q`:

```text
gross P&L = (exit fill price - entry fill price) * q
net P&L   = gross P&L - entry commission - exit commission
            - entry fees - exit fees
return    = net P&L / (entry notional + entry commission + entry fees)
```

Holding period is the number of dataset-session intervals from entry to exit.
An end-of-data holding becomes an explicit open trade with nullable exit and
outcome fields. QF-5 never fabricates a final-close sale; forced liquidation is
disabled and unsupported in this milestone.

## Performance formulas and undefined values

All returns and rates are decimal ratios, not percentages. Metrics exclude open
trades from completed-trade statistics and serialize undefined values as JSON
`null`, never `NaN` or infinity.

| Metric | QF-5 definition and edge policy |
| --- | --- |
| Total return | `ending equity / initial capital - 1`. |
| CAGR | `(ending / initial) ** (365.2425 / elapsed calendar days) - 1`; `null` for zero elapsed days or nonpositive ending equity. |
| Annualized volatility | Sample standard deviation of defined consecutive daily equity returns times `sqrt(annualization_factor)`; `null` with fewer than two defined returns. A return following zero prior equity is undefined and excluded. |
| Sharpe | Daily excess-return mean divided by sample standard deviation of daily excess returns, times `sqrt(annualization_factor)`. Daily risk-free return is `(1 + annual_rate) ** (1 / annualization_factor) - 1`; zero deviation or insufficient observations gives `null`. |
| Sortino | Daily excess-return mean divided by the square root of the mean squared negative excess returns, times `sqrt(annualization_factor)`; zero downside deviation gives `null`. All return sessions are in the downside denominator. |
| Maximum drawdown | Minimum of `equity / running_peak - 1`, reported as zero or a negative decimal. |
| Profit factor | Sum of positive net completed-trade P&L divided by the absolute sum of negative net completed-trade P&L. No losing trades gives `null`, avoiding JSON infinity. |
| Exposure | Fraction of dataset sessions whose end-of-session shares are nonzero; an entry execution session counts. |
| Trade count | Completed round trips only; open trade count is separate. |
| Win rate | Positive-net-P&L completed trades divided by completed trades; no completed trades gives `null`. |

Supporting fields include starting/ending equity, aggregate positive and
negative net completed-trade P&L (`gross_profit` and `gross_loss` in the
performance schema), winning and losing counts, average trade return, and the
benchmark total return. Decimal calculations use a private 34-significant-digit
`ROUND_HALF_EVEN` policy rather than the caller's ambient Decimal context. This
includes all built-in and custom cost-model callbacks used for configuration,
slippage, commissions, fees, affordability, and benchmark execution.

## Buy-and-hold benchmark

The benchmark starts with the same capital and buys the maximum affordable
whole-share position at the first dataset session's open. It uses the exact same
commission, fee, and slippage models, preserves residual cash, marks every
session at the close, and holds through the final session without an invented
sale. Its first daily return therefore measures initial capital through the
first close, including entry costs and the first session's price move, and that
observation participates in volatility, Sharpe, and Sortino. It therefore has
an open trade count when purchased, while trade count, win rate, profit factor,
and average closed-trade return remain unavailable. Its equity curve and
applicable risk/return metrics are included in the result and export. This
return-series convention is benchmark implementation version 2.

## Deterministic identity and export

The SHA-256 run ID is canonical-JSON-derived from the complete QF-4 market-data
reference (QF-3 dataset ID, schema, adjustment mode, and trading calendar), a
separate SHA-256 fingerprint of the actual validated bars, QF-4 strategy name,
explicit implementation version, configuration identity, full configuration,
complete backtest configuration (including each cost-model implementation
version), and engine/result schema versions. QF-5 recomputes the strategy
configuration identity from the exact immutable snapshot used by the run ID and
rejects stale, hard-coded, or initialization-mutated identities. Before
execution, it invokes the authoritative QF-3 complete-dataset validator. It
checks bar types, values,
symbols, ordering, requested and actual bounds, bar count, exchange sessions,
the exact recomputed missing-session set, corporate-action-session structure,
normalized-data digest, dataset ID, schema version, and canonical artifact
paths. A copied or manually constructed dataset whose derivable metadata is
incorrect is rejected before strategy, benchmark, or metric evaluation. The
independent QF-5 bar fingerprint canonicalizes the ordered symbol, session date,
and every OHLCV value and is persisted in the manifest as
`market_data.bars_fingerprint` along with the QF-3 raw and normalized SHA-256
digests.

An in-memory `MarketDataset` does not contain the raw provider bytes, so its raw
digest can be identity-bound but not independently re-read there. QF-3 cache
loading verifies those bytes against `raw_sha256`; complete split and dividend
provenance likewise remains a schema-v3 ingestion/cache guarantee rather than
something inferable from OHLCV alone.

The strategy implementation version is also persisted directly on the result
and every trade. The identity excludes object addresses, QF-3 retrieval time,
optional initiation time, and export time.
Equivalent decimal configuration values serialize identically. Record ordering
and signal, order, fill, trade, and benchmark IDs are derived deterministically
within the run.

Strategy, backtest, and benchmark configurations are canonicalized into deeply
immutable snapshots when the run is initialized. Result properties and manifest
serialization return detached primitive copies, so later mutation of a custom
strategy or cost model cannot rewrite provenance beneath an existing run ID.

`export_backtest_result` creates, without overwriting, this directory beneath a
caller-selected ignored reports root:

```text
<run-id>/
  manifest.json
  signals.csv
  orders.csv
  fills.csv
  positions.csv
  trades.csv
  equity.csv
  benchmark_equity.csv
```

The manifest contains provenance, strategy and backtest configuration,
performance, benchmark metadata and metrics, record counts, warnings, and
limitations. CSV field and row order is stable; nested values use canonical
JSON. Empty record tables retain their complete headers, including the full fill
schema when neither the strategy nor benchmark can afford a share. Dates are ISO
`YYYY-MM-DD`; the optional initiation timestamp must be timezone-aware ISO 8601
with a defined UTC offset. A `tzinfo` object whose `utcoffset()` returns `None`
is rejected as naive. Export builds a temporary directory and renames it into
place only after every file is complete. `load_backtest_manifest` reloads the
complete JSON object.

## Adjustment assumptions and MVP limitations

QF-5 preserves QF-3 provider, symbol, requested and actual range, adjustment,
calendar, timezone, adapter, dataset, schema, raw and normalized content
digests, missing-session, split-session, and dividend-session metadata, and
records its own fingerprint of the validated bars it consumed. It accepts only
schema-version-3 `unadjusted` datasets with no verified split or cash-dividend
session between the first and last observed bars. Legacy schemas without
complete corporate-action provenance, identity-inconsistent copies, and
unadjusted split- or dividend-bearing ranges are rejected before the strategy,
benchmark, or metrics run.

QF-3's `split_adjusted` mode divides earlier prices by later split coefficients
but retains only their effective sessions, not the point-in-time factors needed
for share conversion. Using those prices directly for whole-share fills or
per-share costs would mix post-split-equivalent units with historical share
units and allow a future split to revise earlier execution economics. QF-5
therefore also rejects both `split_adjusted` and
`split_and_dividend_adjusted` datasets.

Supporting a range containing a split requires preserving the point-in-time
factor plus QF-5 quantity and cost-basis transformation at the effective
session. A cash dividend requires explicit ex-date entitlement, payment timing,
withholding, and cash-credit semantics; dividend-adjusted execution likewise
requires an explicit total-return policy. The MVP does not infer either behavior
from normalized prices.

This implementation is intentionally limited to one stock or ETF, long-only,
unlevered whole shares, next-open market orders, full fills, and discrete state
transitions. It has no volume or order-book model, partial fills, intraday
sequencing, taxes, multi-asset allocation, shorting, derivatives, brokerage,
event-driven engine, or forced liquidation. QF-5 itself does not optimize;
QF-6's `quantforge.optimization` package coordinates repeated unchanged QF-5
runs and reads this typed performance model. See `docs/optimization.md`.
