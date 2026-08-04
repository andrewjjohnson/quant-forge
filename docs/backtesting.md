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

The first daily return is exactly zero. Later daily returns are arithmetic
end-of-session equity returns while prior equity is nonzero. After complete
equity depletion, the next return is undefined and stored as `null`, not as an
invented zero or infinity. Undefined observations are omitted from volatility,
Sharpe, and Sortino inputs. The initial-capital value remains the first
running-peak candidate, so entry costs can create an immediate drawdown.

A calendar-resolved signal whose execution session is beyond the dataset is
retained with an `unexecuted_end_of_data` order. An unresolved calendar date or
a missing execution bar is retained as a rejected order with a reason. No later
bar is substituted. Invalid or nonpositive OHLCV is rejected before the
strategy runs.

## Signals, orders, and fills

`SignalRecord` adds a stable signal ID to the original immutable QF-4
`StrategyDecision`; the decision remains the source of strategy identity,
parameter snapshot, indicator observations, target state, target weight, and
timing.

On flat-to-long transitions, target weight defines the fraction of available
cash used as the affordability budget. QF-5 finds the maximum whole-share
quantity whose slipped notional plus commission and fees fits that budget. Cash
may not become negative. On long-to-flat transitions, the engine requests and
sells the entire current quantity. It does not rebalance an existing long
position to its target weight. Repeated already-satisfied targets become
rejected audit orders with an explicit no-op reason. An unaffordable entry
requests zero shares and is rejected; fractional shares are never created.

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
`ROUND_HALF_EVEN` policy rather than the caller's ambient Decimal context.

## Buy-and-hold benchmark

The benchmark starts with the same capital and buys the maximum affordable
whole-share position at the first dataset session's open. It uses the exact same
commission, fee, and slippage models, preserves residual cash, marks every
session at the close, and holds through the final session without an invented
sale. It therefore has an open trade count when purchased, while trade count,
win rate, profit factor, and average closed-trade return remain unavailable. Its
equity curve and applicable risk/return metrics are included in the result and
export.

## Deterministic identity and export

The SHA-256 run ID is canonical-JSON-derived from the QF-3 dataset ID and schema,
QF-4 strategy name, explicit implementation version, configuration identity,
full configuration, complete backtest configuration, and engine/result schema
versions. The implementation version is also persisted directly on the result
and every trade. The identity excludes object addresses, QF-3 retrieval time,
optional initiation time, and export time.
Equivalent decimal configuration values serialize identically. Record ordering
and signal, order, fill, trade, and benchmark IDs are derived deterministically
within the run.

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
JSON. Dates are ISO `YYYY-MM-DD`; the optional initiation timestamp must be
timezone-aware ISO 8601. Export builds a temporary directory and renames it into
place only after every file is complete. `load_backtest_manifest` reloads the
complete JSON object.

## Adjustment assumptions and MVP limitations

QF-5 preserves QF-3 provider, symbol, requested and actual range, adjustment,
calendar, timezone, adapter, dataset, and schema metadata. Split-adjusted bars
embed QF-3's historical split transformations. The engine does not add dividend
cash flows, preventing double counting. QF-3 does not produce coherent
split-and-dividend-adjusted OHLCV, so QF-5 rejects that mode. Unadjusted data is
accepted with a warning because the MVP does not change share quantities for
splits and may therefore be economically misleading across a split.

This implementation is intentionally limited to one stock or ETF, long-only,
unlevered whole shares, next-open market orders, full fills, and discrete state
transitions. It has no volume or order-book model, partial fills, intraday
sequencing, taxes, multi-asset allocation, shorting, derivatives, brokerage,
event-driven engine, optimization, or forced liquidation.
