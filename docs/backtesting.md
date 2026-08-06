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
    DividendPolicy,
    ExplicitZeroFees,
    NextSessionOpenExecution,
    PerShareCommission,
    SplitAccountingPolicy,
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
    dividend_policy=DividendPolicy.PRICE_RETURN_ONLY,
    split_policy=SplitAccountingPolicy(),
    execution=NextSessionOpenExecution(),
    annual_risk_free_rate=Decimal("0.03"),
)
result = run_backtest(dataset, strategy, config)
artifact_path = export_backtest_result(result, Path("reports/backtests"))
```

Commission, additional transaction-fee, and slippage models are required
constructor arguments. `ExplicitZeroFees` records a deliberate zero-fee policy;
zero commission is likewise expressed with a zero-valued commission model. No
cost category has an implicit default. The configuration is frozen and includes
the dividend, split, execution, sizing, risk-free-rate, annualization, long-only,
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

## Dividend and split policies

`DividendPolicy` is independent of the mandatory `SplitAccountingPolicy`:

| Policy | Raw dataset with dividends | Return basis |
| --- | --- | --- |
| `PRICE_RETURN_ONLY` | Preserve events but credit no cash. Record ignored events, their summed per-share amount, and estimated excluded cash based on prior-close holdings. | `price_return` |
| `CASH_DIVIDENDS` | Credit each entitled dividend exactly once and attribute it separately from price-trade P&L. | `total_return_with_cash_dividends` |
| `REJECT_IF_DIVIDENDS` | Reject with instructions to select one of the two economic policies. | `price_return` when no events exist |

`REJECT_IF_DIVIDENDS` is the compatibility and fail-closed default. Callers
should still select a policy explicitly for reportable research. The maintained
SPY example explicitly selects `PRICE_RETURN_ONLY` so early signal research can
use raw executable prices without implementing a synthetic total-return series.
That is a valid price-behavior experiment, but long-period strategy and
buy-and-hold results are understated relative to total economic return and must
not be presented simply as “total return.”

Price-only mode never changes cash, equity, daily returns, CAGR, Sharpe,
Sortino, drawdown, trade P&L, or benchmark performance for a dividend event.
Its estimated ignored cash is informational only. Both accounts expose a typed
`DividendAccountingSummary` containing policy, return basis, events present,
credited and ignored counts, credited cash, ignored per-share amount, estimated
ignored portfolio cash, the corporate-action snapshot identifier, and a warning.
Changing only the policy changes the backtest and benchmark identities.

Split handling is never optional for raw data. Every supported non-unit split
still transforms existing shares and per-share basis before the open, preserves
aggregate basis and cash, and rejects fractional results. Dividend policy cannot
disable or weaken split validation. Raw provider OHLCV remains authoritative for
fills, marks, and portfolio accounting. Before QF-4 indicator calculation, QF-5
also derives an ephemeral causal split-normalized feature view: beginning on an
effective split session, OHLC values are multiplied by the cumulative
shares-after/shares-before factor and volume is divided by that factor. This
keeps moving windows continuous in original-share units without revising any
earlier row when a later split occurs. The fixed transformation is serialized in
`split_policy` and is not persisted as a second QF-3 dataset.

## Chronological convention

QF-5 uses `NextSessionOpenExecution` plus raw-price explicit corporate-action
accounting. Strategy decisions are first calculated from the causal feature view
described above. For every execution session QF-5 then applies this exact order:

1. Snapshot shares held at the previous session's close for dividend entitlement.
2. Apply effective split factors to existing shares and per-share cost basis.
3. Execute eligible next-open orders using only raw session `open`.
4. Apply adverse slippage, commission, and separately auditable fees.
5. Apply the selected dividend policy: credit entitled cash, disclose ignored
   cash, or fail before execution in strict mode.
6. Mark remaining shares using only raw session `close`.
7. Record cash, holdings, equity, return, peak, drawdown, exposure, and action IDs.

A bar for session `t` becomes available only after its close. A QF-4 decision
with `signal_session=t` cannot execute until the first later exchange-calendar
session. Buying at an ex-date open does not earn that dividend; selling at that
open retains entitlement established at the previous close. The dividend
cash-mode credit occurs after opening orders, so it cannot fund an ex-date purchase.
Tiingo's `divCash` identifies the ex-date, not the later real payment date; this
is a deliberate daily-model timing simplification, not exact settlement.

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
ordinary entries are always whole-share. A corporate-action factor that would
produce fractional shares is rejected clearly because cash-in-lieu is not
implemented; it is never rounded or discarded.

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
negative-decimal drawdown, exposure state and weight, and associated order,
fill, and corporate-action IDs. The accounting identity is always:

```text
equity = cash + shares * session close
```

A completed `TradeRecord` traces entry and exit through signal, order, and fill
IDs and preserves the strategy configuration identity. Split-adjusted exit
quantity can differ from entry quantity, so price-trade economics use notionals:

```text
gross price P&L = exit gross notional - entry gross notional
net P&L   = gross P&L - entry commission - exit commission
            - entry fees - exit fees
price return = net P&L / (entry notional + entry commission + entry fees)
total economic P&L = net price P&L + attributed dividend income
total economic return = total economic P&L / total entry cost
```

`DividendCashflowRecord` separately records run, account, action, symbol,
ex-date, entitled shares, per-share amount, total cash, resulting cash, and
source dataset. It is not a fill or trading proceeds. `SplitAdjustmentRecord`
records factor, pre/post shares and average cost, preserved aggregate basis,
unchanged cash, action ID, run/account, and source dataset. Splits create no
realized P&L. `TradeRecord` exposes price net P&L, attributed dividend income,
and total economic P&L/return separately. In cash mode, completed-trade win
rate, profit factor, and average return use total economic values; in price-only
mode, dividend attribution is zero and these equal price-trade economics.
Portfolio metrics always come from the policy-aware equity ledger.

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
| Profit factor | Sum of positive total-economic completed-trade P&L divided by the absolute sum of negative total-economic P&L. No losing trades gives `null`, avoiding JSON infinity. |
| Exposure | Fraction of dataset sessions whose end-of-session shares are nonzero; an entry execution session counts. |
| Trade count | Completed round trips only; open trade count is separate. |
| Win rate | Positive-total-economic-P&L completed trades divided by completed trades; no completed trades gives `null`. |

Supporting fields include starting/ending equity, aggregate positive and
negative net completed-trade P&L (`gross_profit` and `gross_loss` in the
performance schema), winning and losing counts, average trade return, the
benchmark total return, total dividend income, and dividend/split event counts.
Decimal calculations use a private 34-significant-digit
`ROUND_HALF_EVEN` policy rather than the caller's ambient Decimal context. This
includes all built-in and custom cost-model callbacks used for configuration,
slippage, commissions, fees, affordability, and benchmark execution.

## Buy-and-hold benchmark

The benchmark starts with the same capital and buys the maximum affordable
whole-share position at the first dataset session's open. It uses the exact same
raw bars, commission, fee, slippage, dividend-entitlement, split, and
fractional-share policies, preserves residual cash, marks every
session at the close, and holds through the final session without an invented
sale. Its first daily return therefore measures initial capital through the
first close, including entry costs and the first session's price move, and that
observation participates in volatility, Sharpe, and Sortino. It therefore has
an open trade count when purchased, while trade count, win rate, profit factor,
and average closed-trade return remain unavailable. Its equity curve and
applicable risk/return metrics are included in the result and export. This
return-series and action convention is benchmark implementation version 4.

## Deterministic identity and export

The SHA-256 run ID is canonical-JSON-derived from the complete QF-4 market-data
reference (QF-3 dataset ID, schema, adjustment mode, and trading calendar), a
separate SHA-256 fingerprint of the actual validated bars, QF-4 strategy name,
explicit implementation version, configuration identity, full configuration,
complete backtest configuration (including each cost-model implementation
version, dividend policy, and the versioned split policy defining both raw
execution and causal feature bases), corporate-action snapshot ID, and engine/result
schema versions. QF-5 recomputes the strategy
configuration identity from the exact immutable snapshot used by the run ID and
rejects stale, hard-coded, or initialization-mutated identities. Before
execution, it invokes the authoritative QF-3 complete-dataset validator. It
checks bar types, values,
symbols, ordering, requested and actual bounds, bar count, exchange sessions,
the exact recomputed missing-session set, corporate-action records, counts,
snapshot/action identities and session structure, normalized-data digest,
dataset ID, schema version, and canonical artifact
paths. A copied or manually constructed dataset whose derivable metadata is
incorrect is rejected before strategy, benchmark, or metric evaluation. The
independent QF-5 bar fingerprint canonicalizes the ordered symbol, session date,
and every OHLCV value and is persisted in the manifest as
`market_data.bars_fingerprint` along with the QF-3 raw and normalized SHA-256
digests. `fingerprint_market_bars()` exposes that same canonical calculation to
orchestrators that must verify a returned result against their supplied bars.

An in-memory `MarketDataset` does not contain the raw provider bytes, so its raw
digest can be identity-bound but not independently re-read there. QF-3 cache
loading verifies those bytes against `raw_sha256`; complete split and dividend
provenance likewise remains a schema-v4 ingestion/cache guarantee rather than
something inferable from OHLCV alone.

The strategy implementation version is also persisted directly on the result
and every trade. The identity excludes object addresses, optional initiation
time, and export time. Provider retrieval time is included indirectly through
the immutable QF-3 dataset ID.
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
  dividend_cashflows.csv
  split_adjustments.csv
  benchmark_dividend_cashflows.csv
  benchmark_split_adjustments.csv
```

The manifest contains provenance, strategy and backtest configuration,
performance, separate strategy and benchmark dividend-accounting summaries,
record counts, warnings, and limitations. It therefore preserves the selected
policy, return basis, credited/ignored disclosures, and action snapshot even
when the price-only cashflow CSV is empty. CSV field and row order is stable;
nested values use canonical
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
digests, missing-session, action-session, complete action metadata, and
corporate-action snapshot identity, and records its own fingerprint of the
validated bars. It accepts only schema-version-4 `unadjusted` datasets with
complete explicit actions, raw OHLCV basis, and supported dividend/split policies. It
executes dividend- and split-bearing ranges using those records. Legacy schemas,
incomplete or missing actions, identity-inconsistent copies, unknown semantics,
and mixed price bases are rejected before strategy, benchmark, or metrics.

QF-5 rejects `split_adjusted` and `split_and_dividend_adjusted` datasets. Using
adjusted execution prices with cash-dividend accounting could double count total
return; adjusted historical share units are also inconsistent with point-in-time
whole-share fills and per-share costs. QF-5 never infers an action from a price
gap and never substitutes Tiingo adjusted fields.

This implementation is intentionally limited to one stock or ETF, long-only,
unlevered whole shares, next-open market orders, full fills, and discrete state
transitions. Cash-dividend mode credits on ex-date rather than payment date and has no
DRIP, withholding/tax model, cash-in-lieu, volume or order-book model, partial
fills, intraday action sequencing, multi-asset allocation, shorting,
derivatives, brokerage, event-driven engine, or forced liquidation. QF-5 itself does not optimize;
QF-6's `quantforge.optimization` package coordinates repeated unchanged QF-5
runs and reads this typed performance model. See `docs/optimization.md`.
