# Indicator and strategy contracts

QF-4 defines reusable research contracts through the point where a strategy
expresses position-sizing intent. It does not simulate orders, fills, prices,
cash, commissions, slippage, positions, or profit and loss.

```text
QF-3 canonical market data
        |
        v
aligned causal indicators
        |
        v
strategy target-state decisions
        |
        v
normalized target-weight intent
        |
        v
future execution and portfolio components
```

## Indicator contract

`quantforge.indicators.Indicator` is a structural protocol. An implementation
exposes a stable name, immutable typed parameters, required `MarketField`
values, observations required for warm-up, named outputs, missing-value policy,
a stable primitive configuration and its SHA-256 identity, and `calculate()`.
It consumes QF-3 `MarketDataset` objects, not provider responses or SDK models.

`IndicatorOutput` stores the input session dates unchanged and one or more
immutable `IndicatorFieldOutput` series of exactly the same length. `None` is
the explicit unavailable value; every available output must be a finite
`Decimal`. The name `session_date` is reserved for the tabular alignment key and
cannot be used as an indicator output field. Outputs never omit warm-up rows.
The reference simple moving average requires `N` observations; therefore its
first `N - 1` rows are `None`, and its first available result uses exactly the
first `N` source observations. A non-finite/missing source observation makes
every full window containing it unavailable. Values are never filled or
backfilled.

`SimpleMovingAverage` uses `Decimal`, the current observation, and only the
prior `N - 1` observations. Sums and division run in a private local decimal
context with 34 significant digits and `ROUND_HALF_EVEN`; an exact result that
exceeds that precision is rounded by this fixed policy. The complete context
also fixes `Emin=-999999`, `Emax=999999`, `capitals=1`, `clamp=0`, empty initial
flags, and traps only `InvalidOperation`, `DivisionByZero`, and `Overflow`.
Neither the caller's ambient context nor mutable `decimal.DefaultContext` is
used or modified. The primitive indicator configuration records the complete
arithmetic policy alongside the component and contract versions, parameters,
required fields, warm-up observations, output fields, and missing
representation.

```python
from quantforge.indicators import (
    MarketField,
    SimpleMovingAverage,
    SimpleMovingAverageParameters,
)

indicator = SimpleMovingAverage(
    SimpleMovingAverageParameters(window=20, source_field=MarketField.CLOSE)
)
output = indicator.calculate(dataset)
rows = output.to_rows()  # one row for every input session
```

## Strategy contract

`quantforge.strategies.Strategy` is a structural protocol. A strategy declares:

- its stable identifier and immutable parameters;
- an explicit implementation version that changes when decision-relevant
  strategy code changes;
- required market fields and owned indicator definitions;
- the maximum observations required by its indicators;
- asset assumptions and a timing convention;
- a position-sizing policy;
- primitive configuration metadata and a stable configuration identity; and
- a method that returns `StrategyOutput`.

The public contract deliberately makes the strategy responsible for coordinating
its indicators. `run_strategy(strategy, dataset)` only checks input fields and
chronology, invokes the contract, and validates identity, parameter snapshots,
ordering, uniqueness, provenance, and timing. It has no moving-average branches
and is not a backtest engine.

Immutable parameter objects expose `to_primitive()`. Decimal weights are encoded
exactly as context-independent canonical decimal strings, enums as their stable
string values, and optional values as JSON `null`. Canonicalization removes
representation-only fractional zeros without performing decimal arithmetic or
rounding. Decision snapshots must match parameter names, primitive types, and
values exactly; for example, JSON `true` is not interchangeable with integer
`1`. A component configuration is canonical-JSON encoded and SHA-256 hashed, so
equivalent values such as `Decimal("0.50")` and `Decimal("0.5")` receive the same
identity regardless of the caller's active decimal context.

The generic runner recomputes that identity from the concrete configuration
captured before generation, rejects a stale or hard-coded `configuration_id`,
and requires the same identity on the strategy output and every decision. It
also verifies that the strategy configuration remains unchanged through
generation.

The implementation version is a required top-level field in the strategy's
primitive configuration. It therefore participates in the configuration
identity and QF-5 run identity even when parameters are unchanged. The reference
moving-average crossover strategy currently declares implementation version
`1`.

QF-5 canonicalizes the complete strategy configuration into an immutable
snapshot before calculating its run identity and independently requires the
declared configuration ID to match that exact snapshot. A custom strategy may
return an ordinary primitive dictionary, but mutating that dictionary after the
run does not alter the result manifest or exported provenance.

## Decision schema

Every `StrategyDecision` contains:

| Field | Meaning |
| --- | --- |
| `canonical_symbol` | QF-3 normalized symbol |
| `signal_session` | completed daily session whose close made the signal known |
| `earliest_executable_session` | first later exchange session resolved from the QF-3 calendar, or `None` |
| `execution_timing` | `next_session_after_close` |
| `execution_session_status` | `pending` or `unresolved`; never a claim that a fill occurred |
| `target_position` | long-only `long` or `flat` desired state |
| `target_weight` | normalized requested allocation in `[0, 1]` |
| `strategy_id` / `strategy_configuration_id` | reproducible strategy identity |
| `strategy_parameters` | stable primitive parameter snapshot |
| `reason` | optional originating rule |
| `indicator_values` | finite values retained for decision auditability |

`StrategyOutput` also retains the input dataset ID, schema version, adjustment
mode, exchange calendar, and corporate-action snapshot ID. Strategies do not
inspect or apply actions; QF-5 owns that accounting. `to_rows()` supports tabular/vectorized consumers;
iteration over the output supports a future chronological event-driven consumer.
Both see the same immutable decisions. There are no callbacks, engine objects,
provider types, fill prices, or quantities in the schema.

## Daily timing and causality

A daily bar is complete after its exchange session closes. Indicators for
session `t` may use that completed bar. A decision has `signal_session=t` and
cannot execute at that close. The reference timing rule resolves the first
exchange session after `t` from the QF-3 calendar, skipping weekends and market
holidays. Resolution uses the calendar's direct successor operation, so it does
not query past the calendar's supported range. A signal on the dataset's final
bar remains `pending`; a resolved calendar date is eligibility metadata, not
evidence of an order or fill. If a calendar cannot resolve the next session
safely, the reference strategy records the signal as `unresolved` rather than
inventing a date or discarding the signal. The generic runner verifies calendar
resolution independently: it requires the resolved date and `pending` status
whenever resolution succeeds and permits `unresolved` only when it fails.

Rolling windows are trailing and never centered. Indicator rows are never
backfilled. A crossover requires valid fast and slow averages on both the
current and immediately prior session. Tests calculate through a cutoff, append
sentinel future bars, recalculate, and require all historical indicator values
and decisions to remain identical.

QF-3 adjustment metadata is retained by reference. A direct QF-4 invocation uses
exactly the bars supplied by its caller. For QF-5's raw corporate-action path,
the backtester supplies an ephemeral causal split-normalized feature view while
retaining the immutable raw QF-3 dataset reference: OHLC is multiplied by the
cumulative split factor and volume is divided by it beginning on each effective
session. Later splits never revise earlier feature rows. Strategies request
exactly their declared source field and do not inspect or apply corporate-action
records themselves. QF-5 continues to use only the original raw bars for fills,
marks, and portfolio accounting, and exports the feature-basis convention in its
split policy.

## Position-sizing boundary

`PositionSizingPolicy` converts `PositionIntent` plus an optional `SizingContext`
into `TargetWeightIntent`. Future policies may declare a need for available
equity, current position, reference price, or risk budget. The reference
`TargetWeightSizingPolicy` needs none: `long` requests the configured weight and
`flat` requests zero. It never calculates shares, reserves cash, applies
leverage, rounds lots, or assumes a fill.

## Moving-average crossover reference

`MovingAverageCrossoverParameters` declares:

- positive integer `fast_window` and `slow_window`, with fast strictly smaller;
- `source_field`, defaulting exactly to normalized `close`; and
- `target_long_weight` in `(0, 1]`, defaulting to full allocation (`1`).

The strategy begins conceptually flat. It emits `long` once when the fast SMA
moves from less-than-or-equal to strictly above the slow SMA. It emits `flat`
once when the fast SMA moves from greater-than-or-equal to strictly below while
the current target is long. An equality row itself never emits a decision;
leaving equality for a strict relation may confirm a crossover. Remaining above
or below does not repeat a decision. A first valid pair cannot signal without a
valid prior pair. Total warm-up is the slow window's observation requirement.

```python
from decimal import Decimal

from quantforge.strategies import (
    MovingAverageCrossoverParameters,
    MovingAverageCrossoverStrategy,
    run_strategy,
)

strategy = MovingAverageCrossoverStrategy(
    MovingAverageCrossoverParameters(
        fast_window=20,
        slow_window=50,
        target_long_weight=Decimal("0.75"),
    )
)
decisions = run_strategy(strategy, dataset)
```

## Adding an indicator

Create an immutable parameter record with `to_primitive()`, then implement the
`Indicator` protocol. Reuse input and alignment validation, preserve every
session, and make unavailable values explicit. For example, the calculation
shape of a one-period close difference is:

```python
from dataclasses import dataclass
from decimal import Decimal

from quantforge.configuration import PrimitiveMapping
from quantforge.indicators import IndicatorFieldOutput, IndicatorOutput, MarketField
from quantforge.indicators.base import (
    validate_indicator_alignment,
    validate_market_input,
)


@dataclass(frozen=True, slots=True)
class CloseDifferenceParameters:
    source_field: MarketField = MarketField.CLOSE

    def to_primitive(self) -> PrimitiveMapping:
        return {"source_field": self.source_field.value}


# Inside CloseDifference.calculate(dataset):
validate_market_input(dataset, frozenset((MarketField.CLOSE,)))
closes = tuple(bar.close for bar in dataset.bars)
values = (None,) + tuple(
    current - previous for previous, current in zip(closes, closes[1:])
)
result = IndicatorOutput(
    "close_difference",
    configuration_id,
    tuple(bar.session_date for bar in dataset.bars),
    (IndicatorFieldOutput("close_difference", values),),
)
validate_indicator_alignment(dataset, result)
```

The implementation must also supply the protocol's metadata properties and a
stable configuration identity, following `SimpleMovingAverage`.

## Adding a strategy

Create a frozen parameter record, define owned indicators, and structurally
implement `Strategy`. `generate()` calculates those indicators and emits only
engine-neutral state changes:

```python
class CloseAboveAverageStrategy:
    name = "close_above_average"
    implementation_version = "1"
    timing = ExecutionTiming.NEXT_SESSION_AFTER_CLOSE

    # Expose parameters, required_fields, required_indicators,
    # warm_up_observations, sizing_policy, asset_assumptions,
    # configuration(), and configuration_id as required by Strategy.

    def generate(self, dataset: MarketDataset) -> StrategyOutput:
        average = self.required_indicators[0].calculate(dataset)
        # Compare only values at the current/prior session, emit target-state
        # changes, resolve next-session eligibility, and retain audit values.
        return StrategyOutput(...)


output = run_strategy(CloseAboveAverageStrategy(...), dataset)
```

New strategy code belongs in `quantforge.strategies`; it must not change the
generic runner, execution, portfolio, provider, or reporting components.
