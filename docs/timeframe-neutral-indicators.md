# Timeframe-neutral indicators

QF-22 generalizes the existing QF-4 SMA, Wilder RSI, Wilder ATR, and Wilder
directional-movement/ADX formulas for canonical intraday, exchange-session
daily, and exchange-week bars. QF-23 adds EMA and QF-24 adds Bollinger Bands
through the same contract. The indicator layer consumes the leakage-safe
QF-20/QF-21 context boundary and does not retrieve, aggregate, or align market
data itself.

## Contracts

The existing `Indicator.calculate(MarketDataset)` API remains the compatibility
path for QF-3 daily datasets. Its numerical formulas, row alignment, missing
values, and decimal arithmetic are unchanged.

`TimeframeNeutralIndicator` adds two declarations used by multi-timeframe
research:

- `calculate_bar_fields()` applies the same formula to canonical OHLCV bars;
- `developing_bar_support` declares whether a causal developing bar is allowed.

The built-in indicators support QF-21 developing bars because each calculation
uses only its current and preceding input observations. Support is a capability,
not an opt-in: a developing value is visible only when the surrounding context
was explicitly constructed with `DEVELOPING_BAR_AS_OF`.

`evaluate_indicator(indicator, context, timeframe)` is the generic convenience
boundary. `bind_indicator()` returns a `ConfiguredTimeframeIndicator` when a
caller needs to inspect or retain the source-bound configuration before
calculation.

## Bar-count semantics

Every period, window, and warm-up is measured in input observations/bars:

| Indicator | Parameter | First available output |
| --- | --- | --- |
| SMA | `window=N` | bar `N` |
| EMA | `period=N` | bar `N` |
| Bollinger Bands | `period=N` | bar `N` |
| Wilder RSI | `period=N` | bar `N + 1` |
| Wilder ATR | `period=N` | bar `N + 1` |
| +DI / -DI | `period=N` | bar `N + 1` |
| ADX | `period=N` | bar `2N` |

For example, a 20-period SMA consumes 20 five-minute bars on a 5m source and 20
trading weeks on a weekly source. No indicator converts the number 20 into a
duration or session count.

Warm-up rows remain present and use `None`. There is no backfill, centered
window, or partial-window result.

## Exponential moving average

`ExponentialMovingAverageParameters` contains the positive integer `period` and
one canonical OHLCV `source_field`, defaulting to `close`. The source timeframe
and completion policy are supplied by the typed QF-22 binding rather than
duplicated in the formula parameters. Consequently the bound configuration and
identity include period, field, complete timeframe semantics, completion or
developing-bar policy, and dataset-family provenance.

For period `N`, the smoothing factor is:

```text
alpha = 2 / (N + 1)
```

The first EMA is the simple arithmetic mean of the first `N` consecutive finite
source observations. It appears on bar `N`; preceding aligned rows are `None`.
Each later value is calculated recursively under the indicator's serialized
34-digit `Decimal`, round-half-even arithmetic policy:

```text
EMA[current] = alpha * source[current] + (1 - alpha) * EMA[previous]
```

The implementation evaluates the algebraically equivalent rational form
`((N - 1) * EMA[previous] + 2 * source[current]) / (N + 1)`. This avoids
rounding a repeating `alpha` before applying the recurrence while preserving
the standard formula.

A non-finite source observation is treated as unavailable: that row is `None`,
the prior EMA is not carried forward, and initialization restarts. Another
value appears only after `N` new consecutive finite observations. This policy
does not interpolate, forward-fill, or bridge a market-data gap.

EMA declares `DEVELOPING_AS_OF` support. In developing context mode the current
input is the structurally distinct as-of bar already reconstructed by QF-21
from completed lower-timeframe constituents. The recurrence never sees or
infers the eventual completed close.

## Bollinger Bands

`BollingerBandsParameters` contains a positive integer `period`, a finite
positive `Decimal` `standard_deviation_multiplier` (default `2`), and one
canonical OHLCV `source_field` (default `close`). The QF-22 binding supplies the
typed source timeframe and context completion policy. The bound configuration
identity therefore includes period, multiplier, field, complete timeframe
semantics, completed-only or developing-as-of policy, and dataset-family
provenance.

For each full trailing window of `N` consecutive finite observations, the
middle band is the arithmetic mean. The standard deviation is the population
standard deviation (`ddof=0`), not the sample estimate:

```text
middle = sum(window) / N
population_variance = sum((observation - middle) ** 2) / N
population_standard_deviation = sqrt(population_variance)
upper = middle + multiplier * population_standard_deviation
lower = middle - multiplier * population_standard_deviation
bandwidth = (upper - lower) / middle
```

The first `N - 1` aligned rows are `None` for all four outputs. A non-finite
source observation makes every full window containing it unavailable; values
are not filled, backfilled, or carried across the gap. Calculations use the
indicator's serialized 34-digit `Decimal`, round-half-even arithmetic policy,
including square root.

The first complete window establishes its sum and centered sum of squares.
Each later contiguous finite bar removes the expired observation and adds the
new observation with a constant-time sliding-variance update. A missing-value
gap invalidates the statistics; the first complete finite window after the gap
re-establishes them once. To bound fixed-precision cancellation residue, the
calculation maintains monotonic rolling minimum and maximum queues. A centered
sum outside the range-based bound `N * (maximum - minimum) ** 2` is impossible
for a valid window and triggers an immediate direct rebuild. A zero centered
sum with unequal minimum and maximum is likewise treated as cancellation and
rebuilt. The calculation also re-establishes statistics after every `N` rolling
updates and resets exact constant windows in `O(1)`. Monotonic range maintenance
is amortized `O(1)`, and the `O(N)` periodic rebuilds occur at least `N` bars
apart, so mature contiguous calculation remains amortized `O(1)` per bar
instead of rescanning every overlapping period.

Constant-price windows have zero population deviation, equal middle/upper/lower
bands, and bandwidth `0`. More generally, any zero-width band has bandwidth
`0`, including a zero-price window. If a mathematically possible input has a
zero middle but nonzero width, bandwidth is `None` because the ratio is
undefined; the three price bands remain available. Percent-B is not emitted:
QF-24 makes it optional, and QuantForge does not yet have a project-wide
zero-width percent-B convention.

Bollinger Bands declare `DEVELOPING_AS_OF` support. A developing result uses
only the causal as-of bar exposed by QF-21; appending later constituents or the
eventual completed candle cannot revise the earlier historical prefix.

## Source binding and identity

A configured timeframe indicator binds:

- the complete base indicator configuration and identity;
- the complete canonical source `Timeframe` and its configuration identity;
- required source fields, including an indicator's selected `source_field`;
- the context completion policy;
- the indicator's developing-bar declaration;
- the bar observation unit and warm-up count;
- the QF-14 `DatasetFamilyReference` for the selected source series.

The dataset-family reference retains dataset ID, canonical source snapshot,
timeframe identity, and family identity. The family identity already binds the
aggregation policy, feed scope, adjustment basis, source interval, session
policy, and provider provenance. The configured indicator therefore changes
identity when its timeframe, completion policy, field, dataset member, source
snapshot, or aggregation family changes.

Binding and calculation both resolve the exact requested timeframe through the
context. Calculation fails when the timeframe is undeclared, unavailable, or
has a different dataset-family reference. A daily series therefore cannot be
passed silently to an indicator bound to 4h bars.

## Output alignment and metadata

`TimeframeIndicatorOutput` retains one row for every visible input bar and
includes:

- source timeframe and source fields;
- completion policy and developing-bar support;
- warm-up count in bars;
- dataset-family aggregation provenance;
- exact bar IDs, bar-end timestamps, and completion states;
- immutable named indicator value fields.

`to_rows()` uses `bar_id`, `bar_end_timestamp`, and `completion` as the alignment
columns. This permits multiple intraday observations from one exchange session
without ambiguous or duplicate daily keys.

## Causality and developing bars

The evaluator only sees bars already exposed by `MultiTimeframeContext`. In the
default completed-only mode, those bars have terminal completion states and end
at or before the decision timestamp. Appending later bars cannot alter the
historical output prefix because all formulas are trailing and recursive only
from earlier values.

In developing mode, QF-21 may append one reconstructed contextual bar. Its OHLCV
contains only completed lower-timeframe constituents whose ends are at or before
the context `as_of`. The output records that row as `developing`; it is never
presented as the eventual completed candle. An indicator declaring
`COMPLETED_ONLY` rejects that input.

## Deliberate limits

QF-22 through QF-24 do not add MACD, stochastic, volume formulas,
Bollinger-based prediction or squeeze-classification rules, prediction
integration, feature export, or strategy/backtest multi-timeframe integration.
Those are sibling-ticket concerns under QF-12.
