# Timeframe-neutral indicators

QF-22 generalizes the existing QF-4 SMA, Wilder RSI, Wilder ATR, and Wilder
directional-movement/ADX formulas for canonical intraday, exchange-session
daily, and exchange-week bars. It consumes the leakage-safe QF-20/QF-21 context
boundary and does not retrieve, aggregate, or align market data itself.

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
| Wilder RSI | `period=N` | bar `N + 1` |
| Wilder ATR | `period=N` | bar `N + 1` |
| +DI / -DI | `period=N` | bar `N + 1` |
| ADX | `period=N` | bar `2N` |

For example, a 20-period SMA consumes 20 five-minute bars on a 5m source and 20
trading weeks on a weekly source. No indicator converts the number 20 into a
duration or session count.

Warm-up rows remain present and use `None`. There is no backfill, centered
window, or partial-window result.

## Source binding and identity

A configured timeframe indicator binds:

- the complete base indicator configuration and identity;
- the complete canonical source `Timeframe` and its configuration identity;
- required source fields, including an SMA's selected `source_field`;
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

QF-22 does not add EMA, Bollinger Bands, MACD, stochastic, volume formulas,
prediction integration, feature export, or strategy/backtest multi-timeframe
integration. Those are sibling-ticket concerns under QF-12.
