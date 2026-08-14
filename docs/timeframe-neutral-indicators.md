# Timeframe-neutral indicators

QF-22 generalizes the existing QF-4 SMA, Wilder RSI, Wilder ATR, and Wilder
directional-movement/ADX formulas for canonical intraday, exchange-session
daily, and exchange-week bars. QF-23 adds EMA and QF-24 adds Bollinger Bands
through the same contract. The indicator layer consumes the leakage-safe
QF-20/QF-21 context boundary and does not retrieve, aggregate, or align market
data itself.

QF-35 adds a pluggable standard-indicator backend below the same QF-22 binding.
QF-36 exposes the existing SMA, EMA, Wilder RSI, and Wilder ATR definitions
through that backend while preserving their historical native configurations.
QF-37 applies the same contract to directional movement/ADX and Bollinger
Bands, including generic named normalization for multiple backend outputs.

## Standard-indicator backend boundary

`StandardIndicatorDefinition` contains the normalized indicator name,
canonical input fields, normalized parameter names/values, and normalized
output names. It contains no TA-Lib or native implementation class. An
`IndicatorComputationRequest` pairs that definition with canonical bars, and an
`IndicatorBackend` returns an `IndicatorComputationResult` containing aligned
QuantForge fields plus normalized request and backend metadata.

`IndicatorBackendRegistry` resolves stable identifiers:

- `native_v1` maps SMA, EMA, Wilder RSI, Wilder ATR, directional movement/ADX,
  and Bollinger Bands to their historical QuantForge Decimal implementations;
- `talib_v1` maps those same definitions to TA-Lib `SMA`, `EMA`, `RSI`, `ATR`,
  `PLUS_DI`, `MINUS_DI`, `ADX`, and `BBANDS`.

Backend selection occurs when one of those backend-neutral indicators is
configured. The QF-22 `ConfiguredTimeframeIndicator` remains above it and
continues to own the source timeframe, completed/developing policy,
dataset-family lineage, causal bar selection, and outer configuration identity.
Prediction, signal-feature, data, strategy, and backtesting packages consume
`IndicatorOutput` or `TimeframeIndicatorOutput`; they neither import TA-Lib nor
receive TA-Lib arrays.

A future library integrates by implementing one backend adapter and registering
its mappings for normalized definitions. It does not require classes such as
`FutureEMA` or `TalibEMA`. Dynamic third-party discovery is deliberately out of
scope; registries are assembled explicitly.

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

### EMA backend normalization

Both backends consume the same `ExponentialMovingAverage` and
`ExponentialMovingAverageParameters` public contracts. The normalized input is
the configured `MarketField`; normalized parameters are `period` and
`source_field`; the normalized output is `exponential_moving_average`.

`native_v1` retains the formula and 34-digit Decimal policy above, including
restarting its seed after a missing observation. `talib_v1` translates the
selected Decimal input series to a float64 array, translates `period` to
TA-Lib's `timeperiod`, calls `EMA`, maps its single array back to
`exponential_moving_average`, converts finite values with `Decimal(str(value))`,
and converts TA-Lib `NaN` outputs to `None`. Non-finite canonical observations
are passed as unavailable `NaN`, never as infinities. TA-Lib's own missing-gap
propagation and float64 rounding therefore apply only when `talib_v1` is
selected explicitly; `native_v1` values are not changed.

The pinned TA-Lib 0.7.1 `EMA` accepts periods from `1` through `100000`.
`talib_v1` validates that backend-specific range while the indicator is
configured, so a native-valid larger period cannot produce an identity that
will fail only when calculation begins. This restriction does not narrow
`native_v1`; its historical positive-integer parameter contract is unchanged.

The `talib_v1` adapter fails closed unless TA-Lib uses default compatibility
and a zero `EMA` unstable period. This prevents mutable TA-Lib process-global
settings from silently changing a configuration's result.

TA-Lib is pinned to `0.7.1`. Explicit-backend serialization records the backend
id, backend-contract version, Python-wrapper name and exact version, native
TA-Lib C runtime name and exact version, and mapped function name. Both library
versions participate in the base indicator identity and the outer timeframe-
bound identity. `TimeframeIndicatorOutput` also exposes the resolved
`backend_identity` metadata.

For compatibility, constructing EMA without a backend retains the exact QF-23
configuration shape and identity while resolving execution to `native_v1`.
`ExponentialMovingAverage.from_configuration()` treats a historical mapping
without `backend` the same way. An explicitly selected backend uses the new
versioned configuration shape; deserialization fails if its recorded backend
or library identity differs from the installed resolver.

## TA-Lib mappings and migration

QF-36 applies the same normalization contract to the previously existing core
standard indicators:

| QuantForge definition | Normalized inputs | Parameters | Output | TA-Lib mapping |
| --- | --- | --- | --- | --- |
| `simple_moving_average` | configured market field | `window`, `source_field` | `simple_moving_average` | `SMA(real, timeperiod=window)` |
| `exponential_moving_average` | configured market field | `period`, `source_field` | `exponential_moving_average` | `EMA(real, timeperiod=period)` |
| `wilder_relative_strength_index` | `close` | `period` | `wilder_rsi` | `RSI(close, timeperiod=period)` |
| `wilder_average_true_range` | `high`, `low`, `close` | `period` | `wilder_average_true_range` | `ATR(high, low, close, timeperiod=period)` |
| `wilder_directional_movement` | `high`, `low`, `close` | `period` | `positive_directional_indicator`, `negative_directional_indicator`, `average_directional_index` | one request invoking `PLUS_DI`, `MINUS_DI`, and `ADX` once each |
| `bollinger_bands` | configured market field | `period`, `source_field`, `standard_deviation_multiplier` | `bollinger_middle_band`, `bollinger_upper_band`, `bollinger_lower_band`, `bollinger_bandwidth` | `BBANDS(real, timeperiod=period, nbdevup=multiplier, nbdevdn=multiplier)` plus normalized bandwidth |

The adapter converts canonical `Decimal` inputs to float64, converts TA-Lib
`NaN` outputs to aligned `None` values, rejects infinite outputs, and converts
finite outputs with `Decimal(str(value))`. TA-Lib parameter names, arrays, and
function objects remain inside the adapter. `IndicatorComputationResult`
metadata exports the normalized parameters and input/output names together with
the exact backend, mapped function, Python wrapper version, and C runtime
version. The same backend identity and parameters are retained in explicit
indicator configurations embedded by study manifests and timeframe-bound
configurations.

For functions returning tuples, the adapter first assigns backend-local names
to every returned array and then maps those names to QuantForge output names.
Downstream consumers never depend on TA-Lib tuple positions. Bollinger
bandwidth is derived from the named `upper`, `middle`, and `lower` arrays using
the existing QuantForge zero-width and zero-middle policies. A directional
request exposes all three normalized series without recalculating any one
TA-Lib function.

TA-Lib 0.7.1 accepts periods through `100000`. SMA, EMA, and ATR accept a
minimum period of `1`; RSI accepts a minimum period of `2`. These
backend-specific limits are checked during explicit `talib_v1` configuration.
Directional movement and Bollinger Bands accept a minimum period of `2` in
TA-Lib 0.7.1; the historical native implementations still accept period `1`.
The broader historical native contracts are unchanged. The adapter requires
default TA-Lib compatibility and zero unstable periods for EMA, RSI, ATR,
`PLUS_DI`, `MINUS_DI`, and `ADX`. SMA and BBANDS have no TA-Lib unstable-period
setting.

### Numerical and unavailable-region differences

The two backends intentionally do not promise bit-for-bit equality:

- `native_v1` uses the serialized 34-digit Decimal policy; `talib_v1` uses
  TA-Lib float64 arithmetic, so finite values can differ in their final digits.
- SMA has the same `window - 1` leading unavailable rows. The native rolling
  window resumes after a non-finite observation leaves the window, while
  pinned TA-Lib can continue returning `NaN` after a gap. Neither path fills or
  backfills values.
- EMA differences and missing-gap behavior are described in the EMA section
  above.
- Wilder RSI and TA-Lib RSI both first produce a normal period-`N` value on bar
  `N + 1`, but a completely flat initialized series is defined as `50` by the
  historical native implementation and `0` by TA-Lib 0.7.1. Ordinary finite
  fixtures can also differ by float64 rounding.
- Wilder ATR and TA-Lib ATR both first produce a period-`N` value on bar
  `N + 1`; representative finite fixtures agree within float64 precision rather
  than exact Decimal equality.
- Native and TA-Lib +DI/-DI both first become available on bar `N + 1`, and ADX
  first becomes available on bar `2N`, but TA-Lib's directional initialization
  can produce materially different early values from the historical native
  Wilder smoothing. Those values are backend semantics, not normalized toward
  equality.
- Native and TA-Lib Bollinger Bands have the same `N - 1` unavailable leading
  rows for supported periods. Native uses exact rolling moments followed by
  34-digit Decimal arithmetic; TA-Lib uses float64, so band and derived
  bandwidth values can differ in their final digits. TA-Lib does not support
  the native period-`1` configuration.

Constructing SMA, EMA, Wilder RSI, Wilder ATR, directional movement, or
Bollinger Bands without `backend_id` remains the compatibility path. It resolves
execution to `native_v1` but emits the exact pre-backend configuration shape and
deterministic ID. Each class's `from_configuration()` method treats a historical
mapping without `backend` as that legacy path. A newly created study must select
`backend_id="talib_v1"` explicitly. Explicit native and TA-Lib configurations
use contract version `2`, have distinct deterministic IDs, and fail
deserialization if the installed backend or library identity has drifted.

## Bollinger Bands

`BollingerBandsParameters` contains a positive integer `period`, a finite
positive `Decimal` `standard_deviation_multiplier` (default `2`), and one
canonical OHLCV `source_field` (default `close`). The QF-22 binding supplies the
typed source timeframe and context completion policy. The bound configuration
identity therefore includes period, multiplier, field, complete timeframe
semantics, completed-only or developing-as-of policy, and dataset-family
provenance.

Multiplier validation also bounds configuration-serialization resources before
fixed-point formatting: the coefficient may contain at most 68 digits and its
raw fixed-point representation may contain at most 256 characters. Larger
finite values raise a controlled invalid-parameters error instead of attempting
an exponent-sized string allocation. These limits are serialized in the
indicator configuration.

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

The first complete window establishes exact rational sums of observations and
squared observations. Each later contiguous finite bar removes the expired
observation and adds the new observation to both moments without rescanning the
window. Population variance is derived exactly as
`sum_of_squares / N - (sum / N) ** 2`, then the mean and variance cross the
serialized 34-digit `Decimal` boundary once before square root and band
arithmetic. Exact moment accumulation prevents scale-transition cancellation
from making an output depend on bars that are no longer in the active window.
A source value crosses a serialized resource boundary before exact conversion:
its stored and adjusted exponents must remain within `-999999` through `999999`,
its coefficient may contain at most 68 digits, and either integer component of
its exact fraction may require at most 2048 decimal digits (4096 after
squaring). Values outside those bounds raise a controlled indicator-calculation
error before a power-of-ten integer can be materialized. The coefficient
allowance is twice the 34-digit output precision so exact moments retain
lower-order source detail across large scale transitions. The separate 2048
digit allocation ceiling prevents the intentionally broad Decimal exponent
range from becoming an equally broad exact-integer resource allowance. Initial
window moments are accumulated as a stream, so the calculation does not retain
one exact fraction allocation per observation.
A missing-value gap invalidates the moments; the first complete finite window
after the gap establishes them again. Thus mature contiguous calculation uses
`O(1)` rolling operations in the period length rather than rescanning every
overlapping window.

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
- explicit standard-indicator backend id, exact wrapper/runtime library
  versions, and mapped function through the base indicator configuration, when
  present.

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
- resolved backend identity for backend-neutral standard indicators.

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

QF-22 through QF-24 and QF-35 through QF-37 do not add MACD, stochastic, volume
formulas, Bollinger-based prediction or squeeze-classification rules,
prediction integration, feature export, or strategy/backtest multi-timeframe
integration. Those remain sibling-ticket concerns under QF-12.
