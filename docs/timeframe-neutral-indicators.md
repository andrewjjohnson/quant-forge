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
QF-38 adds descriptive parity tooling over that contract; it does not add
backend-specific indicator classes or alter either backend's mathematics.
QF-25 adds a backend-neutral MACD definition whose standard implementation is
the pinned `talib_v1` adapter; it deliberately adds no native MACD formula.
QF-26 adds a backend-neutral slow stochastic oscillator through the same
adapter, with fixed simple-moving-average smoothing and no native stochastic
formula. QF-27 adds native typed volume moving-average and relative-volume
formulas with explicit feed scope and denominator policies.

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
  `PLUS_DI`, `MINUS_DI`, `ADX`, and `BBANDS`, and is the standard implementation
  for the MACD and stochastic definitions through TA-Lib `MACD` and `STOCH`.

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

## Backend parity comparisons

`compare_indicator_backends()` runs one `StandardIndicatorDefinition` against
two explicit backend IDs over the same canonical bars. The comparison binds the
QF-3 bar fingerprint and complete canonical timeframe configuration, then
records normalized parameters, output names, tolerances, and exact backend
wrapper/runtime versions. `compare_standard_indicator_backends()` exposes the
same operation for an already constructed `IndicatorComparisonSource`.

The QF-3 `MarketDataset` convenience path accepts only the matching canonical
one-session, regular-hours, bar-start, completed-only timeframe. QF-3 daily
values cannot be reinterpreted as extended-hours, end-labeled, or developing
bars merely by attaching a different timeframe identity. Consumers of another
canonical bar source must construct an `IndicatorComparisonSource` with that
source's own explicit provenance and timeframe.

Each normalized output name is compared independently. Backend tuple positions
are never used for matching. The field report includes each backend's first
valid timestamp, leading unavailable count, valid count, overlapping valid
count, and timestamps available on only one side. Those availability records
are explicitly excluded from numerical-divergence counts, so different
lookbacks are visible without being mislabeled as formula errors.

For overlapping finite values, absolute difference is `abs(a - b)`. Symmetric
relative difference is `abs(a - b) / max(abs(a), abs(b))`; it is unavailable
when both values are zero. A row diverges when:

```text
absolute_difference > absolute_tolerance
                      + relative_tolerance * max(abs(a), abs(b))
```

Mean and median statistics use a fixed 34-digit Decimal, round-half-even
policy. Appending future bars can add later rows but cannot revise an already
reported historical pair. `export_indicator_backend_comparison()` writes an
immutable directory containing `comparison.json`, `field_summary.csv`,
`divergences.csv`, and `summary.txt`; byte-for-byte validation is available
through `validate_indicator_backend_comparison_export()`.

The artifacts are comparison evidence only. Backend A/B ordering is not a
ranking, no backend is promoted, and legacy omitted-backend configurations
continue to resolve through the unchanged `native_v1` compatibility path.

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
| MACD | `slow_period=S`, `signal_period=G` | bar `S + G - 1` |
| Stochastic | `k_period=K`, `k_smoothing_period=S`, `d_period=D` | bar `K + S + D - 2` |
| Volume moving average | `lookback=N` | bar `N` |
| Relative volume, current-inclusive | `lookback=N` | bar `N` |
| Relative volume, prior-bars-only | `lookback=N` | bar `N + 1` |

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
| `moving_average_convergence_divergence` | configured market field | `fast_period`, `slow_period`, `signal_period`, `source_field` | `macd`, `signal`, `histogram` | `MACD(real, fastperiod=fast_period, slowperiod=slow_period, signalperiod=signal_period)` |
| `stochastic_oscillator` | `high`, `low`, `close` | `k_period`, `k_smoothing_period`, `d_period`, `smoothing_method` | `k`, `d` | `STOCH(high, low, close, fastk_period=k_period, slowk_period=k_smoothing_period, slowk_matype=0, slowd_period=d_period, slowd_matype=0)` |

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
MACD accepts fast and slow periods from `2` through `100000` and a signal period
from `1` through `100000`; QuantForge additionally requires the normalized fast
period to be less than the slow period. Stochastic accepts each of its three
periods from `1` through `100000`; the normalized `simple_moving_average`
smoothing method maps to TA-Lib moving-average type `0` for both smoothing
stages.
The broader historical native contracts are unchanged. The adapter requires
default TA-Lib compatibility and zero unstable periods for EMA, RSI, ATR,
`PLUS_DI`, `MINUS_DI`, and `ADX`. MACD also requires a zero EMA unstable period.
SMA and BBANDS have no TA-Lib unstable-period setting.

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
- MACD has no native comparison. Its first normalized value appears after
  `slow_period + signal_period - 2` leading unavailable rows, following pinned
  TA-Lib's lookback. TA-Lib float64 and missing-gap behavior are retained.
- Stochastic has no native comparison. Its `k_period + k_smoothing_period +
  d_period - 3` leading rows are unavailable. A missing high, low, or close is
  passed to TA-Lib as `NaN`. QuantForge additionally masks every `k` or `d`
  whose causal dependency window includes that row, even when TA-Lib's rolling
  extrema would otherwise ignore a missing high or low. Any larger unavailable
  region produced by pinned TA-Lib is retained without fill or restart.

Constructing SMA, EMA, Wilder RSI, Wilder ATR, directional movement, or
Bollinger Bands without `backend_id` remains the compatibility path. It resolves
execution to `native_v1` but emits the exact pre-backend configuration shape and
deterministic ID. Each class's `from_configuration()` method treats a historical
mapping without `backend` as that legacy path. A newly created study must select
`backend_id="talib_v1"` explicitly. Explicit native and TA-Lib configurations
use contract version `2`, have distinct deterministic IDs, and fail
deserialization if the installed backend or library identity has drifted.

## Moving average convergence/divergence

`MovingAverageConvergenceDivergenceParameters` contains positive integer
`fast_period`, `slow_period`, and `signal_period` values plus one canonical
`source_field`, defaulting to close. The fast period must be strictly less than
the slow period. Defaults are `12`, `26`, and `9`. The QF-22 binding, rather
than the formula parameter record, supplies the complete source timeframe,
completion policy, and dataset-family lineage. The resulting bound identity
therefore includes all three periods, source field, backend id and exact wrapper
and C runtime versions, timeframe, completion policy, and aggregation lineage.

The public `MovingAverageConvergenceDivergence` class owns only the normalized
definition. It defaults to `talib_v1`; selecting `native_v1` fails through the
standard unsupported-backend domain error because QF-25 adds no QuantForge MACD
or EMA calculation. A future backend can support the same definition by adding
one adapter mapping without adding another public MACD class.

`talib_v1` translates the normalized periods to TA-Lib's `fastperiod`,
`slowperiod`, and `signalperiod` arguments, but those names never appear in the
definition or downstream result. TA-Lib's three tuple positions receive
backend-local names before the adapter emits the stable QuantForge fields
`macd`, `signal`, and `histogram`. To preserve the normalized contract exactly,
`histogram` is subtracted from the already normalized Decimal `macd` and
`signal` values under a fully specified 34-digit, round-half-even Decimal
context with fixed exponent bounds and traps. Thus every available row satisfies
`histogram == macd - signal`, independent of a final float64 rounding difference
in TA-Lib's separately returned histogram array.

For slow period `S` and signal period `G`, TA-Lib's lookback is `S + G - 2`
rows. QuantForge therefore reports `S + G - 1` warm-up observations: every
output is `None` for the leading lookback rows and all three first become
available together on bar `S + G - 1`. No partial-window value is emitted.
Canonical non-finite source observations are passed to TA-Lib as `NaN`, never
as infinity, and unavailable TA-Lib values normalize to `None`. With pinned
TA-Lib 0.7.1, a source gap makes that row and subsequent MACD rows unavailable;
QuantForge does not fill, backfill, carry, or independently restart the EMA
state.

MACD calculation fails closed unless TA-Lib uses default compatibility and a
zero global EMA unstable period, because that process-global setting changes
MACD initialization and lookback. The indicator supports a causal QF-21
developing bar: the backend sees only the already reconstructed as-of source
value supplied by the context, never the eventual completed bar.

## Stochastic oscillator

`StochasticOscillatorParameters` contains positive integer `k_period`,
`k_smoothing_period`, and `d_period` values. Defaults are `5`, `3`, and `3`.
The normalized smoothing method is fixed to `simple_moving_average`; it is
serialized with the periods rather than left as hidden library state. The
QF-22 binding supplies the complete source timeframe, completion policy, and
dataset-family lineage. The bound identity therefore includes all three
periods, the smoothing method, backend id and exact wrapper and C runtime
versions, timeframe, completion policy, and aggregation lineage.

The public `StochasticOscillator` class owns only that normalized definition.
It defaults to `talib_v1`; selecting `native_v1` fails through the standard
unsupported-backend domain error because QF-26 adds no QuantForge stochastic
calculation. A future backend can support the definition by adding one adapter
mapping rather than another public stochastic class.

QF-26 deliberately selects TA-Lib's slow stochastic `STOCH` variant. For each
bar, TA-Lib first calculates raw fast %K over the trailing `k_period` high-low
range. The normalized `k` output is a simple moving average of raw %K over
`k_smoothing_period`; normalized `d` is a simple moving average of `k` over
`d_period`. The adapter fixes both TA-Lib moving-average-type arguments to
simple moving average (`0`). TA-Lib parameter names, moving-average codes,
tuple order, and its backend-local `slowk`/`slowd` names are not exposed in the
definition or result.

For periods `K`, `S`, and `D`, TA-Lib's lookback is `K + S + D - 3` rows.
QuantForge reports `K + S + D - 2` warm-up observations: both outputs are
`None` for the leading lookback rows and first become available together on bar
`K + S + D - 2`. No partial-window value is emitted. Canonical non-finite high,
low, or close observations are passed as `NaN`, never infinity. The dependency
mask is field-specific: `k` uses trailing windows of `K + S - 1` bars for high
and low but only `S` bars for close; `d` uses `K + S + D - 2` bars for high and
low but only `S + D - 1` bars for close. Historical closes do not incorrectly
participate in later raw %K extrema. This closes TA-Lib's missing-extrema
behavior without replacing its stochastic calculation. TA-Lib's own
unavailable outputs also normalize to `None`, and neither path fills or
backfills values.

When the complete `k_period` high-low range is zero, pinned TA-Lib 0.7.1 emits
zero for raw %K and consequently zero for the simple-smoothed `k` and `d` once
their windows are available. QuantForge preserves and documents that backend
result; it does not substitute a division-by-zero formula or another neutral
value. Finite TA-Lib float64 values are converted with `Decimal(str(value))`.
The indicator supports a causal QF-21 developing bar and sees only its
already-reconstructed as-of high, low, and close.

## Volume moving average and relative volume

`VolumeMovingAverageParameters` contains a positive integer `lookback` and a
typed QF-14 `FeedScope`. For a full trailing window of `N` bars:

```text
volume_moving_average[t] = sum(volume[t-N+1:t+1]) / N
```

The current bar is always included. The first `N - 1` rows are `None`. Zero is
a valid source volume and participates in the mean; it is not treated as a
missing bar. A nonfinite volume makes every full window containing it
unavailable. Windows are neither partially calculated nor filled.

`RelativeVolumeParameters` contains the same `lookback` and `feed_scope` plus
one `RelativeVolumeDenominatorPolicy`:

```text
INCLUDE_CURRENT_BAR:
    relative_volume[t] = volume[t] / mean(volume[t-N+1:t+1])

EXCLUDE_CURRENT_BAR:
    relative_volume[t] = volume[t] / mean(volume[t-N:t])
```

The inclusive policy first produces a value on bar `N`; the prior-bars-only
policy needs `N` completed denominator bars plus the numerator and first
produces a value on bar `N + 1`. A nonfinite numerator, nonfinite denominator
window, or exactly zero denominator produces `None`. No infinity or NaN is
emitted. Both formulas use a fixed 34-digit `Decimal`, round-half-even policy.

Feed scope is mandatory rather than inferred. Consolidated, single-venue
(including IEX-only), provider-defined, and explicitly unknown observations
therefore have distinct readable configurations and identities. A relative-
volume denominator is calculated internally from the same source-bar tuple as
its numerator, so it cannot silently draw its numerator and denominator from
different feeds. At QF-22 binding, the declared scope must equal the explicit
scope carried by the selected QF-14 dataset-family reference; missing or
mismatched provenance fails before evaluation. `TimeframeIndicatorOutput`
retains that verified scope. The outer identity also binds the exact QF-14
family, whose identity independently includes feed scope and rejects mixed-feed
multi-timeframe contexts.

Both indicators declare causal developing-bar support. In
`DEVELOPING_BAR_AS_OF` mode, the current developing volume is only the sum of
completed lower-timeframe constituents exposed by QF-21 as of the decision
timestamp. Appending later source bars cannot revise the prior output prefix.

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
timeframe identity, family identity, and explicit feed scope. The family
identity also binds the aggregation policy, feed scope, adjustment basis,
source interval, session policy, and provider provenance. The configured
indicator therefore changes identity when its timeframe, completion policy,
field, dataset member, source snapshot, feed scope, or aggregation family
changes.

The timeframe-indicator configuration contract is version `2` as of QF-27.
Version 2 adds explicit feed scope both to `aggregation_provenance` and to the
source configuration. Callers that must identify or compare a persisted QF-22
through QF-26 version-1 artifact can request
`configuration(contract_version="1")` or
`configuration_id_for_contract("1")`; that compatibility path reproduces the
original shape without either feed-scope field. New artifacts always use
version 2 and must not be mislabeled as version 1.

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
- verified provider-neutral feed scope;
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

QF-22 through QF-27 and QF-35 through QF-38 do not add On-Balance Volume,
volume-profile indicators, prediction filters, MACD crossover/divergence
behavior, Bollinger-based prediction or squeeze-classification rules,
prediction integration, feature export, or strategy/backtest multi-timeframe
integration. Those remain sibling-ticket concerns under QF-12.
