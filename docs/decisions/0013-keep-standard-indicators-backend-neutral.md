# ADR 0013: Keep standard indicator definitions backend-neutral

- Status: Accepted
- Date: 2026-08-14
- Jira: [QF-35](https://frostfiredigital-37308542.atlassian.net/browse/QF-35),
  [QF-37](https://frostfiredigital-37308542.atlassian.net/browse/QF-37),
  [QF-25](https://frostfiredigital-37308542.atlassian.net/browse/QF-25)

## Context

QuantForge needs mature standard-indicator mathematics without allowing a
third-party library to define timeframe, provenance, causality, names, or
deterministic study identity. Backend-specific public classes would multiply
the indicator catalog and force downstream consumers to understand the chosen
library. Making TA-Lib the implicit default would also revise QF-23 EMA values
and historical configuration identities.

## Decision

QuantForge owns one backend-neutral standard-indicator definition containing
canonical input fields, normalized parameters, and normalized output names.
The QF-22 `ConfiguredTimeframeIndicator` remains the outer temporal and lineage
boundary. A stable-id registry resolves adapters below it.

QF-35 defines `native_v1` and `talib_v1` and initially maps EMA. QF-36 extends
the same adapters and backend-neutral public classes to SMA, Wilder RSI, and
Wilder ATR. QF-37 extends them to directional movement/ADX and Bollinger Bands.
QF-25 adds MACD through the same definition and makes `talib_v1` its standard
implementation without introducing a native MACD formula.
Multiple library outputs receive backend-local names before they map to stable
QuantForge names, so tuple positions remain adapter details. Each adapter owns
input extraction, parameter translation, library invocation, output
normalization, and its stable function identity. TA-Lib is pinned to `0.7.1`.
Explicit-backend configurations bind the backend id, backend-contract version,
Python-wrapper name and exact version, native runtime library name and exact
version, and mapped function name.

Omitted-backend constructors and serialized historical configurations remain
unchanged and resolve explicitly to `native_v1`. A historical mapping without
a `backend` field deserializes through that compatibility path. New explicit
backend mappings use a new configuration contract and fail closed if the
installed identity does not reproduce the serialized mapping.

## Consequences

Prediction, feature, data, strategy, backtesting, scanner, and reporting code
continues to consume normalized QuantForge outputs and does not import TA-Lib.
A future backend requires an adapter and mappings, not backend-specific EMA,
RSI, or MACD classes. Existing native EMA studies, IDs, values, and predictions
remain reproducible.

TA-Lib uses float64 and its own initialization and missing-gap behavior, so an
explicit `talib_v1` study can differ numerically from `native_v1`. That
difference is visible in deterministic identity. The adapter converts finite
outputs through their stable decimal string and converts `NaN` to QuantForge
`None`. It rejects non-default TA-Lib compatibility or EMA unstable-period
global state rather than letting hidden mutable state alter a study.
The adapter also validates TA-Lib 0.7.1's supported EMA period range of `1`
through `100000` at configuration time. Larger periods remain valid for the
historical native backend and are rejected only when `talib_v1` is selected.

QF-36 maps the existing core SMA, EMA, Wilder RSI, and Wilder ATR definitions.
QF-37 maps directional movement to one normalized three-output request and
Bollinger Bands to normalized middle/upper/lower/bandwidth outputs. Dynamic
third-party plugin loading is not introduced.
QF-25 maps normalized fast, slow, and signal periods plus a source field to
TA-Lib `MACD`, then exposes only `macd`, `signal`, and `histogram`. The histogram
is derived from the normalized lines so the stable result contract preserves
exact subtraction after float-to-Decimal normalization.

## Alternatives considered

- **Create `TalibEMA` and `NativeEMA` classes.** Rejected because each backend
  would duplicate the QuantForge indicator catalog and leak library choice to
  consumers.
- **Make TA-Lib the new implicit EMA implementation.** Rejected because it would
  change historical values and deterministic identities.
- **Put backend selection in prediction or backtesting.** Rejected because it
  would invert dependencies and duplicate normalization outside the indicator
  layer.
- **Normalize TA-Lib until it exactly matches native Decimal math.** Rejected
  because the backend should own its library's standard mathematics and its
  distinct identity already makes the semantics explicit.

## Validation

Offline tests cover stable backend resolution, native versus TA-Lib EMA,
canonical input and parameter translation, normalized output alignment, exact
version identity, unsupported mappings, historical configuration and value
compatibility, QF-22 timeframe/provenance preservation, and a future test
adapter used by the same EMA class.

QF-36 adds the same coverage for SMA, Wilder RSI, and Wilder ATR, including
legacy identity fixtures, explicit-backend round trips, TA-Lib period limits,
representative float64 comparisons, flat-series RSI differences, and
append-future causality.

QF-37 adds named multi-output normalization, directional single-call-count
coverage, native fixture and overnight-study compatibility, Bollinger derived
output coverage, input immutability, backend-specific initialization and
lookback expectations, timeframe lineage metadata, and append-future causality.

QF-25 adds MACD period ordering and backend-range validation, TA-Lib tuple and
parameter translation, exact normalized histogram invariants, explicit lookback
and unavailable behavior, serialization/version identity, timeframe neutrality,
input immutability, and append-future causality coverage.
