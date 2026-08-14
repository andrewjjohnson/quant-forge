# ADR 0013: Keep standard indicator definitions backend-neutral

- Status: Accepted
- Date: 2026-08-14
- Jira: [QF-35](https://frostfiredigital-37308542.atlassian.net/browse/QF-35)

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

QF-35 defines `native_v1` and `talib_v1` and maps only EMA. Each adapter owns
input extraction, parameter translation, library invocation, output
normalization, and its stable function identity. TA-Lib is pinned to `0.7.1`.
Explicit-backend configurations bind the backend id, backend-contract version,
library name, exact installed library version, and mapped function name.

The omitted-backend EMA constructor and serialized QF-23 configuration remain
unchanged and resolve explicitly to `native_v1`. A historical mapping without a
`backend` field deserializes through that compatibility path. New explicit
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

Only EMA is mapped in QF-35. Migrating other indicators remains separate ticket
scope, and dynamic third-party plugin loading is not introduced.

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
