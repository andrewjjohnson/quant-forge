# ADR 0010: Derive daily and weekly bars from exchange sessions

- Status: Accepted
- Date: 2026-08-12
- Jira: [QF-19](https://frostfiredigital-37308542.atlassian.net/browse/QF-19)
- Pull request: [#16](https://github.com/andrewjjohnson/quant-forge/pull/16)

## Context

Higher-timeframe research must not combine provider-native EOD candles with a
different intraday feed, session scope, revision, or adjustment basis. Daily and
weekly boundaries also cannot be represented as elapsed 24-hour or seven-day
windows: exchange holidays, early closes, and configured extended hours change
the actual observations in a period.

## Decision

QuantForge derives one completed daily bar per fully covered exchange session
and one completed weekly bar per fully covered Monday-Sunday exchange trading
week. QF-13 supplies exact sessions and boundaries. QF-17 supplies expected,
missing, and unexpected source-interval evidence. OHLCV is first open, maximum
high, minimum low, final close, and summed volume.

The default strict policy rejects incomplete source coverage. Explicit
diagnostic mode excludes unexpected observations, never fills gaps, and binds
the complete source report and exact missing constituents into the derived
identity. A partial source-range session or week is excluded instead of emitted
as completed. Holiday-shortened weeks are complete when all actual exchange
sessions are present. Every output records ordered source-bar IDs, and dataset
validation rejects counting one source bar twice.

Derived bars identify QuantForge as producer. The deterministic dataset ID
binds the immutable source snapshot, source quality report, target timeframe,
complete regular/extended-hours session scope, aggregation policy, bar content,
and QF-14 family ID. The family preserves feed scope and the source adjustment
and corporate-action basis. Artifacts use a separate content-addressed
`session/derived/` namespace and are rederived on cache load.

## Consequences

Early-close daily bars end at the actual close, and holiday weeks need no
special weekday assumptions. Daily and weekly bars are internally consistent
with their intraday charts and reproducible from one source snapshot. Provider-
native EOD bars cannot alias these datasets merely by matching symbol or date.

The implementation intentionally supports only one-session daily and one-week
weekly targets. Multi-session bars, multi-week bars, larger intraday
aggregation, multi-timeframe alignment, developing bars, and indicators remain
outside QF-19.

## Validation

Deterministic fixture tests hand-audit daily and weekly OHLCV across the July
2024 Independence Day week, including the July 3 early close. Additional tests
cover exact session membership, no duplicate constituent use, strict and
diagnostic missing data, partial-week exclusion, identity sensitivity, provider-
native rejection, immutable cache verification, and target-policy validation.
