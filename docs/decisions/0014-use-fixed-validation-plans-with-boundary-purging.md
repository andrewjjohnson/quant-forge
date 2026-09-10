# ADR 0014: Use fixed validation plans with boundary purging

- Status: Accepted
- Date: 2026-08-25
- Jira: [QF-8](https://frostfiredigital-37308542.atlassian.net/browse/QF-8)

## Context

Prediction and trading/backtest research both need chronological development,
selection, test, and final-holdout partitions. Reimplementing splits inside each
study type would let boundary conventions, dataset provenance, indicator
backends, label horizons, costs, or holdout semantics drift independently.

A chronological split alone is not sufficient. A development or selection
observation can have a future outcome whose endpoint belongs to the following
protected interval. Indicators can also require earlier bars for causal warm-up
without making those bars eligible for selection. Intraday and daily research
cannot safely share naive calendar-date or elapsed-day assumptions.

## Decision

QuantForge uses one versioned, study-neutral `ValidationPlan` definition. Every
plan chooses exactly one observation axis: timezone-aware timestamps normalized
to UTC or actual exchange sessions under one QF-13 session policy. Partition
intervals are closed and inclusive because their boundaries identify discrete
observation membership, not continuous execution time.

Explicit folds contain development, optional selection, and test windows.
Expanding and rolling progression rules are validated from those fixed windows.
One reserved final holdout follows every fold. QF-8 defines membership only and
does not execute folds or consume the holdout.

Every configured outcome records its exact versioned configuration and maximum
future reach. Outcome provenance is captured only from a typed component that
exposes its actual session or elapsed-time horizon; an independent configuration
and shorter declared horizon cannot be paired. The plan's purge horizon must
equal the maximum outcome reach.

Before an earlier partition is used, observations are removed when their label
horizon plus explicit embargo reaches or crosses the next protected boundary.
Exchange-session distances use the configured exchange calendar rather than
weekdays. Warm-up observations are returned in a separate context-only field
that is never eligible for selection. A single-timeframe plan may use a scalar
count; a multi-timeframe plan records and selects an independent count for every
exact source timeframe. Every window must provide enough preceding context for
each indicator and its typed rule or strategy in the timeframe where that
component's count is expressed; the first study row supplies the final
observation needed for its own result. Rule provenance also binds the exact
configuration identities of its required indicators to the environment. For a
multi-timeframe rule, that binding is the exact source-timeframe and indicator
configuration pair, so one instance cannot satisfy the same indicator required
on another source frequency.

One immutable `ResearchEnvironment` is shared across the complete plan. Its
identity binds dataset fingerprint and QF-14 family references, QF-13
timeframes/session policies, aggregation policies, indicator configurations and
QF-35 backend identities, rule or strategy version, outcome definitions, and
applicable execution/cost configuration. Historical implicit-native indicator
configurations remain distinct from new explicit `native_v1` configurations.
Trading/backtest environments factory-capture only the existing validated
`BacktestConfig` rather than permitting a structural lookalike, generic
reference, or identity-free execution defaults. Domain-specific rule-provenance
factories verify the component's own canonical type before capturing it as a
trading strategy or prediction rule.
The configured timeframe set must exactly equal the timeframes supported by its
dataset provenance. The QF-3 standalone adapter captures its canonical daily
timeframe only from a validated dataset and resolves exchange timezone from its
calendar rather than provider serialization metadata. QF-14 members retain
their manifest-bound timeframe identities. If a selected family member is derived,
the environment's aggregation reference is factory-captured from and must
exactly match that family's typed aggregation policy.

## Consequences

Prediction and trading/backtest studies can share leakage controls without
sharing prediction metrics, orders, trades, portfolio state, or equity curves.
Changing any scientific environment field, boundary, horizon, embargo, warm-up,
fold mode, or holdout reservation produces a different deterministic plan ID.
Appending future observations cannot change membership inside an already fixed
historical interval. Multi-timeframe consumers must select warm-up separately
from each source chronology; daily and weekly observation counts are never
treated as interchangeable units.

Consumers must declare the maximum reach of every outcome that influences
training or selection. A mismatch fails plan construction. The closed interval
convention requires adjacent windows to use distinct observation keys. A future
continuous-time consumer that needs half-open intervals must introduce a new
versioned contract rather than reinterpret schema version 1.

QF-8 adds canonical serialization and validation for cache reuse but no new
persistence store. Walk-forward execution, out-of-sample aggregation, and a
holdout-consumption ledger remain later work.

## Alternatives considered

- **Let prediction and backtest packages own separate split types.** Rejected
  because equivalent studies could silently use different boundary, provenance,
  or leakage semantics.
- **Use calendar dates or elapsed durations for every plan.** Rejected because
  exchange holidays, early closes, daylight-saving transitions, and intraday
  timestamps are materially different temporal domains.
- **Purge a fixed number of observed rows.** Rejected because missing rows could
  weaken an exchange-session horizon and because timestamp labels need exact
  elapsed reach.
- **Include warm-up rows in the partition and rely on callers not to select
  them.** Rejected because the unsafe state would be representable and easy to
  misuse.
- **Treat the indicator backend as a tunable validation parameter.** Rejected
  because backend choice is fixed scientific provenance under ADR 0013.

## Validation

Unit fixtures cover prediction and backtest studies, session and timestamp
boundaries, expanding and rolling folds, overlap rejection, outcome-horizon
purging, embargo, warm-up separation, future-data appends, family/timeframe and
manifest identity mismatch, standalone timeframe relabeling, source-specific
multi-timeframe warm-up, and historical native-backend preservation. A
deterministic integration test captures existing QF-11 rule, outcome, and
indicator contracts directly into a validation plan. Canonical manifest tests
reject non-canonical bytes, tampering, and unsafe cross-plan reuse.
