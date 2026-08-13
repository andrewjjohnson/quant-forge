# ADR 0012: Reconstruct developing bars from causal source intervals

- Status: Accepted
- Date: 2026-08-13
- Jira: [QF-21](https://frostfiredigital-37308542.atlassian.net/browse/QF-21)

## Context

Some research questions need the state of a forming larger-intraday, daily, or
weekly candle at a historical decision time. Reusing the eventual completed
bar would leak its later high, low, close, and volume. Relabeling completed
QF-18/QF-19 datasets as developing would also blur immutable dataset semantics
and change established bar identities.

## Decision

`MultiTimeframeContext` retains completed-bars-only as its default. Consumers
must explicitly select `DEVELOPING_BAR_AS_OF`. The primary series must be a
canonical intraday source reloaded through its immutable cache. QF-21 retains
the source dataset's QF-17 expected-interval evidence and reconstructs at most
one developing bar per contextual timeframe from completed expected source bars
whose ends are at or before `as_of`.

Reconstruction supports compatible larger intraday intervals, one exchange
session, and one exchange trading week. Boundaries come from QF-13 session,
anchor, early-close, and DST policies. Missing available constituents fail
closed and unexpected observations are excluded. OHLCV uses first open,
maximum high, minimum low, final close, and summed volume.

The result is a distinct immutable `DevelopingBar` with `complete = false`, an
explicit developing completion state, `as_of`, nominal interval, observed
start/end and sessions, source count and IDs, expected completion boundary,
source dataset-family reference, and a versioned reconstruction policy. Its
full primitive record is embedded in context serialization. Persisted QF-18 and
QF-19 timeframes remain completed-only and their completed bars are neither
copied nor relabeled.

## Consequences

Historical developing values cannot observe later source intervals, and early
close or DST boundaries do not require fixed-time assumptions. Indicators and
studies can distinguish terminal and forming values through both type and the
shared `complete`/`completion` metadata.

The mode requires an immutable canonical intraday primary series even when
completed higher-timeframe artifacts are also supplied. A newly persisted
source artifact retains distinct provenance identity, although later bars do
not change the earlier causal OHLCV or constituent set. Multi-session and
multi-week developing targets remain unsupported until their completed
aggregation semantics are defined.

## Alternatives considered

- **Expose the eventual completed candle before its end.** Rejected as direct
  look-ahead leakage.
- **Truncate a source interval at `as_of`.** Rejected because it invents a
  noncanonical lower-timeframe observation.
- **Mutate QF-18/QF-19 timeframes to include developing values.** Rejected
  because persisted completed artifacts and their identities must remain
  unchanged.
- **Make developing bars the default.** Rejected because completed bars are the
  safer, less ambiguous research boundary.

## Validation

Offline tests cover explicit opt-in, 3:55 p.m. daily and Tuesday weekly
causality, larger intraday reconstruction, future-source invariance, early
close, pre/post-DST boundaries, stable serialization, structural completion
metadata, and QF-18/QF-19 completed-bar regressions.
