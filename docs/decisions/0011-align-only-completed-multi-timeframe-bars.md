# ADR 0011: Align only completed multi-timeframe bars by default

- Status: Accepted
- Date: 2026-08-12
- Jira: [QF-20](https://frostfiredigital-37308542.atlassian.net/browse/QF-20)

## Context

Intraday research needs daily and weekly context without exposing the eventual
high, low, close, or volume of a period that is still forming. Matching symbols
or display interval names also cannot prove that series share a feed, source
revision, session scope, or adjustment basis. The alignment boundary must be
shared by future prediction and strategy consumers without depending on either.

## Decision

QuantForge builds `MultiTimeframeContext` in the data layer from QF-13
timeframes, QF-14 dataset-family references, and canonical QF-15/QF-18/QF-19
bars. The only QF-20 completion policy is completed bars only. A bar becomes
visible when it is terminal and its explicit end timestamp is at or before the
decision `as_of`; its label never changes availability.

Every declared timeframe has explicit available, stale, or missing metadata.
Optional positive maximum ages define staleness without filling or removing
observations. Missing and undeclared access raises dedicated domain errors.
Input series must share one family identity and canonical source snapshot, and
all declared timeframes must share the exact exchange-session policy.
Series are created only from validated source, QF-18, or QF-19 dataset artifacts;
bare family references cannot be paired with unbound bar tuples. The artifact's
recomputed content, provenance, exact dataset ID, and family supply the context
reference. Source artifacts are reloaded through their `IntradayMarketDataCache`
and must equal the supplied in-memory dataset, so the complete content-addressed
manifest identity, raw artifacts, and canonical paths are verified before the
series is exposed. The aligned timeframe and complete context models are
factory-only results of `build_multi_timeframe_context()`; callers cannot attach
unvalidated bars directly to those exported dataclasses.

The context is ordered independently of caller input order. Its deterministic
identity binds `as_of`, primary and required timeframes, freshness limits,
completion policy, family validation, dataset references, availability, and
the visible bar IDs. Serialization references immutable bars rather than
copying OHLCV values.

## Consequences

An intraday decision cannot observe the current session's eventual daily bar or
the current week's eventual weekly bar. Adding future bars to the same validated
family inputs leaves a historical context byte-identical. A newly persisted
content-addressed dataset or family remains distinct provenance and therefore
changes the context identity even when its price prefix is numerically equal.
Consumers must handle missing and stale states explicitly and retain referenced
immutable datasets for reproduction.

Developing-bar reconstruction, indicators, prediction integration, and trade
execution remain separate later boundaries. The completed-only restriction is
deliberately conservative; a future developing-bar policy requires a distinct
identity-bearing implementation rather than changing this schema silently.

## Alternatives considered

- **Select the latest row by label.** Rejected because start labels can expose
  an eventual close before the bar ends.
- **Forward-fill missing context.** Rejected because it hides availability and
  freshness assumptions.
- **Match series by symbol/provider.** Rejected because feed, source revision,
  session, adjustment, and aggregation policies can still differ.
- **Reconstruct developing bars in QF-20.** Rejected as explicit sibling-ticket
  scope and a different causal policy.
- **Place alignment in prediction or backtesting.** Rejected because both must
  consume the same temporal semantics without circular dependencies.

## Validation

Offline tests use weekly, daily, four-hour, and five-minute bars around a
Tuesday decision. Future daily/weekly sentinel bars remain inaccessible. The
fixture includes the July 4 holiday and July 3 early close. Tests also cover
future-appending invariance, deterministic ordering and serialization,
available/stale/missing states, guarded access, incompatible sessions, and
consolidated-versus-IEX family rejection. Regression tests also tamper source
content together with matching digest evidence, forge source dataset IDs,
change family symbols, and alter derived source provenance to prove a reference
cannot authenticate unrelated bars. Direct construction of aligned or complete
context results is also rejected.
