# 0022: Bind prediction membership to exact decision schedules

- Status: accepted
- Jira: QF-48

## Context

QF-8 already supports timestamp boundaries and elapsed purge offsets. The QF-39
prediction adapter used daily QF-3 session membership, while QF-42 already owned
exact historical decisions. QF-46 provides exact anchors, typed elapsed horizons,
conservative reach, and future-observation resolution. Inferring intraday
membership from daily observations would collapse distinct decisions and permit
incompatible artifacts to share validation lineage.

## Decision

Add an explicit, immutable prediction membership source to `ValidationPlan`,
captured from a QF-42 schedule and its canonical completed primary artifact.
Reuse `BoundaryAxis`, existing QF-8 partition/purge/embargo functions, and QF-46
reach. Keep legacy plan serialization unchanged when this source is absent.
Require coverage of every scheduled primary observation rather than silently
removing missing decisions. Validate each window's preceding primary warm-up
against the captured schedule when constructing the plan.

QF-39 restricts its QF-42 schedule to retained QF-8 timestamps. QF-11 supplies a
bounded canonical outcome source only after causal predictions are fixed. QF-7
captures and preserves the original exact anchor across replay. QF-40 reads the
captured evidence and binds it into validation/holdout lineage; it does not build
another membership calculation. Holdout exposure continues to guard complete
exchange sessions conservatively.

Elapsed label callbacks also receive the existing QF-46 `OutcomeResolution` from
the complete source. Its availability evidence agrees with persisted metadata
without exposing prices beyond the bounded source. Labelers must use that evidence
instead of interpreting the bounded slice as the complete artifact.

## Consequences

Timestamp predictions need explicit schedules, exact closed window endpoints,
aware anchors, and typed elapsed outcome components. Membership and provenance
remain reproducible and resumable using existing persistence. Daily prediction
and backtest paths retain their contracts. Missing primary coverage fails closed,
and the existing runner still requires a prior completed QF-3 daily observation
for its metadata input. Concrete intraday outcome math remains downstream work.
