# ADR 0008: Bind intraday quality reports to datasets

- Status: Accepted
- Date: 2026-08-11
- Jira: [QF-17](https://frostfiredigital-37308542.atlassian.net/browse/QF-17)
- Pull request: [#14](https://github.com/andrewjjohnson/quant-forge/pull/14)

## Context

Canonical QF-15 bars validate individual boundaries and OHLCV values, while a
chronologically valid batch can still omit a source interval inside an exchange
session. Weekend, holiday, early-close, extended-hours, and daylight-saving
semantics make fixed weekday or UTC-range gap checks incorrect. Downstream
aggregation and study tickets need auditable coverage evidence without deciding
their tolerance policy in QF-17.

QF-16 dataset manifests did not carry a coverage outcome. Recomputing quality
ad hoc in each consumer would allow inconsistent calendar assumptions and leave
the original dataset identity silent about known gaps.

## Decision

QuantForge derives expected completed source intervals from the request's full
QF-13 timeframe, session policy, half-open range, and exchange calendar. One
immutable report preserves exact missing, unexpected, developing, and
zero-volume interval evidence plus per-session coverage. Strict mode raises on
missing or unexpected completed intervals; diagnostic mode returns the same
facts for caller-owned tolerance decisions. Neither mode mutates bars.

Duplicate, overlapping, out-of-range, out-of-session, invalid-duration, and
invalid OHLCV observations remain hard canonical contract failures. Zero volume
remains valid under QF-15 and is reported as a warning rather than reinterpreted
as a missing bar.

Intraday dataset schema version 2 embeds the diagnostic report and report ID in
the content-addressed manifest. Loads recompute and compare the report before
returning the dataset. The raw snapshot schema remains version 1 because
coverage evidence does not alter provider response content.

## Consequences

Derived datasets and studies receive one deterministic, provider-neutral
quality record through `IntradayDataset.quality_report`. Known gaps participate
in dataset identity, and future consumers can apply stricter domain policies
without losing the original evidence. Calendar changes that alter expected
coverage fail cache verification instead of silently changing interpretation.

Existing schema-version-1 intraday dataset manifests are immutable and are not
upgraded in place. They must be refreshed or re-ingested to produce a
quality-bearing schema-version-2 dataset. Existing raw snapshot content hashes
remain stable.

## Alternatives considered

- **Fill missing bars before validation.** Rejected because invented prices or
  volume would corrupt higher-timeframe indicators and hide source quality.
- **Treat weekdays and fixed UTC hours as expected coverage.** Rejected because
  holidays, early closes, and daylight-saving transitions would be mislabeled.
- **Store only a complete/incomplete boolean.** Rejected because downstream
  research needs exact missing boundaries and session context.
- **Make diagnostic mode accept malformed canonical bars.** Rejected because
  session, range, ordering, overlap, and OHLCV invariants are not tolerance
  choices.
- **Change raw snapshot schema with the dataset manifest.** Rejected because
  provider response content and coverage evidence have separate identities.

## Validation

Deterministic offline tests cover complete XNYS sessions, an exact missing
5-minute interval, holidays, weekends, early closes, completed terminal partial
bars, daylight-saving offsets, regular versus extended-hours scope, strict and
diagnostic behavior, zero-volume warnings, duplicate/overlap/range rejection,
serialization, cache persistence, and report recomputation.
