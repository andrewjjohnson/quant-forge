# ADR 0033: Prepare immutable outcome sources within dataset execution sessions

- Status: Accepted
- Date: 2026-09-27
- Jira: [QF-61](https://frostfiredigital-37308542.atlassian.net/browse/QF-61)

## Context

QF-11 repeatedly scanned the full immutable outcome source after each historical
rule evaluation, including decisions without candidates. Future dense research
also needs many explicit observation requests against one source. Skipping all
validation or tying the source contract to prediction candidates is unsuitable.

## Decision

Own a `PreparedOutcomeSources` registry in each validated prediction dataset
session. Admit only structurally immutable sources and metadata. Bind complete
contents, ancestry, timeframe/session, adjustment and coverage evidence through a
scientific compatibility digest; retain the exact admitted backing as a reusable
capability. New source handles require full content authentication. Retain the
reference validation path for unsupported mutable graphs.

Validate source-wide invariants and build timestamp/session indexes once. Keep
every request's exact anchor, horizon, source resolution, temporal reach, expected
calendar boundaries, coverage/status, path completeness, configuration and price
basis checks. Keep bounded callback detachment and mutation guards. Source
preparation remains after prediction generation; no-candidate decisions create no
requests. Explicit `OutcomeEvaluationRequest` anchors need no rule object.

Do not expose the future-bearing preparation to prediction context or features.
Keep QF-59 context indexes and QF-60 projection preparation under their existing
scopes. Persist no operational state; preserve all scientific identities and
independent offline checks. Resume rebuilds preparation and executes only the
incomplete suffix.

## Consequences

Historical invariant cost becomes one-time; request work is proportional to the
requested bounded path, plus logarithmic position lookup. The registry retains
immutable backing references and indexes, never another price store or results.
New independently constructed source handles must still authenticate content.
Unsupported custom mutable sources remain correct but unoptimized. Preparation
adds measurable startup work, reported separately from steady throughput.

No new outcomes, ML/dense dataset framework, strategy changes, observation modes,
global cache, parallelism, persistence schema or dependency are introduced.
See [implementation and evidence](../prepared-outcome-sources.md).
