# ADR 0021: Define outcome time contracts before timestamp membership

- Status: Accepted
- Date: 2026-09-17
- Jira: [QF-46](https://frostfiredigital-37308542.atlassian.net/browse/QF-46)

## Context

QF-42 supplies exact historical decisions, while legacy QF-11 outcomes and
QF-8/QF-39 execution use positive exchange-session horizons. Intraday outcome
work needs a shared contract without mixing elapsed minutes into session counts,
changing established daily results, or duplicating scheduling and purging.

## Decision

Add explicit session/exact anchors, typed session/elapsed horizons, and an
immutable future-label evaluation request. Dispatch legacy labelers with their
original date arguments; request-aware labelers receive the original supplied
instant. Preserve legacy serialization and identities when no new temporal
configuration is declared.

Resolve elapsed targets to the first expected completed observation boundary at
or after the target using canonical session windows. Require that exact bar;
never skip a missing endpoint. Restrict to the same session, distinguish
unavailability from numeric zero, and expose nominal duration plus one bar
interval through existing QF-8 duration/provenance types. This endpoint contract
does not certify path completeness or select a price.

Provide opt-in future-outcome export metadata and strict generic configuration
integrity hooks. Leave timestamp observation membership, fixed-candidate replay,
and consumer execution integration to QF-48. QF-46 precedes QF-48; QF-49 and
QF-47 then implement concrete returns and paths, followed by QF-45.

## Consequences

Existing session engines and numerical fixtures remain unchanged. New components
can declare exact temporal semantics without inventing a second temporal model.
Conservative reach may over-purge by up to one interval when QF-48 consumes it.
Trailing absent observations are dataset-end status unless later observed coverage
proves an internal gap. Future path labelers must independently verify every
required intermediate observation. Cross-session outcomes and recess-aware
window generation require a separately versioned policy.

See [outcome temporal contracts](../outcome-temporal-contracts.md).
