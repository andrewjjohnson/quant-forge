# ADR 0030: Share immutable prediction source backing during historical execution

- Status: Accepted
- Date: 2026-09-25
- Jira: [QF-58](https://frostfiredigital-37308542.atlassian.net/browse/QF-58)

## Context

QF-42 copied a study into a pristine template and recursively copied that template
for every decision. The QF-45 study includes a 48,480-bar `outcome_source`; its
repeated deep copies dominated execution. Independent mutable components and
QF-11's detached bounded labeler inputs remain necessary.

Frozen dataclasses alone do not establish deep immutability. Constructors accept
some subclasses; exchange-calendars also supplies `pandas.Timestamp` values with
writable instance attributes inside otherwise frozen bars.

## Decision

At the normal QF-42 iterator boundary, inspect only `PredictionStudy.outcome_source`
once. The helper admits exact, reviewed frozen/slotted source record types,
tuples, immutable scalar types and standard immutable timezone types. Unknown
types, subclasses, mutable containers, and cycles fall back to ordinary copying.

Exact calendar `Timestamp` instances without attached attributes may be replaced
by standard-library `datetime` values only when equality and ISO serialization
are exact. Nanosecond rounding is forbidden. Rebuild only enclosing records that
need those immutable replacements, once per iterator; retain all other immutable
leaves. Pandas is already present in the frozen exchange-calendars dependency
graph; no dependency or lockfile is changed.

Supply a fresh deepcopy memo for each template/decision copy, mapping only the
original and normalized source roots to the shared backing. Do not retain a
populated component-copy memo. Do not change `TimeframeBarSeries.__deepcopy__`:
QF-11's small bounded callback copies and mutation detection require its existing
behavior. No global copy override or new study protocol is introduced.

## Consequences

Rules, labelers, evaluators and mutable study extensions still copy independently.
Contexts still use each decision's exact timestamp and existing causal selection.
Standard immutable source records reject ordinary mutation; malicious bypass of
Python frozen records is outside their contract. Existing callback isolation and
mutation checks remain active, including detached source/resolution copies.

The source check does not certify scientific provenance and does not replace any
data/ancestry validation. Source identities, schedules, indicators, numerical
results, compact serialization/hashes and checkpoint formats stay unchanged.
Object IDs are operational copy-memo keys only. The bounded memo has at most two
roots regardless of the number of bars. Unsupported custom sources keep previous
behavior and do not gain the optimization.

## Alternatives considered

A global source `__deepcopy__` override would also share the source handed to
labelers and undermine QF-11's mutation guards. Sharing every frozen dataclass
would incorrectly trust nested mutable values and user subclasses. Cloning only
the three known components would omit mutable extensions on generic studies.
Changing calendar/context models globally would broaden QF-58 unnecessarily.

## Validation

Generic stateful-rule tests compare old copy semantics with sharing for exact
QF-11/context/indicator/outcome payloads and compact bytes/identities. Source-copy
instrumentation proves no per-decision large-source copy. Mutation, subclass,
timestamp precision, causal boundaries and old-prefix resume tests protect the
boundary. A bounded real QF-45 reprofile uses disposable checkpoint copies; see
[the implementation and profile report](../prediction-source-sharing.md).
