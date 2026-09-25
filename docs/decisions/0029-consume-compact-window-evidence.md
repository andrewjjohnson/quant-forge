# ADR 0029: Consume compact window evidence downstream

- Status: Accepted
- Jira: [QF-57](https://frostfiredigital-37308542.atlassian.net/browse/QF-57)

## Context

QF-55 and QF-56 persist historical windows with shared evidence once and lazy
normalized decisions. Legacy downstream assumptions would reproduce the repeated
evidence that blocked QF-45 at 5,760 decisions per window.

## Decision

Reuse `PredictionWindowReader` for embedded v1 and referenced v2 windows.
QF-40 streams observations and stores compact aggregate schema 2 with identity
references instead of observations. Existing exact metrics retain numeric samples
only. Accept a finalized holdout through the original frozen-request validator
and permanent ledger, without rerunning historical evaluation.

QF-9 indexes header, configuration and shared-evidence views of one JSONL file in
its existing graph. Initial inspection validates scientific scope and all records;
replay authenticates immutable file hashes and control records. QF-41 renders
published aggregates and authoritative holdout state without decision iteration.

## Consequences

Research semantics and time boundaries do not change. Physical identities and
aggregate schemas correctly reflect representation differences. Legacy records
remain readable without migration. Schedule metadata and exact-median scalar
buffers grow with membership; full prediction graphs and repeated evidence do not.
Whole-file hashing still costs linear I/O. Bounded scientific verification needs
the trusted canonical parent metadata already used by the validation plan.

The optional finalized holdout path does not authorize prior holdout exploration;
its provenance and exposure history remain the research operator's responsibility.
The ledger cannot restore unseen status or change an existing consumed request.

## Alternatives considered

Expanding v2 into v1 recreates the original scale failure. A third reader, a new
manifest graph or a replacement report engine duplicates existing contracts.
Approximate quantiles would change numerical semantics, so exact scalar buffers
are retained. Embedding every observation in schema 2 needlessly duplicates the
authoritative compact artifacts.

## Validation

Real v1/v2 semantic and report comparisons, full holdout lifecycle, corruption and
replacement checks, blocked-network replay and a 5,760-decision structural fixture
exercise the boundaries. Legacy suites remain mandatory. See
[the consumer contract](../compact-window-consumers.md).
