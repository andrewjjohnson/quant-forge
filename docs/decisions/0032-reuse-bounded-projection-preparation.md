# ADR 0032: Reuse bounded projections and immutable expected lineage locally

- Status: Accepted
- Date: 2026-09-26
- Jira: [QF-60](https://frostfiredigital-37308542.atlassian.net/browse/QF-60)

## Context

After QF-59 the same QF-52 bounded view was constructed at partition entry and
again by each compact trial validator. Completed resume repeated the latter
work. The expected view is independent of strategy parameters, but must remain
bound to source ancestry, cutoff, plan, fold and role. Persisted evidence must
remain independently verifiable after the process ends.

## Decision

Use an explicit `PreparedProjectionRegistry` owned by a sequential walk-forward
or grid invocation. A direct evaluator call owns a smaller scope unless the
caller explicitly opens an enclosing preparation scope. Always clear owned state
on exit, including errors and interruption. Retain at most two bounded views;
evicting an entry causes independent reconstruction on its next use.

Authenticate current canonical metadata and evidence, exact daily content,
plan/fold/window scope, cutoff/start and projection version. Admit only reviewed
immutable carriers. Hash immutable evidence bytes without decoding full source
collections. Never key by memory address, path alone or coincidentally equal bars.

Reuse the existing QF-52 factory result as immutable expected lineage. Validate
supplied bounded records on every use and compare their complete provenance to
that independently prepared expectation. Continue all existing compact, decision,
context, outcome, membership and persisted identity checks. Offline readers keep
the unchanged reconstruct-and-verify implementation.

## Consequences

Compatible trials and repeated validators avoid full canonical reconstruction
and projection. Selection and test retain different scopes; no holdout preparation
is created by walk-forward execution. The registry retains bounded session
metadata/evidence, never a canonical intraday dataset or decision collection.
There is no schema change, persisted handle, global cache or new dependency.
Repeated access after eviction or process restart may rebuild preparation.

Outcome-source validation, indicators, context indexing, per-decision execution,
parallelism, scientific configuration and downstream reports remain unchanged.
See [the implementation and evidence](../prepared-prediction-projections.md).
