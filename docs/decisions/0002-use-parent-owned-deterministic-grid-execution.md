# ADR 0002: Use parent-owned deterministic grid execution

- Status: Accepted
- Date: 2026-08-04
- Jira: QF-6

## Context

QF-6 must execute CPU-bound QF-5 backtests sequentially or with bounded local
parallelism, persist every outcome incrementally, resume safely, and produce the
same scientific result regardless of worker completion order. Allowing workers
to write shared study state would require locking and make duplicate claims,
partial artifacts, scheduling-dependent row order, and recovery behavior harder
to verify. Submitting the complete immutable dataset with every combination
would also add avoidable serialization work.

## Decision

Sequential execution remains the reference behavior. Parallel execution uses a
bounded standard-library process pool. Each process receives the immutable QF-3
dataset, serializable strategy factory, and QF-5 configuration once through its
initializer. Submitted work contains only one normalized parameter mapping.

The parent process is the sole owner of study manifests, trial state
transitions, QF-5 result exports, ranking, stability analysis, and study-level
exports. It persists `RUNNING` before submission and atomically transitions the
record to `SUCCEEDED` or `FAILED` after receiving a result. No more than the
configured worker count is outstanding. Final records are always ordered by
canonical combination index, and ranking has stable metric and identifier tie
breakers.

Execution mode, worker count, and completion timestamps are operational
metadata excluded from scientific study/trial identities. They remain in the
manifest, and exact operational configuration is required when resuming that
physical store. Separate sequential and process stores can therefore prove
equivalent scientific identities and outputs.

## Consequences

Workers cannot race to claim or overwrite the same trial, and a corrupt partial
JSON write cannot be mistaken for completion. Completed successes and
exclusions remain immutable. A crash can lose only active work; stale `RUNNING`
records retry on resume. Failed-trial retry remains explicit and disabled by
default. A broken pool halts new scheduling, records active failures, and leaves
unsubmitted combinations pending for resume. Completion order cannot affect
final ordering, ranking, stability, or recommendation.

The parent receives one complete QF-5 result at a time per active worker so it
can create the immutable QF-5 artifact and persist the compact trial summary.
Full results are not accumulated for the whole study. Local processes still
have per-worker dataset copies according to operating-system process semantics,
and factories/configurations must be serializable. Distributed machines and
database-backed queues remain outside QF-6.

## Alternatives considered

- Threads were rejected as the default because QF-5 work is CPU-oriented and
  the Python runtime does not guarantee useful CPU parallelism for this design.
- Worker-owned file writes were rejected because locking, claim recovery, and
  canonical artifact ordering would add complexity without QF-6 scale needs.
- Submitting every combination at once was rejected because it creates an
  unbounded task queue.
- A database queue and distributed scheduler were rejected as unnecessary for
  a deterministic local Cartesian grid and explicitly outside ticket scope.

## Validation

Tests compare sequential and two-worker process studies for equal study,
combination, trial, and QF-5 run identities, metrics, rankings, stability, and
parameter summaries. Resume tests instrument the QF-5 call count and require a
complete study to perform zero calls. Failure tests prove one trial does not
stop later work, while persistence tests reject corruption and incompatible
manifests. Ranking tests reverse trial completion/input order and require the
same result.
