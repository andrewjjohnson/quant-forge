# ADR 0028: Checkpoint compact prediction journals

- Status: Accepted
- Jira: [QF-56](https://frostfiredigital-37308542.atlassian.net/browse/QF-56)

## Context

QF-55 normalizes historical evidence but legacy execution retains every full
QF-11 result until completion. QF-45 needs 5,760 decisions per trial. Per-decision
durability must not require one filesystem object per decision.

## Decision

Consume QF-42 through a shared sequential iterator. Execute QF-11, validate with
QF-55, normalize, append, checkpoint and release each result. Keep shared
evidence once, an append-only JSONL journal, and an atomic hashed checkpoint.

The checkpoint commits byte offset, count, last ID, scope IDs, the existing
QF-55 prefix hash and counts. Fsync journal bytes before checkpoint replacement.
Resume verifies committed state before truncating only an uncommitted tail.

Stream the same QF-55 final encoding, validate it, then publish without replacing
a prior final. Scientific identity is independent of layout and checkpoint
cadence. V1 remains explicit/default; QF-32 compact analyzers receive readers and
QF-39 stores versioned file references with existing partitions and freeze.

## Consequences

Three staging files and bounded completed-result state per window. Resume
validates records without research recomputation. Compact execution avoids
legacy accumulating caches. Domain metrics need a reader-based analyzer.
Single-writer local filesystem atomicity is required, and final assembly needs
temporary space for staging plus final output. Incompatible evidence is retained.
QF-40/QF-9/QF-41 compact consumption remains QF-57.

See [the execution contract](../incremental-prediction-windows.md).
