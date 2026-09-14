# ADR 0018: Persist holdout exposure before evaluation

- Status: Accepted
- Jira: QF-40

## Context

QF-8 reserves final holdout boundaries and QF-39 freezes selections before testing.
QF-40 must not call a holdout pristine after an interrupted evaluation or let a
cosmetic configuration change erase prior exposure. Results and bookkeeping are
separate files, so marking consumption only after a successful result export is
unsafe. QF-39's single-writer checkpoint is not a concurrent consumption ledger.

## Decision

Use a QF-40-specific local POSIX ledger with canonical JSON/SHA-256 records,
atomic file replacement, file/directory fsync, and an advisory exclusive lock for
the entire consumption attempt. Persist an immutable consumed marker, exact
frozen request, original UTC timestamp and run ID before calling the existing
prediction/backtest evaluator. Persist result references afterward. Uncertain or
interrupted attempts remain consumed. Exact retries cannot replace the freeze or
erase consumption; successful reproducibility reruns must match existing results.

Scientific lineage excludes descriptive labels and execution retry flags, while
retaining material research inputs. Maintain an independent prior-exposure guard
for overlapping holdout calendar days on the same canonical symbol. An incompatible
lineage cannot present an already viewed interval as pristine. Keep the permanent
ledger distinct from reproducible aggregate exports and QF-9's future manifests.

OOS aggregation reads QF-39 test artifacts only and remains independent of current
holdout state. Backtest summaries retain native reset-account results and expose
a dimensionless chain of relative OOS equity; no capital or position carryover is
introduced. Prediction outcomes remain a separate result family.

## Consequences

- A failed attempt can conservatively consume an interval without a usable result.
- Exact recovery is possible without a fresh pristine claim.
- Concurrent local attempts fail explicitly; crashes release the OS lock.
- Deletion, rollback, alternate ledgers, manual inspection and arbitrary callback
  I/O cannot be prevented by a local file ledger. One permanent workspace store
  and preserved evidence remain research responsibilities.
- Network/distributed filesystems and cross-platform lock portability are outside
  this local implementation. QF-9/QF-41 must consume current ledger state rather
  than infer it from an initial reservation or cached aggregate.

See [OOS and holdout contracts](../oos-holdout-aggregation.md) for the formulas,
identities, failure matrix, explicit workflow, and schema.
