# ADR 0016: Isolate backtest evaluation accounts from historical context

- Status: Accepted
- Date: 2026-09-13
- Jira: [QF-43](https://frostfiredigital-37308542.atlassian.net/browse/QF-43)
- Pull request: [#38](https://github.com/andrewjjohnson/quant-forge/pull/38)

## Context

QF-39 needs historical indicator context without training-period holdings,
cashflows, benchmark performance, or metric observations entering its test
account. Removing early bars loses warm-up; slicing an already executed result
retains contaminated portfolio state and benchmark normalization.

## Decision

Add an optional frozen `EvaluationInterval` to `BacktestConfig`. Its dates are
inclusive observed exchange sessions, consistent with QF-8's session membership.
The existing runner validates the entire immutable source, then supplies QF-4
only history through evaluation end. Causal split normalization includes earlier
effective splits. Before orders are constructed, decisions outside the
evaluation interval are excluded. The existing accounting loop and benchmark
receive only evaluation bars/actions and start from configured capital without
positions, P&L, or previous-close dividend entitlement.

Strategy history retains its non-accounting conceptual target state; it is not
reset by modifying QF-4 interfaces. A historical long target creates no synthetic
boundary entry. An evaluation flat target against the fresh account remains an
explicit rejected no-op. New evaluation decisions follow unchanged next-open
semantics, and final eligible opens outside the interval never execute.

The optional version-1 interval configuration records the fixed reset, context,
and signal policies. Complete source provenance and all existing scientific
configuration remain identity-bearing. QF-6 reuses its existing execution and
persistence; bounded study identities additionally bind resolved candidate
strategy configuration IDs, including indicator/backend versions. Absent
intervals serialize no new keys and preserve prior identities and versions.

## Consequences and validation

No second runner, accounting implementation, benchmark formula, strategy
interface, or metric implementation is introduced. Internal source views retain
original provenance and cannot be persisted as independently valid QF-3 data.
Complete-source data-quality and dividend-policy checks remain conservative.
Source appends require new identities even when historical economics are equal.

Tests compare bounded results with independently executed evaluation-only
fixtures, check exact pre-change identities, warm-up availability, boundary
signals/actions/costs, future append causality, and artifact export. QF-6 tests
cover process equivalence, zero-call resume, changed source/context/interval/
capital/backend identity, and altered trial/artifact policies. Only fresh
accounts are supported. QF-39 owns orchestration and parameter freezing;
QF-40 owns aggregate OOS results and holdout consumption.

## Pre-merge review correction: causal strategy metadata

The initial version-1 policy above retained full-source metadata on the truncated
history view. PR #38's review identified that future counts, sessions, actions,
and hashes could consequently reach strategy code, while defensive validation
failed because metadata no longer matched bars. Version 2 supersedes that
strategy-metadata policy with `strategy_metadata=causal_prefix_v1`.

Only the accounting view and final result retain full-source provenance. Raw
strategy history receives self-consistent prefix ranges/counts, filtered action
metadata, and rebuilt QF-3 digests, dataset/action identities, and canonical paths
derived from that prefix. Retrieval time is the epoch unavailable sentinel;
adapter version explicitly identifies a synthetic causal projection. The raw
prefix passes QF-3 validation but creates no cache files or provider response.
The subsequent split-normalized feature view retains its existing ephemeral
contract. No-boundary behavior is unchanged.

Tests reproduce the original metadata mismatch, compare all strategy-visible
metadata/action records under future appends, preserve source provenance, and
reject version-1 bounded artifacts on resume under the corrected policy.
