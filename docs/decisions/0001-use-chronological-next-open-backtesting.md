# ADR 0001: Use chronological next-open backtesting

- Status: Accepted
- Date: 2026-08-03
- Jira: QF-5

## Context

Daily strategy signals are derived from completed close data, while portfolio
accounting is stateful. A fully vectorized accounting implementation would make
execution ordering, cash affordability, costs, position state, and audit links
harder to verify. Filling at the same close would introduce look-ahead bias.

## Decision

QF-5 uses vectorized- and table-compatible QF-3/QF-4 inputs and outputs, but a
deterministic chronological state machine is the source of truth for orders,
fills, cash, positions, and trades. Close-derived decisions may first execute at
the next exchange session's open. The session order is execute at open, apply
costs and accounting, then mark at close. A fixed explicit Decimal policy is
used for calculations.

The MVP uses one symbol, long-only whole shares, market orders, full fills, and
no forced final liquidation. Later execution models must preserve the signal
eligibility boundary and produce equally auditable records.

## Consequences

The implementation favors correctness and traceability over opaque dataframe
expressions. Every fill is causally later than its signal, and every quantity
change is linked to one fill. Commission and additional fees remain separate in
configuration and accounting. Strategy and cost-model implementation versions
participate in run identity, with strategy versions retained in trade
provenance. Canonical configuration snapshots prevent later caller mutation
from rewriting a completed result. Runs are deterministic and their tabular
records remain suitable for vectorized analysis. The state machine is
single-threaded and intentionally does not simulate order books or intraday
events.

Changing the source-of-truth execution ordering or same-bar eligibility is a
research-semantics change requiring a new ADR, schema/version review, golden
fixture review, and explicit compatibility notes.

## Alternatives considered

- Same-close vectorized fills were rejected because a close-derived signal is
  not actionable at that close.
- Fully vectorized cash and trade accounting was rejected because it obscures
  sequential affordability and audit invariants.
- A general event-driven exchange simulator was deferred because partial fills,
  liquidity, order books, and intraday events are outside QF-5.

## Validation

Golden fixture tests verify signal, next-session order/fill, price impact,
commission, fees, cash, trade, equity, drawdown, and metric values. Additional
tests cover weekend/holiday calendar semantics through QF-4, final-session
nonexecution, future-bar causality, accounting invariants, generic strategy
compatibility, strategy-version provenance, complete-equity-depletion returns,
cost-model-version identity, deeply immutable configuration provenance, stable
serialization, and repeated equivalent runs.
