# ADR 0017: Compose walk-forward studies from fixed validation and evaluator contracts

- Status: Accepted
- Jira: QF-39

## Context

QF-8 defines leakage-safe chronological membership; QF-42 supplies historical
prediction collections and QF-32 window analysis; QF-43 separates backtest context
from evaluation. QF-39 needs to select configurations in historical windows and
persist reproducible OOS evidence without duplicating these components.

## Decision

Add `quantforge.walk_forward` with a narrow selection/test adapter protocol,
separate prediction/backtest adapters, immutable candidate/freeze/artifact
records, and a local atomic state machine. Consume the ordered QF-8 folds and
all its membership/purge/context helpers unchanged. Use the existing QF-6/QF-32
candidate enumeration, eligibility, ranking and stability calculations.

The candidate universe captures a finite grid beside the QF-8 plan. The plan's
research environment declares allowed indicator/outcome configurations and
maximum context/horizon requirements. Each candidate is validated against those
limits and the fixed rule family/version/backend. The plan's baseline rule is
not rewritten into a universal optimization configuration.

Selection ranks one permitted partition: optional selection when present,
otherwise development. It freezes one eligible configuration, optionally
requiring an existing stable/non-isolated classification. Development and
selection scores are not pooled; no learned model is fitted.

Project evaluator datasets into independently valid QF-3 in-memory ranges so
protected bars and metadata cannot reach selection callbacks. Preserve original
source provenance in study and frozen-selection identity. This projection uses
existing QF-3 canonical serialization and validation, without recalculating or
filling prices. QF-43 still owns evaluation accounting and QF-42 still owns
historical decision scheduling and QF-11 result identity.

Persist the immutable frozen selection before test execution. Retain explicit
partial/completed/failed states and prior failures. Verify completed evidence
before skipping it; retry failed tests only under declared policy and always
with the same frozen selection. Use typed snapshot artifacts for the original
prediction window or backtest result, retaining the existing offline validators
and backtest exports.

## Consequences

Prediction studies retain their own outcomes and decision evidence. Backtests
retain their own orders, equity and accounting. Neither engine, optimizer nor
validation splitter is redesigned. A small additive QF-32 `candidates` property
exposes existing enumeration for universe capture.

The implementation supports one local writer and one selected configuration per
fold. Adapter and underlying engine versions participate in identity. Trusted
Python component contracts remain necessary; bounded input views are not a
process sandbox.

QF-39 stops at per-fold OOS artifacts. Combining OOS results and holdout
consumption belong to QF-40; generic artifact indexing belongs to QF-9 and report
rendering to QF-41.
