# ADR 0038: Separate non-authoritative rapid exploratory scans

- Status: Accepted
- Date: 2026-09-30
- Jira: [QF-72](https://frostfiredigital-37308542.atlassian.net/browse/QF-72)

## Context

After QF-58–QF-65 the authoritative event path is correct and much faster, but
each scheduled decision still builds a QF-11 study identity over its context
manifest, a QF-64 receipt validated against the schedule, a QF-56 fsynced
journal append and checkpoint, and per-decision isolation copies. On the fixed
QF-45 selection window that is about 16 ms per decision, of which well under
1 ms is the rule's scientific work (causal positions, indicator prefixes, the
predicate). Identity and context cost grows with visible history, so a
multi-year window costs on the order of an hour per configuration. That is
right for verification and wrong for screening many hypotheses.

## Decision

Add `quantforge.rapid`, a distinct exploratory execution boundary that shares
the authoritative scientific components and skips the audit machinery:

- **Same science, prepared once.** A `RapidResearchSession` covers one QF-8 plan
  window. It runs the unchanged QF-8/QF-39 `partition` (retained/purged
  membership, QF-52 bounded view), builds the QF-42 schedule, captures QF-59
  `PreparedPredictionContext` and one QF-63 `PreparedContextScope`, and opens a
  QF-11 dataset session with its QF-61 outcome registry. It joins or owns a
  QF-65 canonical preparation. Nothing is persisted; closing releases it.
- **Shared kernel.** A rule opts in with `rapid_specification()`: the causal
  scalars its predicate reads (indicator outputs or OHLCV fields at a lag), an
  optional decision-clock window, and the **same** predicate and candidate
  builder the authoritative `generate_with_context` calls. QF-45's
  `midday_bullish_cross` is the first admitted kernel. No strategy is
  implemented twice.
- **Exact values by prefix.** Decisions sharing per-timeframe input starts (the
  QF-63 series key) form a segment. The unchanged `build_prediction_rule_context`
  runs once at each segment's last decision, with QF-63 first-use proofs,
  QF-28/QF-52 checks and the values guard. Every member reads
  `values[stop - start - 1 - lag]`, the value its own prefix contains. The
  kernel receives only scalars at or before its cutoff.
- **Outcomes only for triggers.** Triggers construct the unchanged
  `OutcomeEvaluationRequest`, resolve through the QF-61 prepared source, and run
  the configured QF-47/QF-49 labelers and evaluators, keeping their value
  fields. No outcome is labeled for non-trigger decisions.
- **Explicit admission.** Unreviewed rules, indicators (not QF-63 prefix-stable
  exact types), developing-bar or staleness policies, insufficient declared
  warm-up, outcomes not on the canonical primary source or beyond the plan's
  purge horizon, and kernel/candidate disagreement raise errors. Nothing is
  approximated.
- **Holdout isolation, no override.** Only DEVELOPMENT and SELECTION windows are
  accepted. The full footprint (first warm-up bar through the last decision
  plus maximum outcome reach) must avoid the plan's final holdout and every
  reserved or consumed exposure scope for the symbol in the research
  workspace's permanent `HoldoutLedger` at `reports/holdout-ledger` (read
  through a new read-only `exposure_scopes()`). Sessions take the workspace
  root, never a ledger object, and never create a ledger.
- **Unmistakable results.** `RapidScanResult` is unrelated to any authoritative
  type. `authoritative` is a constant `False`, `mode` is `"exploratory"` and
  every serialization carries the notice. It has no result, study, context,
  receipt or checkpoint identity. Exports use the reserved `.rapid.json` suffix
  and QF-9 `verify_artifacts` refuses that suffix and any JSON document declaring
  `"authoritative": false`, so exports can never be indexed, reported or
  published.
- **Promotion by configuration only.** `promote_strategy_configuration` freezes
  the exact rule configuration; `require_same_rule` proves the authoritative
  component built from those parameters has the identical configuration ID.

## Consequences

- Multi-year scans become interactive: one session's preparation, then
  per-configuration cost dominated by first-time series builds and trigger
  outcome labeling. See [rapid strategy scans](../rapid-strategy-scan.md).
- Rapid omits identities, receipts, journals, checkpoints, resume, manifests and
  QF-9/QF-40/QF-41 consumption by design. A promising result must be reproduced
  through the authoritative QF-32/QF-39/QF-40 workflow.
- Rapid sweeps increase selection bias. They are exploratory evidence only.
- Authoritative behavior is unchanged: QF-45's rule refactor preserves its
  configuration, identity and output bytes; the QF-63 admission predicate and
  ledger listing are additive; the QF-9 refusal affects only self-declared
  non-authoritative documents.
- New rules become rapid-compatible only by declaring a reviewed specification
  over admitted prepared inputs; dense all-bars datasets (QF-70) and ML datasets
  (QF-67) remain separate.
