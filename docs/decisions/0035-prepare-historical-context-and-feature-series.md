# ADR 0035: Prepare historical context runs and reusable indicator series

- Status: Accepted
- Date: 2026-09-29
- Jira: [QF-63](https://frostfiredigital-37308542.atlassian.net/browse/QF-63)

## Context

After QF-59 made causal source selection indexed, each historical decision still
rebuilt its complete visible history. It re-sorted and rescanned the selected
bars, recomputed every normalized indicator from scratch, and validated each
output against every bar. It also re-hashed each visible bar's identity about
12 times, walked every bar again for QF-52 lineage, and serialized the complete
rule input three times to detect mutation. `IntradayBar.bar_id` is a property
that reserializes and hashes the bar on every access.

On the frozen QF-45 inputs a late decision (about 4,100 visible 2m bars) cost
1.13 s, of which TA-Lib was about 1%. The history of adjacent decisions differs
by one bar. Separately, every new QF-42 iterator renormalized the study's
outcome source into a fresh QF-58 backing. That missed the QF-61 exact-backing
check, so QF-32 trials 2..k and each QF-7 configured outcome re-authenticated
all 48,480 source bars.

EMA values depend on the exact input start: a later start seeds differently.
QF-45's daily history uses a rolling start that shifts by one bar at the first
in-window close.

## Decision

Add a provider-owned `PreparedContextScope` per QF-39 permitted-context provider,
that is, per plan, window, role and immutable source family.

- **Runs.** For each QF-59 index, prepare the source run
  `[window_start - warm_up - 1, bisect_right(window_end))` once. Validate it once:
  one reviewed exact bar type and timeframe, completed-only, strictly ordered,
  non-overlapping, unique IDs. Hash bar IDs once and capture pristine field
  identities of every bar and nested record. Runs never extend past the window's
  last permitted cutoff.
- **Contexts.** Each decision resolves QF-59 positions `[start, stop)`, slices
  the run and proves the slice is the complete causal prefix with O(1) boundary
  checks. Trusted QF-20 constructors skip only the sorts and scans already proven
  for the whole run. Contexts carry prepared visible IDs and a non-serialized,
  non-compared binding.
- **Series.** Each indicator series is keyed by a content identity: run
  identity, absolute input start, bound QF-22 configuration ID (definition,
  parameters, backend and library versions, timeframe, fields, completion
  policy, warm-up, dataset reference and feed), indicator type and backend
  type. The series is computed once by the unchanged binding and calculation
  path over the run from that start. Each decision receives the prefix ending at
  its cutoff. The rolling daily start therefore creates a second series; no
  full-source series is ever sliced.
- **Admission.** Only reviewed exact indicator types (SMA, EMA, Wilder RSI/ATR/
  DMI, Bollinger, MACD, stochastic) on reviewed exact backends (`talib_v1`,
  `native_v1`) are admitted, and only for completed-only policies. Tests prove
  prefix stability with exact Decimal representation for every admitted pair.
  The first use of every series is also proven equal to the reference
  evaluation of that decision's context; a mismatch rejects the series and
  falls back. Every use, including reuse, repeats the backend's per-evaluation
  checks that `compute()` would apply: TA-Lib's process-global default
  compatibility and zero unstable periods (checked before and after serving,
  as `compute()` checks before and after evaluating), and the current backend
  identity.
  Drift therefore fails exactly as a fresh computation would, regardless of
  cache state.
- **Validation.** QF-28 symbol/basis checks and QF-52 per-bar lineage run once
  per newly visible bar under each exact key. Family, reference and session
  checks still run every decision. A decision fails exactly when the reference
  scan would.
- **Mutation.** Prepared rule contexts skip the three value serializations.
  They carry a guard that holds only visible data: it compares wrapper-field
  identities, the requirement snapshot, and the visible bars' field identities
  against the pristine evidence. A change to shared backing marks the scope
  compromised; every later decision or trial fails closed.
- **Fallback.** Developing-bar policies, unsupported or mutable source graphs,
  unreviewed indicators, output caches and non-QF-39 providers use the unchanged
  reference path.
- **Outcome backing.** The dataset session retains the QF-58 normalized backing
  per original outcome source, so later iterators and QF-7 outcomes hit the QF-61
  exact-backing check.

Preparation never enters scientific identity, results, checkpoints or artifacts.
Resume rebuilds it; QF-56/QF-62 checkpoints stay authoritative.

## Consequences

Per-decision work no longer grows with visible history, except C-level slices and
the O(visible) identity guard. On real QF-45 inputs a late no-candidate decision
fell from 1.13 s to 0.045 s, and QF-32 trials 2..k from about 2.1 s to 0.040 s.
All executed decisions are byte-identical to the completed QF-45 windows. The
dominant remaining costs are persisted-record serialization, QF-55 compact
conversion and append validation; QF-64 targets those for no-candidate
decisions.

Prepared state costs about 5 MB for the QF-45 three-trial grid. Series share the
run's bar and ID objects; no source data is duplicated across trials.

Custom indicators do not gain the optimization until reviewed. The binding can
reach future bars in the run only through private attributes of
`MultiTimeframeContext`, which rules never receive. The rule-facing
`PredictionRuleContext` references only visible data.

No schema, persistence format, indicator formula, EMA math, strategy, sparse
receipt, loading, dense dataset or ML change is made.

## Alternatives considered

- **Full-source EMA then slice.** Rejected: seeding depends on the input start.
- **Memoizing `bar_id` on bars.** Rejected: it would change the reviewed frozen
  record types and QF-58 admission.
- **Removing the post-rule check.** Rejected: bypass mutation of shared backing
  must still be detected.
- **Attaching future-bearing state to the rule context.** Rejected on causality
  grounds.
- **Persisting prepared series.** Rejected: runtime state is operational only.

See [the implementation and evidence](../prepared-historical-features.md).
