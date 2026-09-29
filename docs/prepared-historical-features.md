# Prepared historical context and feature series (QF-63)

QF-63 prepares invariant historical context and indicator work once per QF-39
permitted-context scope, then serves each scheduled decision from exact causal
positions. Scientific semantics, identities, schemas, schedules, checkpoints and
indicator mathematics are unchanged. See
[ADR 0035](decisions/0035-prepare-historical-context-and-feature-series.md).

## What each decision repeated before QF-63

QF-59 made finding the causal bars cheap. Using them still repeated work that
grew with visible history. In a no-candidate decision, per visible bar:

- **Materialization.** `TimeframeBarSeries._from_validated_artifact` re-sorted
  the selected bars, and `build_multi_timeframe_context` plus
  `TimeframeContext`/`MultiTimeframeContext` validation re-sorted and rescanned
  them.
- **Identity hashing.** Sort keys, `visible_bar_ids`, indicator outputs,
  `validate_output`, rule-timeframe manifests and `values_primitive()` each
  called `bar.bar_id`, a property that reserializes and SHA-256-hashes the bar.
  That is about 12 hashes per visible bar, at 16.6 us each (daily: 52 us).
- **Indicators.** Each indicator was recomputed from scratch (TA-Lib itself was
  about 1% of the decision), then validated and compared bar by bar.
- **QF-28 and QF-52 checks.** Symbol, adjustment basis and lineage were
  rechecked for every visible bar, including a feed-scope hash per bar.
- **Mutation snapshots.** `values_primitive()` serialized every bar and every
  indicator row three times: at build, before the rule and after it.

Separately, every new QF-42 iterator normalized the study's outcome source into
a new QF-58 backing. The QF-61 exact-backing check then missed, so QF-32 trials
2..k (and each QF-7 configured outcome) re-authenticated all 48,480 source bars.
That happened on every decision, about 1.1 s each.

## Fresh baseline (main `ab0089f`, after QF-62)

The frozen real QF-45 inputs are 48,480 2m and 250 daily bars, with the fixed
8/48/daily-50 study on the fold-0 selection window (4,095 decisions). The
disposable harness is `reports/qf63-prepared-features/qf63_profile.py` (ignored).

- Every executed decision was compared byte for byte with the completed QF-45
  window; all matched.
- Each timing run uses a fresh dataset session, which models one trial. Figures
  are uninstrumented steady-state means excluding the first decision.

| Position | Visible 2m bars | s/decision |
| --- | ---: | ---: |
| Sequence 0-39 | 62-101 | 0.0805 |
| Sequence 188-199 (daily start shift at 194) | ~250 | 0.1207 |
| Sequence 2000-2039 | ~2,060 | 0.5907 |
| Sequence 4040-4079 | ~4,100 | 1.1309 |
| Candidate decisions (7 sampled, sequence 460-4024) | 520-4,085 | 0.214-1.096 |

The completed QF-45 run averaged 0.617 s/decision for the fixed trial. The
selection trials averaged 0.632, 1.612 and 1.611 s/decision; later trials paid
the per-decision re-authentication. In a shared session, the baseline paid
**20/20** re-authentications in a 20-decision late sample of trials 2 and 3,
about 2.09 s/decision.

Late invocation counts, per decision (mean of 10):

| Operation | Baseline | QF-63 |
| --- | ---: | ---: |
| Intraday `bar_id` hashes | 49,398 | **0** |
| Daily `bar_id` hashes | 700 | **0** |
| `IntradayBar.to_primitive()` | 61,748 | **0** |
| Full `values_primitive()` traversals | 3 | **0** |
| Indicator calculations / TA-Lib computes | 3 / 3 | **0 / 0** |
| Normalized output-array constructions (validated) | 3 | **0** (3 prefix slices) |
| Full context materialization/validation | 1 | **0** (trusted O(1)) |
| Per-decision selected-series constructions | 2 | **0** |
| Canonical JSON encodes | 117,202 (~170 MB) | 230 (~14.2 MB) |
| Snapshot decodes | 123 (13.8 MB) | 83 (13.0 MB) |
| Context identity hash / study identity | 1 / 1 | 1 / 1 |
| Prepared feature lookups | 0 | 3 |

Late cProfile, 30 decisions (seconds per decision, with profiler overhead;
nested rows overlap):

| Component | Baseline | QF-63 |
| --- | ---: | ---: |
| QF-42 decision (iterator) | 2.0160 | 0.0288 |
| `run_prediction_study_in_session` | 2.0048 | 0.0182 |
| `_prepare_prediction_context` | 1.4719 | 0.0042 |
| `build_prediction_rule_context` | 1.0594 | 0.0020 |
| `values_primitive()` x3 | 0.7065 | not called |
| Indicator evaluate (x3) / TA-Lib | 0.2752 / 0.0131 | not called |
| `validate_output` (x3) | 0.2498 | 0.0001 |
| Context selection + construction (`get_context_at`) | 0.2699 | 0.0002 |
| `MultiTimeframeContext.to_primitive` | 0.2504 | 0.0006 |
| Context source validation | 0.0151 | 0.0001 (incremental) |
| Prepared values guard: capture / verify | n/a | 0.0004 / 0.0030 |
| `PredictionWindowDecision` snapshot | not isolated | 0.0082 |
| Study identity (`_stable_id`) | 0.0059 | 0.0060 |
| Compact conversion (`from_embedded`) | 0.0105 | 0.0105 |

## Prepared architecture

```text
QF-39 provider (plan, window, role, immutable family)          once per scope
  QF-59 PreparedPredictionContext: plan ID, window, source indexes
  QF-63 PreparedContextScope
    PreparedTimeframeRun per timeframe
      bars [window_start - warm_up - 1, bisect_right(window_end))
      one-time validation, bar IDs, end timestamps, integrity shadow
    indicator series store (content-keyed), validation spans, statistics

decision N
  QF-59 bounds_for -> [start, stop)                            O(log n)
  run slice + O(1) causal proof -> MultiTimeframeContext         C slices
  prepared series prefix per indicator -> TimeframeIndicatorOutput
  incremental QF-28/QF-52 checks for newly visible bars only
  rule; guard verifies identities of visible data
```

`PreparedPredictionContext.bounds_for` applies every existing `select`
membership check; `select` is now a thin wrapper around it.
`PredictionSourceIndex.extent` gives the smallest run containing every slice of
the window: warm-up plus the pre-window anchor, through the window end.

### Runs

`PreparedTimeframeRun.prepare` admits a run only if all of these hold:

- the bars are one reviewed exact type (`IntradayBar` or `AggregatedSessionBar`),
  on the source's timeframe;
- every bar is completed (no developing bars);
- bars are strictly ordered by end and do not overlap;
- bar IDs are unique;
- every nested value is a reviewed QF-58 record, tuple or immutable scalar.

The source is the QF-58 normalized backing already admitted by QF-59; no price
is copied. Any failure returns `None`, and the provider keeps the reference
path for the whole scope.

Runs end at the window's last permitted cutoff. A selection scope therefore
never physically contains test or holdout bars.

### Contexts

`PreparedContextScope.context_at` converts the QF-59 absolute positions to run
positions. It then requires `end[stop-1] <= as_of < end[stop]` (O(1)) and slices
the run. Because the run was proven sorted, unique and completed-only, the
slice is exactly the reference `completed and end <= as_of` view.

`MultiTimeframeContext._from_prepared_runs` and
`TimeframeContext._from_prepared_run` recheck declarations, family consistency,
manifest, age/availability and the latest bar in O(1) per timeframe. They skip
only the full sorts and scans already proven for the run.

Prepared contexts carry their visible IDs (`TimeframeContext.visible_bar_ids`).
Their `to_primitive()`, `context_id` and the QF-28 manifests reuse those IDs and
never re-hash bars. `MultiTimeframeContext.prepared_evidence` holds the binding.
It is not compared, hashed or serialized, and rules never receive this object.

Reference contexts compute `visible_bar_ids` exactly as before.
`ConfiguredTimeframeIndicator.calculate` reuses the aligned view's IDs only when
no developing bar was filtered, so its output is unchanged in both cases.

### Indicator series and the compatibility key

A series is identified by `configuration_identity` of:

- `quantforge_prepared_indicator_series`, preparation version `1`;
- the **run ID**, which hashes the dataset reference with feed scope, the family
  manifest, the full timeframe/session configuration, the absolute source
  positions and every ordered bar ID;
- the absolute **input start**;
- the bound QF-22 **timeframe indicator configuration ID**, which covers the
  indicator configuration (definition version, parameters, backend identity
  with library and runtime versions, normalization), source fields, timeframe,
  completion policy, warm-up, and the dataset reference and feed scope;
- the exact **indicator type** and **backend type**.

Python object identity is never part of the key. Validation plan, fold, role and
window are bound by scope ownership: each QF-39 provider owns one scope, and
providers are created per selection or test invocation. The two-fold QF-39 test
shows that fold and role scopes never share state.

The series is computed once by the unchanged `bind_indicator(...).calculate(...)`
path. It runs on a one-timeframe context holding exactly `run.bars[start:]`, and
the normal `TimeframeIndicatorOutput` constructor validates it once. A decision
with visible `[start, stop)` receives a structural prefix of `stop - start`
positions: fresh tuple slices and fresh `IndicatorFieldOutput` shells.

### EMA warm-up and seed equivalence

Current EMA values depend on their input start: the seed is the SMA of the first
`period` values. QF-63 never computes a full-source EMA and slices it. The series
input start is the decision's own start, so every value sees exactly the input
the reference sees.

- The 2m start is fixed by the window warm-up.
- The daily start shifts by one bar at the first in-window close (QF-45 sequence
  194). That creates a second daily series. Both starts are exercised by the
  tests and the real data.

This equals the reference only if each output position depends on no later
input. `tests/unit/prediction/test_prepared_features.py` proves this for every
admitted pair, 14 in total: SMA, EMA, Wilder RSI/ATR/DMI, Bollinger, MACD and
stochastic, on `talib_v1` and, where supported, `native_v1`.

- Prefixes of many lengths are compared with exact `Decimal.as_tuple()`
  representation, not only numeric equality.
- Sentinel future bars cannot change earlier values.
- A later start produces a different EMA, which is why the start is part of the
  key.

`talib_v1` already enforces default compatibility and zero unstable period.
Warm-up rows remain `None` in both paths, including insufficient history.

As defense in depth, the **first use** of every series in a scope is proven
equal, including Decimal representation, to the reference evaluation of that
decision's context. On a mismatch the series is rejected and the decision uses
the reference evaluation. Real QF-45 and all fixtures recorded **0** rejections.

### Causality

- Runs stop at the window end; the provider's schedule set still rejects
  non-permitted decisions.
- Each context slice ends at the inclusive bar-end cutoff, proven in O(1).
  Developing bars cannot enter a run. The current session's daily bar becomes
  visible only at its close, exactly as QF-20 requires.
- The rule-facing `PredictionRuleContext` holds only visible bars, their IDs,
  the prefix outputs and a guard over visible data. The guard references a
  data-free counter/flag object, never the scope.
- Future-bearing runs and series stay provider-owned. They are reachable only
  through `MultiTimeframeContext` private state, which rules never receive.
- Outcome sources remain separate (QF-61) and are never attached to contexts.

The sentinel test replaces every 2m and daily bar after a cutoff with absurd
prices. Rule inputs at and before the cutoff stay byte-identical, although the
prepared run physically contains the sentinel bars.

### Incremental validation

QF-28 symbol/adjustment checks and QF-52 per-bar lineage (`_validate_source`)
run only for bars not yet checked under the same exact key. The key is the run,
plus the symbol or basis, or plus the exact retained dataset metadata object.
Because checks are per bar, a decision fails exactly when the reference scan of
its visible bars would. The family manifest, reference binding and session
policy are still checked every decision.

### Mutation safety

Prepared rule contexts replace the three `values_primitive()` snapshots with a
`PreparedValuesGuard`, captured at build and verified after the rule (and by
`validate_values_unchanged()`). It compares three things:

- every per-decision wrapper field by identity: rule context, timeframe inputs,
  named outputs, output shells, field outputs, adjustment basis, snapshot and
  timeframe records;
- the requirement snapshot;
- every visible bar and every nested record reachable from visible bars,
  field by field by identity, against pristine evidence captured at run
  preparation.

Frozen records can change only by bypass (`object.__setattr__`), which replaces a
field object. This detects the same changes as reserialization.

A changed shared bar or record marks the scope compromised. Every later
`context_at` then raises `PreparedFeatureIntegrityError`, so later decisions and
other trials fail closed. This is stricter than the reference path: there, a
bypass mutation made before a later decision's pre-rule snapshot would go
unnoticed. Changes to decision-owned shells fail only that decision; later
decisions get fresh shells. Reference contexts keep the exact existing snapshot
comparison, and study messages are unchanged.

### Explicit fallback

The unchanged reference path (QF-59 selection, QF-20/QF-28 construction,
per-decision evaluation and snapshots) is used when:

- the context policy is developing-bar;
- any source graph fails QF-58/QF-59/QF-63 admission;
- the scope was not prepared;
- an indicator is not a reviewed exact type on a reviewed exact backend (for
  example a subclass or custom indicator);
- a caller supplies an `indicator_output_cache`;
- the provider is not the QF-39 provider;
- a series is rejected at first use.

Each fallback is counted and scientifically identical.

## Scientific identity equivalence

Preparation is operational; no operational ID enters any artifact. The
following are byte-identical between paths (equal serialized QF-42 windows and
schema 2/3 compact bytes):

- context IDs and QF-52 view ancestry;
- normalized indicator configuration IDs;
- study/prediction/candidate/outcome/evaluation IDs;
- QF-48 schedule membership;
- compact results.

Rule inputs, including every bar and indicator value of every decision, are
compared through `values_primitive()` in the equivalence test. On real data, all
executed QF-45 decisions matched the completed windows: fixed early, daily
transition, mid, late, 7 candidates, and all three selection trials.

## Cross-trial reuse and candidate export

`PreparedOutcomeSources.copy_memo` retains the QF-58 normalized backing per
original outcome source, for the life of the dataset session.
`iter_prediction_window_decisions` and `PreparedOutcomeSources.prepare` use it.
Later iterators and QF-7's configured outcomes therefore hit the exact-backing
check. The original is a capability; it was recursively verified immutable when
first normalized. The QF-58 old-copy test hook is preserved.

Real QF-45 three-trial grid, one provider and one shared session (90 decisions):

- 7 series were built: 2m EMA 8/12/40/48/60 and daily EMA50 from two starts.
- They were reused 263 times, with 7 first-use proofs and 0 fallbacks.
- There were 0 outcome re-authentications after the first.

Candidate export (QF-7/QF-29) reuses the same capability automatically: the
QF-39 provider's contexts carry the binding into `build_prediction_rule_context`,
and captured contexts are reused for every configured outcome. Feature schema,
columns, outcomes and manifests are unchanged; exported files are byte-identical.
Candidate export also regained the outcome-backing reuse. On the real QF-45
inputs, three sampled candidates were exported through the unchanged QF-7 builder
with a fresh QF-39 provider, as the QF-45 example does.

| Per three candidates | Baseline | QF-63 |
| --- | ---: | ---: |
| Seconds per candidate | 17.06, 17.12, 17.00 | 9.23, 9.21, 9.57 |
| Indicator calculations | 72 | 6 (3 series builds + 3 first-use proofs) |
| `values_primitive()` traversals | 66 | 0 |
| Intraday `bar_id` hashes | 144,046 | 5,243 |
| `IntradayBar.to_primitive()` | 1,201,440 | 296,123 |

The remaining export time is mainly QF-7's two separate dataset sessions per
candidate. Each authenticates the 48,480-bar outcome source once (two
authentications per candidate, down from seven). Sharing QF-7 sessions across
candidates is feature-export plumbing and is left to follow-up work.

## Runtime after QF-63 (same host, same harness)

| Position | Baseline s/decision | QF-63 s/decision | Speed-up |
| --- | ---: | ---: | ---: |
| Sequence 0-39 | 0.0805 | **0.0215** | 3.7x |
| Sequence 188-199 (daily transition) | 0.1207 | **0.0207** | 5.8x |
| Sequence 2000-2039 | 0.5907 | **0.0278** | 21.2x |
| Sequence 4040-4079 | 1.1309 | **0.0451** | 25.1x |
| Candidate decisions (sequence 460-4024) | 0.214-1.096 | **0.052-0.069** | 4-16x |
| Trials 2-3, late, shared session | ~2.09 | **0.039-0.040** | ~53x |

Components per late decision: compact conversion 0.0130 s (unchanged) and GC
0.0077 s (was 0.0364). QF-56 append was 1.8 ms in the early sample (2.5 ms
before). The remaining time is dominated by persisted-record work:

- study identity encoding of the ~2 MB context manifest;
- `PredictionWindowDecision` snapshot encode/decode;
- QF-55 compact conversion;
- append validation.

About 83 snapshot decodes (13 MB) and 230 encodes (14 MB) remain per late
decision. They are QF-64's target for no-candidate decisions.

In the synthetic large-history test, every decision sees at least 2,500 visible
2m bars. Reference decisions cost 0.681 s and prepared decisions 0.018 s. The
reference computes 31,038 bar IDs per decision; the prepared path computes 33
over 63 decisions, all for the candidate's bounded QF-11 outcome path.

## Preparation cost and memory

QF-63 one-time work on the real QF-45 inputs:

- **Scope capture** takes **0.10 s** per provider. That covers the 4,157-bar 2m
  run and the 72-bar daily run: validation, 4,229 bar-ID hashes, run identity
  and the integrity shadow.
- **Series builds** are lazy, at first use: 3.4-5.5 ms per 2m EMA over the whole
  run, and 0.2 ms per daily EMA50.
- **First-use proofs** evaluate the reference once per series, at the series'
  earliest decision.

Provider construction as a whole was 1.15-1.40 s before (QF-58/59 normalization
and indexes) and 1.46 s after. First-decision latency, dominated by QF-61
outcome preparation, fell from 1.81 s to 1.72 s. QF-52 projection startup is
unchanged (about 55 s per partition on this host).

Prepared state for the real QF-45 three-trial grid (structural bytes owned by
preparation; bars and nested records are shared, never copied):

| Component | Count | Bytes |
| --- | ---: | ---: |
| 2m run: bar IDs (tuple + strings) | 4,157 | 469,781 |
| 2m run: bar/end/completion tuples | 4,157 | 99,888 |
| 2m run: integrity shadow (field identities) | 4,157 bars + provenance | 1,098,528 |
| Daily run: IDs / tuples / shadow | 72 | 8,176 / 1,848 / 13,312 |
| 2m EMA series x5 (8, 12, 40, 48, 60): values / index tuples | 4,156 each | ~528,000 / 99,864 each |
| Daily EMA50 series x2 (two starts) | 72 / 71 | ~4,100 / ~1,800 each |
| **Total structural report** | | **~4.85 MB** |
| Deep traversal of the scope (excluding shared bars and session metadata) | 37,659 objects | 5,594,873 |

Trial series share the run's bar objects, ID strings and timestamps by
identity; the performance test proves this. There is one run per scope, not per
trial, and no duplicate source array.

## Resume

Preparation is rebuilt deterministically after restart and is never persisted.
QF-56/QF-62 checkpoints remain authoritative. The restart test interrupts a
prepared schema 3 run after five durable appends and resumes with a fresh
provider. That provider executes only the 59 remaining decisions and produces
final bytes identical to the reference compact window. A finalized rerun executes
no decisions.

## Tests

- `tests/unit/prediction/test_prepared_features.py`: prefix stability and
  sentinel invariance for all 14 admitted pairs, and start-dependent EMA seeding.
- `tests/integration/test_prepared_feature_execution.py` (QF-45-like synthetic
  cache: real rule, plan, 61/50 warm-ups and 11:00 candidates; window crosses a
  daily close and a session):
  - byte-equal windows and compact schema 2/3, and equal rule inputs at all 64
    decisions;
  - exact causal prefixes, current-session daily visible only at its close, and
    the rolling daily start shift;
  - the sentinel future test;
  - three-trial series reuse with one outcome authentication;
  - series separated by parameter and backend, and never shared across scopes;
  - bar/provenance bypass mutation, which fails the trial and every later trial;
  - identical rejection on the reference path;
  - shell mutation isolated to its decision;
  - the rule-facing object graph contains no future bar, scope or context;
  - subclass indicators, output caches and developing policy use the reference;
  - custom source records disable preparation;
  - restart with zero committed recomputation;
  - byte-identical QF-7 export;
  - two-fold QF-32/QF-39 selection, ranking, OOS and trial artifacts identical,
    with distinct per-fold/role scopes.
- `tests/performance/test_prepared_features.py`: large-history (2,500-bar)
  invocation counts, startup versus steady state, three-trial reuse, structural
  sharing and prepared-state bytes. Measurements are recorded, not thresholds.
- The existing QF-58/59/60/61/62, QF-42/QF-28, QF-39 and QF-45 suites pass
  unchanged. The QF-58 old-copy hook and QF-59 reference toggle still disable
  their layers.

## Limits and follow-ups

- **Reviewed indicators only.** Volume and custom indicators use the reference
  path until they are reviewed and proven prefix stable.
- **Guard cost.** The guard is O(visible bars) at C level, about 3 ms profiled at
  4,100 visible bars. It is the price of bypass-mutation detection over shared
  backing.
- **Remaining per-decision cost.** It is persisted-record encoding and QF-55
  compact validation. QF-64 (sparse decision receipts) and a later
  manifest-normalization ticket own it.
- **Dense-mode foundation.** Series are trigger-independent and positions
  explicit, so a future dense mode can reuse the same prepared series for every
  eligible timestamp.
