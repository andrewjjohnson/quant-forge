# Rapid exploratory strategy scans (QF-72)

> **NON-AUTHORITATIVE / EXPLORATORY.** A rapid scan is hypothesis screening. Its
> results must be reproduced through the authoritative QuantForge pipeline
> (QF-32/QF-39/QF-40) before any research conclusion. Rapid output is never a
> QF-9 artifact, QF-42 window, QF-39 selection, OOS result, holdout result or
> published report, and it can never read the reserved final holdout.

QF-72 adds a separate exploratory execution boundary, `quantforge.rapid`. It is
fast because it skips durability, identity, audit and publication machinery,
**not** because it loosens market, calendar, indicator, rule, signal, outcome or
causal semantics. See [ADR 0038](decisions/0038-separate-rapid-exploratory-scans.md).

```text
                    shared scientific semantics
                    ───────────────────────────
                    QF-65 authenticated canonical data
                    QF-8 plan windows, warm-up, purging
                    QF-59 causal positions, QF-63 prepared series
                    shared rule kernel (same predicate)
                    QF-61 prepared outcomes, QF-47/49 labelers
                            │
             ┌──────────────┴──────────────┐
             │                             │
      RAPID / EXPLORATORY           AUTHORITATIVE
      quantforge.rapid              QF-11/QF-42/QF-32/QF-39
             │                             │
      indexed in-memory scan        per-decision study identity
      triggers only get outcomes    schema-4 receipts, QF-56 fsync
      no journal/checkpoint         checkpoints and resume
      no QF-9 index/publication     QF-9/QF-40/QF-41, holdout
             │                             │
      hypothesis screening          hypothesis verification
```

## Baseline: where authoritative time goes (post-QF-65 main `f07aef2`)

The ignored harness `reports/qf72-rapid-scan/qf72_baseline.py` runs the frozen
QF-45 inputs through the production `load_inputs → prepare_walk_forward(fixed)
→ validate → select` path for the complete fixed 8/48 **selection** window
(2025-05-01 → 2025-05-30, 4,095 decisions, 25 candidates) with schema-4 sparse
receipts. It never touches the test window, the holdout or the ledger. One fresh
process, warm file cache, 14-core host:

| Stage | Seconds |
| --- | ---: |
| Canonical 1m source (QF-65 authentication) | 7.71 |
| Derived 2m (aggregate + persisted-cache verification) | 11.24 |
| Daily | 0.87 |
| Canonical prediction input (incl. QF-51 evidence rebuild) | 16.47 |
| 2m/daily `TimeframeBarSeries` | 0.87 |
| Plan and QF-39 adapter | 2.02 |
| Plan validation | 0.66 |
| **Startup subtotal** | **39.9** |
| QF-8/QF-52 partition (incl. bounded projection) | 2.61 |
| QF-59/QF-63 scope preparation | 1.17 |
| 4,095 decisions (QF-42 → QF-11 → QF-64 receipt → QF-56 append) | ~66.2 |
| Window finalization (independent re-validation) | 0.84 |
| **Total** | **114.5** |

Max RSS 1,862 MB. Per ordinary (no-trigger) decision, uninstrumented component
timers (nested rows overlap; ~16.2 ms/decision in total):

| Work per decision | ms | Exists for |
| --- | ---: | --- |
| QF-11 study identity (`_stable_id` over configuration + ~context manifest) | 5.15 | identity/audit |
| QF-64 receipt (context decode, schedule/scope binding, `validate_no_prediction`) | 2.26 | persistence + offline validation |
| Rule-context preparation (QF-52 source check, QF-63 build, manifest snapshot) | 2.33 | mostly audit; ~0.3 ms positions/slices |
| QF-63 values guard verification (profile) | ~1.2 | mutation audit |
| QF-11 per-decision temporal/labeler configuration re-validation (profile) | ~1.0 | configuration identity |
| QF-42 per-decision study configuration capture | 0.56 | identity |
| QF-56 append: 3 fsyncs, checkpoint publication | 0.63 | durability/resume |
| QF-42 study template deep copy | 0.17 | component isolation |
| Indicator prefix lookups (with backend state checks) | 0.19 | **scientific** |
| Rule evaluation (`generate_with_context`) | 0.25 | **scientific** |

A trigger adds outcome resolution (0.4 ms) and labeling (0.36 ms) plus a rich
observation snapshot and its compact validation. The scientifically necessary
work per decision is well under 1 ms; the rest is identity, persistence,
resume, offline validation and isolation. cProfile shows canonical JSON
encoding (`iterencode`, 42 s of 150 s profiled) as the dominant primitive.

## Rapid boundary

A `RapidResearchSession` is opened for exactly one QF-8 plan window:

```python
from quantforge.oos import HoldoutLedger
from quantforge.rapid import rapid_research_session
from quantforge.validation import PartitionRole

with rapid_research_session(
    plan=config.plan,
    fold_index=0,
    role=PartitionRole.DEVELOPMENT,
    dataset=inputs.dataset,  # QF-65 authenticated canonical input
    series=(inputs.primary, inputs.daily),
    holdout_ledger=HoldoutLedger(Path("reports/holdout-ledger")),
) as session:
    for fast, slow in ((8, 40), (8, 48), (12, 60)):
        result = session.scan(
            EmaSmokeRule(EmaParameters(fast, slow)),
            outcomes=configured_outcomes(inputs.primary),
        )
```

Session preparation (once): the QF-8/QF-39 `partition` (exact retained and
purged membership plus the QF-52 bounded view), the QF-42 schedule, QF-59
`PreparedPredictionContext`, one QF-63 `PreparedContextScope`, the QF-11
dataset session and its QF-61 outcome registry, and the holdout checks. Nothing
is persisted; everything is released on exit.

Each scan: admit the rule and outcomes, select eligible timestamps, resolve
causal positions, read prepared series at those positions, evaluate the shared
kernel, and label outcomes **only for triggers**.

## Shared strategy kernel

The rule owns one pure predicate over causal scalars. For QF-45,
`examples.spy_ema.midday_bullish_cross(clock, previous_fast, previous_slow,
current_fast, current_slow, daily_close, daily_ema50)` is the complete rule. The
authoritative `EmaSmokeRule.generate_with_context` reads those values from its
`PredictionRuleContext` and calls the predicate; its candidate builder is shared
too. The rapid specification declares which causal values the predicate needs
and delegates to the same functions. The rule configuration, implementation
version and identities are unchanged.

## Capability / admission contract

A rule is rapid-compatible only when it declares `rapid_specification()`:

- `RapidIndicatorInput(name, timeframe, alias, output, lag)` or
  `RapidBarInput(name, timeframe, field, lag)` for every value the kernel reads;
- an optional `RapidDecisionWindow(start, end, timezone)` (exchange-local,
  inclusive), frozen before scanning;
- the kernel `decide(clock, values)` and the candidate builder.

Admission fails with `RapidAdmissionError` unless: every declared timeframe is
completed-only with no staleness limit; every indicator is a QF-63 reviewed,
prefix-stable exact type on a reviewed backend; every alias/output/bar field
exists; the plan window's declared warm-up covers each input and lag; and the
source graph passes QF-58/QF-59/QF-63 admission. Outcomes must be exact reviewed
QF-47/QF-49 labeler/evaluator pairs on the session's canonical primary source,
with future reach within the plan's purge horizon. Rules without a
specification (or any unreviewed component) are refused, never approximated.

## Prepared feature access

Positions come from QF-59 `bounds_for` for every eligible decision. Decisions
sharing the same per-timeframe input starts (QF-63's series key; the daily start
shifts once at the first in-window close) form a group. For each group, the
QF-63 prepared rule context is built once, through the unchanged
`build_prediction_rule_context`, at the group's last decision. That includes
first-use series proof, QF-28 symbol/basis checks and QF-52 source checks for
every bar any member can see. Every member decision then reads
`values[stop - start - 1 - lag]`: by QF-63 prefix stability this is exactly the
value the authoritative rule receives at that decision. The kernel only ever
receives scalars at or before its own cutoff. Series are shared across
configurations by QF-63's content key.

## Holdout enforcement

At session entry, before any market value is read:

- only `DEVELOPMENT` and `SELECTION` plan windows are accepted;
  `WALK_FORWARD_TEST` (authoritative OOS; AGENTS.md forbids tuning on it) and
  `FINAL_HOLDOUT` raise `RapidScopeError`/`RapidHoldoutError`;
- the full data footprint (first warm-up bar through the last eligible decision
  plus the maximum configured outcome reach) must not intersect the plan's final
  holdout;
- the footprint's exchange sessions must not overlap **any** holdout exposure
  scope (reserved or consumed, same symbol) in the permanent
  `HoldoutLedger` supplied by the caller.

There is no override flag. The ledger is audited and read once, read-only, when
the session opens (its records can be large); each scan re-checks its own
maximum outcome reach. The research interval itself is checked before any
market data is prepared, and the warm-up footprint right after the QF-59/QF-63
positions are known.

## Result contract

`RapidScanResult` is a new, unrelated type (no subclass or alias of any
authoritative result). `authoritative` is a constant `False` property, `mode` is
`"exploratory"`, and the notice is attached to every serialization. It holds
the exact strategy configuration, source references, research window, counts,
ordered lightweight `RapidEvent`s (timestamp, session, direction, optional rule
values, configured outcome value fields), outcome configurations, exploratory
aggregates and chronological summaries, and profile statistics. It has no
result, study, context, receipt or checkpoint identity.

## Promotion

`promote_strategy_configuration(result)` returns only the frozen rule
configuration (name, version, configuration ID, configuration and parameters)
and the research window it came from. `require_same_rule(rule)` proves that the
authoritative component built from those parameters has the identical
configuration identity. Nothing about the rapid result becomes authoritative.

## Promotion and export workflow

```python
from quantforge.rapid import export_rapid_scan, promote_strategy_configuration

promoted = promote_strategy_configuration(result)  # configuration only
rule = EmaSmokeRule(EmaParameters.from_primitive(promoted.parameters))
promoted.require_same_rule(rule)  # identical config ID
export_rapid_scan(result, Path("reports/exploration/ema.rapid.json"))
```

The authoritative run then builds its study through the normal factory (for
QF-45, `EmaStudyFactory.build({"ema_pair": "8/48"})`, whose rule passes
`require_same_rule`) and runs the QF-32/QF-39/QF-40 workflow. Compare its
triggers with the rapid scan and continue only if they reproduce. Exports carry
the notice, `"authoritative": false`, the rapid schema, the exact strategy
configuration, source references and the research window. They must end in
`.rapid.json` and are refused inside a holdout ledger or an authoritative study
directory. QF-9 refuses to index them.

## Equivalence evidence

Real frozen QF-45 inputs (`reports/qf72-rapid-scan/qf72_rapid.py`, ignored), fold-0
selection window 2025-05-01 to 2025-05-30, real permanent ledger:

| Comparison | Authoritative | Rapid | Result |
| --- | ---: | ---: | --- |
| Permitted decisions (QF-8 retained) | 4,095 | 4,095 | equal |
| Eligible decisions (11:00-14:00 New York) | 1,911 by clock filter | 1,911 | equal set |
| 8/48 triggers vs schema-4 baseline window | 25 | 25 | timestamps, order, direction, all six features, 30m outcome identical |
| 8/40 / 12/60 triggers vs completed QF-45 trial windows | 29 / 18 | 29 / 18 | identical |
| Configured outcomes vs completed QF-45 QF-7 exports | 72 triggers | 72 triggers | 4,824 outcome fields, 0 mismatches |
| Repeated scan | | | scientific primitive identical |

The synthetic QF-45 composition tests (`tests/integration/test_rapid_*.py`)
record every call of the shared predicate on both paths and require rapid's
arguments to equal the authoritative ones, including exact `Decimal`
representation, at **every eligible decision**, not only triggers. They cover a
normal session, the 2024-12-24 13:00 early close, the 2024-12-25 holiday gap, the
first-decision warm-up boundary, the first in-window daily close (the QF-63
rolling daily start), completed-only daily context, a 120-minute path ending
exactly at an early close, two other EMA pairs on the development window, and
sentinel future bars (inputs at or before the cutoff are byte-identical; later
ones change).

After the kernel refactor the authoritative fixed selection window is
byte-identical to the pre-change baseline (6,820,107 bytes, SHA-256
`0b8f4fb187e79631b261b35e455cd632cfcb1b6fd2720bf3d601625da77ae328`).

## Benchmarks

### Real data: frozen QF-45 2025 cache (one process, warm file cache)

| Stage | Seconds |
| --- | ---: |
| QF-65 input preparation (1m, 2m, daily, prediction input, series) | 36.7 |
| Plan | 2.4 |
| Rapid session open (selection window) | 5.3 |
| Scan 8/48 with six outcomes (first; includes 1.19 s one-time QF-61 preparation) | 1.47 |
| Scan 8/40 with six outcomes (marginal) | 0.27 |
| Scan 12/60 with six outcomes (marginal) | 0.17 |
| Scan 8/48 without outcomes | 0.011 |
| Process to first result | ~46 |

The same window through the authoritative path took 114.5 s end to end, about
66 s of it in the 4,095-decision loop. The 2025-03-19 to 04-30 development
window (2,730 eligible decisions) scans in 0.02-0.09 s per pair and produces no
trigger: daily closes were below their EMA50 throughout.

### Four years: synthetic canonical cache

The only cached multi-year SPY 1m dataset (2023-2025) is missing four 1m bars
(2023-06-05 09:52-09:56 ET). Strict QF-18 aggregation and the canonical QF-3
prediction input fail closed on it. Repairing or bypassing that is out of
scope, so the multi-year benchmark (`qf72_synthetic.py`, ignored) builds a
complete 2021-01-04 to 2024-12-31 SPY 1m RTH cache with the same offline
fixture machinery as the tests. It has 1,005 real XNYS sessions (holidays, early
closes, DST) and 390,690 bars, with synthetic prices that trigger at 11:00
each session. It then runs the normal QF-65 load path. This measures speed
only; it is not research evidence.

Development window 2021-03-22 to 2024-10-31: 177,195 permitted and 82,751
eligible decisions, 911 triggers per pair.

| Stage | Seconds |
| --- | ---: |
| Canonical 1m source / derived 2m / daily | 30.6 / 45.4 / 3.4 |
| Canonical prediction input / series / plan | 64.2 / 3.5 / 9.7 |
| Rapid session open (QF-8 partition 8.7, QF-59/QF-63 8.0, ledger 0.9) | 19.9 |
| Scan 8/48 with six outcomes (first) | 17.2 |
| Scan 8/40, 12/60 with six outcomes (marginal) | 8.2, 8.4 |
| Scan 8/48 without outcomes / new pair 9/21 without outcomes | 0.40 / 0.77 |
| Process to all five results | 212 |

Peak RSS was 5.2 GB, mostly the authenticated 1m/2m inputs. The kernel loop over
82,751 decisions takes about 0.05 s (about 0.6 us each). The first scan's
time goes to the one-time outcome-source preparation (4.7 s), positions (1.5 s)
and two anchor contexts with first-use series proofs (2.9 s).

The authoritative QF-42/QF-11 path on the same window, uninstrumented and
excluding receipts, journal fsyncs and finalization, costs 6.7 ms per decision
at the start, 186 ms in the middle and 367 ms at the end. Study identity and
context work grow with visible history. That extrapolates to roughly
177,195 x 0.19 s, about 9 hours per configuration, as a lower bound.

### Remaining rapid cost

- One-time per process: QF-65 canonical preparation (about 157 s for four
  years). Each later configuration reuses it.
- Per configuration with outcomes: QF-46/47/49 labeling of triggers through
  the unchanged request/resolution/label/evaluation contracts, about 1.4 ms per
  configured outcome per trigger (92% of a marginal scan). Configuring fewer
  outcomes scales this down directly.
- Per new indicator period: one QF-63 series build and first-use proof at
  the anchor (about 0.1-0.4 s for four years).

No further optimization is proposed; the scan is interactive.

## Limits

- Only rules that declare a reviewed specification are admitted; QF-45's
  `EmaSmokeRule` is the first. Reviewed indicators are the QF-63 prefix-stable
  set; bar inputs are completed OHLCV fields.
- Rapid scans use plan windows. Sub-intervals need a plan window of their own.
- Outcome resolution shares the authoritative coverage semantics: a missing
  endpoint is classified against the canonical source's final coverage
  timestamp.
- There is no resume, persistence, receipt or audit trail. Rerun the scan
  instead.
- No ML datasets, dense observations, model training or automatic promotion are
  provided (QF-67/QF-70 own those).

## Production files

- `src/quantforge/rapid/__init__.py`, `models.py`, `admission.py`,
  `scope.py`, `session.py`, `export.py` (new subsystem).
- `src/quantforge/examples/spy_ema.py`: extract the shared kernel and candidate
  builder; declare the rapid specification (no semantic change).
- `src/quantforge/prediction/prepared_features.py`: expose the reviewed
  prefix-stable admission predicate (no behavior change).
- `src/quantforge/oos/holdout.py`: read-only exposure-scope listing.
- `src/quantforge/experiments/artifacts.py`: QF-9 refuses JSON documents that
  declare `"authoritative": false`.

No authoritative writer, reader, validator, schema, identity, durability or
walk-forward behavior changes.
