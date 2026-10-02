# Conditional ML smoke study (QF-69)

> A platform acceptance study of the conditional ML research workflow, not a
> trading result. It asks: *given that one frozen strategy triggered, can a
> simple model, trained without leakage, describe stronger or weaker trigger
> instances?* A null, negative or unstable answer is acceptable. Nothing here
> is evidence of an edge, of robustness or of profitability.

The study runs the complete chronological lifecycle on real, authenticated
data and stops at the pre-holdout gate:

```text
frozen specification (persisted first)
      ↓
QF-40 reservation of the plan's final holdout (permanent ledger)
      ↓
QF-39: QF-32 trial on selection → frozen selection → walk-forward test
      ↓
frozen-candidate development evidence (QF-69, ADR 0041)
      ↓
QF-67 event dataset (development / selection / test rows)
      ↓
QF-68 fit on development rows only → selection predictions → freeze → re-read
      ↓
frozen OOS predictions → deterministic rebuild/refit → offline reproduction
      ↓
base-strategy QF-40 aggregate, QF-9 manifest, QF-41 reserved report
      ↓
feature-leakage audit → report → pre-holdout gate (holdout RESERVED / UNCONSUMED)
```

Code: `quantforge.ml.study` (generic lifecycle), `quantforge.ml.study_report`
(descriptive statistics), `quantforge.examples.spy_ema_ml_study` (the frozen
design and row audit) and `quantforge.examples.spy_ema_ml_runner` (the
orchestration). CLI: `scripts/run_conditional_ml_study.py`.

## Frozen specification

`specification.json` is written before any research stage. Every later stage
rebuilds the specification from code and refuses to run if it differs. Its
identity (`specification_id`) is bound to every stage record, the report and
the gate.

### Strategy and population

| Item | Frozen value |
| --- | --- |
| Rule | `spy_midday_ema_smoke` v2, unchanged kernel `spy_ema.midday_bullish_cross` |
| Parameters | EMA 8/48 on completed 2m bars; completed daily EMA50 (`talib_v1`, normalized) |
| Signal | UP only when previous fast <= previous slow, current fast > current slow, and the latest completed daily close > its EMA50 |
| Decision window | Decision bar **end** inclusively 11:00–14:00 America/New_York |
| Candidates | One (`8/48`); no EMA pair is compared |
| QF-39 selection | `BEST_ELIGIBLE` over the single candidate, ranked by event `count` (needs one labeled selection event); no return-based selection |
| Population | `accepted_only`; one row per generated signal; no-trigger decisions are coverage, never negatives |
| Source | SPY, Massive, canonical 1m RTH `7e396b64…` (250 sessions of 2025), QuantForge 2m `0447ab4d…` and daily `18936f68…`; XNYS; raw unadjusted provider OHLCV |
| Timestamps | UTC; a decision is the end of a completed 2m bar |

QF-45's configuration, dates, artifacts and its AGENTS.md selection exception
are untouched. QF-69 is a separate plan and study (its own plan, study, lineage
and holdout identities).

### Windows

All are inclusive XNYS sessions of the complete 2025 cache:

| Role | Sessions | Retained decisions |
| --- | --- | ---: |
| Warm-up / context | 2025-01-02 .. 2025-03-18 (61 2m and 50 daily bars) | — |
| Development (training) | 2025-03-19 .. 2025-06-27 | 13,650 |
| *QF-45 reserved scope (excluded)* | *2025-06-30 .. 2025-07-31* | *none* |
| Selection | 2025-08-01 .. 2025-08-29 | 4,095 |
| Walk-forward test (OOS) | 2025-09-02 .. 2025-10-31 | 8,580 |
| Reserved final holdout | 2025-11-03 .. 2025-12-31 | — |

- **Outcome reach and purge.** The plan declares only the 30-minute forward
  return. Its reach (30 minutes plus one 2m alignment interval) is the QF-8
  label horizon.
- **Embargo** is zero. Partitions are separated by at least a weekend, and no
  decision was purged.
- **How the windows were chosen.** Development spans every session after
  warm-up and before QF-45's reserved holdout scope. Selection, test and
  holdout take the next one, two and two months of the cache. No QF-69
  decision or label falls in the QF-45 scope.
- **Why not more data.** The multi-year 1m caches (`f398093f…`, `d0e05395…`)
  fail completeness (four missing bars on 2023-06-05) and are not used. No
  bars were repaired.

**Planning evidence (exploratory).** Before freezing, a QF-72 rapid scan of
the **development window only**, with no outcome requested, counted 55 causal
triggers (28 in May, 27 in June). It was used only to confirm that the frozen
floors were reachable. No selection or test outcome, or trigger, was used to
choose the windows.

### Prior exposure

- **Development** reuses periods inspected in QF-45: its development
  (never executed), May selection (8/48 fixed and comparison trials
  inspected) and June OOS (12/60 inspected). It is training data only and is
  not described as untouched.
- **Selection, test and holdout** (2025-08-01 .. 12-31) have no recorded
  QuantForge strategy evaluation before QF-69. 2025 market history is public,
  so they are not blind in an absolute sense.
- **Warm-up context** for windows after the QF-45 scope reads completed July
  2025 bars as causal context only (no decision, label or outcome), as the
  QF-67/QF-72 session-granular protected-scope contracts allow.

### Features, target and model

- **Features**: the unchanged QF-67 schema `qf45_ema_trigger_features` v1:
  `ema_fast_previous`, `ema_slow_previous`, `ema_fast`, `ema_slow` (2m) and
  `daily_close`, `daily_ema50` (daily). All are known at the decision from
  completed inputs, and null values are rejected. No feature was added,
  selected or engineered.
- **Target**: QF-67 `forward_return_30m_positive` v1, the QF-49
  `raw_return > 0` (`outcome_close / reference_close - 1`, same session). Zero
  is a negative. Unavailable outcomes stay null with their QF-46 status.
  There is no direction adjustment and there are no costs.
- **Model**: QF-68 `qf69_conditional_logistic` v1. It is the unchanged L2
  logistic regression (`lbfgs`, `C = 1`, `tol = 1e-8`, `max_iter = 1000`,
  `random_state = 0`, one BLAS thread) with training-only standardization
  (`exclude_observation` for nulls). It is probability-only: no class
  threshold, calibration fitting, search or second estimator.
- **Minimums** (frozen before fitting; plumbing floors, not statistical
  adequacy):
  - at least 30 fitting rows (five per standardized feature);
  - at least 10 fitting rows of each class;
  - selection and OOS metrics are undefined below 10 labeled scored rows.

  QF-45's one-observation exception does not apply. A partition below a floor
  is recorded with its status, and nothing is relaxed afterwards.
- **Evaluation**:
  - model metrics: log loss, Brier score, ROC AUC where defined, and
    calibration;
  - baseline: a constant training prevalence on exactly the same labeled
    scored rows;
  - five predeclared equal-width score buckets on `[0, 1]`
    (`min(floor(5p), 4)`), each with counts and raw-return summaries;
  - the score distribution;
  - unfiltered base-strategy raw-return summaries per partition.

  Overlapping intraday labels are not independent; no significance is
  claimed.
- **Holdout policy**: the normal QF-40 reservation, made before research. The
  study stops at the gate with the holdout reserved and unconsumed. Any
  consumption is a later, explicitly authorized QF-40 step, with no retuning
  afterwards.

## The development-observation gap

QF-39 runs its QF-32 trials on a fold's selection window, so a three-role fold
never executes development. QF-69 adds `WalkForwardStudy.evaluate_development`.
It executes the frozen candidate on the frozen development membership through
the ordinary partition execution, and records `folds/<fold>/development.json`.
QF-40 `load_prediction_development_window` verifies it offline, and QF-67
admits it. Nothing is relabeled, built from exports, or taken from another
fold. See [ADR 0041](decisions/0041-record-frozen-candidate-development-evidence.md)
and [walk-forward studies](walk-forward-studies.md#frozen-candidate-development-evidence-qf-69).

## Running it

```bash
uv run --frozen python scripts/run_conditional_ml_study.py --preflight
```

```bash
uv run --frozen python scripts/run_conditional_ml_study.py \
  > reports/qf69-conditional-ml.log 2>&1
```

- **Preconditions.** A clean committed checkout (the QF-9 code provenance is
  captured first, in `execution.json`) and the workspace's existing permanent
  ledger at `reports/holdout-ledger`. The CLI never creates a ledger. The
  output root must sit directly under `reports/`.
- **Re-running** verifies and reuses every completed stage. Every stage record
  is immutable and must reproduce exactly, and mixed-code resume is rejected.
- **Network.** None. Inputs come from the authenticated 2025 cache through
  `load_inputs` (QF-65 preparation).

Output root (ignored by Git), `reports/qf69-conditional-ml/`:

| Path | Content |
| --- | --- |
| `specification.json` | Frozen specification (fingerprinted record) |
| `execution.json` | QF-9 code and dependency provenance |
| `walk-forward/<study-id>/` | QF-39 study, including `folds/<fold>/development.json` and `development/` |
| `datasets/<dataset-id>/` | QF-67 dataset (`manifest.json`, `rows.parquet`, `rows.csv`) |
| `models/<model-id>/model.json` | QF-68 frozen envelope |
| `predictions/<set-id>/` | Selection and OOS prediction sets |
| `stages/01-dataset.json` .. `05-reproduction.json` | Immutable stage records |
| `reproduction/` | Rebuilt dataset and refrozen model used for byte comparisons |
| `oos/`, `manifests/`, `html/` | Base-strategy QF-40 aggregate, QF-9 manifest, QF-41 reserved report |
| `report.json`, `report.md` | Study report (JSON is authoritative) |
| `pre-holdout-gate.json` | Gate record |

## Lifecycle stages and gate

Each stage record binds the specification ID and the identities of earlier
stages. A stage refuses to run before its predecessor:

1. **Dataset.** `build_event_dataset` plus `export_event_dataset`, with file
   digests recorded.
2. **Training.** `fit_event_model` on the fold's development rows, then the
   frozen class floor. Below a floor the status is recorded, and there is no
   model, freeze or OOS.
3. **Freeze.** `predict_selection` exported, `freeze_event_model`, then
   `read_event_model`. The envelope digest is recorded.
4. **Out of sample.** The re-read frozen model (its digest must match),
   `predict_out_of_sample`, exported.
5. **Reproduction.** A rebuild from QF-39 evidence gives the same dataset ID
   and byte-identical files. A refit gives identical model-configuration,
   fitted-state and model IDs and a byte-identical envelope. Rescoring gives
   identical prediction-set IDs, and `reproduce_prediction_set` is exact for
   both sets.

**Feature-leakage audit.** The schema-level checks are:

- every feature is decision-time causal;
- sources are the rule's declared contemporaneous fields;
- no target or partition column is a feature.

For every row, QF-8 selects the exact completed context bars of its role
window at the decision. The audit then:

- asserts every bar ended at or before the decision (the latest 2m bar ends
  exactly at it, and the latest daily bar is the latest completed one);
- recomputes both 2m EMAs and the daily EMA50 with the rule's own indicator
  definitions;
- recomputes the 30-minute raw return from canonical 2m closes under QF-49's
  34-digit half-even policy.

This bypasses QF-59/QF-63 prepared series, QF-42/QF-11 execution and
QF-64/QF-67 persistence and extraction.

**Gate items:**

- dataset validation;
- feature-leakage audit;
- deterministic training;
- persisted frozen model;
- OOS predictions and metrics;
- offline reproduction;
- artifact and manifest validation (ML readers plus the QF-9 verification in
  publication);
- the holdout state from the permanent ledger.

Outcomes are `PRE_HOLDOUT_COMPLETE`, `PRE_HOLDOUT_COMPLETE_INSUFFICIENT_TRAINING`
or `PRE_HOLDOUT_BLOCKED`.

## Results

Real run of 2026-10-01: `PRE_HOLDOUT_COMPLETE`, with the holdout
**RESERVED / UNCONSUMED**.

- **Code.** Commit `ec85211` (QF-9 `execution.json`), Python 3.13 and
  scikit-learn 1.9.1.
- **Runtime.** One process, 1,688 s (28 min) wall clock, peak RSS 3.9 GB. The
  full test suite ran on the same host for about 15 minutes of that, so the
  time is inflated.
- **Network.** None.

The pipeline behaved correctly. The model itself is a clear **negative
result**: it is badly miscalibrated outside its training range and does worse
than the constant training-prevalence baseline. That outcome is acceptable
for this smoke study and changed nothing.

### Identities

| Artifact | Identity |
| --- | --- |
| Specification | `b44979e5e8a5a8e7ee923a3a749f31c5005153a55848c57133ab97b8dd587033` |
| QF-8 plan / final holdout | `10f1c1ba…d817` / `a9b55ee1…9f2c` |
| QF-39 study / fold | `fae4cc8f…e5ef` / `f868a4cd…f60e` |
| Holdout lineage (reserved) | `2c987265…d051` |
| QF-67 dataset / population | `a201e62814ddfe89f5805baf07db13a650c5f9c1fbaa6a90902f914acf6e1caa` / `bc2a5e11…81ab` |
| Model configuration (bound) / fitted state | `be9e4083…1ff4` / `03a2812e…9b26` |
| Frozen model | `ea5b7d6c84490c2d86c4564b38f4991214816c728e692f06326db621f6a35593` |
| Selection / OOS prediction sets | `feac29af…fed9` / `6db9c00d…2145` |
| QF-40 aggregate / QF-9 manifest / QF-41 report | `oos/c093b3e4….json` / `manifests/011ef63c….json` / `html/0484cc44….html` |

### Events and labels

| Partition | Scheduled decisions | Triggers (rows) | Positive / negative | Unavailable | Prevalence | Mean / median 30m raw return |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| Development | 13,650 | 55 | 29 / 26 | 0 | 0.527 | 0.0000518 / 0.000232 |
| Selection | 4,095 | 28 | 17 / 11 | 0 | 0.607 | 0.0000365 / 0.000229 |
| Walk-forward test | 8,580 | 49 | 25 / 24 | 0 | 0.510 | 0.0000726 / 0.0000575 |

The development count equals the planning count (55). Every trigger ends
before 14:00, so every 30-minute label is available. No source is excluded:
development comes from the frozen candidate's verified development evidence.
The base strategy's returns are unfiltered, before costs, and tiny relative
to any realistic trading cost.

### Training and model

- **Fit.** `fitted` on all 55 development rows (29 positive, 26 negative),
  with no exclusions. Both frozen floors were met.
- **Coefficients** (standardized units):
  - intercept 0.118;
  - the four 2m EMA levels about 0.024–0.027 each;
  - `daily_close` 0.272;
  - `daily_ema50` −0.951.

  These are collinear price levels, not interpretable effects.

### Model versus training-prevalence baseline (same rows)

| Partition | Rows | Model log loss | Baseline log loss | Model Brier | Baseline Brier | Model ROC AUC | Mean probability |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Selection | 28 | 2.335 | 0.683 | 0.579 | 0.245 | 0.476 | 0.024 (baseline 0.527) |
| Walk-forward test (OOS) | 49 | 3.104 | 0.694 | 0.507 | 0.250 | 0.383 | 0.0037 (baseline 0.527) |

- **Scores and buckets.** Every selection score is 0.013–0.043, and every OOS
  score is 0.0008–0.0085. All scored rows fall in the first predeclared bucket
  `[0, 0.2)`, so its outcomes equal the unfiltered base strategy and the other
  four buckets are explicitly empty. Calibration (five bins) shows the same
  collapse.
- **Selection evidence** was frozen with the model and already showed the
  miscalibration. Nothing was selected or changed from it: the configuration
  was fixed before fitting.

### Post-hoc diagnostics (read-only, after the gate)

These were computed from the persisted dataset and frozen envelope only. They
change nothing and are not a basis for any retuning of this study.

| Partition | Standardized `daily_ema50` | Standardized `ema_fast` | Linear predictor |
| --- | --- | --- | --- |
| Development (training) | −1.57 .. 1.56 | −2.12 .. 1.33 | −1.03 .. 1.19 |
| Selection | 4.59 .. 6.43 | 2.52 .. 4.26 | −4.36 .. −3.11 |
| Walk-forward test | 6.71 .. 10.36 | 3.49 .. 7.31 | −7.19 .. −4.76 |

The six frozen features are SPY **price levels**. SPY rose from about 560–607
during development to 623–688 in selection and test, so later rows sit 2.5 to
10 training standard deviations outside the fitted range. A linear model on
non-stationary levels extrapolates into near-zero probabilities. This is a
property of the predeclared feature schema, not of the pipeline.

Any scale-free alternative (for example, spreads relative to price) would be
a new, separately frozen study with its own untouched evaluation windows.
These windows cannot be reused for it, now that they have been seen.

### Reproduction, audit and gate

- **Reproduction.** The rebuild from QF-39 evidence gave dataset `a201e628…`
  with byte-identical files. The refit gave identical model-configuration,
  fitted-state and model IDs and a byte-identical `model.json`. Selection and
  OOS rescoring gave identical prediction-set IDs, and
  `reproduce_prediction_set` was exact for 28 and 49 records (maximum
  difference 0.0).
- **Feature-leakage audit.** Passed for all 132 rows, with zero failures.
  Every context bar ended at or before its decision. All six features, and
  all 132 30-minute returns, equal their independent recomputation from
  canonical bars.
- **Gate.** All eight items passed. The QF-9 manifest was verified in
  publication, and the QF-41 report shows `reserved_unconsumed`.
- **Ledger.** `reports/holdout-ledger` now holds the QF-69 reservation
  (sessions 2025-11-03..12-31) beside QF-45's untouched reservation. There is
  no exposure or consumption marker. QF-69 never reads, evaluates or consumes
  either holdout.

The next step is manual review. Consuming the QF-69 holdout is optional and
would be a separate, explicitly authorized QF-40 step with no retuning; for
this frozen negative model it would add no platform evidence beyond the gate.

## Limitations

- **Smoke study.** Tens of events per partition. Metrics and bucket summaries
  are unstable and descriptive only.
- **Conditional population.** Results hold only given that 8/48 triggered
  (UP only, 11:00–14:00).
- **Target.** A raw 30-minute return without costs. Classification metrics
  are not trading performance, and the target is not a trade.
- **Dependence.** Overlapping intraday labels are not independent samples.
  No standard error, interval or test is reported.
- **Features.** The six are highly collinear price levels. Coefficients are
  not interpretable effects, and no feature was engineered.
- **Exposure.** Development reuses QF-45-inspected periods, and later windows
  are public history.
- **Indexing.** QF-9 indexes the base-strategy study, OOS aggregate and
  reserved report. ML artifacts are validated by their own readers and bound
  through the specification and stage records.
- **Out of scope:** dense datasets, feature or strategy search, AutoML,
  neural networks, live inference, execution, position sizing, options,
  brokers and holdout consumption.
