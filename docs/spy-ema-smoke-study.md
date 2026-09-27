# QF-45: shortened SPY EMA platform smoke test

This preparation implements the current QF-45 story. It is a platform acceptance
exercise, not evidence of profitability or strategy robustness. The research run,
manual audit, final gate and explicit holdout consumption are separate from this
implementation PR. No minimum accuracy, return or candidate count is required.

## Frozen configuration

| Role | Inclusive XNYS sessions |
| --- | --- |
| Warm-up/context | 2025-01-02 through 2025-03-18 |
| Development | 2025-03-19 through 2025-04-30 |
| Selection | 2025-05-01 through 2025-05-30 |
| One OOS test | 2025-06-02 through 2025-06-27 |
| Reserved final holdout | 2025-06-30 through 2025-07-31 |

Dates are resolved with the existing XNYS calendar before execution. QF-8 owns
exact timestamp membership and purging. The maximum configured future reach is
122 minutes (120 minutes plus QF-46's conservative 2-minute alignment allowance),
with zero additional embargo. No date is selected from observed returns. The
CLI preflight reports actual retained/purged counts and first/last timestamps.
The verified real preflight required no calendar correction and no purging:
5,850 development timestamps, 4,095 selection timestamps and 3,705 OOS timestamps.
Each role declares 61 completed 2m bars and 50 completed daily bars for warm-up.
Normalized EMA seeding uses the exact context selected by QF-8 for that role;
there is no hidden EMA state carried between trials/windows.
QF-43 is a backtest accounting contract; this prediction-only study has no
orders, fills, fees, equity or benchmark. Its corresponding boundary discipline
is QF-8/QF-48 context-only warm-up versus evaluated decisions.

Use 1m canonical RTH source bars, session-open anchored QuantForge 2m aggregation,
and completed daily context, all XNYS / America/New_York. No cross-session
aggregation, forward filling, missing-bar repair or provider-specific rule logic.
Unadjusted provider OHLC/volume and unavailable intraday corporate actions remain
explicit in the canonical input. The frozen complete 2025 cache contains:

| Artifact | Count | Identity |
| --- | ---: | --- |
| Canonical 1m | 96,960 bars / 250 sessions | `7e396b640d4387c324ab9a25194a5b046d8dcaf1bc7a345783f993f6a2477972` |
| Raw snapshot | 1 logical snapshot | `6d85c98d75be72c7d5e4fba8bbc78066c09967eead34aacca9ad8c673315e029` |
| Derived 2m | 48,480 | `0447ab4da91ae57b6acca49b41f13501140f0e18565afa5c1376dd05c3a0ac8d` |
| Derived daily | 250 | `18936f6834a19be1be8112a7e0756199079a5bf9a0d7f4ffa3ae3b9a6367b08e` |

The loader constructs no network provider. Missing/incompatible cache fails;
it does not refetch. Normal canonical validation and deterministic aggregation
are retained, including QF-58 through QF-61 execution preparation.

Fixed study: fast EMA8, slow EMA48, daily EMA50, normalized `talib_v1`. Emit only
UP when previous fast <= slow, current fast > slow, completed daily close > its
EMA50, and the decision bar **end** is inclusively 11:00–14:00 New York time.
The current session's eventual daily close is never a feature. The previous and
current normalized EMA values and daily close/EMA are preserved as causal
features. QF-29 additionally retains indicator/source/completion provenance.

### QF-45-only selection exception

The owner-approved [AGENTS.md exception](../AGENTS.md#qf-45-only-smoke-test-exception)
allows this platform acceptance smoke test to exercise parameter comparison,
deterministic ranking, frozen selection, OOS evaluation, holdout handling,
manifests, and reporting. It does not establish a robust or deployable
strategy-selection policy. The three configurations, study windows, and
selection metric were fixed before inspecting study outcomes.

Exactly three QF-32 categorical configurations are declared: `8/40`, `8/48`,
`12/60`. There is no Cartesian fast/slow expansion. Daily50 and backend stay
fixed. Selection ranks mean 30m raw return, requires one available prediction,
and uses QF-32's deterministic tie ordering. A single event may determine the
selected configuration. The always-UP matched baseline is descriptive only:
it is identical by construction for this bullish-only event population, does
not measure incremental edge, and is not an independent quality or risk gate.
The selected configuration must not be interpreted as robust, profitable, or
production-worthy. The exception applies only to QF-45; the normal prohibition
on return-only optimization remains in force for ordinary QuantForge research.

Do not add sample-size, risk, stable-region, or other eligibility gates to QF-45
now, because that would change the frozen scientific configuration. In particular,
do not switch to `FIRST_STABLE`: its new eligibility semantics could leave this
intentionally small grid without a selectable configuration. Keep the rule, grid,
dates, ranking metric, and result schemas unchanged. All other research-integrity
requirements remain in force.

Positive-30m fraction means strictly `return > 0`;
zero is not positive. Empty trials are unrankable and retain native evidence.
The fixed-run gate accepts an unrankable result only when its sole persisted
QF-32 trial succeeded and contains completed analysis. A failed trial stops the
command before comparison, even if its compact window was already finalized;
resume preserves that failure rather than treating it as an empty success.
If none is eligible or OOS is empty, QF-39 fails explicitly; do not invent a
selection, change dates or weaken the generic contract to obtain completion.

## Run locally

Use uv **0.12.1** and Python 3.13, with a clean committed checkout. On this
machine the pinned uv executable is `$HOME/.local/bin/uv`; Homebrew's uv differs.
From the repository root:

```bash
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/private/tmp/quantforge-uv-cache
uv --version
uv sync --all-extras --frozen
uv run --frozen python scripts/run_spy_ema_smoke.py --resume \
  > reports/qf45-smoke-2025.log 2>&1
```

An optional configuration-only check is:

```bash
uv run --frozen python scripts/run_spy_ema_smoke.py --preflight
```

The pre-holdout command runs these stages:

1. Validate cached identities, calendar membership, warm-up and outcome reach.
2. Reserve the final holdout using the permanent workspace ledger. Reject prior
   consumption/exposure before running research.
3. Run fixed 8/48 on the May selection interval through the ordinary QF-39/QF-32
   adapter and QF-42 compact execution. Development is permitted prior context,
   not pooled selection scores. Verify QF-9 integrity and replay fixed candidates
   into native QF-7/QF-29 outcome/feature exports before comparison begins.
4. Run the three selection trials, freeze through QF-39, then evaluate June OOS.
5. Export candidate-only supplemental outcomes and QF-29 features for each
   selection trial and the selected OOS configuration.
6. Aggregate **only OOS** through QF-40, index artifacts through QF-9 and render
   standalone QF-41 HTML with authoritative reserved/unconsumed state.
7. Print `PRE_HOLDOUT_COMPLETE` and stop for the manual review gate.

For each candidate, existing labelers produce 10/30/60/120m returns, 60m MFE/MAE
and 60m +0.30% target / -0.20% stop results, including unavailable/ambiguous
statuses. Supplemental QF-7 execution replays only recorded candidate timestamps
under the original QF-39 context and role; it checks timestamp/population/EMA
agreement. Empty decisions never request supplemental outcomes. The primary
QF-42 window preserves original prediction/outcome IDs; QF-7 records its own
linked native identities. No outcome math is implemented in the example.

This is not a dense observation dataset. No ML, strategy search beyond the three
pairs, new result schema, chart framework or research inside the renderer is
introduced. Operational completion/provenance files contain pointers and state;
scientific results remain in native platform artifacts.

## Files, progress and interruption

The default root is `reports/qf45-smoke-2025/` (ignored by Git):

- `execution.json`: original QF-9 execution/code/lock provenance; mixed-code resume
  is rejected. Keep the checkout at the producing commit during the local run.
- `fixed/<grid-id>/artifacts/<trial-id>/`: fixed-run JSONL and native wrapper.
- `walk-forward/<study-id>/folds/<fold-id>/selection/`: three native QF-32 trials.
- `walk-forward/<study-id>/folds/<fold-id>/selection.json`: immutable freeze,
  persisted before test execution.
- `walk-forward/<study-id>/folds/<fold-id>/test/`: compact OOS window.
- Beside each incomplete JSONL, `prediction-window.jsonl.in-progress/` contains
  `shared.json`, `decisions.jsonl`, `checkpoint.json`. QF-56 commits every decision.
- `features/<feature-dataset-id>/`: QF-7/29 CSV, Parquet, schemas,
  manifest and row checkpoints for triggered events only.
- `oos/`, `manifests/`, `html/`: native aggregates, indexes and static reports.
- `pre-holdout-complete.json`: final pointers and pending manual-audit status.

Fixed + three selection trials + OOS total **20,085** decisions. At the prior
QF-61 sample rate (~0.106 seconds/decision), decision work alone is approximately
35 minutes. Cache validation, projections, candidate exports, integrity checks
and report publication add substantial startup/stage time. This is an estimate,
not a measured complete-run duration; expect tens of minutes or longer.

The log is `reports/qf45-smoke-2025.log`. Stage lines describe progress; during a
window its small checkpoint shows durable completed count. Use Ctrl-C to stop,
then repeat the same command with `--resume` (use `>>` to append the log).
QF-56 verifies the exact durable prefix and resumes at the first incomplete
decision. Completed windows/trials/folds are verified, not recalculated. QF-7
checks causal context compatibility before loading a completed feature export.
Do not run two writers against one output directory.

The permanent authority is **`reports/holdout-ledger/`**, shared across output
roots, not recreated per execution. Preserve it, including all exposure markers.
Alternate run roots must remain directly under `reports/` so the existing QF-9
index can reference both study artifacts and the ledger. Never delete it to
restore an unseen status. There is no holdout-consumption CLI flag.

The historical 76-decision checkpoint is under `reports/qf45-compact/`, with
three folds, older dates, another candidate universe and different validation/
window/schedule identities. Its June 27 selection decisions cannot belong to
this May selection. It is deliberately not copied or resumed. QF-58–61 diagnostic
copies are also excluded. The fresh run uses its own native identities.

## After the local command finishes

Return `pre-holdout-complete.json`, the log, and the complete output directory
with relative paths intact. The next action is **review, not consumption**:
open the indexed fixed, parameter and reserved OOS reports; audit at least 10
fixed candidate events across sessions (all if fewer); verify candle boundaries,
previous/current EMAs, daily causality, time window, every outcome alignment and
status, prediction/outcome IDs and source/view ancestry. Record the audited
ordered timestamps and findings. Do not lengthen the study for more events.

Confirm all nine gate items, including the preserved reserved report and safe
resume evidence. Only in that subsequent reviewed step prepare the existing
`HoldoutEvaluation` using the last frozen fold and explicitly call the same
ledger's `consume` once. Regenerate native QF-9/QF-41 artifacts, verify durable
consumption including old-manifest regeneration, and never retune. This
preparation PR does not claim that manual audit or final QF-45 completion passed.

## Local verification

Focused checks:

```bash
uv run --frozen pytest tests/unit/test_spy_ema_smoke.py \
  tests/integration/test_spy_ema_compact.py tests/integration/test_spy_ema_runner.py
```

These use synthetic offline data and the actual normalized EMA/outcome,
compact/resume, QF-32/QF-39/QF-40/QF-9/QF-41 paths. Repository-wide checks follow
`docs/development.md`. GitHub Actions must not be polled or waited on for this PR.
