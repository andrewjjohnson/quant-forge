# QF-45 heavyweight acceptance test

`tests/integration/test_spy_ema_runner.py::test_pre_holdout_pipeline_and_completed_resume`
is the only test that drives the actual QF-45 pre-holdout composition end to
end: the real EMA rule/factory/outcomes and `spy_ema_plan` QF-8/QF-39 wiring,
`spy_ema_runner.run_pre_holdout`, QF-32/39/40/42, QF-7/29 candidate export,
QF-9 manifests and the QF-41 report, followed by a completed resume. It is marked
`heavy_acceptance`: required locally and in CI, dispatched first under xdist, and
run as its own CI job ([tiers and CI layout](development.md#test-tiers-qf-66)).

QF-66 profiled it on `main` (`b749dbd`, after QF-63) and changed only its tier
marker; its body, fixture and assertions are unchanged. Its runtime is almost
entirely production re-authentication of immutable evidence, not test-harness
work; see [findings](#qf-66-profile-and-findings).

## What it proves

Every row is exercised on each run. "Assertion" rows are explicit test asserts,
"guard" rows are harness patches that fail the test, and "runner check" rows are
fail-closed production checks the test executes. No other test covers this
composition, so none of these may move or weaken without an equivalent test.

| # | Invariant | How the test proves it | Kind |
| --- | --- | --- | --- |
| 1 | Fixed QF-45 setup (EMA 8/48, daily EMA50, `talib_v1`, 61 two-minute and 50 daily warm-up bars, real outcomes) | `bounded_plan` calls the real `prepare_walk_forward` and narrows only dates/intervals; `validate(plan)`; the fixed trial must be `SUCCEEDED` | runner check |
| 2 | Deterministic offline cached inputs | Fixture persists synthetic 1m bars to `IntradayMarketDataCache`, reloads them, derives 2m/daily and reloads the prediction dataset from `MarketDataCache`; no provider is constructed | fixture |
| 3 | QF-7/QF-29 candidate replay and export | `export_candidate_features` requires exactly one replayed row per recorded candidate with the same timestamp and EMA features; `features/*/features.csv` exists | runner check + assertion |
| 4 | Exactly three QF-32 trials | 5 windows = fixed + 3 trials + 1 OOS; `SmokePredictionEvaluator` freezes only if all 3 trials succeeded | assertion + runner check |
| 5 | Selection frozen before OOS | QF-39 persists `selection.json` before evaluating the test window; resume and `inspect_validation` re-check it against captured state | runner check |
| 6 | QF-39 OOS execution | The fold must be `COMPLETED`; the OOS window is inspected under `WALK_FORWARD_TEST` membership | runner check |
| 7 | QF-40 receives only OOS observations | `load_oos_source` rejects any outcome reaching the protected holdout; aggregation reads only fold test artifacts | runner check |
| 8 | QF-9 integrity/index | `inspect_study` and `verify_artifacts` for every window, feature dataset and parameter study; `publish_pre_holdout` verifies before writing | runner check |
| 9 | Publish/finalization | Immutable `pre-holdout-complete.json`, manifest and HTML; the report must stay `reserved_unconsumed` | runner check + assertion |
| 10 | Independent persisted-artifact validation | The test re-reads the final manifest from disk and verifies every artifact against the artifact root | assertion |
| 11 | Completed resume, zero recomputation | A second `run_pre_holdout` rebuilds adapters and invocation-scoped registries; `len(calls) == 5` shows only the five QF-7 candidate replays requested context and no QF-42 decision reran | assertion |
| 12 | Equivalent, duplicate-free resume | `replay == completed`; every `prediction-window.jsonl` is byte-identical | assertion |
| 13 | Holdout reserved and unconsumed | `HoldoutLedger.consume` fails the test; no context is requested on the holdout session; no ledger exposure exists; the result and HTML say `reserved_unconsumed` | guard + assertion |

## Fixture size

The fixture has 55 consecutive XNYS sessions from 2025-01-02 with 21,450
synthetic 1m bars. Each session dips before 11:00 and jumps at 11:00, so every
evaluated window has one candidate, while daily closes rise above the EMA50.
`bounded_plan` uses the last four sessions for development, selection, test and
holdout, each a four-minute window from 11:00.

It is at, or within one session of, its minimum. The frozen daily EMA50 needs 50
completed daily bars before the first evaluated decision, plus four window
sessions, so it needs at least 54 sessions. The QF-63 fixtures keep the same one
extra completed daily bar before the first evaluated session. Sessions must be
complete 390-minute RTH sessions: the canonical source is coverage-validated, and
dropping minutes would be missing-data behavior. QF-66 therefore kept the fixture.
Removing one session would save about 2%, which is not enough to justify moving
the exact daily warm-up boundary. Early closes, holidays and DST are not what
this test exists to prove; QF-51/52/59/63 tests own them.

## QF-66 profile and findings

Measured on the same machine with `pytest -n 0`, warm filesystem and dependency
caches, and a phase/invocation probe (not committed). A separate cProfile run
has about 2x overhead; it is used only for attribution.

| Stage | First run (s) | Resume (s) |
| --- | ---: | ---: |
| Synthetic canonical cache fixture | 62.6 | - |
| Plan/adapter preparation and QF-39 `validate`, fixed and comparison | 38.4 | 37.0 |
| Holdout reservation preflight (`load_oos_source`) | 15.3 | 34.5 |
| Fixed trial (QF-32 on the selection window) | 22.0 | 20.5 |
| Fixed inspection (QF-9 and QF-7/29 export) | 45.2 | 39.8 |
| QF-39 study total | 156.7 | 75.8 |
| of which membership | 41.6 | 39.8 |
| of which three trials | 26.6 | not rerun |
| of which OOS evaluate and artifact validation | 23.6 | 10.7 |
| of which other QF-39 work (adapter identity checks, fold state) | 64.9 | 25.3 |
| Selection inspection, three windows | 101.2 | 89.9 |
| OOS inspection | 32.2 | 28.1 |
| Publish: QF-40 aggregate, QF-9 manifest, QF-41 report | 125.5 | 125.1 |
| **Total** | **599.1** incl. fixture | **450.8** |

Whole test: **1,064 s** (fixture 62.6 s, call 1,001 s). All five QF-42 window
executions together took **16.7 s (1.6%)**.

Invocation counts for the whole test:

| Operation | Calls | Seconds |
| --- | ---: | ---: |
| Synthetic source/fixture builds | 1 | 62.6 |
| `validate_market_dataset` (canonical or bounded) | 150 | 603.0 |
| of which from `DatasetProvenance.from_market_dataset` (same canonical dataset) | 35 | ~190 |
| Canonical 1m source-evidence re-derivations (`validate_session_projection_evidence`) | 88 | ~450 |
| QF-52 `bounded_prediction_view` | 38 | 276.8 |
| QF-60 registries / `project` / `verify_lineage` | 4 / 12 / 13 | - / 58.7 / 17.9 |
| Offline lineage (`validate_prediction_view_lineage`) | 24 | 206.5 |
| `partition` | 18 | 213.9 |
| `prepare_prediction_study_dataset` | 27 | 96.7 |
| QF-9 `inspect_study` / `validate_compact_window` / `verify_artifacts` | 24 / 18 / 163 | 214.0 / 179.9 / 116.4 |
| `load_oos_source` | 4 | 60.4 |
| Candidate exports (`build_signal_feature_dataset`) | 10 | 28.3 |
| QF-42 window executions | 5 (first run), 0 (resume) | 16.7 |
| `get_context_at` | 25 = 15 decisions + 5 + 5 candidate replays | <0.1 |

Seconds are outermost inclusive time, so nested rows overlap. Values marked "~"
are cProfile shares scaled to real time.

**Why QF-66 did not shrink the test.** The test harness adds only the fixture
and one final manifest reload/verification, which is a trust boundary. Every
repeated validation above happens inside production code the test must
exercise: runner inspection, QF-9, QF-39, QF-40 and QF-41. The resume run must
repeat that verification from persisted bytes with fresh adapters. Memoizing
validators in the test would let the resume reuse first-run in-memory state,
which the acceptance contract forbids. The fixture cannot shrink materially, as
explained above.

**Production bottleneck (reported, not changed by QF-66).** A canonical intraday
prediction dataset embeds its complete 1m source-bar and session evidence, about
9 MB of canonical JSON here. Every `validate_market_dataset` of such a dataset
calls `validate_prediction_provenance`, which rebuilds and re-hashes all 21,450
source bars, re-resolves exchange sessions, re-checks coverage and re-captures
the evidence. That costs about 5.8 s per call here and would cost about 4.5x more
on the real 96,960-bar QF-45 source. It is reached 88 times through QF-39
identity checks (`DatasetProvenance.from_market_dataset` in `partition`,
`configuration`, `validate` and `prepare_walk_forward`), QF-52 projections, grid
setup and QF-9/QF-40 offline lineage. In cProfile, `validate_prediction_provenance`
accounts for about 63% of the test and QF-9 `safe_metadata` recursion over large
snapshots (164 million calls) for about 25%. The fix is execution-local reuse of
authenticated canonical provenance across these consumers, with trust boundaries
preserved. It belongs with [QF-65](https://frostfiredigital-37308542.atlassian.net/browse/QF-65)
or a separate story, not a test-speed ticket.

## QF-66 results

| Measurement | Before (`main`) | After (QF-66) |
| --- | ---: | ---: |
| QF-45 runner alone, `-n 0` | 1,063.6 s | 1,079.2 s (same work: all 62 probed invocation counts identical) |
| QF-45 runner start in the local full suite | ~500 s in (modeled; it finished last on the only busy worker) | t = 0 |
| Local full suite, `uv run pytest` | 1,606.9 s | 1,201.4 s (-25%); an earlier run measured 1,175.2 s |
| CI test layout | one job; the test step took 4,764-5,398 s | concurrent tier jobs; the critical path is the heavy job (estimated ≤ ~3,190 s) |

The suite now ends about one regression file after the QF-45 runner (see
[tiers](development.md#test-tiers-qf-66)). Further reduction needs the production
follow-up above, not test changes.

## QF-65 follow-up

QF-65 implemented the production fix recommended above. `run_pre_holdout` joins
or owns one execution-local [canonical preparation](prepared-canonical-loading.md).
Each invocation, including the test's completed resume, therefore
authenticates the canonical inputs afresh from persisted bytes. It does not
reuse first-run in-memory state. Within an invocation, the retained QF-51
evidence of the same canonical input is rebuilt once instead of on every
`validate_market_dataset`. The test body, fixture and every assertion are
unchanged. The QF-45 runner alone (`pytest -m heavy_acceptance -n 0`) went from
1,063.6-1,079.2 s to **316.5 s** (call 284.5 s) on the same machine.

## Benchmark methodology

- Same machine for before/after (Apple Silicon, 14 cores), uv 0.12.1, frozen
  lockfile, warm dependency and filesystem caches.
- Heavy test: before with `pytest -n 0 tests/integration/test_spy_ema_runner.py`,
  after with the CI command `pytest -m heavy_acceptance -n 0`. Each shared the
  machine with one other single-process run: cProfile before, and the
  runner-last ordering check after. Full-suite runs had the machine to themselves.
- Full suite: repository defaults (`-n 4 --dist=loadfile`), before from a
  worktree of `main` using the same virtual environment. Durations are
  pytest-reported, on a monotonic clock that excludes system sleep; later runs
  were wrapped in `caffeinate`.
- CI estimates replay per-test JUnit durations from GitHub Actions runs through
  a model of xdist 3.8's `loadfile` scheduler. The model reproduced measured
  wall times within 3%: CI 5,232 s vs 5,398 s and 4,748 s vs 4,764 s; local
  1,605 s vs 1,607 s (before) and 1,200 s vs 1,201 s (after).
