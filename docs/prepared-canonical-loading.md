# Prepared canonical loading (QF-65)

QF-65 authenticates canonical research inputs once per research load session.
Later consumers in that session reuse the proof instead of reconstructing,
revalidating, reserializing and rehashing the same immutable data. No scientific
identity, persisted format, cache, schema, QF-63 feature or QF-64 receipt
changes. See [ADR 0037](decisions/0037-reuse-authenticated-canonical-preparation.md).

## Fresh baseline (main `697b8fe`, after QF-64)

The harness `reports/qf65-canonical-loading/qf65_startup.py` is ignored and not
committed. It loads the frozen QF-45 cache (`data/market-data`: 96,960 1m bars,
48,480 2m, 250 daily). It then drives the fixed QF-45 study (8/48, daily EMA50)
through the production `load_inputs → prepare_walk_forward → validate → select`
path. A sentinel stops at the first `get_context_at` call, so no decision is
evaluated, no window is finalized and the holdout is never touched. Each run is
a fresh process on the same host (14 cores, 36 GB) with a warm OS file cache.
Cold-cache runs need `sudo purge` and were not possible. All 292.8 MB of cache
artifacts read in 0.045 s warm, SHA-256 in 0.108 s, and the 1m JSON parse
takes 0.565 s, so page-cache state cannot move these results materially.

| Stage (fresh process, no profiler) | Baseline (s) |
| --- | ---: |
| Canonical 1m source ready (`get_intraday_bars`) | 27.28 |
| Derived 2m ready (aggregate + persist/verify) | 60.19 |
| Daily ready (aggregate + persist/verify) | 24.06 |
| Prediction input ready (`prediction_dataset_from_intraday`) | 82.71 |
| 2m/daily `TimeframeBarSeries` ready | 8.27 |
| Plan and adapter ready (`prepare_walk_forward`) | 58.15 |
| Plan validated (`adapter.validate`) | 26.87 |
| Select until first scheduled decision (QF-8/52/60/59/63/61) | 67.06 |
| **Total, cached inputs to first decision** | **354.58** |

Within the select stage the partition took 57.05 s, of which the QF-60/QF-52
projection took 29.17 s. Max RSS was 1,826 MB.

## Repeated-work inventory (baseline cProfile, one startup)

| Operation | Calls | What repeated |
| --- | ---: | --- |
| Full 1m decode (`intraday_batch_from_primitive`) | 8 | 2 cache loads plus 6 rebuilds from retained QF-51 evidence |
| `IntradayBar` construction | 824,160 | 8.5 per source bar |
| `IntradayBar.bar_id` | 6,306,557 | 65 per source bar |
| `IntradayBarBatch.batch_id` (full batch serialization) | 54 | same batches re-hashed |
| `Timeframe.configuration_id` | 14,328,399 | every bar primitive re-hashed its timeframe |
| `resolve_exchange_session` | 1,074,948 | for 251 distinct sessions |
| Coverage validation | 24 | same batches |
| `_validate_source_dataset` (aggregation re-authentication) | 3 | 2m, daily, daily again |
| `aggregate_session_dataset` | 2 | re-derived to prove the daily artifact |
| Cache `load` of the same 1m dataset | 2 | from-source series binding re-read it |
| `validate_market_dataset` (canonical input) | 8 | 6 full QF-51 evidence rebuilds |
| `DatasetProvenance.from_market_dataset` | 3 | plan, adapter validate, partition |
| `intraday_session_windows` | 867 | same session/timeframe values |

Identity cost was mostly building and encoding huge primitives:
`configuration_identity` ran 23.8 M times, while hashing itself was minor.

## Trust boundaries kept, repetition removed

| Check | QF-65 treatment |
| --- | --- |
| Cache bytes match persisted identities (manifest, bars, raw extracts) | Always on first read. An identical later load re-reads and re-hashes every file; any change goes back through the full load and fails closed |
| Canonical schema, request, raw snapshot identities, manifest identity | Always on first read, each computed once |
| Per-bar identity, batch identity | Always on first read: one `to_primitive` and one hash per bar, one batch hash |
| Coverage, session and calendar semantics | Always on first read. Sessions and windows are resolved once per exact value |
| Fetch-result raw/bar bindings | Always on first read, with the verified snapshot IDs |
| Presented source equals the authenticated source | Request and metadata equal by value, plus every bar field and nested record field equal to the pristine value. Otherwise the reference re-authentication runs |
| Derived 2m/daily artifact validation | Full `validate()` once per session, then reused for content-identical objects |
| Daily artifact derives from the source | Proven by performing the derivation once; reused for the identical source/target/policy relationship. Otherwise re-derived |
| Derived cache files on disk | Unchanged `_write_once`/`load` byte comparison |
| QF-51 retained evidence of a new canonical input | Full independent rebuild once (at persist) |
| Later validations of the same QF-3 input | Verdict reused only for byte-identical content. The key covers every metadata/provenance field, the SHA-256 of each evidence snapshot and the canonical bars |
| QF-52 records, QF-60 lineage, QF-9/40/41/57 offline checks of persisted artifacts | Unchanged. Offline consumers in their own processes never see a preparation |

## Prepared canonical architecture

`data.prepared_canonical.CanonicalPreparation` is the execution-local session.
`canonical_preparation()` opens it, or joins the active one. Only the outermost
scope owns it and clears everything on exit, including errors and interruption.
It holds:

- `PreparedCanonicalDataset` entries for each canonical intraday source the
  session authenticated from a cache: the dataset object, authenticated
  `batch_id` and per-position `bar_ids`, resolved cache root, SHA-256 of every
  artifact file, and a pristine content-integrity snapshot. Coverage, session,
  feed and adjustment evidence are its authenticated metadata and request.
- Validated derived artifacts and derivation relationships, each with the same
  content-integrity snapshot, shared per object.
- A bounded (64-entry LRU) map from QF-3 prediction-input content fingerprints to
  validation verdicts.
- A `TimeframeMemo` (`quantforge.timeframes`) for exchange sessions and timeframe
  configuration identities, and a session-window map.

Consumers consult `active_canonical_preparation()`; without a session they run
the unchanged reference code. The QF-45 script owns one session for loading,
configuration checks and the pre-holdout run. `load_inputs`, `run_pre_holdout`
and `PredictionEvaluator.preparation_scope` join it, or own one when called
alone. A completed resume is a new process, or a new `run_pre_holdout` call, and
therefore authenticates everything afresh.

The layer is generic: nothing refers to EMA, QF-45, a provider or a timeframe.
Event and future dense research share the same authenticated backing. Feature
preparation (QF-63) and sparse persistence (QF-64) are unchanged and compose
with it.

## Compatibility key

Reuse of a canonical source requires all of the following:

- `(dataset_id, request_id)`. The dataset ID binds the manifest: provider,
  symbol, feed, source interval, session scope, raw snapshots, batch, bar count,
  data digest and coverage report. The request ID binds symbol, bounds,
  timeframe (calendar, timezone, RTH/ETH, anchor, labels), feed, adjustment
  basis and schema.
- Presented `request` and `metadata` equal to the authenticated values.
- Every bar field, and every field of every nested record (provenance, feed
  scope, adjustment basis, timeframe, interval, session policy), equal to the
  pristine value. Equal-but-distinct objects pass; replaced or bypass-mutated
  fields fail.
- For cache reloads, the same resolved cache root and unchanged file digests.

Derived reuse is keyed by derived type and dataset ID, with equal metadata and
request and intact bars. Derivation relationships add the source key, target
timeframe identity and aggregation-policy identity. Session and timeframe
memos are keyed by the complete frozen value. Object identity, paths, file
sizes and mtimes are never a reuse key. Object identity only avoids duplicating
an integrity snapshot, after content has been checked.

## Identity behavior

No identity definition changes. Each identity is computed once from the same
inputs:

- The loader builds each bar's primitive once, derives `bar_id` from it and
  composes the exact `batch_id` from those verified entries.
- Coverage uses the verified batch ID, and the manifest identity uses the
  verified snapshot IDs.
- `IntradayBarBatch.to_primitive` builds each bar primitive once.
- Batch validation hashes the request and timeframe once instead of per bar.
- Bars from one raw chunk share one immutable provenance record.

Derived 2m output hashes its batch once. Its QF-15 digest equals that value for
these NaN-free JSON primitives, and the later independent `validate()` re-checks
both. The real-data run confirms every identity below is byte-identical.

## Results on the real QF-45 inputs

Each column is one fresh process. The second run of each mode repeats the
process with the same warm file cache.

| Stage | Baseline 1 | Baseline 2 | QF-65 1 | QF-65 2 |
| --- | ---: | ---: | ---: | ---: |
| Canonical 1m source | 27.28 | 28.42 | 7.41 | 7.41 |
| Derived 2m | 60.19 | 61.14 | 10.72 | 10.71 |
| Daily | 24.06 | 24.41 | 0.41 | 0.41 |
| Prediction input | 82.71 | 83.27 | 15.24 | 15.15 |
| 2m/daily series | 8.27 | 8.30 | 0.77 | 0.77 |
| Plan and adapter | 58.15 | 58.00 | 2.37 | 2.35 |
| Plan validation | 26.87 | 26.69 | 0.16 | 0.15 |
| Select to first decision | 67.06 | 66.80 | 6.08 | 6.10 |
| of which QF-8/QF-52 partition | 57.05 | 56.73 | 2.48 | 2.49 |
| of which QF-60 projection | 29.17 | 29.01 | 0.80 | 0.81 |
| of which QF-60 lineage verification | 2.00 | 2.04 | 0.09 | 0.09 |
| of which QF-59/QF-63 context provider | 1.25 | 1.27 | 1.13 | 1.14 |
| **Total to first decision** | **354.58** | **357.03** | **43.16** | **43.05** |
| Max RSS (MB) | 1,826 | 1,936 | 1,796 | 1,689 |

Startup falls **87.9%**, from a mean of 355.8 s to 43.1 s. The heavy QF-45
acceptance test (`pytest -m heavy_acceptance -n 0`) falls from 1,063.6-1,079.2 s
([QF-66](qf45-acceptance-test.md)) to 316.5 s on the same machine.

Invocation counts for one full startup (cProfile; nested rows overlap):

| Operation | Baseline | QF-65 |
| --- | ---: | ---: |
| Full 1m decodes | 8 | **2** (trust-boundary load + one independent QF-51 evidence rebuild) |
| `IntradayBar` constructions | 824,160 | 242,400 (2 x 96,960 + 48,480 derived) |
| `IntradayBar.to_primitive` | 11,154,557 | 682,877 |
| `IntradayBar.bar_id` property | 6,306,557 | 198,077 |
| Batch identities | 54 | 7 (4 property + 3 composed from verified entries) |
| `Timeframe.configuration_id` computations | 14,328,399 | **3** (689,448 memo hits) |
| `configuration_identity` | 23,806,783 | 883,726 |
| Exchange-session calendar resolutions | 1,074,948 | **251** (327,218 memo hits) |
| Session-window computations | 867 | 331 (536 reuses) |
| Coverage reports | 24 | 5 |
| Canonical source authentications | 2 loads + 3 re-authentications | **1** (+1 byte-verified reuse, 3 content-verified reuses) |
| Daily derivations | 2 | **1** (+1 derivation-relationship reuse) |
| Derived artifact validations (2m / daily) | 2 / 11 | **1 / 1** (+5 reuses) |
| QF-51 source-evidence rebuilds | 6 | **1** (+5 verdict reuses) |
| Full `validate_market_dataset` runs | 8 | 3 (1 canonical + 2 bounded views) |
| Source-evidence captures | 7 | 2 |
| Raw snapshot identities | 8 | 1 |
| Retained coverage validations | 13 | 3 |

The regression test `tests/performance/test_prepared_canonical_loading.py` pins
the same collapse on a 5-session fixture with an early close and a holiday.
There, 7 decodes become 2, 5 evidence rebuilds become 1, and 14,874 calendar
resolutions become 5 (one per session).

## Memory

`CanonicalPreparation.memory_report()` counts only state the preparation owns.
Bars, metadata and nested records belong to the authenticated datasets, which
the research session holds anyway. No 1m, 2m or daily collection is copied.
Bars from one raw chunk (and all derived 2m bars) share one provenance record.
Each retained derived object has one integrity snapshot.

| Prepared state (real QF-45 startup) | Bytes |
| --- | ---: |
| Authenticated 1m `bar_ids` (96,960) | 10.96 MB |
| 1m content-integrity snapshot | 14.74 MB |
| Derived 2m/daily integrity snapshots | 7.41 MB |
| Session windows (331 session/timeframe values) | 5.78 MB |
| Session memo (251 sessions), timeframe identities (3), market verdicts (1) | negligible |
| **Total owned by the preparation** | **about 38.9 MB** |

Traced Python memory (tracemalloc, after a full GC at each stage; separate
runs from the timing runs):

| Stage | Baseline retained / peak (MB) | QF-65 retained / peak (MB) |
| --- | ---: | ---: |
| Canonical 1m source | 126.4 / 1,227.9 | 109.6 / 1,204.1 |
| Derived 2m | 197.4 / 1,227.9 | 169.1 / 1,204.1 |
| Daily | 218.5 / 1,227.9 | 170.1 / 1,204.1 |
| Prediction input | 258.6 / 1,446.0 | 210.2 / 1,204.1 |
| Series | 214.6 / 1,446.0 | 210.6 / 1,204.1 |
| Plan and validation | 245.7 / 1,446.0 | 231.6 / 1,204.1 |
| First decision | 402.5 / 1,446.0 | **392.0 / 1,204.1** |

Net retained memory is 10.5 MB *lower*, and peak traced memory is 242 MB
lower. The prepared state is more than offset by shared provenance records,
which were previously one object per 1m and 2m bar, and by no longer rebuilding
full-batch primitives in every consumer. Process max RSS in the timing runs was
1,826-1,936 MB before and 1,689-1,796 MB after.

## Session, calendar and derived-data reuse

`TimeframeMemo` maps each exact `(session date, session-policy value)` to its
calendar-resolved `ExchangeSession`, and each timeframe value to its
configuration identity. Invalid dates are never retained and keep failing
through the calendar. Early closes, holidays and DST transitions resolve exactly
as the reference does, because the memo stores the calendar's own answer. Every
bar still passes its full `IntradayBarWindow` validation, including
no-cross-session and completion rules. `intraday_session_windows` is memoized per
exact (session, timeframe) value for aggregation, scheduling and bounded-record
validation.

Aggregation takes source bar identities from the authenticated source by
position. It slices each target window's expected constituents by bisect over the
sorted session intervals. This is equivalent to the old full filter, which
scanned all of a session's intervals for every window (18.9 M comparisons for 2m).
Emitted bars share one derived provenance record.

## Corruption and incompatibility (fail closed)

`tests/integration/test_prepared_canonical_loading.py` covers each case below
inside an active session, on DST and early-close/holiday fixtures:

- **Changed cache files.** Bars, manifest, raw extract or a truncated bars file
  after first authentication are detected by the re-hash on reuse. The full load
  then raises `CacheError`.
- **Wrong identity, root or request.** An unknown dataset ID, a copy under
  another root with changed bytes, or a request with a different adjustment
  basis gets no reuse and is loaded and rejected independently.
- **Changed presented source.** Missing, duplicate, extra and changed bars, and a
  wrong symbol, timeframe, adjustment, feed, bar count or batch ID, are all
  rejected by the reference re-authentication. This holds for both 2m and daily
  derivation.
- **Bypass mutation.** Mutating a retained bar is detected by the integrity
  snapshot, which evicts the entry and revalidates; the batch identity mismatch
  raises. Mutating a prediction input changes its fingerprint and fails full
  validation.
- **Derived artifacts.** A corrupted derived file on disk causes a persist
  collision and a derived-cache load mismatch. A derived artifact rebound to
  another source, or with changed bars, fails validation. A daily artifact from
  another source or with changed bars is re-derived and rejected by
  `prediction_dataset_from_intraday`.
- **Lifecycle.** An interrupted session releases its state. Separate sessions
  never share state and rebuild identically, and inner scopes join the outer
  one.

`tests/unit/data/test_prepared_canonical.py` covers memo exactness (normal,
early close, DST, extended-hours policy), invalid sessions, value-keyed identity
memoization, integrity detection of nested-record mutation, mutable-graph
refusal and the reference path for non-intraday datasets.

## Scientific equivalence

The real-data harness compared baseline and QF-65 runs. Every identity was
byte-identical:

- source dataset, batch and quality-report IDs and all 96,960 bar
  serializations;
- 2m and daily dataset IDs and bars;
- family references and manifests;
- the canonical prediction dataset ID, its bars and its full metadata, including
  all QF-51 evidence;
- the QF-8 plan ID, the QF-32 universe ID and the first scheduled decision
  timestamp.

The synthetic suites also compare reference and prepared loading: source,
derived and prediction datasets, `TimeframeBarSeries`, and QF-52 view identity
and provenance.

## Remaining startup cost and persisted formats

The final cProfile run is scaled to real time (factor 0.65; the rows overlap).
Of the remaining 43.1 s:

- About 11 s is the two unavoidable full 1m decodes (about 5.5 s each). Each
  constructs and domain-validates 96,960 bars and hashes each bar's canonical
  JSON for its identity. One decode is the cache trust boundary. The other is
  the independent QF-51 evidence rebuild of the new canonical prediction input.
- About 10 s is 2m work: derivation (4.0 s), plus its one independent
  validation and existing-file verification (5.8 s).
- About 5 s is coverage work: five coverage reports (2.9 s) and three
  retained-coverage validations of the same manifest (2.1 s).
- About 2.3 s is `TimeframeBarSeries` construction. Ordering hashes every bar
  for a tiebreak that valid series never need.
- About 6 s is outside canonical loading: QF-63 scope capture (2.2 s) and
  QF-58-style deep copies of the 2m source in the adapter and grid (about 4 s).

Parsing the 1m cache JSON takes 0.6 s and file I/O under 0.1 s. A binary or
columnar persisted format would therefore save at most about 1 s: identities are
defined over canonical JSON, so the per-bar identity work remains. **A
persisted-format follow-up is not justified.**

The next smallest measured candidates, not implemented here:

1. Let the new canonical input's QF-51 evidence validation reuse the
   session-authenticated source when its evidence reproduces that source exactly
   (about 7 s). This adds a second evidence-validation path, so it needs its own
   review.
2. Memoize `validate_retained_coverage_report` per manifest content (3 to 1,
   about 1.5 s).
3. Verify an existing derived cache file by digest instead of reserializing
   (about 1.5 s).
4. Remove the `bar_id` tiebreak from `TimeframeBarSeries` ordering, where any tie
   is already rejected (about 1.5 s).
5. Share the immutable 2m source across adapter and grid deep copies.

Reprofile the end-to-end event study before opening performance tickets.

## Scope

There is no cache, schema, identity, provider, QF-63 feature-series or QF-64
receipt change. There is also no binary or columnar storage, parallel loading,
missing-data repair, dense or ML execution, or strategy change. Existing caches
are read unchanged, and nothing needs to be redownloaded or migrated.
