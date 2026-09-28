# Normalized historical-window membership (QF-62)

QF-62 adds compact window representation schema `"3"`. It stores each ordered
visible-bar membership list once, in append-only catalogues owned by the window.
Decisions refer to those lists through exact `[start_index, stop_index)`
references. Schema `"2"` ([QF-55](compact-prediction-windows.md) and
[QF-56](incremental-prediction-windows.md)) stays the default and remains readable.
QF-45's frozen configuration keeps schema `"2"`. New studies opt in with
`PredictionGridConfig(window_schema_version="3")`.

This is a representation change only. Scheduling, contexts, indicators, rules,
outcomes, eligibility, ranking and OOS semantics are unchanged. The design
choice is recorded in [ADR 0034](decisions/0034-normalize-window-membership-catalogues.md).

## What QF-45 exposed

In the completed QF-45 fixed window, 96.36% of the 2,453,216,703 bytes were
`visible_bar_ids` JSON lists (2,364,000,720 bytes). A late no-candidate decision
was 1,148,745 bytes. Each decision's `prediction_study.manifest.prediction_context`
holds the lists in three kinds of location:

| Location | Producer | QF-45 copies |
| --- | --- | ---: |
| `source_context.timeframes[i].visible_bar_ids` | QF-20 `TimeframeContext` | 2m, daily |
| `timeframes[i].visible_bar_ids` | QF-28 `PredictionTimeframeInput` | 2m, daily |
| `timeframes[i].indicators[j].visible_bar_ids` | QF-28 normalized indicator manifest | 2m ×2, daily ×1 |

So the 2m list appears four times and the daily list three times. Offline
validation already requires these copies to agree:

- a completed-only rule list must equal its source list;
- every indicator list must equal its rule list.

Under `DEVELOPING_BAR_AS_OF`, the source list also ends with the developing bar.

All other decision content is decision-specific and stays unchanged. That
includes requirements, dataset references, availability, ages, developing-bar
evidence, signals, rows and outcomes. The lists are not stored in the shared
evidence, header, rows or signals.

Across the 4,094 adjacent fixed-window transitions:

- **2m membership:** prefix-growing every time, from 62 to 4,156 bars.
- **Daily membership:** prefix-growing 4,093 times. At sequence 194
  (`2025-05-01T20:00Z`, the first session close) the start moved by one bar.
  QF-45's permitted-context provider exposes a bounded daily lookback drawn from
  QF-48 validation membership. This is a legitimate rolling start bound, not a gap.

Some lists cannot be a plain range, so the design must not assume one:

- developing-bar tails, which are unique to each as-of time;
- empty lists for missing timeframes;
- rejected foreign contexts retained by skipped decisions;
- entries with no complete dataset/timeframe binding;
- any non-contiguous, reordered or duplicated list, which is invalid. Such a list
  must reach scientific validation unchanged rather than be coerced into a range.

## Schema 3 representation

The finalized file is still header, shared evidence, then ordered decision JSONL.
Both the header and the shared-evidence record carry `"schema_version":"3"`. The
nested QF-42 scientific window identity keeps schema `"1"`.

```text
{"component":"quantforge_prediction_window",...,"membership_catalogues":[
   {"bar_count":4156,"catalogue_content_id":"...","catalogue_id":"..."}, ...],
 "record_type":"header","schema_version":"3",...}
{"record_type":"shared_evidence","schema_version":"3","window_identity":{...}}
{"catalogue_segments":[{"binding":{...},"bar_ids":[62 IDs],"catalogue_id":"C2m","start_index":0},
                       {"binding":{...},"bar_ids":[51 IDs],"catalogue_id":"Cd","start_index":0}],
 ..., "prediction_study":{"manifest":{"prediction_context":{...
    "visible_bar_range":{"catalogue_id":"C2m","start_index":0,"stop_index":62} ...}}}}
{"catalogue_segments":[{"bar_ids":[1 ID],"catalogue_id":"C2m","start_index":62}], ...}
```

### Catalogue

- **Binding.** A catalogue is bound to exactly one binding. The binding is the
  source dataset reference plus the timeframe, including its session policy.
  - The dataset reference has `canonical_source_snapshot_id`, `dataset_id`,
    `family_id` and `timeframe_configuration_id`.
  - The timeframe is `{configuration, configuration_id}`.
  - The binding is accepted only if the timeframe's ID hashes its configuration
    and equals the reference's `timeframe_configuration_id`.
  - An indicator's `feed_scope` stays in the indicator record and is excluded
    from the binding.
- **Identity.** `catalogue_id` hashes component
  `quantforge_membership_catalogue`, catalogue schema `"1"` and the binding.
  Primary and daily data therefore never share a catalogue. A different source,
  family, snapshot, dataset or session gives a different identity.
- **Growth.** Entries are bar IDs in first-appearance order. Each is introduced
  by the decision whose membership first needs it, as a
  `catalogue_segments` item `{catalogue_id, start_index, bar_ids}`.
  - `start_index` is the current catalogue length.
  - A catalogue's first segment also declares its `binding`.
  - Segments are sorted by catalogue ID, with at most one segment per catalogue
    per decision.
  - Each ID is stored exactly once per window.
  - A decision can reference only entries introduced by itself or earlier
    decisions.
- **Header.** `membership_catalogues` gives each complete catalogue's
  `bar_count` and `catalogue_content_id`, which hashes
  `{catalogue_id, bar_ids}`. Readers check it once they reach the end of the file.

### Ranges and exceptions

A membership list becomes `visible_bar_range` only when both of these hold:

- its location has a valid binding;
- the list is one contiguous run of that catalogue, extended only by entries
  not yet in the catalogue.

Starts need not be zero, so rolling lookbacks shift `start_index`.

Everything else stays an explicit `visible_bar_ids` list:

- empty lists;
- lists containing a developing bar ID (developing bars never enter a catalogue);
- lists with no binding;
- non-contiguous, reordered or duplicated lists.

Explicit lists are the versioned exception form. Nothing is coerced.

### Canonical encoding

The encoder visits locations in a fixed order: source timeframes, then each rule
timeframe followed by its indicators. The persisted record must be exactly the
encoding that the preceding catalogue state yields for the membership the record
references.

`MembershipCatalogues.accept` replays that encoding and compares the result. It
rejects every representation of the same list that is not canonical, including:

- ranges that are rebound, foreign, shifted beyond the catalogue or unneeded;
- negative, empty or reversed bounds;
- boolean indices;
- segments that are reordered, truncated, duplicated, unneeded or foreign;
- a location that has both a range and an explicit list;
- a range outside a recognized membership location.

## Identities

| Layer | Identities | Schema 3 behavior |
| --- | --- | --- |
| Scientific | QF-11 `study_id`/`prediction_study_id`, QF-20 `context_id`, row/outcome/evaluation IDs, signals, schedule ID, QF-42 window identity | Unchanged. Validation rebuilds each expanded context and checks the original IDs. |
| Physical | shared-evidence ID, `window_id`, `decision_id`, `window_result_id`, `catalogue_id`, `catalogue_content_id`, header | New for schema 3. `window_result_id` still hashes the ordered normalized records with the QF-55 algorithm. |
| Operational | QF-56 checkpoint (count, byte offset, last ID, prefix hash, counts) | The format is unchanged apart from `"schema_version":"3"`. Checkpoint cadence is unchanged. |

The persisted list is gone, but the QF-20 and QF-11 identities that hashed it are
unchanged. Validation rebuilds the exact expanded list from the catalogue, then
checks those original IDs against it. A rehashed record whose ranges are
self-consistent but select different bars therefore fails with a
study/context-identity mismatch.

## Reader and validator boundary

`PredictionWindowReader` recognizes versions 1, 2 and 3 and rejects everything else.

- Header, evidence and decision forms must agree. A version 2 decision inside a
  version 3 window, or the reverse, fails.
- `from_reference` accepts compact references with schema `"2"` or `"3"`. The
  reader's version must equal the reference's version.
- `iterate_decisions()` accepts each version 3 decision against the traversal's
  catalogues before yielding it.
- Decisions keep their ranges. Each carries a `MembershipView`, and
  `expanded_record()` rebuilds the exact lists only when asked.
- Structural verification (`verify_integrity`) never expands membership.

`validate_prediction_window_reader` and the QF-56 writer pass
`expanded_record()` to the unchanged QF-42/QF-55 decision validator. A record
that still contains ranges fails closed: the validator requires lists.

Memory has four parts:

- the shared evidence and schedule;
- one decision;
- the catalogues, which grow with the source bars observed, not with the number
  of decisions;
- one ID-to-position index per catalogue.

Readers outside the window modules never read `prediction_context` membership.
The QF-32 analyzers, QF-40 streaming observations and source checks, QF-9
indexing, QF-41 reports and QF-45 candidate export read signals, rows, statuses,
IDs and headers. Their only changes are:

- **QF-32 grid** (`grid.py`): accepts `window_schema_version="3"`, records it in
  the grid identity and writes and validates versioned trial references.
- **QF-39 adapter** (`walk_forward/prediction.py`): accepts `"3"` and writes and
  validates versioned OOS references.
- **QF-9 inspection** (`experiments/artifacts.py`, `_compact_window.py`,
  `adapters.py`): accepts compact version `"3"`. Artifact entries record the
  window's physical schema.
- **Fold indexing** (`experiments/validation.py`): indexes every compact version
  instead of only `"2"`.
- **QF-40 aggregation** (`oos/prediction.py`): uses streaming window sources for
  every compact version instead of falling back to v1.
- **Finalized holdout** (`oos/holdout_evaluation.py`): accepts and references
  `"3"`.

## Resume and checkpoints

QF-56 durability is unchanged:

- three staging files;
- per-decision journal fsync followed by an atomic checkpoint;
- the committed prefix is verified before an uncommitted tail is truncated;
- resume restarts at the first incomplete decision;
- the final file is published without overwriting an existing one.

A version 3 journal line holds the decision together with the catalogue segments
it introduces, so both commit atomically. Recovery replays and re-authenticates
the segments of the committed prefix and rebuilds the catalogues. A failed append
or validation rolls the catalogues back with the other writer state. Opening
version 2 staging with a version 3 writer, or the reverse, fails and preserves
the existing bytes.

An uninterrupted run and an interrupted-then-resumed run produce byte-identical
finals. Both are also identical to `CompactPredictionWindowResult.from_window(...,
schema_version="3")`.

## Measurements

### Storage

Before is the completed QF-45 fixed window; after is the same logical records
rewritten as schema 3. No research was recomputed. All 4,095 expanded schema 3
decisions matched the schema 2 decisions exactly, apart from physical IDs. Both
files passed full offline QF-55 scientific validation against the QF-45 canonical
metadata.

Fixed window: 4,095 decisions, of which 25 are candidates. Sizes are in bytes,
including line feeds.

| Category | Schema 2 | Schema 3 |
| --- | ---: | ---: |
| Header | 656 | 1,062 |
| Shared evidence record | 4,369,822 | 4,369,822 |
| Membership evidence (lists; or catalogue segments + ranges) | 2,364,000,720 | 4,086,860 (761,529 + 3,325,331) |
| Other decision content | 84,845,505 | 84,992,925 |
| Candidate decision records (25) | 17,352,458 | 748,998 |
| No-candidate decision records (4,070) | 2,431,493,767 | 88,330,787 |
| **Total window** | **2,453,216,703** | **93,450,669** |

- **Reduction:** 96.19%, or 26.25× smaller.
- **Membership evidence:** 578× smaller. The 2m catalogue holds 4,156 IDs and the
  daily catalogue holds 72 (71 visible at the end, plus the rolled-out bar).
- **Other decision content:** grows by 147,420 bytes, from the longer
  `visible_bar_range` key and the per-decision `catalogue_segments` field.
- **Source of the reduction:** removing duplicate copies, not dropping evidence.
- **Per decision:** a late record is now about 21.7 KB.

All five QF-45 prediction windows, each exactly equivalent after re-encoding:

| Window | Decisions | Schema 2 | Schema 3 | Reduction |
| --- | ---: | ---: | ---: | ---: |
| Fixed | 4,095 | 2,453,216,703 | 93,450,669 | 96.19% |
| Selection (3 trials) | 3 × 4,095 | 7,359,654,067 | 280,355,965 | 96.19% |
| Walk-forward test | 3,705 | 2,026,062,843 | 85,401,186 | 95.78% |
| **Total** | 19,990 | **11,838,933,613** | **459,207,820** | **96.12% (25.78×)** |

### Performance (fixed window, local, descriptive)

| Operation | Schema 2 | Schema 3 |
| --- | ---: | ---: |
| Canonical encode + SHA-256 of all decision records | 3.85 s | 0.31 s |
| Structural `verify_integrity()` | 30.62 s | 5.95 s |
| Full offline scientific validation, including QF-52 lineage | 80.27 s | 53.54 s |
| QF-56 append (validate + journal fsync + checkpoint), per decision | 13.58 ms | 6.30 ms |
| QF-56 reopen and committed-prefix validation (4,095 decisions) | 38.58 s | 20.27 s |
| QF-56 finalization (prefix check, stream, validate, publish) | 91.58 s | 45.76 s |

- **Writer replay.** The QF-56 figures replay the stored decisions through the
  production writer with QF-45's trusted inputs; no research is recomputed. The
  schema 2 writer reproduced the original QF-45 bytes exactly. The schema 3 writer
  produced exactly the converted artifact.
- **Conversion cost.** Rewriting a finalized window took 40–45 s, most of it
  decoding the 2.45 GB source. Normalization itself took 6.12 s and replayed
  acceptance 2.92 s, about 2.2 ms per decision.
- **Dominant cost.** QF-45 execution spends about 0.19 s per decision on context
  and feature preparation. That is QF-63's concern, and this change does not
  affect it.

## Tests

- `tests/unit/prediction/test_window_membership.py`: catalogue identity and
  binding (source, timeframe, session, family), prefix growth, the rolling daily
  start shift, explicit exceptions, developing bars, purity, and 24 fail-closed
  corruption and rollback cases.
- `tests/unit/prediction/test_normalized_prediction_window.py`: exact version 2
  to version 3 equivalence for candidate, rejected-candidate and no-candidate
  windows; storing each ID once without re-expansion; explicit and mixed
  versions; rehashed corruption caught by scientific validation; QF-56
  resume/interruption/rollback; incompatible prefixes.
- `tests/integration/test_normalized_prediction_window.py`: validated intraday
  caches spanning the 2024-07-03 early close and the 2024-07-04 holiday; a
  rolling two-session daily lookback; interruption just after the shift; offline
  validation; QF-9/QF-41 inspection.
- QF-32, QF-39, QF-40, QF-9, QF-41, holdout and offline-replay consumer suites now
  run for schemas `"2"` and `"3"`.
- `tests/performance/test_normalized_window_membership_scale.py`: a
  4,095-decision storage-shape fixture with QF-45's membership pattern.

## Limits and follow-ups

- **Scope.** Only visible-bar membership is normalized. Repeated
  requirement/timeframe/dataset-reference metadata, candidate feature manifests
  and walk-forward study envelopes are unchanged. After normalization the
  remaining 21.7 KB per QF-45 decision is mostly repeated context metadata. That
  is a candidate for later work, not part of QF-62.
- **Validation cost.** Scientific validation still encodes and hashes the
  expanded context to verify QF-11/QF-20 IDs. That cost is now the dominant
  validation cost; hash-state reuse over catalogue slices would be a separate
  optimization.
- **Deferred tickets.** Sparse decision receipts, prepared feature series, EMA
  caching, dataset loading and checkpoint batching belong to QF-63 through QF-65.
