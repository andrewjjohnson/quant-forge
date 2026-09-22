# Intraday prediction-input provenance (QF-51)

Corporate-action **availability** describes whether a dataset path supplies
split/dividend events. **Adjustment compatibility** describes whether its
prices and volumes can be compared under the declared source semantics.
Unavailable events do not imply unknown prices, zero events, or a daily event
policy. They do not permit mixing adjusted and raw prices.

## Canonical input

The intraday contract already records the explicit policy
`not_provided_for_intraday_bars`. QF-51 preserves that string unchanged.
`CorporateActionAvailability` distinguishes `AVAILABLE` from `UNAVAILABLE`;
`DatasetMetadata.corporate_action_availability` exposes this typed state.
Availability is separate from `corporate_actions_complete`.

`quantforge.data.prediction_inputs.prediction_dataset_from_intraday` accepts:

- a canonical `IntradayDataset`;
- its existing QF-19 one-session `AggregatedSessionDataset`;
- the existing `MarketDataCache`;
- the `IntradayMarketDataCache` containing the canonical source and raw extracts;
- optionally the validated QF-20 composed `DatasetFamily` used by the study.

It reloads the source through the intraday cache and requires an exact match to
the supplied dataset before creating any projection artifacts. This verifies
the source identity, normalized bars, and raw-extract checksums. Cache loading
also reconstructs typed raw snapshots and checks their identities against the
normalized bars and retained manifest metadata, including each chunk's endpoint.
The adapter then verifies the session aggregation and family binding through
the existing artifact APIs, and persists those completed
session prices as QF-11's session carrier. This retains the current QF-11/QF-48
session-coverage contract; exact decisions and future labels still consume the
canonical intraday series. No intraday bar is presented as a daily provider bar.
No provider client, network call, or second cache is introduced.
The session aggregation report must be complete before any projection is
persisted. Diagnostic aggregates with missing constituents or incomplete source
coverage are rejected because the session carrier cannot preserve partial-bar
quality evidence. A diagnostic policy is accepted when its report is complete.

```python
from quantforge.data.prediction_inputs import prediction_dataset_from_intraday

prediction_input = prediction_dataset_from_intraday(
    source,
    derived_daily,
    cache=market_cache,
    intraday_cache=intraday_cache,
    family=context_family,
)
```

When combining 2-minute and daily artifacts, pass the **same composed family**
used to bind both context series. Their separate aggregation families are not
interchangeable. Existing QF-20 artifact composition rules remain authoritative.
Prediction and feature generation require the context's exact family manifest ID
to match the projection's retained manifest before checking individual timeframes.
A missing ID, including one caused by mixing series from different graphs, or a
different graph with the same family ID and referenced members is rejected.
The context's serialized `source_context` retains `dataset_family_manifest_id`
as part of its `context_id`. QF-9 requires that ID to match the retained family
manifest before accepting individual timeframe references. For available contexts,
QF-9 also recomputes `context_id` from all remaining source-context fields before
trusting their lineage, including in empty feature snapshots and directories.
Available feature contexts also pass the existing window-context semantic
validator: completion policy, declared source requirements, staleness limits,
aligned timeframes, and selected bars must agree with their captured rule
requirements. Rehashing context and feature identities cannot waive those checks.
Available contexts
must carry source evidence even when a feature dataset has zero candidates;
only explicitly skipped contexts may omit it.

The new optional `IntradayPredictionProvenance` record contains its own schema
version (`5`), explicit event availability, source dataset/request/raw-snapshot
references, session dataset/timeframe references, chosen family identity, feed
identity, session-policy identity, and immutable copies of the selected family
manifest and canonical intraday source manifest. Its immutable `session_evidence`
also retains the original QF-19 session manifest and normalized bar artifact.
Its `source_bar_evidence` retains all canonical source observations with shared
timeframe/provenance templates, allowing reconstruction of the original batch.
The cache's raw extract also
retains these manifests alongside the original session manifest. `provider_name`
remains the original provider; `adapter_version` identifies QuantForge's
projection. Requested session bounds equal the actual first and last projected
completed sessions, and `missing_sessions` is always empty;
the original intraday request bounds remain in the retained source evidence.

This adapter handles sources explicitly reporting unavailable events. It does
not infer event economics for an intraday source that claims an available event
policy without supplying event records. Existing daily inputs with supported
explicit event policies keep their original behavior.

## Compatibility and integrity

The shared prediction-input checks require:

- exact common family and canonical source snapshot;
- matching feed, symbol, and exchange-session policy;
- exact adjustment mode, OHLC basis, volume basis, adjusted-field usage, and
  truthful corporate-action policy;
- canonical raw-bar request/snapshot lineage, or the existing derived-bar link
  to that same canonical source dataset.

Both context construction and the generic timestamp-outcome dispatch use this
contract. QF-49 and QF-47 retain their additional exact price-basis checks.
Outcome-source compatibility is checked on every run, including when a dataset
session reuses a previously validated labeler. This check runs after signals are
fixed and before outcome labeling.
No labeler-specific exception or provider-name condition is added. A future
provider with the same canonical semantics works automatically.

Unavailable events require explicit intraday lineage, `corporate_actions_complete
= false`, empty event records, zero known-event counts, and the deterministic
empty-record snapshot. That snapshot fingerprints the records supplied; it does
**not** certify that no economic events occurred. An unavailable declaration
combined with the daily event policy, complete-event claim, or event records is
invalid. Adjusted source prices can be accepted without event data when their
explicit price/volume semantics and lineage match; this does not create an
adjustment algorithm or authorize corporate-action accounting.

QF-9 preserves the new record in existing `market_data` provenance. Its readers
validate availability, price semantics, and recorded context/outcome source
references without generating predictions or outcomes. Cache and experiment
validation recompute the embedded family and manifest identities and compare
the complete declared adjustment basis to the canonical source committed by
that family. Each outcome's `family_manifest_id` must match the retained exact
lineage graph, including feature templates and individual outcomes. Prediction
generation enforces the same exact-manifest requirement. Each outcome and
available-context reference must identify exactly one member of that graph with
the recorded timeframe, canonical source, and role. A valid family ID cannot
stand in for a missing dataset or a dataset recorded at another timeframe.
Cache and experiment validation also bind the projection's session dataset and
timeframe IDs to its retained one-session artifact and session policy. The
projection's `calendar` and `provider_timezone` must exactly match that policy's
calendar and timezone, even when a different calendar produces the same session
dates over the recorded range. Embedded
timeframe definitions must reproduce their declared IDs. These checks do not
accept a self-consistent hash as proof of a valid graph: `DatasetFamily.from_manifest`
reconstructs every member through the domain types and enforces unique datasets,
one canonical root, valid parents, reciprocal child links, and absence of cycles.
It verifies the complete canonical manifest after construction, including
members not used by the current study.

The retained intraday source manifest must reproduce `source_dataset_id` using
the existing intraday cache identity algorithm. Its request configuration must
reproduce `source_request_id`, and its ordered raw chunk references must equal
`source_raw_snapshot_ids`. Source symbol, provider, feed, timeframe, and price
basis must agree with the family. Rehashing a changed source manifest creates a
different source dataset identity; it cannot retain the original family binding.
Every retained raw chunk must name the same nonempty, trimmed provider endpoint,
matching ingestion's requirement for one endpoint revision per fetch result.
Recomputed source, family, and artifact identities do not waive this constraint.
The projection cache's `provider_symbol` must exactly match the retained source
manifest, including provider aliases that differ from the canonical symbol.
QF-9 market records omit this field and retain it inside the source evidence.
The projection's `retrieved_at` must equal the source manifest's retrieval time
after both timezone-aware timestamps are normalized to UTC. Equivalent offset
representations are accepted; missing, malformed, naive, or different timestamps
are rejected. The source timestamp must also equal the latest retrieval instant
among all retained raw chunks, matching the ingestion producer's rule. Every
chunk must carry an aware timestamp; chunk order and equivalent UTC-offset
representations do not affect the maximum.
Retained request and chunk bounds must use the canonical UTC ISO representation
emitted by intraday ingestion. The request must have a strictly increasing range;
its nonempty chunks must be ordered, contiguous, and cover that exact range.
The request must include the full first and last projected exchange sessions,
resolved under the retained timeframe's session policy. The projection's session
bounds must be ordered, with each requested bound equal to its corresponding
actual bound and no missing sessions. Rehashed projections cannot trim the actual
interval while retaining broader requested session bounds or relabel completed
sessions as missing. The first and last sessions and the integer bar count must
also match all fully requested sessions in the validated source coverage report,
so shrinking both requested and actual bounds to a subset is invalid. Partially
requested edge sessions remain excluded, matching one-session aggregation.
Wider source requests remain valid; calendar dates alone
cannot establish full-session coverage. These checks
apply even when every enclosing request, source, family, and artifact ID has been
recomputed, without fetching data or regenerating research results.
The source manifest's coverage report is also reconstructed and must reproduce
its own `report_id`. Its request, batch, timeframe, feed, session scope, bounds,
and observation count must agree with the source manifest. Calendar-derived
expected intervals constrain session counts and missing/unexpected findings;
session-level findings must reproduce the report's aggregate lists, counts,
completeness, and warning flags. Duplicate, unordered, overlapping, or out-of-range
evidence is rejected. Prediction projections require a complete validated report.
The general intraday manifest reader still accepts truthful incomplete diagnostic
reports and warnings. This metadata-only check cannot authenticate observations
that are absent from the manifest; loading the full cache still verifies the
report against normalized bars and retained raw extracts.
The producer reloads bars/raw extracts through the cache before capturing this
evidence. Cache loading reconstructs the raw snapshots and the fetch result,
then reproduces the manifest from those records; checksum-valid raw files alone
cannot justify endpoint or other acquisition metadata that contradicts their bodies.
Observational readers validate retained evidence without source I/O or rerunning
research. They check the retained constituent-to-session OHLCV relationship;
hashes establish consistency, not provider authenticity.

Session evidence binds the projected OHLCV values to the named QF-19 artifact.
Readers reconstruct typed session bars and their report, verify the original bar
IDs, serialized content digest, dataset identity, family, and source bindings,
then serialize the projected daily bars and compare that digest with QF-3's
`data_sha256` or QF-9's `bars_fingerprint`. Rehashing a changed projection or
replacing its evidence prices cannot preserve the original session artifact ID.
Projection decimal strings use the existing canonical exact-decimal formatter;
removing representation-only zeros changes no numerical values. This makes the
fingerprint reproducible from canonical session evidence.
Readers also bind the reconstructed session artifact to the selected family
through the same QF-20 artifact API used at runtime. A composed family must name
the original session-family manifest ID in its supported composition policy;
an uncomposed family must equal the artifact's own family. A valid generic DAG
and recomputed family hashes alone do not establish this artifact binding.

Source-bar evidence must reproduce both the canonical source's `batch_id` and
`data_sha256`. Session constituent IDs must equal the authenticated source IDs in
the same session and order; matching counts alone is insufficient. Forged IDs,
swapped sessions, reordered constituents, and changed observations cannot retain
the original source binding. The compact representation preserves exact decimal
strings and all canonical bar fields while storing repeated metadata once.
Readers reconstruct the typed intraday request, bars, and batch through the
existing ingestion decoder before accepting that digest. Domain validation and
canonical reserialization reject invalid prices, timestamps, extra fields,
duplicate bars, ordering errors, and inconsistent request bindings even when
all enclosing hashes have been recomputed.
Every bar must name one retained raw chunk and start within that chunk's
half-open request interval. Its provider, provider symbol, adapter version,
request ID, and retrieval instant must match the corresponding acquisition
metadata, using the same relationships enforced by `IntradayFetchResult`.
Retrieval times are compared as instants, including equivalent offset notation.
Readers also run the existing diagnostic coverage validator on the reconstructed
batch and require exact agreement with the retained report. Missing, unexpected,
developing, and zero-volume findings and their warnings must describe the actual
observations; self-consistent report hashes alone are insufficient.

Each retained session's OHLCV must also equal the reduction of its ordered source
constituents: first open, maximum high, minimum low, last close, and summed
volume. The reader and QF-19 producer share this reduction. Volume summation
reserves enough Decimal precision for an exact total, independent of the
caller's arithmetic context. This preserves ordinary existing values and
identities; previously rounded session volumes must be regenerated from their
canonical source before projection. The evidence format remains version 5.
This check uses only retained observations and does not fetch data, reconstruct
corporate actions, or execute contexts, signals, or outcome labels.

Prediction and feature
manifests bind every recorded outcome source and available context source to the
input's family, canonical snapshot, and feed. Every aligned timeframe in an
available context must retain its dataset reference, even in zero-row artifacts;
visible-bar evidence cannot replace missing or null dataset provenance.
Available context timeframes must also retain the input's exchange-session policy,
including calendar and timezone.
Outcome references must explicitly record their feed scope. Compact context
references obtain it from the captured requirements, whose primary, contextual,
and selected-timeframe feeds must all match the input provenance. A missing feed
or a different feed cannot be accepted by omitting it from a source reference.
Each aligned context reference must also match the timeframe of its own
requirement, whose definition must reproduce its declared configuration ID.
Another valid member of the same family cannot substitute a daily source for
an intraday requirement, or the reverse.
Every outcome source's timeframe identity must match its labeler's declared
observation timeframe. Intraday-backed outcome timeframes must also retain the
input's session policy. This covers the feature study template and every exported
outcome independently; rehashing both a source reference and its enclosing
configuration does not waive those bindings.
Typed elapsed-duration labelers require a canonical outcome source in prediction
and feature manifests, including the feature study template and each configured
outcome. Missing or null sources are rejected even when the artifact has no rows.
Legacy session labelers and custom feature components retain their existing
source-free contracts; this check does not add typed declarations to old artifacts.
These checks still apply when artifact hashes are recomputed; skipped contexts
retain their rejected evidence for auditability. Existing outer manifest,
producer identity, hash, and row checks still apply.

## Serialization, identity, and resume

Legacy daily schema-4 metadata omits `intraday_provenance` entirely. Its original
serialized bytes and dataset identity remain unchanged, as do context-free
daily prediction identities. Artifact-bound multi-timeframe context identities
now include the exact family manifest ID, changing dependent study/feature IDs.
Older intraday studies/features without that context evidence must be regenerated
from their existing source caches. Contexts without a common family manifest
retain their previous serialization, but are not valid for intraday-backed inputs.
Projection dataset IDs use `intraday-projection-<sha256>`; their original 256-bit
digest still commits to the complete QF-3 metadata and content. That namespace
requires intraday provenance independently of the event-policy declaration.
The QF-3 adapter marker also requires provenance, and the cache checks origin in
the checksum-verified raw extract even if manifest metadata is relabeled.
Removing provenance and claiming a daily policy cannot downgrade the same
dataset reference. Replacing all evidence and the dataset reference is a new
artifact, whose external authenticity cannot be proved by standalone hashes.
An absent `intraday_provenance` field is accepted only for the established daily
contract without projection markers. Explicit null or malformed new records are rejected
by the cache reader. The additive intraday record round-trips with canonical
sorted JSON and participates in dataset, study, feature, and checkpoint identity.
Changes to adjustment semantics or source/event provenance cannot alias an
existing artifact. Compatible runs use existing cache/resume validation.
Earlier intraday provenance versions 1–4 lack the complete required source-bar
evidence and are rejected. Version 5 changes intraday projection, study,
feature, and checkpoint identities. Regenerate those projections from the existing
intraday cache and session aggregates, then recreate dependent studies/features;
do not relabel or reuse their old checkpoint identities. Legacy daily artifacts
remain unchanged. Manifests now include completed session OHLCV, constituent
identities, and compact canonical intraday observations, increasing their size
and validation cost with the source history. Raw provider responses remain in
the source cache and are not copied into experiment manifests.

The original QF-45 cache-only reproducer used 8,190 Tiingo SPY one-minute bars
for January 2–31, 2024, 4,095 derived two-minute bars, and 21 derived session bars.
The old QF-3 daily-only policy check and complete `AdjustmentBasis` comparison
rejected the input before research could execute. QF-51 supplies a truthful
input that satisfies those semantics, with stronger source binding. It does
not implement the QF-45 EMA study, change outcome mathematics, or change
QF-46/QF-48 temporal validation.

Ordinary tests use synthetic prices through the actual intraday cache,
aggregation, prediction, feature export, and manifest APIs. They need no
credentials and contain no licensed real market data:

```bash
uv run --frozen pytest tests/integration/test_intraday_prediction_provenance.py \
  tests/integration/test_intraday_prediction_manifest_integrity.py \
  tests/integration/test_intraday_prediction_feed_integrity.py \
  tests/integration/test_intraday_prediction_family_integrity.py \
  tests/integration/test_intraday_prediction_lineage_integrity.py \
  tests/integration/test_intraday_prediction_evidence_integrity.py \
  tests/integration/test_intraday_prediction_price_integrity.py \
  tests/integration/test_intraday_prediction_session_evidence.py \
  tests/integration/test_intraday_prediction_composition_integrity.py \
  tests/integration/test_intraday_prediction_context_semantics.py \
  tests/integration/test_intraday_prediction_origin_integrity.py \
  tests/integration/test_intraday_prediction_constituent_integrity.py \
  tests/integration/test_intraday_prediction_source_bindings.py
```
