# Bounded prediction inputs (QF-52)

QF-60 adds [invocation-local projection/lineage preparation reuse](prepared-prediction-projections.md).
The QF-52 factory, identities and independent offline validation below remain
authoritative; operational preparation is never persisted.

Canonical QF-51 inputs describe complete immutable intraday sources. They retain
their original retrieval metadata, source manifest, coverage report, session
artifact, source observations, and strict validation. `BoundedPredictionProvenance`
is a separate, versioned variant on the existing prediction dataset carrier.
It does not claim to be another canonical source.

```python
from quantforge.data.prediction_views import (
    bounded_prediction_view,
    validate_bounded_prediction_ancestry,
)

view = bounded_prediction_view(canonical_input, cutoff)
validate_bounded_prediction_ancestry(view, canonical_input)
```

The factory validates the full canonical input first. The returned `MarketDataset`
passes the existing dataset validation entry point through an explicit bounded
branch. Canonical provenance version 5 still uses all its complete-source checks.
The bounded identity namespace and adapter marker independently require the new
variant; removing provenance cannot downgrade a view to legacy daily semantics.

## Visibility and evidence

The cutoff is an aware UTC instant, inclusive at actual exchange close. Session
labels do not advance availability. At `2024-01-03T16:00:00+00:00`, only January 2
completed daily metadata is visible. January 3 enters at its actual close.
Early closes follow the same calendar rules.

The view retains only permitted daily bars and the corresponding original QF-19
session-bar records: exact OHLCV, completed boundaries, source identity, and
ordered constituent IDs. Local validation checks bar identities, session/calendar
boundaries, constituent count and uniqueness, ordered contiguous session coverage,
bounded range/count, and the canonical daily-content digest. Future session
evidence fails even when enclosing hashes are recomputed.

The canonical source manifest, request bounds, raw chunk list, coverage report,
source observations, and full session evidence are absent. No parent object or
cache handle is attached. The retained family manifest contains static policies
and artifact identity relationships, not observation history or coverage counts.

This remains a completed-session metadata carrier. Exact intraday membership
belongs to QF-42/QF-48, per-decision features to QF-20/QF-28, and future-label slices
to QF-46. QF-52 does not infer another schedule or attach future label observations.
The factory supports an observed starting session and equal/earlier reprojection;
a bounded view cannot widen its cutoff. At least one completed session is required,
preserving the current QF-11 contract.

## Ancestry, identity, and integrity

`canonical_input_id` references the full QF-51 input; `canonical_provenance_id`
commits to its original provenance. Source/request/session artifact, family, feed,
and session-policy identities remain explicit. Corporate actions stay unavailable
under the original adjustment basis. `source_retrieved_at` preserves the canonical
retrieval instant, also retained on the carrier metadata, separately from cutoff.

`bounded-intraday-<sha256>` hashes the complete canonical bounded record: ancestry,
cutoff, local evidence, content digest, retrieval identity, and material policies.
Identical inputs are deterministic. Every cutoff change produces a different
identity, even when the completed subset is unchanged. There is no projection
creation timestamp.

Local validation proves internal consistency and structural source compatibility.
An opaque flat SHA-256 ancestor reference alone cannot authenticate arbitrary
rehashed replacement observations. `validate_bounded_prediction_ancestry` checks
exact subset membership against an independently supplied canonical parent.
QF-39/QF-40 offline readers do this through `validate_prediction_view_lineage`,
using canonical metadata/evidence already in the validation plan and the expected
cutoff from the exact decision schedule. Rehashed ancestry, prices/evidence, or
cutoff changes are rejected. The parent stays outside the evaluator. No provider
request or research callback is needed.

For session-based plans, both offline readers derive the start from the plan's
warm-up requirement and retained membership, and the end from the last retained
evaluation session plus the configured label horizon in canonical observation
order. Captured warm-up keys must match the plan's preceding observations. The
candidate market record's own range never supplies its expected ancestry bounds;
even a valid, fully rehashed canonical subview is rejected if its range differs.

## Integration and compatibility

`prediction_metadata_prefix` routes intraday inputs to the factory at the first
permitted decision. QF-39 training, selection, and test partitions and QF-40
holdout preparation share that path. Partition records retain bounded identity
and digest; prediction manifests retain the complete bounded record and ancestry.
The enclosing plan remains the independent source authority.

QF-9 uses existing plan/source/frozen-selection/window/holdout relationships. No
artifact type or manifest schema is added. Existing identity/resume comparisons
reject incompatible records; compatible work resumes without evaluation. There
is no new cache: views are in-memory inputs serialized in research artifacts,
not provider-cache acquisitions.

QF-40 still owns permanent consumption. Preparing a view does not consume the
holdout; explicit consumption persists before evaluation and retries stay consumed.
Outcome compatibility retains source/family/request/feed/session and adjustment
bindings. Canonical raw-chunk checks remain strict; bounded inputs use the existing
artifact-certified source reference instead of retaining future chunk metadata.
QF-49/QF-47 calculations do not change.

Legacy daily/session projections retain previous identities, retrieval behavior,
and execution semantics. Canonical QF-51 cache artifacts need no regeneration.
Old invalid intraday bounded artifacts cannot reuse the new view identities.

## Offline regression evidence

Both existing QF-45 reproducers were run before and after the change:

| Source | Before | After | Visible at cutoff |
| --- | --- | --- | --- |
| Cached Tiingo/SPY, January 2–31, 2024, 21 sessions | Canonical passes; projection fails on retrieval mismatch | Both pass | January 2 only |
| Synthetic two-session immutable cache | Same failure | Both pass | January 2 only |

Automated tests use synthetic local caches. They cover boundaries, future evidence,
malformed provenance, outcomes, independent ancestry validation, actual timestamp
walk-forward execution, offline inspection, resume, and durable holdout consumption.
No Tiingo acquisition/parser changes or QF-45 EMA logic are included.
