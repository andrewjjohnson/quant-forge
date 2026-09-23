"""Causal views of validated canonical prediction inputs, without parent objects.

The factory validates the complete canonical input before retaining only local
QF-19 observations. Offline local validation proves view consistency; checking
membership against an independently supplied ancestor additionally authenticates
the projection relationship. An opaque SHA-256 reference alone cannot prove
membership in the original flat-hashed source without reading that source.
"""

from dataclasses import replace
from datetime import UTC, date, datetime
from typing import cast

from quantforge.configuration import (
    Primitive,
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.data.exceptions import ValidationError
from quantforge.data.identity import (
    serialize_bars_csv,
    sha256_hex,
)
from quantforge.data.models import (
    BOUNDED_PREDICTION_ADAPTER_VERSION,
    BoundedPredictionProvenance,
    DatasetMetadata,
    IntradayPredictionProvenance,
    MarketDataset,
)
from quantforge.data.prediction_inputs import validate_bounded_prediction_record
from quantforge.data.prediction_session_evidence import (
    session_bar_from_evidence,
    session_projection_bars,
)


def bounded_prediction_view(
    dataset: MarketDataset, cutoff: datetime, *, start: date | None = None
) -> MarketDataset:
    """Validate a canonical input, then detach a completed-session causal view.

    ``cutoff`` is inclusive at actual session close, independent of bar labels
    and source retrieval time. Reprojection cannot widen an existing view.
    Intraday feature/label observations remain owned by QF-42/QF-46/QF-48.
    """
    from quantforge.data.validate import validate_market_dataset

    validate_market_dataset(dataset)
    if not isinstance(cast(object, cutoff), datetime) or cutoff.utcoffset() is None:
        raise ValidationError("bounded prediction cutoff must be timezone-aware")
    cutoff = cutoff.astimezone(UTC)
    original = dataset.metadata.intraday_provenance
    if isinstance(original, IntradayPredictionProvenance):
        evidence = cast(
            PrimitiveMapping, original.session_evidence.to_primitive()["bars"]
        )
        entries = cast(list[Primitive], evidence["bars"])
        ancestry = {
            "canonical_input_id": dataset.metadata.dataset_id,
            "canonical_provenance_id": configuration_identity(original.to_primitive()),
        }
    elif isinstance(original, BoundedPredictionProvenance):
        if cutoff > original.causal_cutoff:
            raise ValidationError("bounded prediction views cannot widen their cutoff")
        entries = cast(
            list[Primitive], original.session_evidence.to_primitive()["bars"]
        )
        ancestry = {
            "canonical_input_id": original.canonical_input_id,
            "canonical_provenance_id": original.canonical_provenance_id,
        }
    else:
        raise ValidationError(
            "bounded intraday views require canonical intraday ancestry"
        )
    selected = tuple(
        bar
        for entry in entries
        for bar in (session_bar_from_evidence(entry),)
        if bar.end_timestamp <= cutoff
        and (start is None or bar.session_dates[0] >= start)
    )
    if not selected:
        raise ValidationError(
            "bounded prediction requires prior completed session context"
        )
    if start is not None and selected[0].session_dates[0] != start:
        raise ValidationError(
            "bounded prediction start must be a visible observed session"
        )
    bars = session_projection_bars(selected)
    digest = sha256_hex(serialize_bars_csv(bars))
    provenance = BoundedPredictionProvenance(
        canonical_input_id=ancestry["canonical_input_id"],
        canonical_provenance_id=ancestry["canonical_provenance_id"],
        family_id=original.family_id,
        source_dataset_id=original.source_dataset_id,
        source_request_id=original.source_request_id,
        session_dataset_id=original.session_dataset_id,
        session_timeframe_configuration_id=original.session_timeframe_configuration_id,
        session_policy_id=original.session_policy_id,
        feed_scope_id=original.feed_scope_id,
        provider_symbol=dataset.metadata.provider_symbol,
        source_retrieved_at=dataset.metadata.retrieved_at,
        causal_cutoff=cutoff,
        family_manifest=original.family_manifest,
        session_evidence=PrimitiveMappingSnapshot.capture(
            {
                "bars": [
                    {"bar_id": bar.bar_id, "bar": bar.to_primitive()}
                    for bar in selected
                ]
            }
        ),
        bars_fingerprint=digest,
    )
    view_id = provenance.view_id
    raw_digest = configuration_identity(provenance.to_primitive())
    metadata = replace(
        dataset.metadata,
        dataset_id=view_id,
        requested_start=bars[0].session_date,
        actual_first_session=bars[0].session_date,
        requested_end=bars[-1].session_date,
        actual_last_session=bars[-1].session_date,
        bar_count=len(bars),
        data_sha256=digest,
        raw_sha256=raw_digest,
        missing_sessions=(),
        raw_location=f"raw/{raw_digest}.json",
        normalized_location=f"datasets/{view_id}/bars.csv",
        corporate_actions_location=f"datasets/{view_id}/corporate_actions.json",
        adapter_version=BOUNDED_PREDICTION_ADAPTER_VERSION,
        intraday_provenance=provenance,
    )
    view = MarketDataset(bars, metadata)
    validate_market_dataset(view)
    return view


def validate_bounded_prediction_ancestry(
    view: MarketDataset, canonical: MarketDataset
) -> None:
    """Check exact subset membership against an independently retained ancestor.

    The ancestor stays with the caller/inspection layer, never inside the view.
    This requires no provider access or research calculation.
    """
    provenance = view.metadata.intraday_provenance
    if not isinstance(provenance, BoundedPredictionProvenance) or not isinstance(
        canonical.metadata.intraday_provenance, IntradayPredictionProvenance
    ):
        raise ValidationError("bounded ancestry validation requires a canonical parent")
    expected = bounded_prediction_view(
        canonical, provenance.causal_cutoff, start=view.metadata.actual_first_session
    )
    if expected != view:
        raise ValidationError(
            "bounded prediction view differs from its canonical ancestry"
        )


def validate_prediction_view_lineage(
    record: PrimitiveMapping,
    canonical_metadata: DatasetMetadata,
    cutoff: datetime,
    *,
    start: date | None = None,
) -> None:
    """Bind an offline QF-39/QF-40 view to its independently retained plan source.

    Canonical evidence is owned by the inspection layer. It is never attached
    to the view or supplied to research callbacks. Legacy daily inputs retain
    their established integrity checks.
    """
    original = canonical_metadata.intraday_provenance
    if original is None:
        return
    if not isinstance(original, IntradayPredictionProvenance):
        raise ValidationError("prediction plan must retain a canonical input")
    provenance = validate_bounded_prediction_record(record)
    evidence = cast(PrimitiveMapping, original.session_evidence.to_primitive()["bars"])
    bars = session_projection_bars(
        tuple(
            session_bar_from_evidence(entry)
            for entry in cast(list[Primitive], evidence["bars"])
        )
    )
    canonical = MarketDataset(bars, canonical_metadata)
    expected = bounded_prediction_view(canonical, cutoff, start=start)
    if expected.metadata.intraday_provenance != provenance:
        raise ValidationError(
            "bounded prediction view differs from canonical plan ancestry or cutoff"
        )
