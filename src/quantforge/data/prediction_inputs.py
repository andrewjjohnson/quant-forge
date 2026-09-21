"""Truthful intraday provenance at the existing session-based prediction boundary."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, cast

from quantforge.configuration import (
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.data.corporate_actions import corporate_action_snapshot_id
from quantforge.data.exceptions import ValidationError
from quantforge.data.intraday import IntradayBar
from quantforge.data.intraday_ingestion import IntradayDataset, IntradayMarketDataCache
from quantforge.data.lineage import (
    DATASET_FAMILY_SCHEMA_VERSION,
    AdjustmentBasis,
    DatasetFamily,
    DatasetFamilyReference,
)
from quantforge.data.models import (
    AdjustmentMode,
    CorporateActionAvailability,
    DailyBar,
    IntradayPredictionProvenance,
    JsonValue,
    MarketDataset,
    ProviderResponse,
)
from quantforge.data.multi_timeframe import (
    ContextBar,
    MultiTimeframeContext,
    TimeframeBarSeries,
)
from quantforge.data.session_aggregation import (
    AggregatedSessionDataset,
    aggregate_session_dataset,
)
from quantforge.timeframes import SessionInterval, Timeframe

if TYPE_CHECKING:
    from quantforge.data.cache import MarketDataCache

DAILY_CORPORATE_ACTION_POLICY = "separate_provider_reported_cash_dividends_and_splits"
INTRADAY_CORPORATE_ACTION_POLICY = "not_provided_for_intraday_bars"


def validate_prediction_provenance(
    record: PrimitiveMapping,
) -> IntradayPredictionProvenance | None:
    """Validate event availability separately from price-basis compatibility.

    Missing new fields mean legacy daily semantics only. An unavailable event
    source must carry explicit lineage and cannot claim a complete event history.
    Used by both dataset validation and observational manifest inspection.
    """
    provenance = (
        IntradayPredictionProvenance.from_primitive(record["intraday_provenance"])
        if "intraday_provenance" in record
        else None
    )
    policy = record.get("corporate_action_policy")
    if provenance is None:
        if policy != DAILY_CORPORATE_ACTION_POLICY:
            raise ValidationError("unsupported corporate-action dataset policy")
        return None
    if (
        provenance.corporate_action_availability
        is not CorporateActionAvailability.UNAVAILABLE
        or policy != INTRADAY_CORPORATE_ACTION_POLICY
        or record.get("corporate_actions_complete") is not False
        or any(
            type(record.get(name)) is not int or record[name] != 0
            for name in ("corporate_action_count", "dividend_count", "split_count")
        )
        or record.get("corporate_action_snapshot_id")
        != corporate_action_snapshot_id(())
    ):
        raise ValidationError("contradictory intraday corporate-action provenance")
    basis = AdjustmentBasis(
        AdjustmentMode(cast(str, record.get("adjustment_mode"))),
        cast(str, record.get("ohlc_basis")),
        cast(str, record.get("volume_basis")),
        cast(str, policy),
        cast(bool, record.get("adjusted_fields_used")),
    )
    _validate_family_evidence(provenance, record, basis)
    return provenance


def _validate_family_evidence(
    provenance: IntradayPredictionProvenance,
    record: PrimitiveMapping,
    basis: AdjustmentBasis,
) -> None:
    """Bind declared semantics to the source committed by the family identity."""
    manifest = provenance.family_manifest.to_primitive()
    if (
        set(manifest)
        != {
            "schema_version",
            "canonical_source",
            "source_consistency",
            "family_id",
            "lineage",
            "manifest_id",
        }
        or manifest.get("schema_version") != DATASET_FAMILY_SCHEMA_VERSION
        or manifest.get("family_id") != provenance.family_id
        or configuration_identity(
            {
                key: value
                for key, value in manifest.items()
                if key not in {"family_id", "lineage", "manifest_id"}
            }
        )
        != provenance.family_id
        or configuration_identity(
            {key: value for key, value in manifest.items() if key != "manifest_id"}
        )
        != manifest.get("manifest_id")
    ):
        raise ValidationError(
            "prediction input source lineage family manifest is invalid"
        )
    source = manifest.get("canonical_source")
    if not isinstance(source, dict):
        raise ValidationError("prediction input source lineage is invalid")
    if source.get("adjustment_basis") != basis.to_primitive():
        raise ValidationError(
            "prediction input adjustment basis differs from source family"
        )
    feed_scope = source.get("feed_scope")
    if (
        source.get("snapshot_id") != provenance.source_dataset_id
        or source.get("symbol") != record.get("canonical_symbol", record.get("symbol"))
        or source.get("provider") != record.get("provider_name")
        or not isinstance(feed_scope, dict)
        or configuration_identity(feed_scope) != provenance.feed_scope_id
    ):
        raise ValidationError("prediction input source lineage is incompatible")


def prediction_dataset_from_intraday(
    source: IntradayDataset,
    sessions: AggregatedSessionDataset,
    *,
    cache: MarketDataCache,
    intraday_cache: IntradayMarketDataCache,
    family: DatasetFamily | None = None,
) -> MarketDataset:
    """Persist completed session bars with their original intraday source lineage.

    This is the session carrier required by QF-11, not an intraday schedule or a
    new aggregation algorithm. Exact decisions and labels still use canonical
    intraday series. The source must exactly match its immutable intraday cache
    artifact. No provider client, credentials, or network is involved.
    """
    target = sessions.metadata.target_timeframe
    if target.interval != SessionInterval(1):
        raise ValidationError("prediction input requires one-session derived bars")
    family = sessions.dataset_family if family is None else family
    TimeframeBarSeries.from_source_dataset(source, family=family, cache=intraday_cache)
    expected = aggregate_session_dataset(
        source, target, policy=sessions.metadata.aggregation_policy
    )
    if expected != sessions:
        raise ValidationError(
            "prediction session dataset differs from its intraday source"
        )
    TimeframeBarSeries.from_aggregated_session_dataset(sessions, family=family)
    basis = source.request.adjustment_basis
    if basis.corporate_action_policy != INTRADAY_CORPORATE_ACTION_POLICY:
        raise ValidationError(
            "intraday prediction projection requires explicit unavailable events"
        )
    bars = tuple(
        DailyBar(
            bar.symbol,
            bar.session_dates[0],
            bar.open,
            bar.high,
            bar.low,
            bar.close,
            bar.volume,
        )
        for bar in sessions.bars
    )
    if not bars:
        raise ValidationError("prediction input requires completed session coverage")
    provenance = IntradayPredictionProvenance(
        family.family_id,
        source.metadata.dataset_id,
        source.request.request_id,
        source.metadata.raw_snapshot_ids,
        sessions.metadata.dataset_id,
        target.configuration_id,
        configuration_identity(target.session_policy.to_primitive()),
        configuration_identity(family.feed_scope.to_primitive()),
        CorporateActionAvailability.UNAVAILABLE,
        PrimitiveMappingSnapshot.capture(family.to_manifest()),
    )
    # The immutable raw extract is the original derived artifact manifest. It
    # records the full source family and aggregation evidence, not invented events.
    response = ProviderResponse(
        source.metadata.provider_name,
        source.metadata.provider_symbol,
        source.metadata.retrieved_at,
        source.request.timeframe.session_policy.timezone_name,
        basis.adjustment_mode,
        (),
        cast(
            dict[str, JsonValue],
            {
                "component": "quantforge_intraday_prediction_input",
                "session_dataset": sessions.to_manifest(),
                "dataset_family": family.to_manifest(),
            },
        ),
        "quantforge_intraday_prediction_input_v1",
    )
    metadata: dict[str, object] = {
        "canonical_symbol": source.request.symbol,
        "provider_name": response.provider_name,
        "provider_symbol": response.provider_symbol,
        "retrieved_at": response.retrieved_at,
        "requested_start": bars[0].session_date,
        "requested_end": bars[-1].session_date,
        "actual_first_session": bars[0].session_date,
        "actual_last_session": bars[-1].session_date,
        "calendar": target.session_policy.calendar_name,
        "provider_timezone": response.provider_timezone,
        "adjustment_mode": basis.adjustment_mode,
        "bar_count": len(bars),
        "missing_sessions": (),
        "split_sessions": (),
        "dividend_sessions": (),
        "corporate_actions_complete": False,
        "corporate_action_count": 0,
        "dividend_count": 0,
        "split_count": 0,
        "corporate_action_snapshot_id": corporate_action_snapshot_id(()),
        "ohlc_basis": basis.ohlc_basis,
        "volume_basis": basis.volume_basis,
        "adjusted_fields_used": basis.adjusted_fields_used,
        "corporate_action_policy": basis.corporate_action_policy,
        "adapter_version": response.adapter_version,
        "intraday_provenance": provenance,
    }
    key = configuration_identity(
        {
            "component": response.adapter_version,
            "provenance": provenance.to_primitive(),
            "adjustment_basis": basis.to_primitive(),
        }
    )
    return cache.persist(response, bars, (), metadata, key)


def validate_prediction_source_reference(
    provenance: IntradayPredictionProvenance,
    reference: PrimitiveMapping,
) -> None:
    """Require common family, immutable source, and an explicit matching feed."""
    feed_scope = reference.get("feed_scope")
    if (
        reference.get("family_id") != provenance.family_id
        or reference.get("canonical_source_snapshot_id") != provenance.source_dataset_id
        or not isinstance(feed_scope, dict)
        or configuration_identity(feed_scope) != provenance.feed_scope_id
    ):
        raise ValidationError("prediction input source lineage is incompatible")


def _validate_source(
    dataset: MarketDataset,
    reference: DatasetFamilyReference,
    timeframe: Timeframe,
    bars: Iterable[ContextBar],
) -> None:
    metadata = dataset.metadata
    provenance = metadata.intraday_provenance
    if provenance is None:
        # Preserve the existing daily/context and concrete labeler validation.
        return
    validate_prediction_source_reference(
        provenance, reference.to_primitive(include_feed_scope=True)
    )
    if (
        configuration_identity(timeframe.session_policy.to_primitive())
        != provenance.session_policy_id
    ):
        raise ValidationError("prediction input session policy is incompatible")
    basis = AdjustmentBasis(
        metadata.adjustment_mode,
        metadata.ohlc_basis,
        metadata.volume_basis,
        metadata.corporate_action_policy,
        metadata.adjusted_fields_used,
    )
    for bar in bars:
        if bar.symbol != metadata.canonical_symbol:
            raise ValidationError("prediction input source symbol is incompatible")
        if isinstance(bar, IntradayBar):
            original = bar.provenance
            if original.adjustment_basis != basis:
                raise ValidationError(
                    "prediction input source adjustment basis is incompatible"
                )
            canonical_source = (
                original.provider_name == metadata.provider_name
                and original.source_request_id == provenance.source_request_id
                and original.source_snapshot_id in provenance.source_raw_snapshot_ids
            )
            derived_source = (
                reference.dataset_id != provenance.source_dataset_id
                and original.source_snapshot_id == provenance.source_dataset_id
            )
            if (
                not (canonical_source or derived_source)
                or configuration_identity(original.feed_scope.to_primitive())
                != provenance.feed_scope_id
            ):
                raise ValidationError(
                    "prediction input bar source lineage is incompatible"
                )


def validate_prediction_source(
    dataset: MarketDataset, source: TimeframeBarSeries
) -> None:
    """Shared future-label input validation for every timestamp labeler."""
    _validate_source(dataset, source.dataset_reference, source.timeframe, source.bars)
    provenance = dataset.metadata.intraday_provenance
    if (
        provenance is not None
        and source.dataset_family_manifest_id
        != (provenance.family_manifest.to_primitive()["manifest_id"])
    ):
        raise ValidationError("outcome source family manifest is incompatible")


def validate_prediction_context_sources(
    dataset: MarketDataset, context: MultiTimeframeContext
) -> None:
    """Bind visible context sources without rewriting or stripping provenance."""
    if not isinstance(cast(object, context), MultiTimeframeContext):
        raise ValidationError("prediction context provider returned an invalid context")
    for item in context.timeframes:
        if item.dataset_reference is not None:
            _validate_source(
                dataset, item.dataset_reference, item.requirement.timeframe, item.bars
            )
