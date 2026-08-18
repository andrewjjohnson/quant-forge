"""QF-29 flattened feature requests over validated prediction contexts."""

from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from typing import cast

from quantforge.configuration import (
    Primitive,
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.data.developing_bars import DevelopingBar
from quantforge.data.models import MarketDataset
from quantforge.prediction.context import PredictionRuleContext
from quantforge.prediction.errors import SignalFeatureDatasetError
from quantforge.prediction.signal_feature_models import (
    SchemaField,
    SchemaFieldCategory,
)
from quantforge.timeframes import Timeframe

MULTI_TIMEFRAME_FEATURE_CONTRACT_VERSION = "1"
MULTI_TIMEFRAME_FEATURE_DATASET_ENGINE_VERSION = "35"
MULTI_TIMEFRAME_FEATURE_ENGINE_VERSION = "1"


def _valid_namespace(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and value[0].isalpha()
        and value == value.lower()
        and all(character.isalnum() or character == "_" for character in value)
    )


def _timeframe_primitive(timeframe: Timeframe) -> PrimitiveMapping:
    return {
        "configuration_id": timeframe.configuration_id,
        "configuration": timeframe.to_primitive(),
    }


@dataclass(frozen=True, slots=True)
class MultiTimeframeFeatureRequest:
    """One normalized indicator output to flatten under a timeframe namespace."""

    timeframe_name: str
    timeframe: Timeframe
    indicator_alias: str
    normalized_output_name: str
    unit: str
    nullable: bool = True

    def __post_init__(self) -> None:
        if (
            not _valid_namespace(self.timeframe_name)
            or not _valid_namespace(self.indicator_alias)
            or not _valid_namespace(self.normalized_output_name)
            or not isinstance(cast(object, self.unit), str)
            or not self.unit
            or not isinstance(cast(object, self.timeframe), Timeframe)
            or not isinstance(cast(object, self.nullable), bool)
        ):
            raise SignalFeatureDatasetError(
                "multi-timeframe feature requests require lowercase namespaces, "
                "a timeframe, indicator alias, normalized output, unit, and "
                "nullability"
            )

    @property
    def name(self) -> str:
        suffix = (
            self.indicator_alias
            if self.indicator_alias == self.normalized_output_name
            else f"{self.indicator_alias}_{self.normalized_output_name}"
        )
        return f"{self.timeframe_name}_{suffix}"

    @property
    def metadata_name(self) -> str:
        return f"{self.name}__metadata"

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "contract_version": MULTI_TIMEFRAME_FEATURE_CONTRACT_VERSION,
            "indicator_alias": self.indicator_alias,
            "name": self.name,
            "normalized_output_name": self.normalized_output_name,
            "nullable": self.nullable,
            "timeframe": _timeframe_primitive(self.timeframe),
            "timeframe_name": self.timeframe_name,
            "unit": self.unit,
        }


@dataclass(frozen=True, slots=True)
class _CapturedMultiTimeframeColumn:
    """One builder-owned scalar or metadata column fixed at a causal decision."""

    name: str
    definition: SchemaField
    value: Primitive | Decimal
    decision_session: date
    _configuration_snapshot: PrimitiveMappingSnapshot = field(repr=False)

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    def configuration(self) -> PrimitiveMapping:
        return self._configuration_snapshot.to_primitive()

    def value_from_history(self, history: MarketDataset) -> Primitive | Decimal:
        if history.bars[-1].session_date != self.decision_session:
            raise SignalFeatureDatasetError(
                "captured multi-timeframe feature is misaligned with its candidate"
            )
        return self.value


def capture_multi_timeframe_features(
    requests: tuple[MultiTimeframeFeatureRequest, ...],
    context: PredictionRuleContext,
) -> tuple[_CapturedMultiTimeframeColumn, ...]:
    """Resolve requests only from normalized QF-28 indicator outputs."""
    if any(
        not isinstance(item, MultiTimeframeFeatureRequest)
        for item in cast(tuple[object, ...], requests)
    ):
        raise SignalFeatureDatasetError(
            "multi-timeframe features must be typed requests"
        )
    ordered = tuple(sorted(requests, key=lambda item: item.name))
    names = tuple(item.name for item in ordered)
    if len(names) != len(set(names)):
        raise SignalFeatureDatasetError(
            "multi-timeframe feature requests must produce unique names"
        )
    columns: list[_CapturedMultiTimeframeColumn] = []
    for request in ordered:
        columns.extend(_capture_request(request, context))
    return tuple(sorted(columns, key=lambda item: item.name))


def _capture_request(
    request: MultiTimeframeFeatureRequest,
    context: PredictionRuleContext,
) -> tuple[_CapturedMultiTimeframeColumn, _CapturedMultiTimeframeColumn]:
    timeframe_input = next(
        (
            item
            for item in context.timeframes
            if item.requirement.timeframe.configuration_id
            == request.timeframe.configuration_id
        ),
        None,
    )
    if timeframe_input is None:
        raise SignalFeatureDatasetError(
            f"feature timeframe was not declared by the prediction rule: {request.name}"
        )
    indicator_requirement = next(
        (
            item
            for item in timeframe_input.requirement.indicators
            if item.alias == request.indicator_alias
        ),
        None,
    )
    named_output = next(
        (
            item
            for item in timeframe_input.indicators
            if item.alias == request.indicator_alias
        ),
        None,
    )
    if indicator_requirement is None or named_output is None:
        raise SignalFeatureDatasetError(
            f"feature indicator alias was not declared: {request.name}"
        )
    output = named_output.output
    output_names = tuple(item.name for item in output.fields)
    if request.normalized_output_name not in output_names:
        raise SignalFeatureDatasetError(
            f"normalized indicator output was not declared: {request.name}"
        )
    if not output.bar_ids:
        raise SignalFeatureDatasetError(
            f"multi-timeframe indicator has no causal source bar: {request.name}"
        )
    value = output.values_for(request.normalized_output_name)[-1]
    if value is None and not request.nullable:
        raise SignalFeatureDatasetError(
            f"multi-timeframe feature is unavailable but non-nullable: {request.name}"
        )
    latest_bar = timeframe_input.bars[-1]
    observed_through = output.bar_end_timestamps[-1]
    completion_timestamp = (
        latest_bar.expected_completion_boundary
        if isinstance(latest_bar, DevelopingBar)
        else observed_through
    )
    staleness = context.as_of - observed_through
    staleness_microseconds = (
        staleness.days * 86_400 + staleness.seconds
    ) * 1_000_000 + staleness.microseconds
    backend = output.backend_identity
    reference = output.dataset_reference
    metadata: PrimitiveMapping = {
        "timeframe_name": request.timeframe_name,
        "timeframe": _timeframe_primitive(request.timeframe),
        "context_id": context.context_id,
        "context_as_of": context.as_of.isoformat(),
        "completion_policy": output.completion_policy.value,
        "source_bar_id": output.bar_ids[-1],
        "source_bar_completion_timestamp": completion_timestamp.isoformat(),
        "source_bar_observed_through_timestamp": observed_through.isoformat(),
        "completion_state": output.completion_states[-1].value,
        "staleness_microseconds": staleness_microseconds,
        "indicator_alias": request.indicator_alias,
        "indicator_name": output.indicator_name,
        "indicator_configuration_id": indicator_requirement.configuration_id,
        "timeframe_indicator_configuration_id": output.configuration_id,
        "indicator_backend": (None if backend is None else backend.to_primitive()),
        "normalized_output_name": request.normalized_output_name,
        "dataset_family_id": reference.family_id,
        "source_dataset_id": reference.dataset_id,
        "canonical_source_snapshot_id": reference.canonical_source_snapshot_id,
        "feed_scope": reference.feed_scope.to_primitive(),
    }
    provenance = PrimitiveMappingSnapshot.capture(metadata)
    timing = (
        "available at the signal decision timestamp from the QF-20/QF-21 "
        "causal context only"
    )
    configuration_base: PrimitiveMapping = {
        "component_type": "multi_timeframe_signal_contextual_feature",
        "contract_version": MULTI_TIMEFRAME_FEATURE_CONTRACT_VERSION,
        "implementation_version": MULTI_TIMEFRAME_FEATURE_ENGINE_VERSION,
        "request": request.to_primitive(),
        "resolved_provenance": metadata,
    }

    def captured_column(
        *,
        name: str,
        data_type: str,
        unit: str,
        nullable: bool,
        value: Primitive | Decimal,
        role: str,
        source: str,
    ) -> _CapturedMultiTimeframeColumn:
        configuration = {
            **configuration_base,
            "column_name": name,
            "column_role": role,
        }
        return _CapturedMultiTimeframeColumn(
            name,
            SchemaField(
                name,
                SchemaFieldCategory.CONTEMPORANEOUS_FEATURE,
                data_type,
                unit,
                nullable,
                source,
                timing,
                provenance,
            ),
            value,
            context.decision_session,
            PrimitiveMappingSnapshot.capture(configuration),
        )

    return (
        captured_column(
            name=request.name,
            data_type="decimal",
            unit=request.unit,
            nullable=request.nullable,
            value=value,
            role="normalized_indicator_value",
            source=(
                "normalized QuantForge indicator result; no backend-specific "
                "library object or direct call"
            ),
        ),
        captured_column(
            name=request.metadata_name,
            data_type="object",
            unit="context_provenance",
            nullable=False,
            value=metadata,
            role="causal_provenance",
            source=(
                "QF-20/QF-21 source bar, staleness, QF-35 backend, and "
                "dataset-family lineage"
            ),
        ),
    )


__all__ = [
    "MULTI_TIMEFRAME_FEATURE_CONTRACT_VERSION",
    "MULTI_TIMEFRAME_FEATURE_DATASET_ENGINE_VERSION",
    "MULTI_TIMEFRAME_FEATURE_ENGINE_VERSION",
    "MultiTimeframeFeatureRequest",
]
