"""Timeframe-bound indicator evaluation over causal multi-timeframe contexts."""

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from itertools import pairwise
from typing import cast

from quantforge.configuration import (
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
    decimal_to_primitive,
)
from quantforge.data.developing_bars import DevelopingBar
from quantforge.data.lineage import DatasetFamilyReference
from quantforge.data.multi_timeframe import (
    ContextCompletionPolicy,
    MultiTimeframeContext,
    MultiTimeframeContextError,
)
from quantforge.indicators.base import (
    DevelopingBarSupport,
    IndicatorBar,
    TimeframeNeutralIndicator,
)
from quantforge.indicators.exceptions import (
    IndicatorSourceError,
    MisalignedIndicatorOutputError,
    UnsupportedDevelopingBarError,
)
from quantforge.indicators.models import IndicatorFieldOutput, MarketField
from quantforge.timeframes import BarCompletion, Timeframe

TIMEFRAME_INDICATOR_CONTRACT_VERSION = "1"
_RESERVED_ROW_FIELDS = frozenset(("bar_id", "bar_end_timestamp", "completion"))


def _timeframe_primitive(timeframe: Timeframe) -> PrimitiveMapping:
    return {
        "configuration_id": timeframe.configuration_id,
        "configuration": timeframe.to_primitive(),
    }


@dataclass(frozen=True, slots=True)
class TimeframeIndicatorOutput:
    """Immutable indicator values aligned to exact causal source bars."""

    indicator_name: str
    configuration_id: str
    source_timeframe: Timeframe
    source_fields: tuple[MarketField, ...]
    completion_policy: ContextCompletionPolicy
    developing_bar_support: DevelopingBarSupport
    dataset_reference: DatasetFamilyReference
    warm_up_bars: int
    bar_ids: tuple[str, ...]
    bar_end_timestamps: tuple[datetime, ...]
    completion_states: tuple[BarCompletion, ...]
    fields: tuple[IndicatorFieldOutput, ...]

    def __post_init__(self) -> None:
        if not self.indicator_name or not self.configuration_id:
            raise MisalignedIndicatorOutputError(
                "timeframe indicator name and configuration identity are required"
            )
        count = len(self.bar_ids)
        if (
            len(set(self.bar_ids)) != count
            or len(self.bar_end_timestamps) != count
            or len(self.completion_states) != count
            or any(len(output.values) != count for output in self.fields)
        ):
            raise MisalignedIndicatorOutputError(
                "timeframe indicator output must align one-to-one with source bars"
            )
        if any(
            current >= following
            for current, following in pairwise(self.bar_end_timestamps)
        ):
            raise MisalignedIndicatorOutputError(
                "timeframe indicator bar ends must be strictly chronological"
            )
        field_names = tuple(output.name for output in self.fields)
        if (
            not field_names
            or len(field_names) != len(set(field_names))
            or _RESERVED_ROW_FIELDS.intersection(field_names)
        ):
            raise MisalignedIndicatorOutputError(
                "timeframe indicator fields must be nonempty, unique, and unreserved"
            )
        if any(
            value is not None and not value.is_finite()
            for output in self.fields
            for value in output.values
        ):
            raise MisalignedIndicatorOutputError(
                "timeframe indicator values must be finite decimals or None"
            )
        if self.warm_up_bars < 1:
            raise MisalignedIndicatorOutputError(
                "timeframe indicator warm-up must be a positive bar count"
            )

    @property
    def aggregation_provenance(self) -> PrimitiveMapping:
        """Return the compact dataset-family lineage binding for this evaluation."""
        return self.dataset_reference.to_primitive()

    def values_for(self, field_name: str) -> tuple[Decimal | None, ...]:
        """Return one named value series."""
        for output in self.fields:
            if output.name == field_name:
                return output.values
        raise MisalignedIndicatorOutputError(
            f"timeframe indicator output does not contain field: {field_name}"
        )

    def to_rows(self) -> list[PrimitiveMapping]:
        """Return one deterministic row for every exact input bar."""
        rows: list[PrimitiveMapping] = []
        for index, bar_id in enumerate(self.bar_ids):
            row: PrimitiveMapping = {
                "bar_id": bar_id,
                "bar_end_timestamp": self.bar_end_timestamps[index].isoformat(),
                "completion": self.completion_states[index].value,
            }
            for output in self.fields:
                value = output.values[index]
                row[output.name] = (
                    None if value is None else decimal_to_primitive(value)
                )
            rows.append(row)
        return rows


@dataclass(frozen=True, slots=True, init=False)
class ConfiguredTimeframeIndicator:
    """One indicator bound to a timeframe, completion policy, and lineage source."""

    indicator: TimeframeNeutralIndicator
    source_timeframe: Timeframe
    completion_policy: ContextCompletionPolicy
    dataset_reference: DatasetFamilyReference
    _indicator_configuration: PrimitiveMappingSnapshot = field(repr=False)
    _indicator_configuration_id: str = field(repr=False)
    _indicator_name: str = field(repr=False)
    _source_fields: tuple[MarketField, ...] = field(repr=False)
    _output_fields: tuple[str, ...] = field(repr=False)
    _developing_bar_support: DevelopingBarSupport = field(repr=False)
    _warm_up_bars: int = field(repr=False)

    @classmethod
    def from_context(
        cls,
        indicator: TimeframeNeutralIndicator,
        context: MultiTimeframeContext,
        source_timeframe: Timeframe,
    ) -> "ConfiguredTimeframeIndicator":
        """Bind an indicator only after resolving an exact declared context source."""
        context_value = cast(object, context)
        source_timeframe_value = cast(object, source_timeframe)
        if not isinstance(context_value, MultiTimeframeContext):
            raise IndicatorSourceError("indicator context is invalid")
        if not isinstance(source_timeframe_value, Timeframe):
            raise IndicatorSourceError("indicator source timeframe is invalid")
        try:
            metadata = context.metadata_for(source_timeframe)
        except MultiTimeframeContextError as error:
            raise IndicatorSourceError(str(error)) from error
        if metadata.dataset_reference is None:
            raise IndicatorSourceError(
                "indicator source timeframe has no dataset lineage reference"
            )
        try:
            snapshot = PrimitiveMappingSnapshot.capture(indicator.configuration())
            indicator_id = indicator.configuration_id
            support = indicator.developing_bar_support
            required_fields = indicator.required_fields
            output_fields = indicator.output_fields
            warm_up = indicator.warm_up_observations
        except (AttributeError, TypeError, ValueError) as error:
            raise IndicatorSourceError(
                "indicator does not implement the timeframe-neutral contract"
            ) from error
        support_value = cast(object, support)
        required_fields_value = cast(object, required_fields)
        output_fields_value = cast(object, output_fields)
        warm_up_value = cast(object, warm_up)
        untyped_required_fields: frozenset[object] = (
            cast(frozenset[object], required_fields_value)
            if isinstance(required_fields_value, frozenset)
            else frozenset()
        )
        if (
            not isinstance(support_value, DevelopingBarSupport)
            or not isinstance(required_fields_value, frozenset)
            or any(
                not isinstance(item, MarketField) for item in untyped_required_fields
            )
            or not isinstance(output_fields_value, tuple)
            or not output_fields_value
            or isinstance(warm_up_value, bool)
            or not isinstance(warm_up_value, int)
            or warm_up < 1
            or configuration_identity(snapshot.to_primitive()) != indicator_id
        ):
            raise IndicatorSourceError(
                "indicator timeframe metadata or configuration identity is invalid"
            )
        instance = object.__new__(cls)
        object.__setattr__(instance, "indicator", indicator)
        object.__setattr__(instance, "source_timeframe", source_timeframe)
        object.__setattr__(instance, "completion_policy", context.completion_policy)
        object.__setattr__(instance, "dataset_reference", metadata.dataset_reference)
        object.__setattr__(instance, "_indicator_configuration", snapshot)
        object.__setattr__(instance, "_indicator_configuration_id", indicator_id)
        object.__setattr__(instance, "_indicator_name", indicator.name)
        object.__setattr__(
            instance,
            "_source_fields",
            tuple(sorted(required_fields, key=lambda item: item.value)),
        )
        object.__setattr__(instance, "_output_fields", output_fields)
        object.__setattr__(instance, "_developing_bar_support", support)
        object.__setattr__(instance, "_warm_up_bars", warm_up)
        return instance

    @property
    def source_fields(self) -> tuple[MarketField, ...]:
        return self._source_fields

    def configuration(self) -> PrimitiveMapping:
        """Return formula plus complete temporal and lineage source configuration."""
        return {
            "component_type": "timeframe_indicator",
            "contract_version": TIMEFRAME_INDICATOR_CONTRACT_VERSION,
            "indicator": {
                "configuration_id": self._indicator_configuration_id,
                "configuration": self._indicator_configuration.to_primitive(),
            },
            "source": {
                "timeframe": _timeframe_primitive(self.source_timeframe),
                "fields": [field.value for field in self.source_fields],
                "completion_policy": self.completion_policy.value,
                "developing_bar_support": self._developing_bar_support.value,
                "observation_unit": "bar",
                "warm_up_bars": self._warm_up_bars,
                "aggregation_provenance": self.dataset_reference.to_primitive(),
            },
        }

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    def calculate(self, context: MultiTimeframeContext) -> TimeframeIndicatorOutput:
        """Evaluate the exact configured timeframe or fail closed on source drift."""
        context_value = cast(object, context)
        if not isinstance(context_value, MultiTimeframeContext):
            raise IndicatorSourceError("indicator context is invalid")
        if context.completion_policy is not self.completion_policy:
            raise IndicatorSourceError(
                "indicator completion policy does not match its configured source"
            )
        try:
            metadata = context.metadata_for(self.source_timeframe)
            bars = context.bars_for(self.source_timeframe)
        except MultiTimeframeContextError as error:
            raise IndicatorSourceError(str(error)) from error
        if metadata.dataset_reference != self.dataset_reference:
            raise IndicatorSourceError(
                "indicator dataset lineage does not match its configured source"
            )
        if any(bar.timeframe != self.source_timeframe for bar in bars):
            raise IndicatorSourceError(
                "indicator source contains a bar from another timeframe"
            )
        developing = tuple(bar for bar in bars if isinstance(bar, DevelopingBar))
        if developing and (
            self.completion_policy is not ContextCompletionPolicy.DEVELOPING_BAR_AS_OF
            or self._developing_bar_support is not DevelopingBarSupport.DEVELOPING_AS_OF
        ):
            raise UnsupportedDevelopingBarError(
                "developing-bar evaluation requires explicit context and "
                "indicator support"
            )
        current_configuration = PrimitiveMappingSnapshot.capture(
            self.indicator.configuration()
        )
        if (
            current_configuration != self._indicator_configuration
            or self.indicator.configuration_id != self._indicator_configuration_id
            or self._indicator_configuration_id
            != configuration_identity(current_configuration.to_primitive())
            or self.indicator.name != self._indicator_name
            or tuple(
                sorted(self.indicator.required_fields, key=lambda item: item.value)
            )
            != self._source_fields
            or self.indicator.output_fields != self._output_fields
            or self.indicator.developing_bar_support is not self._developing_bar_support
            or self.indicator.warm_up_observations != self._warm_up_bars
        ):
            raise IndicatorSourceError(
                "indicator configuration changed after source binding"
            )
        try:
            fields = self.indicator.calculate_bar_fields(
                cast(tuple[IndicatorBar, ...], bars)
            )
        except (AttributeError, TypeError, ValueError) as error:
            if isinstance(error, IndicatorSourceError):
                raise
            raise IndicatorSourceError(
                "indicator failed its timeframe-neutral bar calculation"
            ) from error
        if tuple(output.name for output in fields) != self._output_fields:
            raise MisalignedIndicatorOutputError(
                "timeframe indicator fields do not match its declared outputs"
            )
        return TimeframeIndicatorOutput(
            indicator_name=self._indicator_name,
            configuration_id=self.configuration_id,
            source_timeframe=self.source_timeframe,
            source_fields=self.source_fields,
            completion_policy=self.completion_policy,
            developing_bar_support=self._developing_bar_support,
            dataset_reference=self.dataset_reference,
            warm_up_bars=self._warm_up_bars,
            bar_ids=tuple(bar.bar_id for bar in bars),
            bar_end_timestamps=tuple(bar.end_timestamp for bar in bars),
            completion_states=tuple(bar.completion for bar in bars),
            fields=fields,
        )


def bind_indicator(
    indicator: TimeframeNeutralIndicator,
    context: MultiTimeframeContext,
    source_timeframe: Timeframe,
) -> ConfiguredTimeframeIndicator:
    """Bind one indicator to an exact requested timeframe series."""
    return ConfiguredTimeframeIndicator.from_context(
        indicator, context, source_timeframe
    )


def evaluate_indicator(
    indicator: TimeframeNeutralIndicator,
    context: MultiTimeframeContext,
    source_timeframe: Timeframe,
) -> TimeframeIndicatorOutput:
    """Bind and evaluate one indicator against a requested context timeframe."""
    return bind_indicator(indicator, context, source_timeframe).calculate(context)


__all__ = [
    "TIMEFRAME_INDICATOR_CONTRACT_VERSION",
    "ConfiguredTimeframeIndicator",
    "TimeframeIndicatorOutput",
    "bind_indicator",
    "evaluate_indicator",
]
