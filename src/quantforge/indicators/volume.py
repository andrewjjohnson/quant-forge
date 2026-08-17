"""Typed causal volume average and relative-volume indicators."""

from dataclasses import dataclass
from decimal import (
    ROUND_HALF_EVEN,
    Context,
    Decimal,
    DecimalException,
    DivisionByZero,
    InvalidOperation,
    Overflow,
    localcontext,
)
from enum import StrEnum
from typing import cast

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.data.lineage import (
    DatasetFamilyValidationError,
    FeedCoverage,
    FeedScope,
)
from quantforge.data.models import MarketDataset
from quantforge.indicators.base import (
    DevelopingBarSupport,
    IndicatorBar,
    IndicatorParameters,
    validate_indicator_alignment,
    validate_indicator_bars,
    validate_market_input,
)
from quantforge.indicators.exceptions import (
    IndicatorCalculationError,
    InvalidIndicatorParametersError,
)
from quantforge.indicators.models import (
    IndicatorFieldOutput,
    IndicatorOutput,
    IndicatorValue,
    MarketField,
)

VOLUME_MOVING_AVERAGE_OUTPUT = "volume_moving_average"
RELATIVE_VOLUME_OUTPUT = "relative_volume"
_DECIMAL_PRECISION = 34
_DECIMAL_EMIN = -999_999
_DECIMAL_EMAX = 999_999
_DECIMAL_TRAPS: tuple[type[DecimalException], ...] = (
    DivisionByZero,
    InvalidOperation,
    Overflow,
)


class RelativeVolumeDenominatorPolicy(StrEnum):
    """Which bars participate in a relative-volume denominator."""

    INCLUDE_CURRENT_BAR = "include_current_bar"
    EXCLUDE_CURRENT_BAR = "exclude_current_bar"


@dataclass(frozen=True, slots=True)
class VolumeMovingAverageParameters:
    """Trailing bar lookback and exact market-volume feed scope."""

    lookback: int
    feed_scope: FeedScope

    def __post_init__(self) -> None:
        _validate_lookback(self.lookback)
        _validate_feed_scope(self.feed_scope)

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "feed_scope": self.feed_scope.to_primitive(),
            "lookback": self.lookback,
        }


@dataclass(frozen=True, slots=True)
class RelativeVolumeParameters:
    """Trailing denominator lookback, convention, and exact feed scope."""

    lookback: int
    feed_scope: FeedScope
    denominator_policy: RelativeVolumeDenominatorPolicy = (
        RelativeVolumeDenominatorPolicy.INCLUDE_CURRENT_BAR
    )

    def __post_init__(self) -> None:
        _validate_lookback(self.lookback)
        _validate_feed_scope(self.feed_scope)
        if not isinstance(
            cast(object, self.denominator_policy), RelativeVolumeDenominatorPolicy
        ):
            raise InvalidIndicatorParametersError(
                "denominator_policy must be a relative-volume denominator policy"
            )

    @property
    def includes_current_bar(self) -> bool:
        """Whether the numerator bar also participates in its denominator."""
        return (
            self.denominator_policy
            is RelativeVolumeDenominatorPolicy.INCLUDE_CURRENT_BAR
        )

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "denominator_policy": self.denominator_policy.value,
            "feed_scope": self.feed_scope.to_primitive(),
            "lookback": self.lookback,
        }


class VolumeMovingAverage:
    """Full-window arithmetic mean of canonical volume observations."""

    name = "volume_moving_average"
    implementation_version = "1"
    output_fields = (VOLUME_MOVING_AVERAGE_OUTPUT,)
    missing_value = None
    developing_bar_support = DevelopingBarSupport.DEVELOPING_AS_OF

    def __init__(self, parameters: VolumeMovingAverageParameters) -> None:
        self._parameters = parameters

    @property
    def parameters(self) -> IndicatorParameters:
        return self._parameters

    @property
    def required_fields(self) -> frozenset[MarketField]:
        return frozenset((MarketField.VOLUME,))

    @property
    def warm_up_observations(self) -> int:
        return self._parameters.lookback

    def configuration(self) -> PrimitiveMapping:
        return _configuration(
            component_name=self.name,
            parameters=self._parameters.to_primitive(),
            warm_up_observations=self.warm_up_observations,
            output_fields=self.output_fields,
            formula={
                "denominator": "trailing_lookback_bars_including_current_bar",
                "missing_value_policy": (
                    "emit_none_for_warm_up_or_any_nonfinite_window_observation"
                ),
                "zero_volume_policy": "retain_zero_as_a_valid_observation",
            },
        )

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    @classmethod
    def from_configuration(
        cls, configuration: PrimitiveMapping
    ) -> "VolumeMovingAverage":
        """Restore one exact serialized volume-average configuration."""
        parameters = _serialized_parameters(configuration)
        indicator = cls(
            VolumeMovingAverageParameters(
                _serialized_lookback(parameters),
                _serialized_feed_scope(parameters),
            )
        )
        _require_exact_configuration(configuration, indicator.configuration(), cls.name)
        return indicator

    def calculate(self, dataset: MarketDataset) -> IndicatorOutput:
        """Return daily-compatible output aligned with every canonical session."""
        validate_market_input(dataset, self.required_fields)
        fields = self.calculate_bar_fields(cast(tuple[IndicatorBar, ...], dataset.bars))
        output = IndicatorOutput(
            self.name,
            self.configuration_id,
            tuple(bar.session_date for bar in dataset.bars),
            fields,
        )
        validate_indicator_alignment(dataset, output)
        return output

    def calculate_bar_fields(
        self, bars: tuple[IndicatorBar, ...]
    ) -> tuple[IndicatorFieldOutput, ...]:
        """Calculate on any already-validated canonical timeframe."""
        validate_indicator_bars(bars, self.required_fields, require_finite=False)
        averages = _trailing_volume_average(
            tuple(bar.volume for bar in bars),
            self._parameters.lookback,
            exclude_current=False,
        )
        return (IndicatorFieldOutput(VOLUME_MOVING_AVERAGE_OUTPUT, averages),)


class RelativeVolume:
    """Current volume divided by an explicit trailing-volume denominator."""

    name = "relative_volume"
    implementation_version = "1"
    output_fields = (RELATIVE_VOLUME_OUTPUT,)
    missing_value = None
    developing_bar_support = DevelopingBarSupport.DEVELOPING_AS_OF

    def __init__(self, parameters: RelativeVolumeParameters) -> None:
        self._parameters = parameters

    @property
    def parameters(self) -> IndicatorParameters:
        return self._parameters

    @property
    def required_fields(self) -> frozenset[MarketField]:
        return frozenset((MarketField.VOLUME,))

    @property
    def warm_up_observations(self) -> int:
        return self._parameters.lookback + (
            0 if self._parameters.includes_current_bar else 1
        )

    def configuration(self) -> PrimitiveMapping:
        denominator = (
            "trailing_lookback_bars_including_current_bar"
            if self._parameters.includes_current_bar
            else "prior_lookback_bars_excluding_current_bar"
        )
        return _configuration(
            component_name=self.name,
            parameters=self._parameters.to_primitive(),
            warm_up_observations=self.warm_up_observations,
            output_fields=self.output_fields,
            formula={
                "numerator": "current_bar_volume",
                "denominator": denominator,
                "missing_value_policy": (
                    "emit_none_for_warm_up_nonfinite_numerator_or_any_nonfinite_"
                    "denominator_observation"
                ),
                "zero_denominator_policy": "emit_none",
            },
        )

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    @classmethod
    def from_configuration(cls, configuration: PrimitiveMapping) -> "RelativeVolume":
        """Restore one exact serialized relative-volume configuration."""
        parameters = _serialized_parameters(configuration)
        denominator_policy = parameters.get("denominator_policy")
        try:
            normalized_policy = RelativeVolumeDenominatorPolicy(denominator_policy)
        except (TypeError, ValueError) as error:
            raise InvalidIndicatorParametersError(
                "serialized relative-volume denominator policy is invalid"
            ) from error
        indicator = cls(
            RelativeVolumeParameters(
                _serialized_lookback(parameters),
                _serialized_feed_scope(parameters),
                normalized_policy,
            )
        )
        _require_exact_configuration(configuration, indicator.configuration(), cls.name)
        return indicator

    def calculate(self, dataset: MarketDataset) -> IndicatorOutput:
        """Return daily-compatible output aligned with every canonical session."""
        validate_market_input(dataset, self.required_fields)
        fields = self.calculate_bar_fields(cast(tuple[IndicatorBar, ...], dataset.bars))
        output = IndicatorOutput(
            self.name,
            self.configuration_id,
            tuple(bar.session_date for bar in dataset.bars),
            fields,
        )
        validate_indicator_alignment(dataset, output)
        return output

    def calculate_bar_fields(
        self, bars: tuple[IndicatorBar, ...]
    ) -> tuple[IndicatorFieldOutput, ...]:
        """Calculate on any already-validated canonical timeframe."""
        validate_indicator_bars(bars, self.required_fields, require_finite=False)
        volumes = tuple(bar.volume for bar in bars)
        averages = _trailing_volume_average(
            volumes,
            self._parameters.lookback,
            exclude_current=not self._parameters.includes_current_bar,
        )
        values: list[IndicatorValue] = []
        try:
            with localcontext(_arithmetic_context()):
                for volume, average in zip(volumes, averages, strict=True):
                    if (
                        not volume.is_finite()
                        or average is None
                        or average == Decimal(0)
                    ):
                        values.append(None)
                    else:
                        values.append(volume / average)
        except DecimalException as error:
            raise IndicatorCalculationError(
                "relative volume arithmetic failed under its configured decimal policy"
            ) from error
        return (IndicatorFieldOutput(RELATIVE_VOLUME_OUTPUT, tuple(values)),)


def _trailing_volume_average(
    volumes: tuple[Decimal, ...],
    lookback: int,
    *,
    exclude_current: bool,
) -> tuple[IndicatorValue, ...]:
    normalized = tuple(value if value.is_finite() else None for value in volumes)
    divisor = Decimal(lookback)
    values: list[IndicatorValue] = []
    try:
        with localcontext(_arithmetic_context()):
            for index in range(len(normalized)):
                window_end = index if exclude_current else index + 1
                window_start = window_end - lookback
                if window_start < 0:
                    values.append(None)
                    continue
                observations = normalized[window_start:window_end]
                if any(observation is None for observation in observations):
                    values.append(None)
                    continue
                total = sum(
                    (value for value in observations if value is not None), Decimal(0)
                )
                values.append(total / divisor)
    except DecimalException as error:
        raise IndicatorCalculationError(
            "volume moving average arithmetic failed under its configured decimal "
            "policy"
        ) from error
    return tuple(values)


def _configuration(
    *,
    component_name: str,
    parameters: PrimitiveMapping,
    warm_up_observations: int,
    output_fields: tuple[str, ...],
    formula: PrimitiveMapping,
) -> PrimitiveMapping:
    return {
        "component_type": "indicator",
        "component_name": component_name,
        "contract_version": "1",
        "implementation_version": "1",
        "parameters": parameters,
        "required_fields": [MarketField.VOLUME.value],
        "warm_up_observations": warm_up_observations,
        "output_fields": list(output_fields),
        "missing_value": None,
        "formula": formula,
        "arithmetic": {
            "decimal_precision": _DECIMAL_PRECISION,
            "rounding": ROUND_HALF_EVEN,
            "decimal_emin": _DECIMAL_EMIN,
            "decimal_emax": _DECIMAL_EMAX,
            "capitals": 1,
            "clamp": 0,
            "initial_flags": [],
            "traps": [signal.__name__ for signal in _DECIMAL_TRAPS],
        },
    }


def _arithmetic_context() -> Context:
    context = Context(
        prec=_DECIMAL_PRECISION,
        rounding=ROUND_HALF_EVEN,
        Emin=_DECIMAL_EMIN,
        Emax=_DECIMAL_EMAX,
        capitals=1,
        clamp=0,
    )
    context.clear_flags()
    for signal in _DECIMAL_TRAPS:
        context.traps[signal] = True
    return context


def _validate_lookback(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidIndicatorParametersError("lookback must be an integer")
    if value < 1:
        raise InvalidIndicatorParametersError("lookback must be greater than zero")


def _validate_feed_scope(value: object) -> None:
    if not isinstance(value, FeedScope):
        raise InvalidIndicatorParametersError(
            "feed_scope must be a provider-neutral feed scope"
        )


def _serialized_parameters(configuration: PrimitiveMapping) -> PrimitiveMapping:
    parameters = configuration.get("parameters")
    if not isinstance(parameters, dict):
        raise InvalidIndicatorParametersError(
            "serialized volume indicator parameters must be an object"
        )
    return parameters


def _serialized_lookback(parameters: PrimitiveMapping) -> int:
    lookback = parameters.get("lookback")
    if isinstance(lookback, bool) or not isinstance(lookback, int):
        raise InvalidIndicatorParametersError(
            "serialized volume indicator lookback must be an integer"
        )
    return lookback


def _serialized_feed_scope(parameters: PrimitiveMapping) -> FeedScope:
    feed_scope = parameters.get("feed_scope")
    if not isinstance(feed_scope, dict):
        raise InvalidIndicatorParametersError(
            "serialized volume indicator feed scope must be an object"
        )
    try:
        coverage = FeedCoverage(feed_scope.get("coverage"))
        market_center = feed_scope.get("market_center")
        provider_scope = feed_scope.get("provider_scope")
        if market_center is not None and not isinstance(market_center, str):
            raise TypeError
        if provider_scope is not None and not isinstance(provider_scope, str):
            raise TypeError
        return FeedScope(coverage, market_center, provider_scope)
    except (DatasetFamilyValidationError, TypeError, ValueError) as error:
        raise InvalidIndicatorParametersError(
            "serialized volume indicator feed scope is invalid"
        ) from error


def _require_exact_configuration(
    serialized: PrimitiveMapping,
    restored: PrimitiveMapping,
    indicator_name: str,
) -> None:
    if serialized != restored:
        raise InvalidIndicatorParametersError(
            f"serialized {indicator_name} configuration is unsupported or incomplete"
        )


__all__ = [
    "RELATIVE_VOLUME_OUTPUT",
    "VOLUME_MOVING_AVERAGE_OUTPUT",
    "RelativeVolume",
    "RelativeVolumeDenominatorPolicy",
    "RelativeVolumeParameters",
    "VolumeMovingAverage",
    "VolumeMovingAverageParameters",
]
