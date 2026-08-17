"""Backend-neutral moving average convergence/divergence indicator."""

from dataclasses import dataclass
from typing import cast

from quantforge.configuration import (
    Primitive,
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.data.models import MarketDataset
from quantforge.indicators.backends import (
    TALIB_INDICATOR_BACKEND,
    IndicatorBackend,
    IndicatorBackendIdentity,
    IndicatorBackendRegistry,
    IndicatorComputationRequest,
    StandardIndicatorDefinition,
    default_indicator_backend_registry,
)
from quantforge.indicators.base import (
    DevelopingBarSupport,
    IndicatorBar,
    IndicatorParameters,
    validate_indicator_alignment,
    validate_indicator_bars,
    validate_market_input,
)
from quantforge.indicators.exceptions import (
    IndicatorBackendVersionError,
    InvalidIndicatorBackendError,
    InvalidIndicatorParametersError,
)
from quantforge.indicators.models import (
    IndicatorFieldOutput,
    IndicatorOutput,
    MarketField,
)

MACD_OUTPUT = "macd"
MACD_SIGNAL_OUTPUT = "signal"
MACD_HISTOGRAM_OUTPUT = "histogram"


@dataclass(frozen=True, slots=True)
class MovingAverageConvergenceDivergenceParameters:
    """Positive MACD periods and one canonical source field."""

    fast_period: int = 12
    slow_period: int = 26
    signal_period: int = 9
    source_field: MarketField = MarketField.CLOSE

    def __post_init__(self) -> None:
        _validate_period("fast_period", self.fast_period)
        _validate_period("slow_period", self.slow_period)
        _validate_period("signal_period", self.signal_period)
        if self.fast_period >= self.slow_period:
            raise InvalidIndicatorParametersError(
                "fast_period must be less than slow_period"
            )
        _validate_source_field(cast(object, self.source_field))

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "fast_period": self.fast_period,
            "signal_period": self.signal_period,
            "slow_period": self.slow_period,
            "source_field": self.source_field.value,
        }


class MovingAverageConvergenceDivergence:
    """Aligned MACD outputs delegated to a configured standard backend.

    QuantForge owns the normalized definition, output names, source binding,
    and deterministic identity. The selected backend owns the MACD calculation.
    """

    name = "moving_average_convergence_divergence"
    implementation_version = "1"
    output_fields = (MACD_OUTPUT, MACD_SIGNAL_OUTPUT, MACD_HISTOGRAM_OUTPUT)
    missing_value = None
    developing_bar_support = DevelopingBarSupport.DEVELOPING_AS_OF

    def __init__(
        self,
        parameters: MovingAverageConvergenceDivergenceParameters | None = None,
        *,
        backend_id: str = TALIB_INDICATOR_BACKEND,
        backend_registry: IndicatorBackendRegistry | None = None,
    ) -> None:
        self._parameters = (
            MovingAverageConvergenceDivergenceParameters()
            if parameters is None
            else parameters
        )
        selected_backend_id = _validate_backend_id(backend_id)
        self._definition = StandardIndicatorDefinition(
            name=self.name,
            parameters=PrimitiveMappingSnapshot.capture(
                self._parameters.to_primitive()
            ),
            input_fields=(self._parameters.source_field,),
            output_fields=self.output_fields,
        )
        registry = backend_registry or default_indicator_backend_registry()
        self._backend: IndicatorBackend = registry.resolve(selected_backend_id)
        self._backend_identity = self._backend.identity_for(self._definition)
        if self._backend_identity.backend_id != selected_backend_id:
            raise InvalidIndicatorBackendError(
                "indicator backend identity does not match the selected registry id: "
                f"{selected_backend_id}"
            )

    @property
    def parameters(self) -> IndicatorParameters:
        return self._parameters

    @property
    def required_fields(self) -> frozenset[MarketField]:
        return frozenset((self._parameters.source_field,))

    @property
    def warm_up_observations(self) -> int:
        """Bars required for TA-Lib's slow EMA plus signal-line lookback."""
        return self._parameters.slow_period + self._parameters.signal_period - 1

    @property
    def standard_definition(self) -> StandardIndicatorDefinition:
        """Return the canonical MACD definition shared by every backend."""
        return self._definition

    @property
    def backend_identity(self) -> IndicatorBackendIdentity:
        """Return the resolved backend and exact library identity."""
        return self._backend_identity

    def configuration(self) -> PrimitiveMapping:
        """Return stable normalized parameters and exact backend provenance."""
        normalized_input_fields: list[Primitive] = [
            field.value for field in self._definition.input_fields
        ]
        normalized_parameter_names: list[Primitive] = []
        normalized_parameter_names.extend(sorted(self._parameters.to_primitive()))
        return {
            "component_type": "indicator",
            "component_name": self.name,
            "contract_version": "1",
            "definition_version": self.implementation_version,
            "parameters": self._parameters.to_primitive(),
            "required_fields": [field.value for field in sorted(self.required_fields)],
            "warm_up_observations": self.warm_up_observations,
            "output_fields": list(self.output_fields),
            "missing_value": None,
            "backend": self._backend_identity.to_primitive(),
            "normalization": {
                "input_fields": normalized_input_fields,
                "parameter_names": normalized_parameter_names,
                "output_fields": list(self.output_fields),
                "unavailable_output": None,
            },
        }

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    @classmethod
    def from_configuration(
        cls,
        configuration: PrimitiveMapping,
        *,
        backend_registry: IndicatorBackendRegistry | None = None,
    ) -> "MovingAverageConvergenceDivergence":
        """Restore serialized normalized parameters and exact backend semantics."""
        parameters_value = configuration.get("parameters")
        if not isinstance(parameters_value, dict):
            raise InvalidIndicatorBackendError(
                "serialized MACD parameters must be an object"
            )
        fast_period_value = parameters_value.get("fast_period")
        slow_period_value = parameters_value.get("slow_period")
        signal_period_value = parameters_value.get("signal_period")
        source_field = parameters_value.get("source_field")
        if any(
            isinstance(period, bool) or not isinstance(period, int)
            for period in (fast_period_value, slow_period_value, signal_period_value)
        ):
            raise InvalidIndicatorBackendError(
                "serialized MACD periods must be integers"
            )
        fast_period = cast(int, fast_period_value)
        slow_period = cast(int, slow_period_value)
        signal_period = cast(int, signal_period_value)
        if not isinstance(source_field, str):
            raise InvalidIndicatorBackendError(
                "serialized MACD source field is invalid"
            )
        try:
            normalized_source_field = MarketField(source_field)
        except ValueError as error:
            raise InvalidIndicatorBackendError(
                "serialized MACD source field is invalid"
            ) from error
        backend_value = configuration.get("backend")
        if not isinstance(backend_value, dict):
            raise InvalidIndicatorBackendError(
                "serialized MACD backend must be an object"
            )
        backend_id = backend_value.get("backend_id")
        if not isinstance(backend_id, str):
            raise InvalidIndicatorBackendError("serialized MACD backend id is invalid")
        indicator = cls(
            MovingAverageConvergenceDivergenceParameters(
                fast_period,
                slow_period,
                signal_period,
                normalized_source_field,
            ),
            backend_id=backend_id,
            backend_registry=backend_registry,
        )
        if indicator.configuration() != configuration:
            raise IndicatorBackendVersionError(
                "serialized MACD configuration does not match the installed backend"
            )
        return indicator

    def calculate(self, dataset: MarketDataset) -> IndicatorOutput:
        """Return causally aligned MACD fields for a canonical daily dataset."""
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
        """Calculate against canonical bars whose timeframe is validated upstream."""
        validate_indicator_bars(bars, self.required_fields, require_finite=False)
        result = self._backend.compute(
            IndicatorComputationRequest(self._definition, bars)
        )
        if result.observation_count != len(bars):
            raise InvalidIndicatorBackendError(
                "indicator backend result observation count does not match the "
                "canonical input bars"
            )
        if (
            result.backend_identity != self._backend_identity
            or result.definition_name != self.name
            or result.normalized_parameters != self._definition.parameters
            or result.normalized_input_fields != self._definition.input_fields
            or tuple(field.name for field in result.fields) != self.output_fields
        ):
            raise InvalidIndicatorBackendError(
                "indicator backend result metadata changed during calculation"
            )
        return result.fields


def _validate_period(parameter_name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidIndicatorParametersError(f"{parameter_name} must be an integer")
    if value < 1:
        raise InvalidIndicatorParametersError(
            f"{parameter_name} must be greater than zero"
        )


def _validate_backend_id(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise InvalidIndicatorBackendError(
            "indicator backend id must be a non-empty string"
        )
    return value


def _validate_source_field(value: object) -> None:
    if not isinstance(value, MarketField):
        raise InvalidIndicatorParametersError(
            "source_field must be a normalized market field"
        )


__all__ = [
    "MACD_HISTOGRAM_OUTPUT",
    "MACD_OUTPUT",
    "MACD_SIGNAL_OUTPUT",
    "MovingAverageConvergenceDivergence",
    "MovingAverageConvergenceDivergenceParameters",
]
