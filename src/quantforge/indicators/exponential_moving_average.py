"""SMA-seeded, causal exponential moving average."""

from dataclasses import dataclass
from decimal import (
    ROUND_HALF_EVEN,
    DecimalException,
    DivisionByZero,
    InvalidOperation,
    Overflow,
)

from quantforge.configuration import (
    Primitive,
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.data.models import MarketDataset
from quantforge.indicators.backends import (
    NATIVE_INDICATOR_BACKEND,
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

EXPONENTIAL_MOVING_AVERAGE_OUTPUT = "exponential_moving_average"
_DECIMAL_PRECISION = 34
_DECIMAL_ROUNDING = ROUND_HALF_EVEN
_DECIMAL_EMIN = -999_999
_DECIMAL_EMAX = 999_999
_DECIMAL_CAPITALS = 1
_DECIMAL_CLAMP = 0
_DECIMAL_TRAPS: tuple[type[DecimalException], ...] = (
    DivisionByZero,
    InvalidOperation,
    Overflow,
)
_DECIMAL_TRAP_NAMES = tuple(signal.__name__ for signal in _DECIMAL_TRAPS)


@dataclass(frozen=True, slots=True)
class ExponentialMovingAverageParameters:
    """Bar period and canonical OHLCV source for an EMA."""

    period: int
    source_field: MarketField = MarketField.CLOSE

    def __post_init__(self) -> None:
        _validate_period_type(self.period)
        if self.period < 1:
            raise InvalidIndicatorParametersError("period must be at least 1")
        _validate_source_field(self.source_field)

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "period": self.period,
            "source_field": self.source_field.value,
        }


class ExponentialMovingAverage:
    """Backend-neutral EMA initialized by a full-period simple average.

    The smoothing factor is ``2 / (period + 1)``. A missing source value emits
    ``None``. The omitted-backend compatibility path resolves explicitly to
    historical ``native_v1`` semantics and retains its serialized identity.
    """

    name = "exponential_moving_average"
    implementation_version = "1"
    output_fields = (EXPONENTIAL_MOVING_AVERAGE_OUTPUT,)
    missing_value = None
    developing_bar_support = DevelopingBarSupport.DEVELOPING_AS_OF

    def __init__(
        self,
        parameters: ExponentialMovingAverageParameters,
        *,
        backend_id: str | None = None,
        backend_registry: IndicatorBackendRegistry | None = None,
    ) -> None:
        self._parameters = parameters
        self._legacy_native_configuration = backend_id is None
        selected_backend_id = _validate_backend_id(
            NATIVE_INDICATOR_BACKEND if backend_id is None else backend_id
        )
        self._definition = StandardIndicatorDefinition(
            name=self.name,
            parameters=PrimitiveMappingSnapshot.capture(parameters.to_primitive()),
            input_fields=(parameters.source_field,),
            output_fields=self.output_fields,
        )
        registry = backend_registry or default_indicator_backend_registry()
        self._backend: IndicatorBackend = registry.resolve(selected_backend_id)
        self._backend_identity = self._backend.identity_for(self._definition)

    @property
    def parameters(self) -> IndicatorParameters:
        return self._parameters

    @property
    def required_fields(self) -> frozenset[MarketField]:
        return frozenset((self._parameters.source_field,))

    @property
    def warm_up_observations(self) -> int:
        """Consecutive observations required to seed or reseed the EMA."""
        return self._parameters.period

    @property
    def standard_definition(self) -> StandardIndicatorDefinition:
        """Return the canonical EMA definition shared by every backend."""
        return self._definition

    @property
    def backend_identity(self) -> IndicatorBackendIdentity:
        """Return the resolved backend and exact library identity."""
        return self._backend_identity

    @property
    def uses_legacy_native_configuration(self) -> bool:
        """Whether serialization intentionally preserves pre-QF-35 identity."""
        return self._legacy_native_configuration

    def configuration(self) -> PrimitiveMapping:
        legacy_configuration: PrimitiveMapping = {
            "component_type": "indicator",
            "component_name": self.name,
            "contract_version": "1",
            "implementation_version": self.implementation_version,
            "parameters": self._parameters.to_primitive(),
            "required_fields": [field.value for field in sorted(self.required_fields)],
            "warm_up_observations": self.warm_up_observations,
            "output_fields": list(self.output_fields),
            "missing_value": None,
            "formula": {
                "smoothing_factor": "2 / (period + 1)",
                "initialization": (
                    "simple_average_of_first_period_consecutive_observations"
                ),
                "recurrence": (
                    "((period - 1) * previous + 2 * current) / (period + 1)"
                ),
                "missing_value_policy": "emit_none_and_restart_initialization",
            },
            "arithmetic": {
                "decimal_precision": _DECIMAL_PRECISION,
                "rounding": _DECIMAL_ROUNDING,
                "decimal_emin": _DECIMAL_EMIN,
                "decimal_emax": _DECIMAL_EMAX,
                "capitals": _DECIMAL_CAPITALS,
                "clamp": _DECIMAL_CLAMP,
                "initial_flags": [],
                "traps": list(_DECIMAL_TRAP_NAMES),
            },
        }
        if self._legacy_native_configuration:
            return legacy_configuration
        normalized_input_fields: list[Primitive] = [
            field.value for field in self._definition.input_fields
        ]
        normalized_parameter_names: list[Primitive] = []
        normalized_parameter_names.extend(sorted(self._parameters.to_primitive()))
        normalized_output_fields: list[Primitive] = list(self.output_fields)
        normalization: PrimitiveMapping = {
            "input_fields": normalized_input_fields,
            "parameter_names": normalized_parameter_names,
            "output_fields": normalized_output_fields,
            "unavailable_output": None,
        }
        return {
            "component_type": "indicator",
            "component_name": self.name,
            "contract_version": "2",
            "definition_version": self.implementation_version,
            "parameters": self._parameters.to_primitive(),
            "required_fields": [field.value for field in sorted(self.required_fields)],
            "warm_up_observations": self.warm_up_observations,
            "output_fields": list(self.output_fields),
            "missing_value": None,
            "backend": self._backend_identity.to_primitive(),
            "normalization": normalization,
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
    ) -> "ExponentialMovingAverage":
        """Restore legacy/native or explicit backend semantics without drift."""
        parameters_value = configuration.get("parameters")
        if not isinstance(parameters_value, dict):
            raise InvalidIndicatorBackendError(
                "serialized EMA parameters must be an object"
            )
        period = parameters_value.get("period")
        source_field = parameters_value.get("source_field")
        if isinstance(period, bool) or not isinstance(period, int):
            raise InvalidIndicatorBackendError(
                "serialized EMA period must be an integer"
            )
        try:
            normalized_source_field = MarketField(source_field)
        except (TypeError, ValueError) as error:
            raise InvalidIndicatorBackendError(
                "serialized EMA source_field is invalid"
            ) from error
        backend_value = configuration.get("backend")
        if backend_value is None:
            indicator = cls(
                ExponentialMovingAverageParameters(period, normalized_source_field),
                backend_registry=backend_registry,
            )
        elif isinstance(backend_value, dict):
            backend_id = backend_value.get("backend_id")
            if not isinstance(backend_id, str):
                raise InvalidIndicatorBackendError(
                    "serialized EMA backend id is invalid"
                )
            indicator = cls(
                ExponentialMovingAverageParameters(period, normalized_source_field),
                backend_id=backend_id,
                backend_registry=backend_registry,
            )
        else:
            raise InvalidIndicatorBackendError(
                "serialized EMA backend must be an object"
            )
        if indicator.configuration() != configuration:
            raise IndicatorBackendVersionError(
                "serialized EMA configuration does not match the installed backend"
            )
        return indicator

    def calculate(self, dataset: MarketDataset) -> IndicatorOutput:
        """Return one causally aligned EMA value or ``None`` per daily bar."""
        validate_market_input(dataset, self.required_fields)
        fields = self.calculate_bar_fields(dataset.bars)
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
        """Calculate through the selected backend over canonical timeframe bars."""
        validate_indicator_bars(bars, self.required_fields, require_finite=False)
        result = self._backend.compute(
            IndicatorComputationRequest(self._definition, bars)
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


def _validate_period_type(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidIndicatorParametersError("period must be an integer")


def _validate_source_field(value: object) -> None:
    if not isinstance(value, MarketField):
        raise InvalidIndicatorParametersError(
            "source_field must be a normalized market field"
        )


def _validate_backend_id(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise InvalidIndicatorBackendError(
            "indicator backend id must be a non-empty string"
        )
    return value


__all__ = [
    "EXPONENTIAL_MOVING_AVERAGE_OUTPUT",
    "ExponentialMovingAverage",
    "ExponentialMovingAverageParameters",
]
