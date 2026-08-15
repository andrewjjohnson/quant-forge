"""Full-window, population-standard-deviation Bollinger Bands."""

from dataclasses import dataclass
from decimal import (
    ROUND_HALF_EVEN,
    Decimal,
    DecimalException,
    DivisionByZero,
    InvalidOperation,
    Overflow,
)
from typing import cast

from quantforge.configuration import (
    Primitive,
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
    decimal_to_primitive,
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

BOLLINGER_MIDDLE_BAND_OUTPUT = "bollinger_middle_band"
BOLLINGER_UPPER_BAND_OUTPUT = "bollinger_upper_band"
BOLLINGER_LOWER_BAND_OUTPUT = "bollinger_lower_band"
BOLLINGER_BANDWIDTH_OUTPUT = "bollinger_bandwidth"
_DECIMAL_PRECISION = 34
_DECIMAL_ROUNDING = ROUND_HALF_EVEN
_DECIMAL_EMIN = -999_999
_DECIMAL_EMAX = 999_999
_DECIMAL_CAPITALS = 1
_DECIMAL_CLAMP = 0
_EXACT_MOMENT_MAX_COEFFICIENT_DIGITS = _DECIMAL_PRECISION * 2
_EXACT_MOMENT_MAX_SOURCE_INTEGER_DIGITS = 2_048
_MULTIPLIER_MAX_COEFFICIENT_DIGITS = _DECIMAL_PRECISION * 2
_MULTIPLIER_MAX_FIXED_POINT_CHARACTERS = 256
_DECIMAL_TRAPS: tuple[type[DecimalException], ...] = (
    DivisionByZero,
    InvalidOperation,
    Overflow,
)
_DECIMAL_TRAP_NAMES = tuple(signal.__name__ for signal in _DECIMAL_TRAPS)


@dataclass(frozen=True, slots=True)
class BollingerBandsParameters:
    """Bar period, population-deviation multiplier, and canonical source field."""

    period: int
    standard_deviation_multiplier: Decimal = Decimal(2)
    source_field: MarketField = MarketField.CLOSE

    def __post_init__(self) -> None:
        _validate_period_type(self.period)
        if self.period < 1:
            raise InvalidIndicatorParametersError("period must be at least 1")
        _validate_multiplier(self.standard_deviation_multiplier)
        _validate_source_field(self.source_field)

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "period": self.period,
            "source_field": self.source_field.value,
            "standard_deviation_multiplier": decimal_to_primitive(
                self.standard_deviation_multiplier
            ),
        }


class BollingerBands:
    """Trailing mean and population-deviation bands over a complete bar window.

    A full window containing a non-finite source emits ``None`` for every field.
    Bandwidth is ``(upper - lower) / middle``. A zero-width band has bandwidth
    zero, including when the middle is zero; a nonzero width with a zero middle
    has unavailable bandwidth rather than an invented ratio.
    """

    name = "bollinger_bands"
    implementation_version = "8"
    output_fields = (
        BOLLINGER_MIDDLE_BAND_OUTPUT,
        BOLLINGER_UPPER_BAND_OUTPUT,
        BOLLINGER_LOWER_BAND_OUTPUT,
        BOLLINGER_BANDWIDTH_OUTPUT,
    )
    missing_value = None
    developing_bar_support = DevelopingBarSupport.DEVELOPING_AS_OF

    def __init__(
        self,
        parameters: BollingerBandsParameters,
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
        """Consecutive observations required for the first complete window."""
        return self._parameters.period

    @property
    def standard_definition(self) -> StandardIndicatorDefinition:
        """Return the canonical Bollinger definition shared by every backend."""
        return self._definition

    @property
    def backend_identity(self) -> IndicatorBackendIdentity:
        """Return the resolved backend and exact library identity."""
        return self._backend_identity

    @property
    def uses_legacy_native_configuration(self) -> bool:
        """Whether serialization intentionally preserves the pre-QF-37 identity."""
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
            "parameter_bounds": {
                "standard_deviation_multiplier": {
                    "maximum_coefficient_digits": (_MULTIPLIER_MAX_COEFFICIENT_DIGITS),
                    "maximum_fixed_point_characters": (
                        _MULTIPLIER_MAX_FIXED_POINT_CHARACTERS
                    ),
                },
            },
            "formula": {
                "middle_band": "sum(window) / period",
                "standard_deviation": (
                    "sqrt(sum((observation - middle_band) ** 2) / period)"
                ),
                "standard_deviation_degrees_of_freedom": 0,
                "window_update": "exact_rational_rolling_sum_and_sum_of_squares",
                "upper_band": (
                    "middle_band + standard_deviation_multiplier * standard_deviation"
                ),
                "lower_band": (
                    "middle_band - standard_deviation_multiplier * standard_deviation"
                ),
                "bandwidth": "(upper_band - lower_band) / middle_band",
                "zero_width_policy": "bandwidth_zero",
                "zero_middle_nonzero_width_policy": "bandwidth_none",
                "missing_value_policy": "full_window_containing_missing_is_none",
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
                "rolling_moment_accumulation": "exact_rational",
                "exact_moment_input_bounds": {
                    "maximum_coefficient_digits": (
                        _EXACT_MOMENT_MAX_COEFFICIENT_DIGITS
                    ),
                    "maximum_source_integer_digits": (
                        _EXACT_MOMENT_MAX_SOURCE_INTEGER_DIGITS
                    ),
                    "maximum_squared_integer_digits": (
                        _EXACT_MOMENT_MAX_SOURCE_INTEGER_DIGITS * 2
                    ),
                    "minimum_stored_exponent": _DECIMAL_EMIN,
                    "maximum_stored_exponent": _DECIMAL_EMAX,
                    "minimum_adjusted_exponent": _DECIMAL_EMIN,
                    "maximum_adjusted_exponent": _DECIMAL_EMAX,
                },
            },
        }
        if self._legacy_native_configuration:
            return legacy_configuration
        normalized_input_fields: list[Primitive] = [
            field.value for field in self._definition.input_fields
        ]
        normalized_parameter_names: list[Primitive] = []
        normalized_parameter_names.extend(sorted(self._parameters.to_primitive()))
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
    ) -> "BollingerBands":
        """Restore legacy/native or explicit backend semantics without drift."""
        parameters_value = configuration.get("parameters")
        if not isinstance(parameters_value, dict):
            raise InvalidIndicatorBackendError(
                "serialized Bollinger Bands parameters must be an object"
            )
        period = parameters_value.get("period")
        source_field = parameters_value.get("source_field")
        multiplier = parameters_value.get("standard_deviation_multiplier")
        if isinstance(period, bool) or not isinstance(period, int):
            raise InvalidIndicatorBackendError(
                "serialized Bollinger Bands period must be an integer"
            )
        if not isinstance(source_field, str) or not isinstance(multiplier, str):
            raise InvalidIndicatorBackendError(
                "serialized Bollinger Bands field or multiplier is invalid"
            )
        try:
            normalized_source_field = MarketField(source_field)
            normalized_multiplier = Decimal(multiplier)
        except (InvalidOperation, ValueError) as error:
            raise InvalidIndicatorBackendError(
                "serialized Bollinger Bands parameters are invalid"
            ) from error
        backend_value = configuration.get("backend")
        if backend_value is None:
            backend_id = None
        elif isinstance(backend_value, dict):
            backend_id = backend_value.get("backend_id")
            if not isinstance(backend_id, str):
                raise InvalidIndicatorBackendError(
                    "serialized Bollinger Bands backend id is invalid"
                )
        else:
            raise InvalidIndicatorBackendError(
                "serialized Bollinger Bands backend must be an object"
            )
        indicator = cls(
            BollingerBandsParameters(
                period,
                normalized_multiplier,
                normalized_source_field,
            ),
            backend_id=backend_id,
            backend_registry=backend_registry,
        )
        if indicator.configuration() != configuration:
            raise IndicatorBackendVersionError(
                "serialized Bollinger Bands configuration does not match the "
                "installed backend"
            )
        return indicator

    def calculate(self, dataset: MarketDataset) -> IndicatorOutput:
        """Return causally aligned bands without partial or filled windows."""
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


def _validate_period_type(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidIndicatorParametersError("period must be an integer")


def _validate_multiplier(value: object) -> None:
    if not isinstance(value, Decimal):
        raise InvalidIndicatorParametersError(
            "standard_deviation_multiplier must be a Decimal"
        )
    if not value.is_finite() or value <= Decimal(0):
        raise InvalidIndicatorParametersError(
            "standard_deviation_multiplier must be finite and greater than zero"
        )
    _, coefficient_digits, stored_exponent = value.as_tuple()
    if not isinstance(stored_exponent, int):
        raise InvalidIndicatorParametersError(
            "standard_deviation_multiplier must be a finite Decimal"
        )
    if (
        len(coefficient_digits) > _MULTIPLIER_MAX_COEFFICIENT_DIGITS
        or _fixed_point_render_length(coefficient_digits, stored_exponent)
        > _MULTIPLIER_MAX_FIXED_POINT_CHARACTERS
    ):
        raise InvalidIndicatorParametersError(
            "standard_deviation_multiplier exceeds serialization resource bounds"
        )


def _fixed_point_render_length(
    coefficient_digits: tuple[int, ...], stored_exponent: int
) -> int:
    """Return the allocation length of ``format(value, 'f')`` without formatting."""
    coefficient_length = len(coefficient_digits)
    if stored_exponent >= 0:
        return coefficient_length + stored_exponent
    decimal_point_position = coefficient_length + stored_exponent
    if decimal_point_position > 0:
        return coefficient_length + 1
    return 2 - decimal_point_position + coefficient_length


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
    "BOLLINGER_BANDWIDTH_OUTPUT",
    "BOLLINGER_LOWER_BAND_OUTPUT",
    "BOLLINGER_MIDDLE_BAND_OUTPUT",
    "BOLLINGER_UPPER_BAND_OUTPUT",
    "BollingerBands",
    "BollingerBandsParameters",
]
