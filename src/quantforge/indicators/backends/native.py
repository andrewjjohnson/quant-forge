"""Adapter for historical QuantForge-native standard indicator mathematics."""

from collections.abc import Callable
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

from quantforge.configuration import PrimitiveMapping
from quantforge.indicators.backends.base import (
    NATIVE_INDICATOR_BACKEND,
    IndicatorBackendIdentity,
    IndicatorComputationRequest,
    IndicatorComputationResult,
    StandardIndicatorDefinition,
)
from quantforge.indicators.exceptions import (
    IndicatorCalculationError,
    MissingMarketFieldError,
    UnsupportedIndicatorBackendError,
)
from quantforge.indicators.models import IndicatorFieldOutput, IndicatorValue

_EMA_NAME = "exponential_moving_average"
_EMA_OUTPUT = "exponential_moving_average"
_DECIMAL_PRECISION = 34
_DECIMAL_EMIN = -999_999
_DECIMAL_EMAX = 999_999
_DECIMAL_TRAPS: tuple[type[DecimalException], ...] = (
    DivisionByZero,
    InvalidOperation,
    Overflow,
)
type _NativeFunction = Callable[
    [tuple[tuple[Decimal, ...], ...], PrimitiveMapping],
    tuple[tuple[IndicatorValue, ...], ...],
]


@dataclass(frozen=True, slots=True)
class _NativeMapping:
    function_name: str
    implementation_version: str
    parameter_names: frozenset[str]
    function: _NativeFunction


_MAPPINGS = {
    _EMA_NAME: _NativeMapping(
        function_name="quantforge_native_ema",
        implementation_version="1",
        parameter_names=frozenset(("period", "source_field")),
        function=lambda inputs, parameters: (_native_ema(inputs[0], parameters),),
    )
}


class NativeIndicatorBackend:
    """Execute mapped standard indicators with versioned QuantForge math."""

    backend_id = NATIVE_INDICATOR_BACKEND

    def identity_for(
        self, definition: StandardIndicatorDefinition
    ) -> IndicatorBackendIdentity:
        mapping = _mapping_for(definition)
        return IndicatorBackendIdentity(
            backend_id=self.backend_id,
            library_name="quantforge_native_indicators",
            library_version=mapping.implementation_version,
            function_name=mapping.function_name,
        )

    def compute(
        self, request: IndicatorComputationRequest
    ) -> IndicatorComputationResult:
        """Translate canonical bars and parameters, then normalize native outputs."""
        definition = request.definition
        mapping = _mapping_for(definition)
        parameters = definition.parameters.to_primitive()
        if frozenset(parameters) != mapping.parameter_names:
            raise UnsupportedIndicatorBackendError(
                f"{self.backend_id} parameter mapping is unavailable for "
                f"indicator: {definition.name}"
            )
        inputs = tuple(
            _decimal_input(request, field.value) for field in definition.input_fields
        )
        raw_outputs = mapping.function(inputs, parameters)
        if len(raw_outputs) != len(definition.output_fields):
            raise IndicatorCalculationError(
                f"{self.backend_id} returned an unexpected output count for "
                f"indicator: {definition.name}"
            )
        fields = tuple(
            IndicatorFieldOutput(name, values)
            for name, values in zip(definition.output_fields, raw_outputs, strict=True)
        )
        return IndicatorComputationResult(
            definition_name=definition.name,
            backend_identity=self.identity_for(definition),
            normalized_parameters=definition.parameters,
            normalized_input_fields=definition.input_fields,
            fields=fields,
            observation_count=len(request.bars),
        )


def _mapping_for(definition: StandardIndicatorDefinition) -> _NativeMapping:
    try:
        mapping = _MAPPINGS[definition.name]
    except KeyError as error:
        raise UnsupportedIndicatorBackendError(
            f"{NATIVE_INDICATOR_BACKEND} does not support indicator: {definition.name}"
        ) from error
    if definition.output_fields != (_EMA_OUTPUT,):
        raise UnsupportedIndicatorBackendError(
            f"{NATIVE_INDICATOR_BACKEND} output mapping is unavailable for "
            f"indicator: {definition.name}"
        )
    return mapping


def _decimal_input(
    request: IndicatorComputationRequest, field_name: str
) -> tuple[Decimal, ...]:
    values: list[Decimal] = []
    for bar in request.bars:
        try:
            raw_value = getattr(bar, field_name)
        except AttributeError as error:
            raise MissingMarketFieldError(
                f"market data is missing required field: {field_name}"
            ) from error
        if not isinstance(raw_value, Decimal):
            raise MissingMarketFieldError(
                f"market field must be a compatible Decimal: {field_name}"
            )
        values.append(raw_value)
    return tuple(values)


def _native_ema(
    source_values: tuple[Decimal, ...], parameters: PrimitiveMapping
) -> tuple[IndicatorValue, ...]:
    period_value = parameters.get("period")
    if isinstance(period_value, bool) or not isinstance(period_value, int):
        raise IndicatorCalculationError("native EMA period must be an integer")
    period = period_value
    values: list[IndicatorValue] = []
    seed: list[Decimal] = []
    previous: Decimal | None = None

    try:
        with localcontext(_arithmetic_context()):
            two = Decimal(2)
            previous_weight = Decimal(period - 1)
            recurrence_divisor = Decimal(period + 1)
            seed_divisor = Decimal(period)
            for source in source_values:
                if not source.is_finite():
                    values.append(None)
                    seed.clear()
                    previous = None
                    continue
                if previous is None:
                    seed.append(source)
                    if len(seed) < period:
                        values.append(None)
                        continue
                    previous = sum(seed, Decimal(0)) / seed_divisor
                else:
                    previous = (
                        previous_weight * previous + two * source
                    ) / recurrence_divisor
                values.append(previous)
    except DecimalException as error:
        raise IndicatorCalculationError(
            "exponential moving average arithmetic failed under its configured "
            "decimal policy"
        ) from error
    return tuple(values)


def _arithmetic_context() -> Context:
    return Context(
        prec=_DECIMAL_PRECISION,
        rounding=ROUND_HALF_EVEN,
        Emin=_DECIMAL_EMIN,
        Emax=_DECIMAL_EMAX,
        capitals=1,
        clamp=0,
        flags=[],
        traps=list(_DECIMAL_TRAPS),
    )


__all__ = ["NativeIndicatorBackend"]
