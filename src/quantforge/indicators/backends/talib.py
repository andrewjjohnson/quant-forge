"""TA-Lib adapter for mapped backend-neutral standard indicators."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from math import isfinite, isnan
from typing import cast

import numpy as np
import numpy.typing as npt
import talib

from quantforge.configuration import PrimitiveMapping
from quantforge.indicators.backends.base import (
    TALIB_INDICATOR_BACKEND,
    IndicatorBackendIdentity,
    IndicatorComputationRequest,
    IndicatorComputationResult,
    StandardIndicatorDefinition,
)
from quantforge.indicators.exceptions import (
    IndicatorCalculationError,
    InvalidIndicatorBackendError,
    MissingMarketFieldError,
    UnsupportedIndicatorBackendError,
)
from quantforge.indicators.models import IndicatorFieldOutput, IndicatorValue

_EMA_NAME = "exponential_moving_average"
_EMA_MINIMUM_PERIOD = 1
_EMA_MAXIMUM_PERIOD = 100_000
type _TalibOutput = npt.NDArray[np.float64] | tuple[npt.NDArray[np.float64], ...]
type _TalibFunction = Callable[..., _TalibOutput]
type _ParameterValidator = Callable[[PrimitiveMapping], None]


@dataclass(frozen=True, slots=True)
class _TalibMapping:
    function_name: str
    function: _TalibFunction
    parameter_names: Mapping[str, str]
    input_parameter_names: frozenset[str]
    output_names: tuple[str, ...]
    validate_parameters: _ParameterValidator


_MAPPINGS = {
    _EMA_NAME: _TalibMapping(
        function_name="EMA",
        function=cast(_TalibFunction, talib.EMA),
        parameter_names={"period": "timeperiod"},
        input_parameter_names=frozenset(("source_field",)),
        output_names=("exponential_moving_average",),
        validate_parameters=lambda parameters: _validate_ema_parameters(parameters),
    )
}


class TalibIndicatorBackend:
    """Translate canonical definitions to TA-Lib and normalize its arrays."""

    backend_id = TALIB_INDICATOR_BACKEND

    def identity_for(
        self, definition: StandardIndicatorDefinition
    ) -> IndicatorBackendIdentity:
        mapping = _mapping_for(definition)
        return IndicatorBackendIdentity(
            backend_id=self.backend_id,
            library_name="TA-Lib",
            library_version=talib.__version__,
            function_name=mapping.function_name,
            runtime_library_name="TA-Lib C",
            runtime_library_version=_talib_runtime_library_version(),
        )

    def compute(
        self, request: IndicatorComputationRequest
    ) -> IndicatorComputationResult:
        """Translate inputs and parameters, invoke TA-Lib, and align output."""
        definition = request.definition
        mapping = _mapping_for(definition)
        _validate_global_state(mapping)
        parameters = definition.parameters.to_primitive()
        inputs = tuple(
            _float_input(request, field.value) for field in definition.input_fields
        )
        translated_parameters = {
            backend_name: parameters[normalized_name]
            for normalized_name, backend_name in mapping.parameter_names.items()
        }
        try:
            raw_output = mapping.function(*inputs, **translated_parameters)
        except (ArithmeticError, RuntimeError, TypeError, ValueError) as error:
            raise IndicatorCalculationError(
                f"{self.backend_id} {mapping.function_name} calculation failed"
            ) from error
        finally:
            _validate_global_state(mapping)
        raw_arrays = (raw_output,) if isinstance(raw_output, np.ndarray) else raw_output
        if len(raw_arrays) != len(mapping.output_names):
            raise IndicatorCalculationError(
                f"{self.backend_id} returned an unexpected output count for "
                f"indicator: {definition.name}"
            )
        fields = tuple(
            IndicatorFieldOutput(name, _normalize_array(values, len(request.bars)))
            for name, values in zip(mapping.output_names, raw_arrays, strict=True)
        )
        return IndicatorComputationResult(
            definition_name=definition.name,
            backend_identity=self.identity_for(definition),
            normalized_parameters=definition.parameters,
            normalized_input_fields=definition.input_fields,
            fields=fields,
            observation_count=len(request.bars),
        )


def _mapping_for(definition: StandardIndicatorDefinition) -> _TalibMapping:
    try:
        mapping = _MAPPINGS[definition.name]
    except KeyError as error:
        raise UnsupportedIndicatorBackendError(
            f"{TALIB_INDICATOR_BACKEND} does not support indicator: {definition.name}"
        ) from error
    if definition.output_fields != mapping.output_names:
        raise UnsupportedIndicatorBackendError(
            f"{TALIB_INDICATOR_BACKEND} output mapping is unavailable for "
            f"indicator: {definition.name}"
        )
    parameters = definition.parameters.to_primitive()
    _validate_parameter_mapping(definition, mapping, parameters)
    mapping.validate_parameters(parameters)
    return mapping


def _talib_runtime_library_version() -> str:
    raw_version = cast(
        object,
        talib.__ta_version__,  # pyright: ignore[reportUnknownMemberType]
    )
    try:
        decoded = (
            raw_version.decode("ascii")
            if isinstance(raw_version, bytes)
            else raw_version
        )
    except UnicodeDecodeError as error:
        raise InvalidIndicatorBackendError(
            "talib_v1 native TA-Lib version is not valid ASCII"
        ) from error
    if not isinstance(decoded, str) or not decoded.strip():
        raise InvalidIndicatorBackendError(
            "talib_v1 native TA-Lib version is unavailable"
        )
    return decoded.split(maxsplit=1)[0]


def _validate_global_state(mapping: _TalibMapping) -> None:
    get_compatibility = cast(
        Callable[[], int],
        talib.get_compatibility,  # pyright: ignore[reportUnknownMemberType]
    )
    get_unstable_period = cast(
        Callable[[str], int],
        talib.get_unstable_period,  # pyright: ignore[reportUnknownMemberType]
    )
    compatibility = get_compatibility()
    unstable_period = get_unstable_period(mapping.function_name)
    if compatibility != 0 or unstable_period != 0:
        raise InvalidIndicatorBackendError(
            "talib_v1 requires TA-Lib default compatibility and zero unstable period"
        )


def _validate_parameter_mapping(
    definition: StandardIndicatorDefinition,
    mapping: _TalibMapping,
    parameters: PrimitiveMapping,
) -> None:
    mapped = frozenset(mapping.parameter_names).union(mapping.input_parameter_names)
    if frozenset(parameters) != mapped:
        raise UnsupportedIndicatorBackendError(
            f"{TALIB_INDICATOR_BACKEND} parameter mapping is unavailable for "
            f"indicator: {definition.name}"
        )
    source_field = parameters.get("source_field")
    if (
        len(definition.input_fields) != 1
        or source_field != definition.input_fields[0].value
    ):
        raise UnsupportedIndicatorBackendError(
            f"{TALIB_INDICATOR_BACKEND} input mapping is unavailable for "
            f"indicator: {definition.name}"
        )


def _validate_ema_parameters(parameters: PrimitiveMapping) -> None:
    period = parameters.get("period")
    if (
        isinstance(period, bool)
        or not isinstance(period, int)
        or not _EMA_MINIMUM_PERIOD <= period <= _EMA_MAXIMUM_PERIOD
    ):
        raise UnsupportedIndicatorBackendError(
            f"{TALIB_INDICATOR_BACKEND} EMA period must be from "
            f"{_EMA_MINIMUM_PERIOD} through {_EMA_MAXIMUM_PERIOD}"
        )


def _float_input(
    request: IndicatorComputationRequest, field_name: str
) -> npt.NDArray[np.float64]:
    values: list[float] = []
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
        if not raw_value.is_finite():
            values.append(np.nan)
            continue
        try:
            converted = float(raw_value)
        except (InvalidOperation, OverflowError, ValueError) as error:
            raise IndicatorCalculationError(
                f"{TALIB_INDICATOR_BACKEND} cannot represent market field as "
                f"float64: {field_name}"
            ) from error
        if not isfinite(converted):
            raise IndicatorCalculationError(
                f"{TALIB_INDICATOR_BACKEND} cannot represent market field as "
                f"float64: {field_name}"
            )
        values.append(converted)
    return np.asarray(values, dtype=np.float64)


def _normalize_array(
    values: npt.NDArray[np.float64], expected_count: int
) -> tuple[IndicatorValue, ...]:
    if values.ndim != 1 or len(values) != expected_count:
        raise IndicatorCalculationError(
            f"{TALIB_INDICATOR_BACKEND} output does not align with canonical input"
        )
    normalized: list[IndicatorValue] = []
    for raw_value in values:
        value = float(raw_value)
        if isnan(value):
            normalized.append(None)
        elif not isfinite(value):
            raise IndicatorCalculationError(
                f"{TALIB_INDICATOR_BACKEND} returned a non-finite indicator value"
            )
        else:
            normalized.append(Decimal(str(value)))
    return tuple(normalized)


__all__ = ["TalibIndicatorBackend"]
