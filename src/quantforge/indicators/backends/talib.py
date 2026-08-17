"""TA-Lib adapter for mapped backend-neutral standard indicators."""

from collections.abc import Callable, Mapping
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

_SMA_NAME = "simple_moving_average"
_EMA_NAME = "exponential_moving_average"
_RSI_NAME = "wilder_relative_strength_index"
_ATR_NAME = "wilder_average_true_range"
_DIRECTIONAL_MOVEMENT_NAME = "wilder_directional_movement"
_BOLLINGER_NAME = "bollinger_bands"
_MACD_NAME = "moving_average_convergence_divergence"
_STOCHASTIC_NAME = "stochastic_oscillator"
_SIMPLE_MOVING_AVERAGE = "simple_moving_average"
_MAXIMUM_PERIOD = 100_000
_DECIMAL_PRECISION = 34
_DECIMAL_EMIN = -999_999
_DECIMAL_EMAX = 999_999
_DECIMAL_TRAPS: tuple[type[DecimalException], ...] = (
    DivisionByZero,
    InvalidOperation,
    Overflow,
)
type _TalibOutput = npt.NDArray[np.float64] | tuple[npt.NDArray[np.float64], ...]
type _TalibFunction = Callable[..., _TalibOutput]
type _ParameterValidator = Callable[[PrimitiveMapping], None]
type _ParameterNormalizer = Callable[[object], int | float]
type _DerivedOutput = Callable[
    [Mapping[str, npt.NDArray[np.float64]]], npt.NDArray[np.float64]
]
type _NormalizedDerivedOutput = Callable[
    [Mapping[str, tuple[IndicatorValue, ...]]], tuple[IndicatorValue, ...]
]


@dataclass(frozen=True, slots=True)
class _TalibMapping:
    function_name: str
    function: _TalibFunction
    parameter_mappings: tuple[tuple[str, tuple[str, ...], _ParameterNormalizer], ...]
    input_parameter_names: frozenset[str]
    input_fields: tuple[str, ...] | None
    source_field_parameter: str | None
    backend_output_names: tuple[str, ...]
    output_sources: tuple[tuple[str, str], ...]
    derived_outputs: tuple[tuple[str, _DerivedOutput], ...]
    normalized_derived_outputs: tuple[tuple[str, _NormalizedDerivedOutput], ...]
    unstable_function_names: tuple[str, ...]
    validate_parameters: _ParameterValidator

    @property
    def output_names(self) -> tuple[str, ...]:
        return (
            tuple(name for name, _ in self.output_sources)
            + tuple(name for name, _ in self.derived_outputs)
            + tuple(name for name, _ in self.normalized_derived_outputs)
        )


def _integer_parameter(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("TA-Lib integer parameter is invalid")
    return value


def _decimal_parameter(value: object) -> float:
    if not isinstance(value, str):
        raise TypeError("TA-Lib decimal parameter is invalid")
    converted = float(value)
    if not isfinite(converted):
        raise ValueError("TA-Lib decimal parameter cannot be represented as float64")
    return converted


def _simple_moving_average_type(value: object) -> int:
    if value != _SIMPLE_MOVING_AVERAGE:
        raise TypeError("TA-Lib stochastic smoothing method is invalid")
    return 0


def _talib_directional_movement(
    high: npt.NDArray[np.float64],
    low: npt.NDArray[np.float64],
    close: npt.NDArray[np.float64],
    *,
    timeperiod: int,
) -> tuple[npt.NDArray[np.float64], ...]:
    plus_di = cast(_TalibFunction, talib.PLUS_DI)(
        high, low, close, timeperiod=timeperiod
    )
    minus_di = cast(_TalibFunction, talib.MINUS_DI)(
        high, low, close, timeperiod=timeperiod
    )
    adx = cast(_TalibFunction, talib.ADX)(high, low, close, timeperiod=timeperiod)
    if not all(isinstance(output, np.ndarray) for output in (plus_di, minus_di, adx)):
        raise TypeError("TA-Lib directional movement returned invalid outputs")
    return cast(tuple[npt.NDArray[np.float64], ...], (plus_di, minus_di, adx))


def _bollinger_bandwidth(
    outputs: Mapping[str, npt.NDArray[np.float64]],
) -> npt.NDArray[np.float64]:
    upper = outputs["upper"]
    middle = outputs["middle"]
    lower = outputs["lower"]
    width = upper - lower
    bandwidth = np.full(middle.shape, np.nan, dtype=np.float64)
    zero_width = np.isfinite(width) & (width == 0)
    defined = np.isfinite(width) & np.isfinite(middle) & (middle != 0)
    bandwidth[zero_width] = 0
    bandwidth[defined] = width[defined] / middle[defined]
    return bandwidth


def _normalized_macd_histogram(
    outputs: Mapping[str, tuple[IndicatorValue, ...]],
) -> tuple[IndicatorValue, ...]:
    macd = outputs["macd"]
    signal = outputs["signal"]
    histogram: list[IndicatorValue] = []
    with localcontext(_normalization_context()):
        for macd_value, signal_value in zip(macd, signal, strict=True):
            histogram.append(
                None
                if macd_value is None or signal_value is None
                else macd_value - signal_value
            )
    return tuple(histogram)


def _normalization_context() -> Context:
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


def _validate_bollinger_parameters(parameters: PrimitiveMapping) -> None:
    _validate_period_parameters(
        parameters,
        parameter_name="period",
        indicator_label="Bollinger Bands",
        minimum_period=2,
    )
    raw_multiplier = parameters.get("standard_deviation_multiplier")
    try:
        multiplier = _decimal_parameter(raw_multiplier)
    except (TypeError, ValueError) as error:
        raise UnsupportedIndicatorBackendError(
            f"{TALIB_INDICATOR_BACKEND} Bollinger Bands multiplier must be a "
            "positive finite float64 value"
        ) from error
    if multiplier <= 0:
        raise UnsupportedIndicatorBackendError(
            f"{TALIB_INDICATOR_BACKEND} Bollinger Bands multiplier must be a "
            "positive finite float64 value"
        )


_MAPPINGS = {
    _SMA_NAME: _TalibMapping(
        function_name="SMA",
        function=cast(_TalibFunction, talib.SMA),
        parameter_mappings=(("window", ("timeperiod",), _integer_parameter),),
        input_parameter_names=frozenset(("source_field",)),
        input_fields=None,
        source_field_parameter="source_field",
        backend_output_names=("real",),
        output_sources=(("simple_moving_average", "real"),),
        derived_outputs=(),
        normalized_derived_outputs=(),
        unstable_function_names=(),
        validate_parameters=lambda parameters: _validate_period_parameters(
            parameters, parameter_name="window", indicator_label="SMA"
        ),
    ),
    _EMA_NAME: _TalibMapping(
        function_name="EMA",
        function=cast(_TalibFunction, talib.EMA),
        parameter_mappings=(("period", ("timeperiod",), _integer_parameter),),
        input_parameter_names=frozenset(("source_field",)),
        input_fields=None,
        source_field_parameter="source_field",
        backend_output_names=("real",),
        output_sources=(("exponential_moving_average", "real"),),
        derived_outputs=(),
        normalized_derived_outputs=(),
        unstable_function_names=("EMA",),
        validate_parameters=lambda parameters: _validate_period_parameters(
            parameters, parameter_name="period", indicator_label="EMA"
        ),
    ),
    _RSI_NAME: _TalibMapping(
        function_name="RSI",
        function=cast(_TalibFunction, talib.RSI),
        parameter_mappings=(("period", ("timeperiod",), _integer_parameter),),
        input_parameter_names=frozenset(),
        input_fields=("close",),
        source_field_parameter=None,
        backend_output_names=("real",),
        output_sources=(("wilder_rsi", "real"),),
        derived_outputs=(),
        normalized_derived_outputs=(),
        unstable_function_names=("RSI",),
        validate_parameters=lambda parameters: _validate_period_parameters(
            parameters,
            parameter_name="period",
            indicator_label="RSI",
            minimum_period=2,
        ),
    ),
    _ATR_NAME: _TalibMapping(
        function_name="ATR",
        function=cast(_TalibFunction, talib.ATR),
        parameter_mappings=(("period", ("timeperiod",), _integer_parameter),),
        input_parameter_names=frozenset(),
        input_fields=("high", "low", "close"),
        source_field_parameter=None,
        backend_output_names=("real",),
        output_sources=(("wilder_average_true_range", "real"),),
        derived_outputs=(),
        normalized_derived_outputs=(),
        unstable_function_names=("ATR",),
        validate_parameters=lambda parameters: _validate_period_parameters(
            parameters, parameter_name="period", indicator_label="ATR"
        ),
    ),
    _DIRECTIONAL_MOVEMENT_NAME: _TalibMapping(
        function_name="PLUS_DI+MINUS_DI+ADX",
        function=_talib_directional_movement,
        parameter_mappings=(("period", ("timeperiod",), _integer_parameter),),
        input_parameter_names=frozenset(),
        input_fields=("high", "low", "close"),
        source_field_parameter=None,
        backend_output_names=("plus_di", "minus_di", "adx"),
        output_sources=(
            ("positive_directional_indicator", "plus_di"),
            ("negative_directional_indicator", "minus_di"),
            ("average_directional_index", "adx"),
        ),
        derived_outputs=(),
        normalized_derived_outputs=(),
        unstable_function_names=("PLUS_DI", "MINUS_DI", "ADX"),
        validate_parameters=lambda parameters: _validate_period_parameters(
            parameters,
            parameter_name="period",
            indicator_label="directional movement",
            minimum_period=2,
        ),
    ),
    _BOLLINGER_NAME: _TalibMapping(
        function_name="BBANDS",
        function=cast(_TalibFunction, talib.BBANDS),
        parameter_mappings=(
            ("period", ("timeperiod",), _integer_parameter),
            (
                "standard_deviation_multiplier",
                ("nbdevup", "nbdevdn"),
                _decimal_parameter,
            ),
        ),
        input_parameter_names=frozenset(("source_field",)),
        input_fields=None,
        source_field_parameter="source_field",
        backend_output_names=("upper", "middle", "lower"),
        output_sources=(
            ("bollinger_middle_band", "middle"),
            ("bollinger_upper_band", "upper"),
            ("bollinger_lower_band", "lower"),
        ),
        derived_outputs=(("bollinger_bandwidth", _bollinger_bandwidth),),
        normalized_derived_outputs=(),
        unstable_function_names=(),
        validate_parameters=_validate_bollinger_parameters,
    ),
    _MACD_NAME: _TalibMapping(
        function_name="MACD",
        function=cast(_TalibFunction, talib.MACD),
        parameter_mappings=(
            ("fast_period", ("fastperiod",), _integer_parameter),
            ("slow_period", ("slowperiod",), _integer_parameter),
            ("signal_period", ("signalperiod",), _integer_parameter),
        ),
        input_parameter_names=frozenset(("source_field",)),
        input_fields=None,
        source_field_parameter="source_field",
        backend_output_names=("macd", "signal", "talib_histogram"),
        output_sources=(
            ("macd", "macd"),
            ("signal", "signal"),
        ),
        derived_outputs=(),
        normalized_derived_outputs=(("histogram", _normalized_macd_histogram),),
        unstable_function_names=("EMA",),
        validate_parameters=lambda parameters: _validate_macd_parameters(parameters),
    ),
    _STOCHASTIC_NAME: _TalibMapping(
        function_name="STOCH",
        function=cast(_TalibFunction, talib.STOCH),
        parameter_mappings=(
            ("k_period", ("fastk_period",), _integer_parameter),
            ("k_smoothing_period", ("slowk_period",), _integer_parameter),
            ("d_period", ("slowd_period",), _integer_parameter),
            (
                "smoothing_method",
                ("slowk_matype", "slowd_matype"),
                _simple_moving_average_type,
            ),
        ),
        input_parameter_names=frozenset(),
        input_fields=("high", "low", "close"),
        source_field_parameter=None,
        backend_output_names=("slow_k", "slow_d"),
        output_sources=(("k", "slow_k"), ("d", "slow_d")),
        derived_outputs=(),
        normalized_derived_outputs=(),
        unstable_function_names=(),
        validate_parameters=lambda parameters: _validate_stochastic_parameters(
            parameters
        ),
    ),
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
            backend_name: normalize(parameters[normalized_name])
            for normalized_name, backend_names, normalize in mapping.parameter_mappings
            for backend_name in backend_names
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
        if len(raw_arrays) != len(mapping.backend_output_names):
            raise IndicatorCalculationError(
                f"{self.backend_id} returned an unexpected output count for "
                f"indicator: {definition.name}"
            )
        backend_outputs = dict(
            zip(mapping.backend_output_names, raw_arrays, strict=True)
        )
        if any(
            not _is_aligned_talib_array(values, len(request.bars))
            for values in backend_outputs.values()
        ):
            raise IndicatorCalculationError(
                f"{self.backend_id} output does not align with canonical input"
            )
        raw_normalized_outputs = {
            normalized_name: backend_outputs[backend_name]
            for normalized_name, backend_name in mapping.output_sources
        }
        raw_normalized_outputs.update(
            {
                normalized_name: derive(backend_outputs)
                for normalized_name, derive in mapping.derived_outputs
            }
        )
        normalized_outputs = {
            name: _normalize_array(values, len(request.bars))
            for name, values in raw_normalized_outputs.items()
        }
        normalized_outputs.update(
            {
                name: derive(normalized_outputs)
                for name, derive in mapping.normalized_derived_outputs
            }
        )
        fields = tuple(
            IndicatorFieldOutput(
                name,
                normalized_outputs[name],
            )
            for name in definition.output_fields
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
    unstable_periods = tuple(
        get_unstable_period(function_name)
        for function_name in mapping.unstable_function_names
    )
    if compatibility != 0 or any(period != 0 for period in unstable_periods):
        raise InvalidIndicatorBackendError(
            "talib_v1 requires TA-Lib default compatibility and zero unstable period"
        )


def _validate_parameter_mapping(
    definition: StandardIndicatorDefinition,
    mapping: _TalibMapping,
    parameters: PrimitiveMapping,
) -> None:
    mapped = frozenset(
        normalized_name for normalized_name, _, _ in mapping.parameter_mappings
    ).union(mapping.input_parameter_names)
    if frozenset(parameters) != mapped:
        raise UnsupportedIndicatorBackendError(
            f"{TALIB_INDICATOR_BACKEND} parameter mapping is unavailable for "
            f"indicator: {definition.name}"
        )
    actual_input_fields = tuple(field.value for field in definition.input_fields)
    expected_input_fields = mapping.input_fields
    if mapping.source_field_parameter is not None:
        source_field = parameters.get(mapping.source_field_parameter)
        expected_input_fields = (source_field,) if isinstance(source_field, str) else ()
    if actual_input_fields != expected_input_fields:
        raise UnsupportedIndicatorBackendError(
            f"{TALIB_INDICATOR_BACKEND} input mapping is unavailable for "
            f"indicator: {definition.name}"
        )


def _validate_period_parameters(
    parameters: PrimitiveMapping,
    *,
    parameter_name: str,
    indicator_label: str,
    minimum_period: int = 1,
) -> None:
    period = parameters.get(parameter_name)
    if (
        isinstance(period, bool)
        or not isinstance(period, int)
        or not minimum_period <= period <= _MAXIMUM_PERIOD
    ):
        raise UnsupportedIndicatorBackendError(
            f"{TALIB_INDICATOR_BACKEND} {indicator_label} {parameter_name} must be "
            f"from {minimum_period} through {_MAXIMUM_PERIOD}"
        )


def _validate_macd_parameters(parameters: PrimitiveMapping) -> None:
    _validate_period_parameters(
        parameters,
        parameter_name="fast_period",
        indicator_label="MACD",
        minimum_period=2,
    )
    _validate_period_parameters(
        parameters,
        parameter_name="slow_period",
        indicator_label="MACD",
        minimum_period=2,
    )
    _validate_period_parameters(
        parameters,
        parameter_name="signal_period",
        indicator_label="MACD",
    )
    fast_period = parameters["fast_period"]
    slow_period = parameters["slow_period"]
    if not isinstance(fast_period, int) or not isinstance(slow_period, int):
        raise UnsupportedIndicatorBackendError(
            f"{TALIB_INDICATOR_BACKEND} MACD periods must be integers"
        )
    if fast_period >= slow_period:
        raise UnsupportedIndicatorBackendError(
            f"{TALIB_INDICATOR_BACKEND} MACD fast_period must be less than slow_period"
        )


def _validate_stochastic_parameters(parameters: PrimitiveMapping) -> None:
    for parameter_name in ("k_period", "k_smoothing_period", "d_period"):
        _validate_period_parameters(
            parameters,
            parameter_name=parameter_name,
            indicator_label="stochastic",
        )
    if parameters.get("smoothing_method") != _SIMPLE_MOVING_AVERAGE:
        raise UnsupportedIndicatorBackendError(
            f"{TALIB_INDICATOR_BACKEND} stochastic smoothing_method must be "
            f"{_SIMPLE_MOVING_AVERAGE}"
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


def _is_aligned_talib_array(values: object, expected_count: int) -> bool:
    if not isinstance(values, np.ndarray):
        return False
    array = cast(npt.NDArray[np.float64], values)
    return array.ndim == 1 and array.shape == (expected_count,)


__all__ = ["TalibIndicatorBackend"]
