"""Full-window, causal simple moving average."""

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
from typing import cast

from quantforge.configuration import (
    PrimitiveMapping,
    configuration_identity,
)
from quantforge.data.models import MarketDataset
from quantforge.indicators.base import (
    IndicatorParameters,
    validate_indicator_alignment,
    validate_market_input,
)
from quantforge.indicators.exceptions import (
    IndicatorCalculationError,
    InvalidIndicatorParametersError,
    MissingMarketFieldError,
)
from quantforge.indicators.models import (
    IndicatorFieldOutput,
    IndicatorOutput,
    IndicatorValue,
    MarketField,
)

SIMPLE_MOVING_AVERAGE_OUTPUT = "simple_moving_average"
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
class SimpleMovingAverageParameters:
    """Parameters for a full-window simple moving average."""

    window: int
    source_field: MarketField = MarketField.CLOSE

    def __post_init__(self) -> None:
        _validate_window_type(self.window)
        if self.window < 1:
            raise InvalidIndicatorParametersError("window must be at least 1")
        _validate_source_field(self.source_field)

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "source_field": self.source_field.value,
            "window": self.window,
        }


class SimpleMovingAverage:
    """Arithmetic mean over the current and prior ``window - 1`` observations."""

    name = "simple_moving_average"
    output_fields = (SIMPLE_MOVING_AVERAGE_OUTPUT,)
    missing_value = None

    def __init__(self, parameters: SimpleMovingAverageParameters) -> None:
        self._parameters = parameters

    @property
    def parameters(self) -> IndicatorParameters:
        return self._parameters

    @property
    def required_fields(self) -> frozenset[MarketField]:
        return frozenset((self._parameters.source_field,))

    @property
    def warm_up_observations(self) -> int:
        """Observations required for the first available result."""
        return self._parameters.window

    def configuration(self) -> PrimitiveMapping:
        return {
            "component_type": "indicator",
            "component_name": self.name,
            "contract_version": "1",
            "parameters": self._parameters.to_primitive(),
            "required_fields": [field.value for field in sorted(self.required_fields)],
            "warm_up_observations": self.warm_up_observations,
            "output_fields": list(self.output_fields),
            "missing_value": None,
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

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    def calculate(self, dataset: MarketDataset) -> IndicatorOutput:
        """Return aligned deterministic means without partial or filled windows."""
        validate_market_input(dataset, self.required_fields)
        source: list[Decimal | None] = []
        for bar in dataset.bars:
            try:
                raw_value = getattr(bar, self._parameters.source_field.value)
            except AttributeError as error:
                raise MissingMarketFieldError(
                    "market data is missing required field: "
                    f"{self._parameters.source_field.value}"
                ) from error
            if not isinstance(raw_value, Decimal):
                raise IndicatorCalculationError(
                    f"{self._parameters.source_field.value} must be a Decimal"
                )
            source.append(raw_value if raw_value.is_finite() else None)

        window = self._parameters.window
        divisor = Decimal(window)
        values: list[IndicatorValue] = []
        arithmetic_context = _arithmetic_context()
        with localcontext(arithmetic_context):
            for index in range(len(source)):
                if index + 1 < window:
                    values.append(None)
                    continue
                observations = source[index + 1 - window : index + 1]
                if any(observation is None for observation in observations):
                    values.append(None)
                    continue
                try:
                    total = sum(
                        (cast(Decimal, value) for value in observations),
                        start=Decimal(0),
                    )
                    values.append(total / divisor)
                except DecimalException as error:
                    raise IndicatorCalculationError(
                        "simple moving average arithmetic failed under its "
                        "configured decimal policy"
                    ) from error

        output = IndicatorOutput(
            self.name,
            self.configuration_id,
            tuple(bar.session_date for bar in dataset.bars),
            (IndicatorFieldOutput(SIMPLE_MOVING_AVERAGE_OUTPUT, tuple(values)),),
        )
        validate_indicator_alignment(dataset, output)
        return output


def _arithmetic_context() -> Context:
    """Build the complete deterministic Decimal policy for SMA arithmetic."""
    return Context(
        prec=_DECIMAL_PRECISION,
        rounding=_DECIMAL_ROUNDING,
        Emin=_DECIMAL_EMIN,
        Emax=_DECIMAL_EMAX,
        capitals=_DECIMAL_CAPITALS,
        clamp=_DECIMAL_CLAMP,
        flags=[],
        traps=list(_DECIMAL_TRAPS),
    )


def _validate_window_type(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidIndicatorParametersError("window must be an integer")


def _validate_source_field(value: object) -> None:
    if not isinstance(value, MarketField):
        raise InvalidIndicatorParametersError(
            "source_field must be a normalized market field"
        )
