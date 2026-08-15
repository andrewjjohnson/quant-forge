"""RSI, DMI, and ADX overnight-gap direction prediction strategy."""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import cast

from quantforge.configuration import (
    PrimitiveMapping,
    configuration_identity,
    decimal_to_primitive,
)
from quantforge.data.models import MarketDataset
from quantforge.indicators import (
    AVERAGE_DIRECTIONAL_INDEX_OUTPUT,
    NEGATIVE_DIRECTIONAL_INDICATOR_OUTPUT,
    POSITIVE_DIRECTIONAL_INDICATOR_OUTPUT,
    WILDER_RSI_OUTPUT,
    Indicator,
    IndicatorBackendRegistry,
    WilderDirectionalMovement,
    WilderDirectionalMovementParameters,
    WilderRelativeStrengthIndex,
    WilderRelativeStrengthIndexParameters,
)
from quantforge.prediction.errors import InvalidPredictionConfigurationError
from quantforge.prediction.models import (
    PredictionDirection,
    PredictionFeature,
    PredictionParameter,
    PredictionSignal,
    PredictionStrategyOutput,
)


@dataclass(frozen=True, slots=True)
class OvernightGapPredictionParameters:
    """Thresholds and session exclusions for the baseline gap rules."""

    rsi_period: int = 2
    adx_period: int = 5
    lower_rsi: Decimal = Decimal(15)
    upper_rsi: Decimal = Decimal(85)
    maximum_adx: Decimal = Decimal(60)
    excluded_weekdays: tuple[int, ...] = (4,)

    def __post_init__(self) -> None:
        _validate_period("rsi_period", self.rsi_period)
        _validate_period("adx_period", self.adx_period)
        lower_rsi = _decimal_parameter("lower_rsi", self.lower_rsi)
        upper_rsi = _decimal_parameter("upper_rsi", self.upper_rsi)
        maximum_adx = _decimal_parameter("maximum_adx", self.maximum_adx)
        if not Decimal(0) <= lower_rsi < upper_rsi <= Decimal(100):
            raise InvalidPredictionConfigurationError(
                "RSI thresholds must satisfy 0 <= lower < upper <= 100"
            )
        if not Decimal(0) <= maximum_adx <= Decimal(100):
            raise InvalidPredictionConfigurationError(
                "maximum_adx must be between 0 and 100"
            )
        weekdays = tuple(cast(tuple[object, ...], self.excluded_weekdays))
        if any(
            isinstance(day, bool) or not isinstance(day, int) or not 0 <= day <= 6
            for day in weekdays
        ):
            raise InvalidPredictionConfigurationError(
                "excluded weekdays must use unique integers from Monday=0 to Sunday=6"
            )
        if len(weekdays) != len(set(weekdays)):
            raise InvalidPredictionConfigurationError(
                "excluded weekdays must not contain duplicates"
            )
        object.__setattr__(self, "lower_rsi", lower_rsi)
        object.__setattr__(self, "upper_rsi", upper_rsi)
        object.__setattr__(self, "maximum_adx", maximum_adx)
        object.__setattr__(
            self,
            "excluded_weekdays",
            tuple(sorted(cast(tuple[int, ...], weekdays))),
        )

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "adx_period": self.adx_period,
            "excluded_weekdays": ",".join(str(day) for day in self.excluded_weekdays),
            "lower_rsi": decimal_to_primitive(self.lower_rsi),
            "maximum_adx": decimal_to_primitive(self.maximum_adx),
            "rsi_period": self.rsi_period,
            "upper_rsi": decimal_to_primitive(self.upper_rsi),
        }


class OvernightGapPredictionStrategy:
    """Predict each eligible next-open gap from completed-session features."""

    name = "overnight_gap_direction"
    implementation_version = "1"

    def __init__(
        self,
        parameters: OvernightGapPredictionParameters,
        *,
        backend_id: str | None = None,
        backend_registry: IndicatorBackendRegistry | None = None,
    ) -> None:
        self._parameters = parameters
        self._required_indicators: tuple[Indicator, ...] = (
            WilderRelativeStrengthIndex(
                WilderRelativeStrengthIndexParameters(parameters.rsi_period),
                backend_id=backend_id,
                backend_registry=backend_registry,
            ),
            WilderDirectionalMovement(
                WilderDirectionalMovementParameters(parameters.adx_period),
                backend_id=backend_id,
                backend_registry=backend_registry,
            ),
        )

    @property
    def parameters(self) -> OvernightGapPredictionParameters:
        return self._parameters

    @property
    def required_indicators(self) -> tuple[Indicator, ...]:
        return self._required_indicators

    @property
    def warm_up_observations(self) -> int:
        return (
            max(
                indicator.warm_up_observations for indicator in self.required_indicators
            )
            + 1
        )

    def configuration(self) -> PrimitiveMapping:
        return {
            "component_type": "prediction_strategy",
            "component_name": self.name,
            "contract_version": "1",
            "implementation_version": self.implementation_version,
            "parameters": self.parameters.to_primitive(),
            "required_indicators": [
                indicator.configuration() for indicator in self.required_indicators
            ],
            "warm_up_observations": self.warm_up_observations,
            "signal_timestamp": "after_completed_session_close",
            "outcome_label": "next_exchange_session_open_vs_signal_close",
        }

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    def generate(self, dataset: MarketDataset) -> PredictionStrategyOutput:
        rsi_output = self.required_indicators[0].calculate(dataset)
        dmi_output = self.required_indicators[1].calculate(dataset)
        rsi_values = rsi_output.values_for(WILDER_RSI_OUTPUT)
        positive_di_values = dmi_output.values_for(
            POSITIVE_DIRECTIONAL_INDICATOR_OUTPUT
        )
        negative_di_values = dmi_output.values_for(
            NEGATIVE_DIRECTIONAL_INDICATOR_OUTPUT
        )
        adx_values = dmi_output.values_for(AVERAGE_DIRECTIONAL_INDEX_OUTPUT)
        parameter_snapshot = _parameter_snapshot(self.parameters.to_primitive())
        signals: list[PredictionSignal] = []

        for index in range(1, len(dataset.bars)):
            bar = dataset.bars[index]
            if bar.session_date.weekday() in self.parameters.excluded_weekdays:
                continue
            values = (
                rsi_values[index - 1],
                rsi_values[index],
                positive_di_values[index - 1],
                positive_di_values[index],
                negative_di_values[index - 1],
                negative_di_values[index],
                adx_values[index - 1],
                adx_values[index],
            )
            if any(value is None for value in values):
                continue
            (
                previous_rsi,
                current_rsi,
                previous_positive_di,
                current_positive_di,
                previous_negative_di,
                current_negative_di,
                previous_adx,
                current_adx,
            ) = (value for value in values if value is not None)

            prediction = predict_overnight_gap_direction(
                self.parameters,
                previous_rsi=previous_rsi,
                current_rsi=current_rsi,
                previous_positive_di=previous_positive_di,
                current_positive_di=current_positive_di,
                previous_negative_di=previous_negative_di,
                current_negative_di=current_negative_di,
                previous_adx=previous_adx,
                current_adx=current_adx,
                session_open=bar.open,
                session_close=bar.close,
            )
            if prediction is None:
                continue
            direction, reason = prediction
            features = {
                "adx": current_adx,
                "close": bar.close,
                "minus_di": current_negative_di,
                "open": bar.open,
                "plus_di": current_positive_di,
                "previous_adx": previous_adx,
                "previous_minus_di": previous_negative_di,
                "previous_plus_di": previous_positive_di,
                "previous_rsi": previous_rsi,
                "rsi": current_rsi,
            }
            signals.append(
                PredictionSignal(
                    symbol=dataset.metadata.canonical_symbol,
                    signal_session=bar.session_date,
                    direction=direction,
                    strategy_id=self.name,
                    strategy_implementation_version=self.implementation_version,
                    strategy_configuration_id=self.configuration_id,
                    strategy_parameters=parameter_snapshot,
                    reason=reason,
                    feature_values=tuple(
                        PredictionFeature(name, value)
                        for name, value in sorted(features.items())
                    ),
                )
            )

        return PredictionStrategyOutput(
            strategy_id=self.name,
            strategy_configuration_id=self.configuration_id,
            dataset_id=dataset.metadata.dataset_id,
            signals=tuple(signals),
        )


def _parameter_snapshot(
    values: PrimitiveMapping,
) -> tuple[PredictionParameter, ...]:
    snapshot: list[PredictionParameter] = []
    for name, value in sorted(values.items()):
        if isinstance(value, (str, int, float, bool)) or value is None:
            snapshot.append(PredictionParameter(name, value))
        else:
            raise InvalidPredictionConfigurationError(
                "prediction parameters must serialize to primitive scalar values"
            )
    return tuple(snapshot)


def predict_overnight_gap_direction(
    parameters: OvernightGapPredictionParameters,
    *,
    previous_rsi: Decimal,
    current_rsi: Decimal,
    previous_positive_di: Decimal,
    current_positive_di: Decimal,
    previous_negative_di: Decimal,
    current_negative_di: Decimal,
    previous_adx: Decimal,
    current_adx: Decimal,
    session_open: Decimal,
    session_close: Decimal,
) -> tuple[PredictionDirection, str] | None:
    """Apply the documented veto, overrides, and base rule in priority order."""
    evaluation = evaluate_overnight_gap_rules(
        parameters,
        previous_rsi=previous_rsi,
        current_rsi=current_rsi,
        previous_positive_di=previous_positive_di,
        current_positive_di=current_positive_di,
        previous_negative_di=previous_negative_di,
        current_negative_di=current_negative_di,
        previous_adx=previous_adx,
        current_adx=current_adx,
        session_open=session_open,
        session_close=session_close,
    )
    if evaluation.direction is None or evaluation.selected_reason is None:
        return None
    return evaluation.direction, evaluation.selected_reason


@dataclass(frozen=True, slots=True)
class OvernightGapRuleEvaluation:
    """Deterministic baseline-rule trace without any future outcome values."""

    direction: PredictionDirection | None
    selected_reason: str | None
    matched_reasons: tuple[str, ...]
    veto_reason: str | None = None


def evaluate_overnight_gap_rules(
    parameters: OvernightGapPredictionParameters,
    *,
    previous_rsi: Decimal,
    current_rsi: Decimal,
    previous_positive_di: Decimal,
    current_positive_di: Decimal,
    previous_negative_di: Decimal,
    current_negative_di: Decimal,
    previous_adx: Decimal,
    current_adx: Decimal,
    session_open: Decimal,
    session_close: Decimal,
) -> OvernightGapRuleEvaluation:
    """Return the original prediction plus every matched rule in priority order."""
    if current_adx > parameters.maximum_adx:
        return OvernightGapRuleEvaluation(None, None, (), "adx_above_maximum_veto")

    current_di_low = min(current_positive_di, current_negative_di)
    current_di_high = max(current_positive_di, current_negative_di)
    previous_di_low = min(previous_positive_di, previous_negative_di)
    previous_di_high = max(previous_positive_di, previous_negative_di)
    adx_is_in_zone = current_di_low <= current_adx <= current_di_high
    previous_adx_was_in_zone = previous_di_low <= previous_adx <= previous_di_high
    matched: list[str] = []
    if adx_is_in_zone and not previous_adx_was_in_zone:
        if current_positive_di > current_negative_di:
            matched.append("adx_entered_di_zone_plus_di_on_top")
        if current_negative_di > current_positive_di:
            matched.append("adx_entered_di_zone_minus_di_on_top")

    stabbed_from_above = (
        previous_rsi > previous_di_high and current_rsi <= current_di_high
    )
    stabbed_from_below = (
        previous_rsi < previous_di_low and current_rsi >= current_di_low
    )
    if stabbed_from_above:
        matched.append("rsi_stabbed_di_zone_from_above")
    if stabbed_from_below:
        matched.append("rsi_stabbed_di_zone_from_below")

    if current_rsi < parameters.lower_rsi:
        matched.append("rsi_below_lower_threshold")
    elif current_rsi > parameters.upper_rsi:
        matched.append("rsi_above_upper_threshold")
    elif session_close > session_open:
        matched.append("bullish_candle_in_middle_rsi_range")
    elif session_close < session_open:
        matched.append("bearish_candle_in_middle_rsi_range")

    selected_reason = matched[0] if matched else None
    if selected_reason is None:
        return OvernightGapRuleEvaluation(None, None, ())
    direction = (
        PredictionDirection.DOWN
        if selected_reason
        in {
            "adx_entered_di_zone_minus_di_on_top",
            "rsi_stabbed_di_zone_from_below",
            "rsi_above_upper_threshold",
            "bearish_candle_in_middle_rsi_range",
        }
        else PredictionDirection.UP
    )
    return OvernightGapRuleEvaluation(direction, selected_reason, tuple(matched))


def _decimal_parameter(name: str, value: object) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise InvalidPredictionConfigurationError(f"{name} must be numeric") from error
    if not result.is_finite():
        raise InvalidPredictionConfigurationError(f"{name} must be finite")
    return result


def _validate_period(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise InvalidPredictionConfigurationError(f"{name} must be a positive integer")
