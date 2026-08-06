"""Focused overnight-gap experiments that leave the QF-11 baseline unchanged."""

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

FOCUSED_REASONS = (
    "adx_entered_di_zone_plus_di_on_top",
    "rsi_below_lower_threshold",
    "rsi_stabbed_di_zone_from_above",
    "rsi_stabbed_di_zone_from_below",
)


@dataclass(frozen=True, slots=True)
class FocusedGapPredictionParameters:
    """Parameters for the explicitly narrower multi-rule experiment."""

    rsi_period: int = 2
    adx_period: int = 5
    lower_rsi: Decimal = Decimal(15)
    maximum_adx: Decimal = Decimal(60)
    excluded_weekdays: tuple[int, ...] = (4,)
    included_weekdays: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        _validate_period("rsi_period", self.rsi_period)
        _validate_period("adx_period", self.adx_period)
        lower_rsi = _decimal_parameter("lower_rsi", self.lower_rsi)
        maximum_adx = _decimal_parameter("maximum_adx", self.maximum_adx)
        if not Decimal(0) <= lower_rsi <= Decimal(100):
            raise InvalidPredictionConfigurationError(
                "lower_rsi must be between 0 and 100"
            )
        if not Decimal(0) <= maximum_adx <= Decimal(100):
            raise InvalidPredictionConfigurationError(
                "maximum_adx must be between 0 and 100"
            )
        excluded, included = _weekday_parameters(
            self.excluded_weekdays, self.included_weekdays
        )
        object.__setattr__(self, "lower_rsi", lower_rsi)
        object.__setattr__(self, "maximum_adx", maximum_adx)
        object.__setattr__(self, "excluded_weekdays", excluded)
        object.__setattr__(self, "included_weekdays", included)

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "adx_period": self.adx_period,
            "excluded_weekdays": _weekdays_primitive(self.excluded_weekdays),
            "included_weekdays": _weekdays_primitive(self.included_weekdays),
            "lower_rsi": decimal_to_primitive(self.lower_rsi),
            "maximum_adx": decimal_to_primitive(self.maximum_adx),
            "rsi_period": self.rsi_period,
        }


class FocusedGapPredictionStrategy:
    """Predict only the four explicitly retained focused rule families."""

    name = "focused_rules"
    implementation_version = "1"

    def __init__(self, parameters: FocusedGapPredictionParameters) -> None:
        self._parameters = parameters
        self._required_indicators: tuple[Indicator, ...] = (
            WilderRelativeStrengthIndex(
                WilderRelativeStrengthIndexParameters(parameters.rsi_period)
            ),
            WilderDirectionalMovement(
                WilderDirectionalMovementParameters(parameters.adx_period)
            ),
        )

    @property
    def parameters(self) -> FocusedGapPredictionParameters:
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
        return _strategy_configuration(
            self.name,
            self.implementation_version,
            self.parameters.to_primitive(),
            self.required_indicators,
            self.warm_up_observations,
        )

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
            if not _weekday_is_eligible(
                bar.session_date.weekday(),
                self.parameters.excluded_weekdays,
                self.parameters.included_weekdays,
            ):
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
            prediction = predict_focused_gap_direction(
                self.parameters,
                previous_rsi=previous_rsi,
                current_rsi=current_rsi,
                previous_positive_di=previous_positive_di,
                current_positive_di=current_positive_di,
                previous_negative_di=previous_negative_di,
                current_negative_di=current_negative_di,
                previous_adx=previous_adx,
                current_adx=current_adx,
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
                _signal(
                    dataset,
                    bar.session_date,
                    direction,
                    self.name,
                    self.implementation_version,
                    self.configuration_id,
                    parameter_snapshot,
                    reason,
                    features,
                )
            )

        return PredictionStrategyOutput(
            self.name,
            self.configuration_id,
            dataset.metadata.dataset_id,
            tuple(signals),
        )


def predict_focused_gap_direction(
    parameters: FocusedGapPredictionParameters,
    *,
    previous_rsi: Decimal,
    current_rsi: Decimal,
    previous_positive_di: Decimal,
    current_positive_di: Decimal,
    previous_negative_di: Decimal,
    current_negative_di: Decimal,
    previous_adx: Decimal,
    current_adx: Decimal,
) -> tuple[PredictionDirection, str] | None:
    """Apply only retained baseline predicates in their original priority order."""
    if current_adx > parameters.maximum_adx:
        return None

    current_di_low = min(current_positive_di, current_negative_di)
    current_di_high = max(current_positive_di, current_negative_di)
    previous_di_low = min(previous_positive_di, previous_negative_di)
    previous_di_high = max(previous_positive_di, previous_negative_di)
    adx_is_in_zone = current_di_low <= current_adx <= current_di_high
    previous_adx_was_in_zone = previous_di_low <= previous_adx <= previous_di_high
    if (
        adx_is_in_zone
        and not previous_adx_was_in_zone
        and current_positive_di > current_negative_di
    ):
        return PredictionDirection.UP, "adx_entered_di_zone_plus_di_on_top"

    stabbed_from_above = (
        previous_rsi > previous_di_high and current_rsi <= current_di_high
    )
    stabbed_from_below = (
        previous_rsi < previous_di_low and current_rsi >= current_di_low
    )
    if stabbed_from_above:
        return PredictionDirection.UP, "rsi_stabbed_di_zone_from_above"
    if stabbed_from_below:
        return PredictionDirection.DOWN, "rsi_stabbed_di_zone_from_below"
    if current_rsi < parameters.lower_rsi:
        return PredictionDirection.UP, "rsi_below_lower_threshold"
    return None


@dataclass(frozen=True, slots=True)
class RsiOversoldUpParameters:
    """Configurable strict RSI-oversold UP-only experiment."""

    rsi_period: int = 2
    lower_rsi: Decimal = Decimal(15)
    excluded_weekdays: tuple[int, ...] = (4,)
    included_weekdays: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        _validate_period("rsi_period", self.rsi_period)
        lower_rsi = _decimal_parameter("lower_rsi", self.lower_rsi)
        if not Decimal(0) <= lower_rsi <= Decimal(100):
            raise InvalidPredictionConfigurationError(
                "lower_rsi must be between 0 and 100"
            )
        excluded, included = _weekday_parameters(
            self.excluded_weekdays, self.included_weekdays
        )
        object.__setattr__(self, "lower_rsi", lower_rsi)
        object.__setattr__(self, "excluded_weekdays", excluded)
        object.__setattr__(self, "included_weekdays", included)

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "excluded_weekdays": _weekdays_primitive(self.excluded_weekdays),
            "included_weekdays": _weekdays_primitive(self.included_weekdays),
            "lower_rsi": decimal_to_primitive(self.lower_rsi),
            "rsi_period": self.rsi_period,
        }


class RsiOversoldUpPredictionStrategy:
    """Predict UP only when completed-session RSI is strictly below threshold."""

    name = "rsi_oversold_up"
    implementation_version = "1"

    def __init__(self, parameters: RsiOversoldUpParameters) -> None:
        self._parameters = parameters
        self._required_indicators: tuple[Indicator, ...] = (
            WilderRelativeStrengthIndex(
                WilderRelativeStrengthIndexParameters(parameters.rsi_period)
            ),
        )

    @property
    def parameters(self) -> RsiOversoldUpParameters:
        return self._parameters

    @property
    def required_indicators(self) -> tuple[Indicator, ...]:
        return self._required_indicators

    @property
    def warm_up_observations(self) -> int:
        return self.required_indicators[0].warm_up_observations

    def configuration(self) -> PrimitiveMapping:
        return _strategy_configuration(
            self.name,
            self.implementation_version,
            self.parameters.to_primitive(),
            self.required_indicators,
            self.warm_up_observations,
        )

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    def generate(self, dataset: MarketDataset) -> PredictionStrategyOutput:
        rsi_values = (
            self.required_indicators[0].calculate(dataset).values_for(WILDER_RSI_OUTPUT)
        )
        parameter_snapshot = _parameter_snapshot(self.parameters.to_primitive())
        signals: list[PredictionSignal] = []
        for bar, rsi in zip(dataset.bars, rsi_values, strict=True):
            if (
                rsi is None
                or rsi >= self.parameters.lower_rsi
                or not _weekday_is_eligible(
                    bar.session_date.weekday(),
                    self.parameters.excluded_weekdays,
                    self.parameters.included_weekdays,
                )
            ):
                continue
            signals.append(
                _signal(
                    dataset,
                    bar.session_date,
                    PredictionDirection.UP,
                    self.name,
                    self.implementation_version,
                    self.configuration_id,
                    parameter_snapshot,
                    "rsi_below_lower_threshold",
                    {"close": bar.close, "open": bar.open, "rsi": rsi},
                )
            )
        return PredictionStrategyOutput(
            self.name,
            self.configuration_id,
            dataset.metadata.dataset_id,
            tuple(signals),
        )


@dataclass(frozen=True, slots=True)
class AlwaysUpParameters:
    """Weekday eligibility for the structural upward-gap baseline."""

    excluded_weekdays: tuple[int, ...] = (4,)
    included_weekdays: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        excluded, included = _weekday_parameters(
            self.excluded_weekdays, self.included_weekdays
        )
        object.__setattr__(self, "excluded_weekdays", excluded)
        object.__setattr__(self, "included_weekdays", included)

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "excluded_weekdays": _weekdays_primitive(self.excluded_weekdays),
            "included_weekdays": _weekdays_primitive(self.included_weekdays),
        }


class AlwaysUpPredictionStrategy:
    """Predict UP on every eligible completed session without indicators."""

    name = "always_up"
    implementation_version = "1"
    required_indicators: tuple[Indicator, ...] = ()
    warm_up_observations = 1

    def __init__(self, parameters: AlwaysUpParameters) -> None:
        self._parameters = parameters

    @property
    def parameters(self) -> AlwaysUpParameters:
        return self._parameters

    def configuration(self) -> PrimitiveMapping:
        return _strategy_configuration(
            self.name,
            self.implementation_version,
            self.parameters.to_primitive(),
            self.required_indicators,
            self.warm_up_observations,
        )

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    def generate(self, dataset: MarketDataset) -> PredictionStrategyOutput:
        parameters = _parameter_snapshot(self.parameters.to_primitive())
        signals = tuple(
            _signal(
                dataset,
                bar.session_date,
                PredictionDirection.UP,
                self.name,
                self.implementation_version,
                self.configuration_id,
                parameters,
                "always_up_baseline",
                {"close": bar.close, "open": bar.open},
            )
            for bar in dataset.bars
            if _weekday_is_eligible(
                bar.session_date.weekday(),
                self.parameters.excluded_weekdays,
                self.parameters.included_weekdays,
            )
        )
        return PredictionStrategyOutput(
            self.name,
            self.configuration_id,
            dataset.metadata.dataset_id,
            signals,
        )


def _strategy_configuration(
    name: str,
    implementation_version: str,
    parameters: PrimitiveMapping,
    indicators: tuple[Indicator, ...],
    warm_up_observations: int,
) -> PrimitiveMapping:
    return {
        "component_type": "prediction_strategy",
        "component_name": name,
        "contract_version": "1",
        "implementation_version": implementation_version,
        "parameters": parameters,
        "required_indicators": [indicator.configuration() for indicator in indicators],
        "warm_up_observations": warm_up_observations,
        "signal_timestamp": "after_completed_session_close",
        "outcome_label": "next_exchange_session_open_vs_signal_close",
    }


def _signal(
    dataset: MarketDataset,
    signal_session: object,
    direction: PredictionDirection,
    strategy_id: str,
    implementation_version: str,
    configuration_id: str,
    parameters: tuple[PredictionParameter, ...],
    reason: str,
    features: dict[str, Decimal],
) -> PredictionSignal:
    from datetime import date

    if not isinstance(signal_session, date):
        raise InvalidPredictionConfigurationError("signal session must be a date")
    return PredictionSignal(
        symbol=dataset.metadata.canonical_symbol,
        signal_session=signal_session,
        direction=direction,
        strategy_id=strategy_id,
        strategy_implementation_version=implementation_version,
        strategy_configuration_id=configuration_id,
        strategy_parameters=parameters,
        reason=reason,
        feature_values=tuple(
            PredictionFeature(name, value) for name, value in sorted(features.items())
        ),
    )


def _parameter_snapshot(values: PrimitiveMapping) -> tuple[PredictionParameter, ...]:
    snapshot: list[PredictionParameter] = []
    for name, value in sorted(values.items()):
        if isinstance(value, (str, int, float, bool)) or value is None:
            snapshot.append(PredictionParameter(name, value))
        else:
            raise InvalidPredictionConfigurationError(
                "prediction parameters must serialize to primitive scalar values"
            )
    return tuple(snapshot)


def _weekday_parameters(
    excluded_value: object, included_value: object
) -> tuple[tuple[int, ...], tuple[int, ...] | None]:
    excluded = _validated_weekdays("excluded_weekdays", excluded_value)
    included = (
        None
        if included_value is None
        else _validated_weekdays("included_weekdays", included_value)
    )
    if included is not None and set(excluded) & set(included):
        raise InvalidPredictionConfigurationError(
            "included and excluded weekdays must not overlap"
        )
    return excluded, included


def _validated_weekdays(name: str, value: object) -> tuple[int, ...]:
    if not isinstance(value, tuple):
        raise InvalidPredictionConfigurationError(f"{name} must be a tuple")
    weekdays = cast(tuple[object, ...], value)
    if any(
        isinstance(day, bool) or not isinstance(day, int) or not 0 <= day <= 6
        for day in weekdays
    ):
        raise InvalidPredictionConfigurationError(
            f"{name} must use integers from Monday=0 to Sunday=6"
        )
    if len(weekdays) != len(set(weekdays)):
        raise InvalidPredictionConfigurationError(f"{name} must be unique")
    return tuple(sorted(cast(tuple[int, ...], weekdays)))


def _weekday_is_eligible(
    weekday: int,
    excluded_weekdays: tuple[int, ...],
    included_weekdays: tuple[int, ...] | None,
) -> bool:
    return weekday not in excluded_weekdays and (
        included_weekdays is None or weekday in included_weekdays
    )


def _weekdays_primitive(value: tuple[int, ...] | None) -> str:
    return "all" if value is None else ",".join(str(day) for day in value)


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
