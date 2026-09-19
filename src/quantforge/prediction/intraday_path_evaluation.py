"""Direction-aware intraday excursions and honest OHLC target/stop ordering."""

from dataclasses import dataclass
from decimal import Decimal, DecimalException
from enum import StrEnum
from typing import cast

from quantforge.configuration import (
    PrimitiveMapping,
    configuration_identity,
    decimal_to_primitive,
)
from quantforge.prediction._arithmetic import arithmetic
from quantforge.prediction.contracts import PredictionOutcome
from quantforge.prediction.errors import (
    InvalidPredictionConfigurationError,
    InvalidPredictionOutputError,
)
from quantforge.prediction.feature_outcomes import (
    TargetStopLabel,
    _positive_percentage,  # pyright: ignore[reportPrivateUsage]
)
from quantforge.prediction.intraday_path import IntradayPathRange, IntradayPathValues
from quantforge.prediction.models import PredictionDirection
from quantforge.prediction.signal_feature_models import SignalFeatureCandidate


class SameBarConflictPolicy(StrEnum):
    """OHLC cannot certify intrabar order; no heuristic resolution is supported."""

    AMBIGUOUS = "ambiguous"


def _configuration(name: str, parameters: PrimitiveMapping) -> PrimitiveMapping:
    return {
        "component_name": name,
        "component_type": "prediction_evaluator",
        "contract_version": "1",
        "implementation_version": "1",
        "result_schema_version": "1",
        "parameters": parameters,
    }


def _optional_decimal(amount: Decimal | None) -> str | None:
    return None if amount is None else decimal_to_primitive(amount)


def _directional_metadata(
    path: IntradayPathValues,
    direction: PredictionDirection | None,
    configuration_id: str,
) -> PrimitiveMapping:
    return {
        **path.metadata_primitive(),
        "direction": None if direction is None else direction.value,
        "evaluation_configuration_id": configuration_id,
        "available": path.available and direction is not None,
        "unavailable_reason": (
            path.status.value
            if not path.available
            else "candidate_direction_unavailable"
            if direction is None
            else None
        ),
    }


@dataclass(frozen=True, slots=True)
class IntradayExcursionEvaluationValues:
    """Existing unclamped decimal-ratio conventions with exact extreme bars."""

    path: IntradayPathValues
    direction: PredictionDirection | None
    evaluation_configuration_id: str
    mfe_percentage: Decimal | None
    mae_percentage: Decimal | None
    mfe_bar: IntradayPathRange | None
    mae_bar: IntradayPathRange | None

    def to_primitive(self) -> PrimitiveMapping:
        return {
            **_directional_metadata(
                self.path, self.direction, self.evaluation_configuration_id
            ),
            "mfe_percentage": _optional_decimal(self.mfe_percentage),
            "mae_percentage": _optional_decimal(self.mae_percentage),
            "mfe_timestamp": None
            if self.mfe_bar is None
            else self.mfe_bar.end_timestamp.isoformat(),
            "mae_timestamp": None
            if self.mae_bar is None
            else self.mae_bar.end_timestamp.isoformat(),
            "mfe_observation_id": None
            if self.mfe_bar is None
            else self.mfe_bar.observation_id,
            "mae_observation_id": None
            if self.mae_bar is None
            else self.mae_bar.observation_id,
        }


class IntradayExcursionEvaluator:
    """Orient complete future extremes using the unchanged daily formulas."""

    name = "intraday_directional_mfe_mae_evaluator"
    implementation_version = "1"
    result_schema_version = "1"

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    def configuration(self) -> PrimitiveMapping:
        return _configuration(
            self.name,
            {
                "up_mfe_formula": "maximum_high / reference_close - 1",
                "up_mae_formula": "minimum_low / reference_close - 1",
                "down_mfe_formula": "1 - minimum_low / reference_close",
                "down_mae_formula": "1 - maximum_high / reference_close",
                "return_unit": "ratio",
                "clamp_to_zero": False,
                "extreme_tie_policy": "earliest_completed_bar",
            },
        )

    def evaluate(
        self,
        signal: SignalFeatureCandidate,
        outcome: PredictionOutcome[IntradayPathValues],
    ) -> IntradayExcursionEvaluationValues:
        path = outcome.values
        direction = signal.direction
        mfe = mae = None
        mfe_bar = mae_bar = None
        if path.available and direction is not None:
            maximum = max(path.future_ranges, key=lambda bar: bar.high)
            minimum = min(path.future_ranges, key=lambda bar: bar.low)
            try:
                with arithmetic():
                    if direction is PredictionDirection.UP:
                        mfe = maximum.high / path.reference_price - Decimal(1)
                        mae = minimum.low / path.reference_price - Decimal(1)
                        mfe_bar, mae_bar = maximum, minimum
                    else:
                        mfe = Decimal(1) - minimum.low / path.reference_price
                        mae = Decimal(1) - maximum.high / path.reference_price
                        mfe_bar, mae_bar = minimum, maximum
            except DecimalException as error:
                raise InvalidPredictionOutputError(
                    "intraday MFE/MAE arithmetic failed under its configured policy"
                ) from error
        return IntradayExcursionEvaluationValues(
            path, direction, self.configuration_id, mfe, mae, mfe_bar, mae_bar
        )


@dataclass(frozen=True, slots=True)
class IntradayTargetStopEvaluationValues:
    """Typed first-bar result; collision evidence never implies an event order."""

    path: IntradayPathValues
    direction: PredictionDirection | None
    evaluation_configuration_id: str
    target_percentage: Decimal
    stop_percentage: Decimal
    same_bar_conflict_policy: SameBarConflictPolicy
    label: TargetStopLabel
    target_level: Decimal | None
    stop_level: Decimal | None
    event_bar: IntradayPathRange | None
    ambiguous_bar: IntradayPathRange | None

    def to_primitive(self) -> PrimitiveMapping:
        event, ambiguous = self.event_bar, self.ambiguous_bar
        return {
            **_directional_metadata(
                self.path, self.direction, self.evaluation_configuration_id
            ),
            "target_percentage": decimal_to_primitive(self.target_percentage),
            "stop_percentage": decimal_to_primitive(self.stop_percentage),
            "same_bar_conflict_policy": self.same_bar_conflict_policy.value,
            "label": self.label.value,
            "target_level": _optional_decimal(self.target_level),
            "stop_level": _optional_decimal(self.stop_level),
            "event_timestamp": None
            if event is None
            else event.end_timestamp.isoformat(),
            "event_observation_id": None if event is None else event.observation_id,
            "ambiguous_timestamp": None
            if ambiguous is None
            else ambiguous.end_timestamp.isoformat(),
            "ambiguous_observation_id": None
            if ambiguous is None
            else ambiguous.observation_id,
            "ambiguous_high": None
            if ambiguous is None
            else decimal_to_primitive(ambiguous.high),
            "ambiguous_low": None
            if ambiguous is None
            else decimal_to_primitive(ambiguous.low),
        }


@dataclass(frozen=True, slots=True)
class IntradayTargetStopEvaluator:
    """Inclusive first-touch thresholds, requiring the entire configured path."""

    target_percentage: Decimal
    stop_percentage: Decimal
    same_bar_conflict_policy: SameBarConflictPolicy = SameBarConflictPolicy.AMBIGUOUS

    name = "intraday_directional_target_stop_evaluator"
    implementation_version = "1"
    result_schema_version = "1"

    def __post_init__(self) -> None:
        for name in ("target_percentage", "stop_percentage"):
            object.__setattr__(
                self, name, _positive_percentage(name, getattr(self, name))
            )
        if not isinstance(
            cast(object, self.same_bar_conflict_policy), SameBarConflictPolicy
        ):
            raise InvalidPredictionConfigurationError(
                "same-bar conflict policy must explicitly preserve ambiguity"
            )

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    def configuration(self) -> PrimitiveMapping:
        return _configuration(
            self.name,
            {
                "target_percentage": decimal_to_primitive(self.target_percentage),
                "stop_percentage": decimal_to_primitive(self.stop_percentage),
                "same_bar_conflict_policy": self.same_bar_conflict_policy.value,
                "threshold_touch": "inclusive",
                "up_target_formula": "reference_close * (1 + target_percentage)",
                "up_stop_formula": "reference_close * (1 - stop_percentage)",
                "down_target_formula": "reference_close * (1 - target_percentage)",
                "down_stop_formula": "reference_close * (1 + stop_percentage)",
                "return_unit": "ratio",
                "event_timestamp_convention": "completed_bar_end_not_exact_hit_time",
            },
        )

    def evaluate(
        self,
        signal: SignalFeatureCandidate,
        outcome: PredictionOutcome[IntradayPathValues],
    ) -> IntradayTargetStopEvaluationValues:
        path = outcome.values
        direction = signal.direction
        target = stop = None
        event = ambiguous = None
        label = TargetStopLabel.UNAVAILABLE
        if path.available and direction is not None:
            try:
                with arithmetic():
                    sign = (
                        Decimal(1)
                        if direction is PredictionDirection.UP
                        else Decimal(-1)
                    )
                    target = path.reference_price * (1 + sign * self.target_percentage)
                    stop = path.reference_price * (1 - sign * self.stop_percentage)
            except DecimalException as error:
                raise InvalidPredictionOutputError(
                    "intraday target/stop arithmetic failed under its configured policy"
                ) from error
            label = TargetStopLabel.NEITHER
            for bar in path.future_ranges:
                target_hit = (
                    bar.high >= target
                    if direction is PredictionDirection.UP
                    else bar.low <= target
                )
                stop_hit = (
                    bar.low <= stop
                    if direction is PredictionDirection.UP
                    else bar.high >= stop
                )
                if target_hit and stop_hit:
                    label, ambiguous = TargetStopLabel.BOTH_SAME_BAR, bar
                    break
                if target_hit or stop_hit:
                    label = (
                        TargetStopLabel.TARGET_FIRST
                        if target_hit
                        else TargetStopLabel.STOP_FIRST
                    )
                    event = bar
                    break
        return IntradayTargetStopEvaluationValues(
            path,
            direction,
            self.configuration_id,
            self.target_percentage,
            self.stop_percentage,
            self.same_bar_conflict_policy,
            label,
            target,
            stop,
            event,
            ambiguous,
        )
