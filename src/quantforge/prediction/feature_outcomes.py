"""Reusable QF-11-compatible forward outcomes for QF-7 datasets."""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, DecimalException, InvalidOperation
from enum import StrEnum
from typing import cast

from quantforge.configuration import (
    PrimitiveMapping,
    configuration_identity,
    decimal_to_primitive,
)
from quantforge.data.models import AdjustmentMode, DailyBar, MarketDataset
from quantforge.prediction._arithmetic import arithmetic
from quantforge.prediction.contracts import OutcomeLabel, PredictionOutcome
from quantforge.prediction.errors import (
    InvalidPredictionConfigurationError,
    InvalidPredictionDataError,
    InvalidPredictionOutputError,
)
from quantforge.prediction.models import PredictionDirection
from quantforge.prediction.signal_feature_models import SignalFeatureCandidate

DEFAULT_FORWARD_RETURN_HORIZONS = (1, 2, 5, 10, 20)


@dataclass(frozen=True, slots=True)
class ForwardReturnValues:
    """Completed-close return at one exact future exchange-session horizon."""

    available: bool
    horizon_sessions: int
    reference_price: Decimal
    outcome_price: Decimal
    raw_return: Decimal

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "available": self.available,
            "horizon_sessions": self.horizon_sessions,
            "outcome_price": decimal_to_primitive(self.outcome_price),
            "raw_return": decimal_to_primitive(self.raw_return),
            "reference_price": decimal_to_primitive(self.reference_price),
        }


class ForwardReturnOutcomeLabeler:
    """Label close-to-future-close arithmetic return in trading sessions."""

    name = "forward_close_return"
    implementation_version = "1"
    result_schema_version = "1"
    required_market_fields = ("close",)

    def __init__(self, horizon_sessions: int) -> None:
        _validate_horizon(horizon_sessions)
        self.required_future_sessions = horizon_sessions

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    def configuration(self) -> PrimitiveMapping:
        return _labeler_configuration(
            self.name,
            self.implementation_version,
            self.result_schema_version,
            self.required_market_fields,
            {
                "future_sessions": self.required_future_sessions,
                "outcome_field": "close",
                "reference_field": "completed_signal_session_close",
                "return_formula": "outcome_close / reference_close - 1",
            },
        )

    def validate_dataset(self, dataset: MarketDataset) -> None:
        _validate_price_basis(dataset)

    def label(
        self, dataset: MarketDataset, signal_session: date
    ) -> OutcomeLabel[ForwardReturnValues] | None:
        signal_index = _signal_index(dataset, signal_session)
        outcome_index = signal_index + self.required_future_sessions
        if outcome_index >= len(dataset.bars):
            return None
        reference_price = dataset.bars[signal_index].close
        outcome_bar = dataset.bars[outcome_index]
        try:
            with arithmetic():
                raw_return = outcome_bar.close / reference_price - Decimal(1)
        except DecimalException as error:
            raise InvalidPredictionOutputError(
                "forward-return arithmetic failed under its configured policy"
            ) from error
        return OutcomeLabel(
            signal_session,
            outcome_bar.session_date,
            ForwardReturnValues(
                True,
                self.required_future_sessions,
                reference_price,
                outcome_bar.close,
                raw_return,
            ),
        )


class ForwardReturnEvaluator:
    """Pass through an already-built forward return without reading market data."""

    name = "forward_return_evaluator"
    implementation_version = "1"
    result_schema_version = "1"

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    def configuration(self) -> PrimitiveMapping:
        return _evaluator_configuration(
            self.name,
            self.implementation_version,
            self.result_schema_version,
            {"evaluation": "identity_projection_of_forward_return_label"},
        )

    def evaluate(
        self,
        signal: SignalFeatureCandidate,
        outcome: PredictionOutcome[ForwardReturnValues],
    ) -> ForwardReturnValues:
        del signal
        return outcome.values


@dataclass(frozen=True, slots=True)
class ExcursionPathValues:
    """Direction-neutral future extremes before directional evaluation."""

    horizon_sessions: int
    reference_price: Decimal
    maximum_high: Decimal
    maximum_high_session: date
    minimum_low: Decimal
    minimum_low_session: date

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "horizon_sessions": self.horizon_sessions,
            "maximum_high": decimal_to_primitive(self.maximum_high),
            "maximum_high_session": self.maximum_high_session.isoformat(),
            "minimum_low": decimal_to_primitive(self.minimum_low),
            "minimum_low_session": self.minimum_low_session.isoformat(),
            "reference_price": decimal_to_primitive(self.reference_price),
        }


@dataclass(frozen=True, slots=True)
class ExcursionEvaluationValues:
    """Direction-aware MFE/MAE research labels, not executable returns."""

    available: bool
    horizon_sessions: int
    reference_price: Decimal
    mfe_percentage: Decimal | None
    mae_percentage: Decimal | None
    mfe_session: date | None
    mae_session: date | None
    unavailable_reason: str | None

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "available": self.available,
            "horizon_sessions": self.horizon_sessions,
            "mae_percentage": _optional_decimal(self.mae_percentage),
            "mae_session": _optional_date(self.mae_session),
            "mfe_percentage": _optional_decimal(self.mfe_percentage),
            "mfe_session": _optional_date(self.mfe_session),
            "reference_price": decimal_to_primitive(self.reference_price),
            "unavailable_reason": self.unavailable_reason,
        }


class ExcursionOutcomeLabeler:
    """Capture direction-neutral future high/low extremes over exact sessions."""

    name = "future_high_low_excursion_path"
    implementation_version = "1"
    result_schema_version = "1"
    required_market_fields = ("close", "high", "low")

    def __init__(self, horizon_sessions: int) -> None:
        _validate_horizon(horizon_sessions)
        self.required_future_sessions = horizon_sessions

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    def configuration(self) -> PrimitiveMapping:
        return _labeler_configuration(
            self.name,
            self.implementation_version,
            self.result_schema_version,
            self.required_market_fields,
            {
                "future_sessions": self.required_future_sessions,
                "future_window": "sessions t+1 through t+h inclusive",
                "reference_field": "completed_signal_session_close",
            },
        )

    def validate_dataset(self, dataset: MarketDataset) -> None:
        _validate_price_basis(dataset)

    def label(
        self, dataset: MarketDataset, signal_session: date
    ) -> OutcomeLabel[ExcursionPathValues] | None:
        signal_index = _signal_index(dataset, signal_session)
        outcome_index = signal_index + self.required_future_sessions
        if outcome_index >= len(dataset.bars):
            return None
        future_bars = dataset.bars[signal_index + 1 : outcome_index + 1]
        maximum = max(
            future_bars, key=lambda bar: (bar.high, -bar.session_date.toordinal())
        )
        minimum = min(future_bars, key=lambda bar: (bar.low, bar.session_date))
        return OutcomeLabel(
            signal_session,
            dataset.bars[outcome_index].session_date,
            ExcursionPathValues(
                self.required_future_sessions,
                dataset.bars[signal_index].close,
                maximum.high,
                maximum.session_date,
                minimum.low,
                minimum.session_date,
            ),
        )


class DirectionalExcursionEvaluator:
    """Orient future extremes as favorable-positive and adverse-negative."""

    name = "directional_mfe_mae_evaluator"
    implementation_version = "1"
    result_schema_version = "1"

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    def configuration(self) -> PrimitiveMapping:
        return _evaluator_configuration(
            self.name,
            self.implementation_version,
            self.result_schema_version,
            {
                "down_mae_formula": "1 - maximum_high / reference_close",
                "down_mfe_formula": "1 - minimum_low / reference_close",
                "up_mae_formula": "minimum_low / reference_close - 1",
                "up_mfe_formula": "maximum_high / reference_close - 1",
            },
        )

    def evaluate(
        self,
        signal: SignalFeatureCandidate,
        outcome: PredictionOutcome[ExcursionPathValues],
    ) -> ExcursionEvaluationValues:
        values = outcome.values
        if signal.direction is None:
            return ExcursionEvaluationValues(
                False,
                values.horizon_sessions,
                values.reference_price,
                None,
                None,
                None,
                None,
                "candidate_direction_unavailable",
            )
        try:
            with arithmetic():
                if signal.direction is PredictionDirection.UP:
                    mfe = values.maximum_high / values.reference_price - Decimal(1)
                    mae = values.minimum_low / values.reference_price - Decimal(1)
                    mfe_session = values.maximum_high_session
                    mae_session = values.minimum_low_session
                else:
                    mfe = Decimal(1) - values.minimum_low / values.reference_price
                    mae = Decimal(1) - values.maximum_high / values.reference_price
                    mfe_session = values.minimum_low_session
                    mae_session = values.maximum_high_session
        except DecimalException as error:
            raise InvalidPredictionOutputError(
                "MFE/MAE arithmetic failed under its configured policy"
            ) from error
        return ExcursionEvaluationValues(
            True,
            values.horizon_sessions,
            values.reference_price,
            mfe,
            mae,
            mfe_session,
            mae_session,
            None,
        )


class SameSessionConflictPolicy(StrEnum):
    """Explicit treatment when daily high/low touch both thresholds."""

    AMBIGUOUS = "ambiguous"
    CONSERVATIVE_STOP_FIRST = "conservative_stop_first"


class TargetStopLabel(StrEnum):
    """Stable target-versus-stop path label."""

    TARGET_FIRST = "target_first"
    STOP_FIRST = "stop_first"
    NEITHER = "neither"
    BOTH_SAME_SESSION = "both_same_session"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class FutureSessionRange:
    """One future daily range retained for honest path classification."""

    session: date
    high: Decimal
    low: Decimal

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "high": decimal_to_primitive(self.high),
            "low": decimal_to_primitive(self.low),
            "session": self.session.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class TargetStopPathValues:
    """Direction-neutral future daily ranges for target/stop evaluation."""

    horizon_sessions: int
    reference_price: Decimal
    future_ranges: tuple[FutureSessionRange, ...]

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "future_ranges": [item.to_primitive() for item in self.future_ranges],
            "horizon_sessions": self.horizon_sessions,
            "reference_price": decimal_to_primitive(self.reference_price),
        }


@dataclass(frozen=True, slots=True)
class TargetStopEvaluationValues:
    """Direction-aware threshold label with preserved daily-bar ambiguity."""

    available: bool
    horizon_sessions: int
    label: TargetStopLabel
    reference_price: Decimal
    target_percentage: Decimal
    stop_percentage: Decimal
    target_level: Decimal | None
    stop_level: Decimal | None
    event_session: date | None
    ambiguous_session: date | None
    ambiguous_high: Decimal | None
    ambiguous_low: Decimal | None
    same_session_conflict_policy: SameSessionConflictPolicy
    unavailable_reason: str | None

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "ambiguous_high": _optional_decimal(self.ambiguous_high),
            "ambiguous_low": _optional_decimal(self.ambiguous_low),
            "ambiguous_session": _optional_date(self.ambiguous_session),
            "available": self.available,
            "event_session": _optional_date(self.event_session),
            "horizon_sessions": self.horizon_sessions,
            "label": self.label.value,
            "reference_price": decimal_to_primitive(self.reference_price),
            "same_session_conflict_policy": self.same_session_conflict_policy.value,
            "stop_level": _optional_decimal(self.stop_level),
            "stop_percentage": decimal_to_primitive(self.stop_percentage),
            "target_level": _optional_decimal(self.target_level),
            "target_percentage": decimal_to_primitive(self.target_percentage),
            "unavailable_reason": self.unavailable_reason,
        }


class TargetStopOutcomeLabeler:
    """Capture the exact daily ranges needed for target/stop classification."""

    name = "future_target_stop_path"
    implementation_version = "1"
    result_schema_version = "1"
    required_market_fields = ("close", "high", "low")

    def __init__(
        self,
        horizon_sessions: int,
        target_percentage: Decimal,
        stop_percentage: Decimal,
        same_session_conflict_policy: SameSessionConflictPolicy = (
            SameSessionConflictPolicy.AMBIGUOUS
        ),
    ) -> None:
        _validate_horizon(horizon_sessions)
        self.required_future_sessions = horizon_sessions
        self.target_percentage = _positive_percentage(
            "target_percentage", target_percentage
        )
        self.stop_percentage = _positive_percentage("stop_percentage", stop_percentage)
        policy_value = cast(object, same_session_conflict_policy)
        if not isinstance(policy_value, SameSessionConflictPolicy):
            raise InvalidPredictionConfigurationError(
                "same-session conflict policy is unsupported"
            )
        self.same_session_conflict_policy = policy_value

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    def configuration(self) -> PrimitiveMapping:
        return _labeler_configuration(
            self.name,
            self.implementation_version,
            self.result_schema_version,
            self.required_market_fields,
            {
                "future_sessions": self.required_future_sessions,
                "same_session_conflict_policy": (
                    self.same_session_conflict_policy.value
                ),
                "stop_percentage": decimal_to_primitive(self.stop_percentage),
                "target_percentage": decimal_to_primitive(self.target_percentage),
            },
        )

    def validate_dataset(self, dataset: MarketDataset) -> None:
        _validate_price_basis(dataset)

    def label(
        self, dataset: MarketDataset, signal_session: date
    ) -> OutcomeLabel[TargetStopPathValues] | None:
        signal_index = _signal_index(dataset, signal_session)
        outcome_index = signal_index + self.required_future_sessions
        if outcome_index >= len(dataset.bars):
            return None
        future_bars = dataset.bars[signal_index + 1 : outcome_index + 1]
        return OutcomeLabel(
            signal_session,
            dataset.bars[outcome_index].session_date,
            TargetStopPathValues(
                self.required_future_sessions,
                dataset.bars[signal_index].close,
                tuple(_future_range(bar) for bar in future_bars),
            ),
        )


class TargetStopEvaluator:
    """Classify target/stop order without guessing intraday daily-bar order."""

    name = "directional_target_stop_evaluator"
    implementation_version = "1"
    result_schema_version = "1"

    def __init__(
        self,
        target_percentage: Decimal,
        stop_percentage: Decimal,
        same_session_conflict_policy: SameSessionConflictPolicy = (
            SameSessionConflictPolicy.AMBIGUOUS
        ),
    ) -> None:
        self.target_percentage = _positive_percentage(
            "target_percentage", target_percentage
        )
        self.stop_percentage = _positive_percentage("stop_percentage", stop_percentage)
        policy_value = cast(object, same_session_conflict_policy)
        if not isinstance(policy_value, SameSessionConflictPolicy):
            raise InvalidPredictionConfigurationError(
                "same-session conflict policy is unsupported"
            )
        self.same_session_conflict_policy = policy_value

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    def configuration(self) -> PrimitiveMapping:
        return _evaluator_configuration(
            self.name,
            self.implementation_version,
            self.result_schema_version,
            {
                "same_session_conflict_policy": (
                    self.same_session_conflict_policy.value
                ),
                "stop_percentage": decimal_to_primitive(self.stop_percentage),
                "target_percentage": decimal_to_primitive(self.target_percentage),
                "threshold_touch": "inclusive",
            },
        )

    def evaluate(
        self,
        signal: SignalFeatureCandidate,
        outcome: PredictionOutcome[TargetStopPathValues],
    ) -> TargetStopEvaluationValues:
        values = outcome.values
        if signal.direction is None:
            return self._result(
                values,
                TargetStopLabel.UNAVAILABLE,
                None,
                None,
                None,
                None,
                None,
                None,
                "candidate_direction_unavailable",
            )
        try:
            with arithmetic():
                if signal.direction is PredictionDirection.UP:
                    target_level = values.reference_price * (
                        Decimal(1) + self.target_percentage
                    )
                    stop_level = values.reference_price * (
                        Decimal(1) - self.stop_percentage
                    )
                else:
                    target_level = values.reference_price * (
                        Decimal(1) - self.target_percentage
                    )
                    stop_level = values.reference_price * (
                        Decimal(1) + self.stop_percentage
                    )
        except DecimalException as error:
            raise InvalidPredictionOutputError(
                "target/stop level arithmetic failed under its configured policy"
            ) from error

        for future_range in values.future_ranges:
            if signal.direction is PredictionDirection.UP:
                target_touched = future_range.high >= target_level
                stop_touched = future_range.low <= stop_level
            else:
                target_touched = future_range.low <= target_level
                stop_touched = future_range.high >= stop_level
            if target_touched and stop_touched:
                label = (
                    TargetStopLabel.BOTH_SAME_SESSION
                    if self.same_session_conflict_policy
                    is SameSessionConflictPolicy.AMBIGUOUS
                    else TargetStopLabel.STOP_FIRST
                )
                return self._result(
                    values,
                    label,
                    target_level,
                    stop_level,
                    future_range.session,
                    future_range.session,
                    future_range.high,
                    future_range.low,
                    None,
                )
            if target_touched:
                return self._result(
                    values,
                    TargetStopLabel.TARGET_FIRST,
                    target_level,
                    stop_level,
                    future_range.session,
                    None,
                    None,
                    None,
                    None,
                )
            if stop_touched:
                return self._result(
                    values,
                    TargetStopLabel.STOP_FIRST,
                    target_level,
                    stop_level,
                    future_range.session,
                    None,
                    None,
                    None,
                    None,
                )
        return self._result(
            values,
            TargetStopLabel.NEITHER,
            target_level,
            stop_level,
            None,
            None,
            None,
            None,
            None,
        )

    def _result(
        self,
        values: TargetStopPathValues,
        label: TargetStopLabel,
        target_level: Decimal | None,
        stop_level: Decimal | None,
        event_session: date | None,
        ambiguous_session: date | None,
        ambiguous_high: Decimal | None,
        ambiguous_low: Decimal | None,
        unavailable_reason: str | None,
    ) -> TargetStopEvaluationValues:
        return TargetStopEvaluationValues(
            label is not TargetStopLabel.UNAVAILABLE,
            values.horizon_sessions,
            label,
            values.reference_price,
            self.target_percentage,
            self.stop_percentage,
            target_level,
            stop_level,
            event_session,
            ambiguous_session,
            ambiguous_high,
            ambiguous_low,
            self.same_session_conflict_policy,
            unavailable_reason,
        )


def _labeler_configuration(
    name: str,
    implementation_version: str,
    result_schema_version: str,
    required_market_fields: tuple[str, ...],
    parameters: PrimitiveMapping,
) -> PrimitiveMapping:
    return {
        "component_name": name,
        "component_type": "prediction_outcome_labeler",
        "contract_version": "1",
        "implementation_version": implementation_version,
        "parameters": parameters,
        "required_market_fields": list(required_market_fields),
        "result_schema_version": result_schema_version,
    }


def _evaluator_configuration(
    name: str,
    implementation_version: str,
    result_schema_version: str,
    parameters: PrimitiveMapping,
) -> PrimitiveMapping:
    return {
        "component_name": name,
        "component_type": "prediction_evaluator",
        "contract_version": "1",
        "implementation_version": implementation_version,
        "parameters": parameters,
        "result_schema_version": result_schema_version,
    }


def _validate_horizon(value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise InvalidPredictionConfigurationError(
            "future-session horizon must be a positive integer"
        )


def _positive_percentage(name: str, value: object) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise InvalidPredictionConfigurationError(f"{name} must be numeric") from error
    if not result.is_finite() or not Decimal(0) < result < Decimal(1):
        raise InvalidPredictionConfigurationError(
            f"{name} must be finite and strictly between zero and one"
        )
    return result


def _signal_index(dataset: MarketDataset, signal_session: date) -> int:
    indexes = {bar.session_date: index for index, bar in enumerate(dataset.bars)}
    index = indexes.get(signal_session)
    if index is None:
        raise InvalidPredictionOutputError(
            "outcome signal session is absent from the dataset"
        )
    return index


def _validate_price_basis(dataset: MarketDataset) -> None:
    metadata = dataset.metadata
    if metadata.adjustment_mode is AdjustmentMode.UNADJUSTED and metadata.split_count:
        raise InvalidPredictionDataError(
            "raw unadjusted datasets containing stock splits cannot produce "
            "multi-session price outcomes without a conversion policy"
        )


def _future_range(bar: DailyBar) -> FutureSessionRange:
    return FutureSessionRange(bar.session_date, bar.high, bar.low)


def _optional_decimal(value: Decimal | None) -> str | None:
    return None if value is None else decimal_to_primitive(value)


def _optional_date(value: date | None) -> str | None:
    return None if value is None else value.isoformat()
