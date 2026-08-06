"""Immutable prediction signals, forward labels, metrics, and results."""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import StrEnum
from typing import cast

from quantforge.configuration import (
    Primitive,
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    PrimitiveScalar,
    decimal_to_primitive,
)
from quantforge.data.models import DatasetMetadata
from quantforge.prediction.errors import InvalidPredictionOutputError


class PredictionDirection(StrEnum):
    """Direction predicted for the next session's open versus the current close."""

    UP = "up"
    DOWN = "down"


@dataclass(frozen=True, slots=True)
class PredictionParameter:
    """One stable primitive strategy parameter captured with a signal."""

    name: str
    value: PrimitiveScalar

    def __post_init__(self) -> None:
        if not self.name:
            raise InvalidPredictionOutputError(
                "prediction parameter names must not be empty"
            )


@dataclass(frozen=True, slots=True)
class PredictionFeature:
    """One finite contemporaneous feature available after the signal close."""

    name: str
    value: Decimal

    def __post_init__(self) -> None:
        if not self.name or not self.value.is_finite():
            raise InvalidPredictionOutputError(
                "prediction features require a name and finite decimal value"
            )


@dataclass(frozen=True, slots=True)
class PredictionSignal:
    """A direction guess containing no forward-looking price or outcome label."""

    symbol: str
    signal_session: date
    direction: PredictionDirection
    strategy_id: str
    strategy_implementation_version: str
    strategy_configuration_id: str
    strategy_parameters: tuple[PredictionParameter, ...]
    reason: str
    feature_values: tuple[PredictionFeature, ...]

    def __post_init__(self) -> None:
        if (
            not self.symbol
            or not self.strategy_id
            or not self.strategy_implementation_version
            or not self.strategy_configuration_id
            or not self.reason
        ):
            raise InvalidPredictionOutputError(
                "prediction signal identity and reason are required"
            )
        direction = cast(object, self.direction)
        if not isinstance(direction, PredictionDirection):
            raise InvalidPredictionOutputError(
                "prediction signal direction is unsupported"
            )
        parameter_names = tuple(item.name for item in self.strategy_parameters)
        feature_names = tuple(item.name for item in self.feature_values)
        if parameter_names != tuple(sorted(parameter_names)) or len(
            parameter_names
        ) != len(set(parameter_names)):
            raise InvalidPredictionOutputError(
                "prediction parameters must be sorted and unique"
            )
        if feature_names != tuple(sorted(feature_names)) or len(feature_names) != len(
            set(feature_names)
        ):
            raise InvalidPredictionOutputError(
                "prediction features must be sorted and unique"
            )

    def parameters_primitive(self) -> PrimitiveMapping:
        return {item.name: item.value for item in self.strategy_parameters}

    def features_primitive(self) -> PrimitiveMapping:
        return {
            item.name: decimal_to_primitive(item.value) for item in self.feature_values
        }


@dataclass(frozen=True, slots=True)
class PredictionStrategyOutput:
    """Ordered causal signals produced without inspecting outcome prices."""

    strategy_id: str
    strategy_configuration_id: str
    dataset_id: str
    signals: tuple[PredictionSignal, ...]
    contract_version: str = "1"


@dataclass(frozen=True, slots=True)
class PredictionMarketData:
    """QF-3 provenance required to reproduce a prediction analysis."""

    dataset_id: str
    bars_fingerprint: str
    schema_version: str
    symbol: str
    provider_name: str
    requested_start: date
    requested_end: date
    actual_first_session: date
    actual_last_session: date
    calendar: str
    adjustment_mode: str
    bar_count: int
    corporate_actions_complete: bool
    corporate_action_count: int
    dividend_count: int
    split_count: int
    corporate_action_snapshot_id: str
    ohlc_basis: str

    @classmethod
    def from_qf3(cls, metadata: DatasetMetadata) -> "PredictionMarketData":
        return cls(
            dataset_id=metadata.dataset_id,
            bars_fingerprint=metadata.data_sha256,
            schema_version=metadata.schema_version,
            symbol=metadata.canonical_symbol,
            provider_name=metadata.provider_name,
            requested_start=metadata.requested_start,
            requested_end=metadata.requested_end,
            actual_first_session=metadata.actual_first_session,
            actual_last_session=metadata.actual_last_session,
            calendar=metadata.calendar,
            adjustment_mode=metadata.adjustment_mode.value,
            bar_count=metadata.bar_count,
            corporate_actions_complete=metadata.corporate_actions_complete,
            corporate_action_count=metadata.corporate_action_count,
            dividend_count=metadata.dividend_count,
            split_count=metadata.split_count,
            corporate_action_snapshot_id=metadata.corporate_action_snapshot_id,
            ohlc_basis=metadata.ohlc_basis,
        )

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "dataset_id": self.dataset_id,
            "bars_fingerprint": self.bars_fingerprint,
            "schema_version": self.schema_version,
            "symbol": self.symbol,
            "provider_name": self.provider_name,
            "requested_start": self.requested_start.isoformat(),
            "requested_end": self.requested_end.isoformat(),
            "actual_first_session": self.actual_first_session.isoformat(),
            "actual_last_session": self.actual_last_session.isoformat(),
            "calendar": self.calendar,
            "adjustment_mode": self.adjustment_mode,
            "bar_count": self.bar_count,
            "corporate_actions_complete": self.corporate_actions_complete,
            "corporate_action_count": self.corporate_action_count,
            "dividend_count": self.dividend_count,
            "split_count": self.split_count,
            "corporate_action_snapshot_id": self.corporate_action_snapshot_id,
            "ohlc_basis": self.ohlc_basis,
        }


@dataclass(frozen=True, slots=True)
class PredictionRow:
    """One causal signal paired later with its next-session opening outcome."""

    prediction_id: str
    dataset_id: str
    dataset_fingerprint: str
    symbol: str
    signal_session: date
    outcome_session: date
    direction: PredictionDirection
    strategy_id: str
    strategy_implementation_version: str
    strategy_configuration_id: str
    strategy_parameters: tuple[PredictionParameter, ...]
    reason: str
    feature_values: tuple[PredictionFeature, ...]
    signal_close: Decimal
    next_open: Decimal
    overnight_gap_percentage: Decimal
    gap_size_percentage: Decimal
    signed_prediction_return: Decimal
    correct: bool

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "prediction_id": self.prediction_id,
            "dataset_id": self.dataset_id,
            "dataset_fingerprint": self.dataset_fingerprint,
            "symbol": self.symbol,
            "signal_session": self.signal_session.isoformat(),
            "outcome_session": self.outcome_session.isoformat(),
            "direction": self.direction.value,
            "strategy_id": self.strategy_id,
            "strategy_implementation_version": (self.strategy_implementation_version),
            "strategy_configuration_id": self.strategy_configuration_id,
            "strategy_parameters": {
                item.name: item.value for item in self.strategy_parameters
            },
            "reason": self.reason,
            "feature_values": {
                item.name: decimal_to_primitive(item.value)
                for item in self.feature_values
            },
            "signal_close": decimal_to_primitive(self.signal_close),
            "next_open": decimal_to_primitive(self.next_open),
            "overnight_gap_percentage": decimal_to_primitive(
                self.overnight_gap_percentage
            ),
            "gap_size_percentage": decimal_to_primitive(self.gap_size_percentage),
            "signed_prediction_return": decimal_to_primitive(
                self.signed_prediction_return
            ),
            "correct": self.correct,
        }


@dataclass(frozen=True, slots=True)
class PredictionMetrics:
    """Direction accuracy and average gap magnitudes by outcome."""

    prediction_count: int
    correct_count: int
    incorrect_count: int
    accuracy: Decimal | None
    average_gap_size_correct: Decimal | None
    average_gap_size_incorrect: Decimal | None
    average_signed_return_correct: Decimal | None
    average_signed_return_incorrect: Decimal | None

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "prediction_count": self.prediction_count,
            "correct_count": self.correct_count,
            "incorrect_count": self.incorrect_count,
            "accuracy": _optional_decimal(self.accuracy),
            "average_gap_size_correct": _optional_decimal(
                self.average_gap_size_correct
            ),
            "average_gap_size_incorrect": _optional_decimal(
                self.average_gap_size_incorrect
            ),
            "average_signed_return_correct": _optional_decimal(
                self.average_signed_return_correct
            ),
            "average_signed_return_incorrect": _optional_decimal(
                self.average_signed_return_incorrect
            ),
        }


@dataclass(frozen=True, slots=True)
class PredictionAnalysisResult:
    """Complete deterministic analysis manifest and labeled prediction rows."""

    analysis_id: str
    engine_version: str
    result_schema_version: str
    market_data: PredictionMarketData
    strategy_id: str
    strategy_implementation_version: str
    strategy_configuration_id: str
    strategy_configuration_snapshot: PrimitiveMappingSnapshot
    strategy_warm_up_observations: int
    analysis_configuration_snapshot: PrimitiveMappingSnapshot
    generated_signal_count: int
    unlabeled_end_of_data_count: int
    rows: tuple[PredictionRow, ...]
    metrics: PredictionMetrics
    limitations: tuple[str, ...]

    @property
    def strategy_configuration(self) -> PrimitiveMapping:
        return self.strategy_configuration_snapshot.to_primitive()

    @property
    def analysis_configuration(self) -> PrimitiveMapping:
        return self.analysis_configuration_snapshot.to_primitive()

    def manifest_primitive(self) -> PrimitiveMapping:
        return {
            "analysis_id": self.analysis_id,
            "engine_version": self.engine_version,
            "result_schema_version": self.result_schema_version,
            "analysis_configuration": self.analysis_configuration,
            "market_data": self.market_data.to_primitive(),
            "strategy": {
                "strategy_id": self.strategy_id,
                "strategy_implementation_version": (
                    self.strategy_implementation_version
                ),
                "strategy_configuration_id": self.strategy_configuration_id,
                "configuration": self.strategy_configuration,
                "warm_up_observations": self.strategy_warm_up_observations,
            },
            "metrics": self.metrics.to_primitive(),
            "record_counts": {
                "generated_signals": self.generated_signal_count,
                "labeled_predictions": len(self.rows),
                "unlabeled_end_of_data": self.unlabeled_end_of_data_count,
            },
            "limitations": list(self.limitations),
        }

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "manifest": self.manifest_primitive(),
            "predictions": cast(
                list[Primitive], [row.to_primitive() for row in self.rows]
            ),
        }


def _optional_decimal(value: Decimal | None) -> str | None:
    return None if value is None else decimal_to_primitive(value)
