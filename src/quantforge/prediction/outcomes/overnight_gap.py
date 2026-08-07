"""Next-session opening-gap outcome and directional evaluation."""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, DecimalException

from quantforge.configuration import (
    PrimitiveMapping,
    configuration_identity,
    decimal_to_primitive,
)
from quantforge.data.calendar import next_session_after
from quantforge.data.models import AdjustmentMode, MarketDataset
from quantforge.prediction._arithmetic import (
    DECIMAL_CAPITALS,
    DECIMAL_CLAMP,
    DECIMAL_EMAX,
    DECIMAL_EMIN,
    DECIMAL_PRECISION,
    DECIMAL_ROUNDING,
    DECIMAL_TRAPS,
    arithmetic,
)
from quantforge.prediction.base import PredictionStrategy
from quantforge.prediction.contracts import (
    OutcomeLabel,
    PredictionOutcome,
    PredictionStudy,
)
from quantforge.prediction.errors import (
    InvalidPredictionDataError,
    InvalidPredictionOutputError,
)
from quantforge.prediction.models import PredictionDirection, PredictionSignal


@dataclass(frozen=True, slots=True)
class NextSessionOpenGapValues:
    """Typed values produced by the concrete next-open gap labeler."""

    signal_close: Decimal
    next_open: Decimal
    overnight_gap_percentage: Decimal
    gap_size_percentage: Decimal

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "gap_size_percentage": decimal_to_primitive(self.gap_size_percentage),
            "next_open": decimal_to_primitive(self.next_open),
            "overnight_gap_percentage": decimal_to_primitive(
                self.overnight_gap_percentage
            ),
            "signal_close": decimal_to_primitive(self.signal_close),
        }


@dataclass(frozen=True, slots=True)
class OvernightGapDirectionEvaluationValues:
    """Classification values derived after a gap outcome already exists."""

    predicted_direction: PredictionDirection
    actual_direction: PredictionDirection | None
    signed_prediction_return: Decimal
    direction_correct: bool

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "actual_direction": (
                None if self.actual_direction is None else self.actual_direction.value
            ),
            "direction_correct": self.direction_correct,
            "predicted_direction": self.predicted_direction.value,
            "signed_prediction_return": decimal_to_primitive(
                self.signed_prediction_return
            ),
        }


class NextSessionOpenGapOutcomeLabeler:
    """Label a fixed signal from the immediate next exchange-session open."""

    name = "next_session_open_gap"
    implementation_version = "1"
    result_schema_version = "1"
    required_future_sessions = 1
    required_market_fields = ("close", "open")

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    def configuration(self) -> PrimitiveMapping:
        return {
            "component_name": self.name,
            "component_type": "prediction_outcome_labeler",
            "contract_version": "1",
            "implementation_version": self.implementation_version,
            "parameters": {
                "cash_dividend_label_policy": (
                    "observed_underlying_price_gap_includes_ex_dividend_effect"
                ),
                "future_sessions": self.required_future_sessions,
                "outcome_field": "open",
                "reference_field": "close",
                "stock_split_label_policy": ("reject_raw_unadjusted_split_datasets"),
            },
            "required_market_fields": list(self.required_market_fields),
            "result_schema_version": self.result_schema_version,
        }

    def validate_dataset(self, dataset: MarketDataset) -> None:
        metadata = dataset.metadata
        if (
            metadata.adjustment_mode is AdjustmentMode.UNADJUSTED
            and metadata.split_count > 0
        ):
            raise InvalidPredictionDataError(
                "raw unadjusted datasets containing stock splits are unsupported "
                "because a mechanical split must not be labeled as an overnight gap"
            )

    def label(
        self, dataset: MarketDataset, signal_session: date
    ) -> OutcomeLabel[NextSessionOpenGapValues] | None:
        bar_indexes = {
            bar.session_date: index for index, bar in enumerate(dataset.bars)
        }
        signal_index = bar_indexes.get(signal_session)
        if signal_index is None:
            raise InvalidPredictionOutputError(
                "outcome signal session is absent from the dataset"
            )
        if signal_index == len(dataset.bars) - 1:
            return None
        signal_bar = dataset.bars[signal_index]
        outcome_bar = dataset.bars[signal_index + 1]
        expected_outcome_session = _next_session(
            signal_session, dataset.metadata.calendar
        )
        if outcome_bar.session_date != expected_outcome_session:
            raise InvalidPredictionDataError(
                "a prediction outcome requires the immediate next exchange session"
            )
        try:
            with arithmetic():
                overnight_gap = outcome_bar.open / signal_bar.close - Decimal(1)
                gap_size = abs(overnight_gap)
        except DecimalException as error:
            raise InvalidPredictionOutputError(
                "prediction label arithmetic failed under its configured policy"
            ) from error
        return OutcomeLabel(
            signal_session,
            outcome_bar.session_date,
            NextSessionOpenGapValues(
                signal_bar.close,
                outcome_bar.open,
                overnight_gap,
                gap_size,
            ),
        )

    def legacy_analysis_configuration(self) -> PrimitiveMapping:
        """Preserve the public QF-11 v1 analysis identity and manifest exactly."""
        return {
            "outcome_timing": "immediate_next_exchange_session_open",
            "reference_price": "completed_signal_session_close",
            "flat_gap_policy": "incorrect",
            "cash_dividend_label_policy": (
                "observed_underlying_price_gap_includes_ex_dividend_effect"
            ),
            "stock_split_label_policy": "reject_raw_unadjusted_split_datasets",
            "label_arithmetic": {
                "decimal_precision": DECIMAL_PRECISION,
                "rounding": DECIMAL_ROUNDING,
                "decimal_emin": DECIMAL_EMIN,
                "decimal_emax": DECIMAL_EMAX,
                "capitals": DECIMAL_CAPITALS,
                "clamp": DECIMAL_CLAMP,
                "initial_flags": [],
                "traps": [signal.__name__ for signal in DECIMAL_TRAPS],
            },
        }


class OvernightGapDirectionEvaluator:
    """Compare a directional signal with an already-generated gap outcome."""

    name = "overnight_gap_direction_evaluator"
    implementation_version = "1"
    result_schema_version = "1"

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    def configuration(self) -> PrimitiveMapping:
        return {
            "component_name": self.name,
            "component_type": "prediction_evaluator",
            "contract_version": "1",
            "implementation_version": self.implementation_version,
            "parameters": {
                "flat_gap_policy": "incorrect",
                "prediction_type": "direction",
                "signed_value": "overnight_gap_percentage",
            },
            "result_schema_version": self.result_schema_version,
        }

    def evaluate(
        self,
        signal: PredictionSignal,
        outcome: PredictionOutcome[NextSessionOpenGapValues],
    ) -> OvernightGapDirectionEvaluationValues:
        gap = outcome.values.overnight_gap_percentage
        try:
            with arithmetic():
                signed_return = (
                    gap if signal.direction is PredictionDirection.UP else -gap
                )
        except DecimalException as error:
            raise InvalidPredictionOutputError(
                "prediction evaluation arithmetic failed under its configured policy"
            ) from error
        actual_direction = (
            PredictionDirection.UP
            if gap > 0
            else PredictionDirection.DOWN
            if gap < 0
            else None
        )
        return OvernightGapDirectionEvaluationValues(
            signal.direction,
            actual_direction,
            signed_return,
            signed_return > 0,
        )


def create_overnight_gap_prediction_study(
    strategy: PredictionStrategy,
) -> PredictionStudy[
    PredictionSignal,
    NextSessionOpenGapValues,
    OvernightGapDirectionEvaluationValues,
]:
    """Compose the existing QF-11 study from generic typed components."""
    return PredictionStudy[
        PredictionSignal,
        NextSessionOpenGapValues,
        OvernightGapDirectionEvaluationValues,
    ].create(
        strategy,
        NextSessionOpenGapOutcomeLabeler(),
        OvernightGapDirectionEvaluator(),
    )


def _next_session(signal_session: date, calendar: str) -> date:
    try:
        return next_session_after(signal_session, calendar)
    except Exception as error:
        raise InvalidPredictionDataError(
            f"cannot resolve next session for calendar: {calendar}"
        ) from error
