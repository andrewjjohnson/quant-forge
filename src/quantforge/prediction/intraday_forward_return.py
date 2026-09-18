"""Endpoint-only intraday returns through the QF-46/QF-48 outcome contract."""

from dataclasses import dataclass
from datetime import timedelta
from decimal import Decimal, DecimalException

from quantforge.configuration import (
    PrimitiveMapping,
    configuration_identity,
    decimal_to_primitive,
)
from quantforge.data import (
    AdjustmentBasis,
    IntradayBar,
    MarketDataset,
    TimeframeBarSeries,
)
from quantforge.prediction._arithmetic import arithmetic
from quantforge.prediction.contracts import (
    OutcomeLabel,
    PredictionOutcome,
    PredictionRecord,
)
from quantforge.prediction.errors import (
    InvalidPredictionConfigurationError,
    InvalidPredictionDataError,
    InvalidPredictionOutputError,
)
from quantforge.prediction.outcome_resolution import (
    OutcomeEvaluationRequest,
    OutcomeResolution,
    OutcomeResolutionStatus,
)
from quantforge.prediction.outcome_temporal import (
    ElapsedDurationHorizon,
    OutcomeTemporalConfiguration,
)

REFERENCE_PRICE_CONVENTION = "completed_decision_observation_close"
FUTURE_PRICE_CONVENTION = "resolved_completed_observation_close"


@dataclass(frozen=True, slots=True)
class IntradayForwardReturnValues:
    """Decimal ratio and exact endpoint evidence; unavailable is never zero."""

    resolution: OutcomeResolution
    reference_observation_id: str
    reference_price: Decimal
    outcome_price: Decimal | None
    raw_return: Decimal | None

    @property
    def available(self) -> bool:
        return self.resolution.available

    @property
    def status(self) -> OutcomeResolutionStatus:
        return self.resolution.status

    def to_primitive(self) -> PrimitiveMapping:
        reference = self.resolution.request.source_reference
        assert reference is not None
        return {
            **self.resolution.metadata_primitive(),
            "reference_observation_id": self.reference_observation_id,
            "reference_price": decimal_to_primitive(self.reference_price),
            "reference_price_convention": REFERENCE_PRICE_CONVENTION,
            "future_price_convention": FUTURE_PRICE_CONVENTION,
            "outcome_price": None
            if self.outcome_price is None
            else decimal_to_primitive(self.outcome_price),
            "raw_return": None
            if self.raw_return is None
            else decimal_to_primitive(self.raw_return),
            "source_reference": reference.to_primitive(include_feed_scope=True),
        }


@dataclass(frozen=True, slots=True)
class IntradayForwardReturnOutcomeLabeler:
    """Use exact decision close and the runner's QF-46-resolved future close.

    Both prices must belong to the same canonical source and exchange session,
    with symbol and adjustment basis matching the prediction dataset.
    A missing/developing decision observation is invalid input, not a substitute
    reference price. Future availability is exclusively the supplied resolution.
    """

    temporal_configuration: OutcomeTemporalConfiguration

    name = "intraday_forward_close_return"
    implementation_version = "2"
    result_schema_version = "1"
    required_market_fields = ("close",)

    def __post_init__(self) -> None:
        if not isinstance(self.temporal_configuration.horizon, ElapsedDurationHorizon):
            raise InvalidPredictionConfigurationError(
                "intraday forward returns require an elapsed-duration horizon"
            )

    @property
    def required_future_duration(self) -> timedelta:
        return self.temporal_configuration.required_future_duration

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    def configuration(self) -> PrimitiveMapping:
        return {
            "component_name": self.name,
            "component_type": "prediction_outcome_labeler",
            "contract_version": "1",
            "implementation_version": self.implementation_version,
            "result_schema_version": self.result_schema_version,
            "required_market_fields": list(self.required_market_fields),
            "temporal_configuration": self.temporal_configuration.to_primitive(),
            "parameters": {
                "reference_field": REFERENCE_PRICE_CONVENTION,
                "outcome_field": FUTURE_PRICE_CONVENTION,
                "return_formula": "outcome_close / reference_close - 1",
                "return_unit": "ratio",
                "price_basis": "same_session_same_canonical_source",
            },
        }

    def validate_dataset(self, dataset: MarketDataset) -> None:
        # QF-3 is validated metadata for this path. Neither price comes from its
        # daily bars; canonical intraday artifact construction validates prices.
        # Endpoint compatibility is checked in label_request, which has a source.
        del dataset

    def label_request(
        self,
        dataset: MarketDataset,
        request: OutcomeEvaluationRequest,
        *,
        source: TimeframeBarSeries,
        resolution: OutcomeResolution,
    ) -> OutcomeLabel[IntradayForwardReturnValues]:
        # Generic dispatch owns request/source/configuration validation and calls
        # the QF-46 resolver on full coverage before bounding the callback source.
        reference = next(
            (
                bar
                for bar in source.bars
                if isinstance(bar, IntradayBar)
                and bar.end_timestamp == request.anchor.decision_timestamp
                and bar.session_date == request.anchor.signal_session
                and bar.complete
            ),
            None,
        )
        if reference is None:
            raise InvalidPredictionDataError(
                "intraday forward return requires the exact completed decision "
                "observation in the outcome source"
            )
        future = resolution.observation
        metadata = dataset.metadata
        adjustment_basis = AdjustmentBasis(
            adjustment_mode=metadata.adjustment_mode,
            ohlc_basis=metadata.ohlc_basis,
            volume_basis=metadata.volume_basis,
            corporate_action_policy=metadata.corporate_action_policy,
            adjusted_fields_used=metadata.adjusted_fields_used,
        )
        # Direct studies and fixed-candidate replay need this check even when no
        # context provider validates the source against the prediction dataset.
        for endpoint in (reference, future):
            if endpoint is None:
                continue
            if endpoint.symbol != metadata.canonical_symbol:
                raise InvalidPredictionDataError(
                    "intraday forward return source symbol is incompatible with "
                    "the prediction dataset"
                )
            if endpoint.provenance.adjustment_basis != adjustment_basis:
                raise InvalidPredictionDataError(
                    "intraday forward return source adjustment basis is "
                    "incompatible with the prediction dataset"
                )
        raw_return = None
        if future is not None:
            try:
                with arithmetic():
                    raw_return = future.close / reference.close - Decimal(1)
            except DecimalException as error:
                raise InvalidPredictionOutputError(
                    "forward-return arithmetic failed under its configured policy"
                ) from error
        return OutcomeLabel(
            request.anchor.signal_session,
            request.anchor.signal_session,
            IntradayForwardReturnValues(
                resolution,
                reference.bar_id,
                reference.close,
                None if future is None else future.close,
                raw_return,
            ),
        )


class IntradayForwardReturnEvaluator:
    """Project the typed return and audit metadata without reading market data."""

    name = "intraday_forward_return_evaluator"
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
            "result_schema_version": self.result_schema_version,
            "parameters": {
                "evaluation": "identity_projection_of_intraday_forward_return_label"
            },
        }

    def evaluate(
        self,
        signal: PredictionRecord,
        outcome: PredictionOutcome[IntradayForwardReturnValues],
    ) -> IntradayForwardReturnValues:
        del signal
        return outcome.values
