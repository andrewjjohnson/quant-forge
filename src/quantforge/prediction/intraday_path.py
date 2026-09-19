"""Complete canonical future paths on the existing QF-46 elapsed contract."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal

from quantforge.configuration import (
    PrimitiveMapping,
    configuration_identity,
    decimal_to_primitive,
)
from quantforge.data import IntradayBar, MarketDataset, TimeframeBarSeries
from quantforge.data.intraday_aggregation import intraday_session_windows
from quantforge.prediction._intraday_reference import (
    REFERENCE_PRICE_CONVENTION,
    completed_decision_reference,
    validate_intraday_price_basis,
)
from quantforge.prediction.contracts import OutcomeLabel
from quantforge.prediction.errors import InvalidPredictionConfigurationError
from quantforge.prediction.outcome_resolution import (
    OutcomeEvaluationRequest,
    OutcomeResolution,
    OutcomeResolutionStatus,
)
from quantforge.prediction.outcome_temporal import (
    ElapsedDurationHorizon,
    OutcomeTemporalConfiguration,
)


@dataclass(frozen=True, slots=True)
class IntradayPathRange:
    """One completed future bar; its end is observation time, not exact hit time."""

    start_timestamp: datetime
    end_timestamp: datetime
    observation_id: str
    high: Decimal
    low: Decimal

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "start_timestamp": self.start_timestamp.isoformat(),
            "end_timestamp": self.end_timestamp.isoformat(),
            "observation_id": self.observation_id,
            "high": decimal_to_primitive(self.high),
            "low": decimal_to_primitive(self.low),
        }


@dataclass(frozen=True, slots=True)
class IntradayPathValues:
    """QF-46 endpoint evidence plus stricter, separately certified path coverage."""

    resolution: OutcomeResolution
    reference_observation_id: str
    reference_price: Decimal
    status: OutcomeResolutionStatus
    missing_observation_timestamp: datetime | None
    future_ranges: tuple[IntradayPathRange, ...]

    @property
    def available(self) -> bool:
        return self.status is OutcomeResolutionStatus.AVAILABLE

    def metadata_primitive(self) -> PrimitiveMapping:
        request = self.resolution.request
        reference = request.source_reference
        assert reference is not None
        decision = request.anchor.decision_timestamp
        assert decision is not None
        endpoint = self.resolution.expected_observation_timestamp
        return {
            **self.resolution.metadata_primitive(),
            "endpoint_status": self.resolution.status.value,
            "endpoint_available": self.resolution.available,
            "status": self.status.value,
            "available": self.available,
            "reference_observation_id": self.reference_observation_id,
            "reference_price": decimal_to_primitive(self.reference_price),
            "reference_price_convention": REFERENCE_PRICE_CONVENTION,
            "source_reference": reference.to_primitive(include_feed_scope=True),
            "path_start_timestamp": decision.isoformat(),
            "path_end_timestamp": None if endpoint is None else endpoint.isoformat(),
            "missing_observation_timestamp": (
                None
                if self.missing_observation_timestamp is None
                else self.missing_observation_timestamp.isoformat()
            ),
            "unavailable_reason": None if self.available else self.status.value,
        }

    def to_primitive(self) -> PrimitiveMapping:
        return {
            **self.metadata_primitive(),
            "future_ranges": [bar.to_primitive() for bar in self.future_ranges],
        }


@dataclass(frozen=True, slots=True)
class IntradayPathOutcomeLabeler:
    """Capture a full path once for directional excursion or target/stop evaluation.

    The decision bar is excluded. Every expected window starting at/after its
    end and ending at/before QF-46's endpoint must be present and completed.
    Endpoint failures take precedence; an available endpoint alone never proves
    path completeness. No partial path is exposed as a successful label.
    """

    temporal_configuration: OutcomeTemporalConfiguration

    name = "intraday_complete_high_low_path"
    implementation_version = "1"
    result_schema_version = "1"
    required_market_fields = ("close", "high", "low")

    def __post_init__(self) -> None:
        if not isinstance(self.temporal_configuration.horizon, ElapsedDurationHorizon):
            raise InvalidPredictionConfigurationError(
                "intraday paths require an elapsed-duration horizon"
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
                "price_basis": "same_session_same_canonical_source",
                "path_interval": "decision_exclusive_resolved_endpoint_inclusive",
                "coverage_policy": "all_expected_completed_windows_required",
                "unavailable_policy": "endpoint_status_then_first_interior_failure",
            },
        }

    def validate_dataset(self, dataset: MarketDataset) -> None:
        # Canonical construction owns OHLC/calendar/lineage validation. QF-3
        # supplies metadata only, as in QF-49; compare each used bar below.
        del dataset

    def label_request(
        self,
        dataset: MarketDataset,
        request: OutcomeEvaluationRequest,
        *,
        source: TimeframeBarSeries,
        resolution: OutcomeResolution,
    ) -> OutcomeLabel[IntradayPathValues]:
        reference = completed_decision_reference(request, source)
        validate_intraday_price_basis(dataset, (reference,))
        status = resolution.status
        missing = None
        ranges: list[IntradayPathRange] = []
        if resolution.available:
            endpoint = resolution.expected_observation_timestamp
            assert endpoint is not None
            bars_by_start = {
                bar.start_timestamp: bar
                for bar in source.bars
                if isinstance(bar, IntradayBar)
                and bar.session_date == request.anchor.signal_session
                and reference.end_timestamp <= bar.start_timestamp
                and bar.end_timestamp <= endpoint
            }
            windows = (
                window
                for window in intraday_session_windows(
                    request.anchor.signal_session, source.timeframe
                )
                if reference.end_timestamp <= window.start_timestamp
                and window.end_timestamp <= endpoint
            )
            for window in windows:
                bar = bars_by_start.get(window.start_timestamp)
                if (
                    bar is None
                    or not bar.complete
                    or bar.end_timestamp != window.end_timestamp
                ):
                    status = (
                        OutcomeResolutionStatus.INCOMPLETE
                        if bar is not None and not bar.complete
                        else OutcomeResolutionStatus.MISSING_OBSERVATION
                    )
                    missing = window.end_timestamp
                    ranges.clear()
                    break
                validate_intraday_price_basis(dataset, (bar,))
                ranges.append(
                    IntradayPathRange(
                        bar.start_timestamp,
                        bar.end_timestamp,
                        bar.bar_id,
                        bar.high,
                        bar.low,
                    )
                )
        return OutcomeLabel(
            request.anchor.signal_session,
            request.anchor.signal_session,
            IntradayPathValues(
                resolution,
                reference.bar_id,
                reference.close,
                status,
                missing,
                tuple(ranges),
            ),
        )
