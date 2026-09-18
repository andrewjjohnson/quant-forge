"""Elapsed request wiring fixtures. Labels contain availability, never price math."""

from dataclasses import dataclass, replace
from datetime import UTC, datetime, time, timedelta
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.data import (
    DatasetFamily,
    DatasetLineage,
    MarketDataset,
    TimeframeBarSeries,
)
from quantforge.data.intraday_aggregation import intraday_session_windows
from quantforge.prediction import (
    PredictionContextRequirements,
    PredictionDecisionSchedule,
    PredictionSignal,
    PredictionStudy,
)
from quantforge.prediction.contracts import (
    OutcomeLabel,
    PredictionOutcome,
    PredictionRecord,
)
from quantforge.prediction.outcome_resolution import (
    OutcomeEvaluationRequest,
    OutcomeResolution,
)
from quantforge.prediction.outcome_temporal import OutcomeTemporalConfiguration
from quantforge.timeframes import IntradayInterval, Timeframe
from quantforge.validation import (
    DatasetProvenance,
    FinalHoldout,
    IndicatorComponent,
    IndicatorProvenance,
    OutcomeProvenance,
    PartitionRole,
    PredictionMembershipSource,
    PurgePolicy,
    ResearchRuleProvenance,
    TemporalOffset,
    TimeframeWarmUpRequirement,
    TimestampBoundary,
    ValidationFold,
    ValidationInterval,
    ValidationPlan,
    ValidationWindow,
)
from quantforge.walk_forward import PredictionEvaluator, WalkForwardConfig
from tests.unit.helpers import SESSIONS
from tests.unit.walk_forward.fixtures import (
    DAILY,
    StudyFactory,
    StudyRule,
    _adjustment_basis,  # pyright: ignore[reportPrivateUsage]
    _family,  # pyright: ignore[reportPrivateUsage]
    _intraday_bar,  # pyright: ignore[reportPrivateUsage]
    _session_bar,  # pyright: ignore[reportPrivateUsage]
    prediction_fixture,
)

PRIMARY = Timeframe.us_equity(IntradayInterval(timedelta(minutes=5)))


def instant(day: int, clock: str) -> datetime:
    return datetime.combine(
        SESSIONS[day], time.fromisoformat(clock), ZoneInfo("America/New_York")
    ).astimezone(UTC)


@dataclass(frozen=True)
class AvailabilityValues:
    available: bool
    status: str

    def to_primitive(self) -> PrimitiveMapping:
        return {"available": self.available, "status": self.status}


@dataclass(frozen=True)
class FixtureEvaluation:
    # Fixed fixture scoring exercises selection plumbing only.
    direction_correct: bool = True

    def to_primitive(self) -> PrimitiveMapping:
        return {"direction_correct": self.direction_correct}


@dataclass(frozen=True)
class MetadataLabeler:
    temporal: OutcomeTemporalConfiguration
    name = "qf48_metadata_only"
    implementation_version = "1"
    result_schema_version = "1"
    required_market_fields = ("close",)

    @property
    def required_future_duration(self) -> timedelta:
        return self.temporal.required_future_duration

    def configuration(self) -> PrimitiveMapping:
        return {
            "component_name": self.name,
            "implementation_version": "1",
            "temporal_configuration": self.temporal.to_primitive(),
            "required_market_fields": ["close"],
            "result_schema_version": "1",
        }

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    def validate_dataset(self, dataset: MarketDataset) -> None:
        assert dataset.bars

    def label_request(
        self,
        dataset: MarketDataset,
        request: OutcomeEvaluationRequest,
        *,
        source: TimeframeBarSeries,
        resolution: OutcomeResolution,
    ) -> OutcomeLabel[AvailabilityValues]:
        assert dataset.metadata.dataset_id == request.dataset_id
        decision = request.anchor.decision_timestamp
        assert decision is not None
        assert all(
            bar.end_timestamp <= decision + self.required_future_duration
            for bar in source.bars
        )
        assert resolution.request == request
        assert resolution.observation is None or resolution.observation in source.bars
        assert all(
            getattr(bar, "session_date", None) == request.anchor.signal_session
            for bar in source.bars
        )
        return OutcomeLabel(
            request.anchor.signal_session,
            request.anchor.signal_session,
            AvailabilityValues(resolution.available, resolution.status.value),
        )


class MetadataEvaluator:
    name = "qf48_fixture_score"
    implementation_version = "1"
    result_schema_version = "1"

    def configuration(self) -> PrimitiveMapping:
        return {"component_name": self.name, "implementation_version": "1"}

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    def evaluate(
        self, signal: PredictionRecord, outcome: PredictionOutcome[AvailabilityValues]
    ) -> FixtureEvaluation:
        assert signal.signal_session == outcome.signal_session
        return FixtureEvaluation()


class TimestampFactory(StudyFactory):
    def __init__(self, source: TimeframeBarSeries, horizon: timedelta) -> None:
        self.source = source
        self.horizon = horizon

    def build(self, parameters: PrimitiveMapping) -> PredictionStudy[Any, Any, Any]:
        session_study = super().build(parameters)
        requirements = cast(
            PredictionContextRequirements,
            getattr(session_study.strategy, "context_requirements"),
        )
        requirements = replace(
            requirements, primary=replace(requirements.primary, timeframe=PRIMARY)
        )
        return PredictionStudy[
            PredictionSignal, AvailabilityValues, FixtureEvaluation
        ].create(
            StudyRule(requirements),
            MetadataLabeler(
                OutcomeTemporalConfiguration.elapsed_duration(self.horizon, PRIMARY)
            ),
            MetadataEvaluator(),
            outcome_source=self.source,
        )


def timestamp_fixture(
    tmp_path: Path,
    *,
    embargo: timedelta = timedelta(0),
    horizon: timedelta = timedelta(minutes=30),
    source_id: str = "timestamp-source",
) -> tuple[WalkForwardConfig, PredictionEvaluator]:
    legacy, old_adapter = prediction_fixture(tmp_path)
    base = _family()
    family = DatasetFamily(
        canonical_symbol="SPY",
        provider_name=base.provider_name,
        feed_scope=base.feed_scope,
        adjustment_basis=_adjustment_basis(),
        aggregation_policy=base.aggregation_policy,
        canonical_source_snapshot_id=source_id,
        datasets=(
            DatasetLineage(
                source_id,
                PRIMARY,
                source_id,
                None,
                ("timestamp-daily",),
            ),
            DatasetLineage("timestamp-daily", DAILY, source_id, source_id),
        ),
    )
    bars = tuple(
        replace(
            _intraday_bar(
                PRIMARY,
                interval.start_timestamp,
                interval.end_timestamp,
                interval.completion,
            ),
            provenance=replace(
                _intraday_bar(
                    PRIMARY,
                    interval.start_timestamp,
                    interval.end_timestamp,
                    interval.completion,
                ).provenance,
                adjustment_basis=family.adjustment_basis,
            ),
        )
        for session in SESSIONS[:7]
        for interval in intraday_session_windows(session, PRIMARY)
    )
    source = TimeframeBarSeries._from_validated_artifact(  # pyright: ignore[reportPrivateUsage]
        family.reference(source_id),
        PRIMARY,
        bars,
        dataset_family_manifest_id=family.manifest_id,
    )
    daily = TimeframeBarSeries._from_validated_artifact(  # pyright: ignore[reportPrivateUsage]
        family.reference("timestamp-daily"),
        DAILY,
        tuple(_session_bar(DAILY, s) for s in SESSIONS[:7]),
        dataset_family_manifest_id=family.manifest_id,
    )
    factory = TimestampFactory(source, horizon)
    adapter = PredictionEvaluator(
        dataset=old_adapter.dataset,
        series=(source, daily),
        primary_timeframe=PRIMARY,
        study_factory=factory,
        analyzer=old_adapter.analyzer,
        indicator_backend=old_adapter.backend,
        grid_config=old_adapter.grid_config,
    )
    studies = [
        factory.build(candidate.parameters.to_primitive())
        for candidate in adapter.universe.candidates
    ]
    indicators = {
        (
            req.timeframe.configuration_id,
            indicator.configuration_id,
        ): IndicatorProvenance.capture(
            cast(IndicatorComponent, indicator.indicator), req.timeframe
        )
        for study in studies
        for req in cast(
            PredictionContextRequirements,
            getattr(study.strategy, "context_requirements"),
        ).all_timeframes
        for indicator in req.indicators
    }
    environment = replace(
        legacy.plan.environment,
        dataset=DatasetProvenance.from_dataset_family(
            family, (source_id, "timestamp-daily")
        ),
        timeframes=(PRIMARY, DAILY),
        research_rule=ResearchRuleProvenance.capture_prediction(studies[0].strategy),
        indicators=tuple(indicators.values()),
        outcomes=(OutcomeProvenance.capture_timestamp(studies[0].outcome_labeler),),
    )
    schedule = PredictionDecisionSchedule(
        PRIMARY, bars[0].end_timestamp, instant(5, "16:00")
    )
    membership = PredictionMembershipSource.capture(schedule, source)

    def window(
        name: str, role: PartitionRole, day: int, start: str, end: str
    ) -> ValidationWindow:
        return ValidationWindow(
            name,
            role,
            ValidationInterval(
                TimestampBoundary(instant(day, start)),
                TimestampBoundary(instant(day, end)),
            ),
            warm_up_by_timeframe=(
                TimeframeWarmUpRequirement(PRIMARY, 3),
                TimeframeWarmUpRequirement(DAILY, 3),
            ),
        )

    folds = tuple(
        ValidationFold(
            f"fold-{day}",
            window(f"dev-{day}", PartitionRole.DEVELOPMENT, day, "10:00", "11:55"),
            window(
                f"test-{day}", PartitionRole.WALK_FORWARD_TEST, day, "13:00", "13:55"
            ),
            window(f"select-{day}", PartitionRole.SELECTION, day, "12:00", "12:55"),
        )
        for day in (4, 5)
    )
    plan = ValidationPlan(
        "QF-48 timestamp fixture",
        environment,
        folds,
        FinalHoldout(
            window("reserved", PartitionRole.FINAL_HOLDOUT, 5, "14:00", "15:55"),
            "untouched",
        ),
        PurgePolicy(
            environment.outcomes[0].future_horizon, TemporalOffset.duration(embargo)
        ),
        legacy.plan.training_window_mode,
        prediction_membership=membership,
    )
    return WalkForwardConfig("QF-48 timestamp fixture", plan), adapter
