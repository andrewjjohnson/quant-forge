"""Historical scheduling and collections of unchanged QF-11 decision results."""

from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import UTC, date, datetime
from typing import Protocol, cast
from zoneinfo import ZoneInfo

from quantforge.configuration import (
    Primitive,
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.data.calendar import calendar_has_intraday_recesses, expected_sessions
from quantforge.data.intraday_aggregation import intraday_session_windows
from quantforge.data.models import MarketDataset
from quantforge.data.multi_timeframe import (
    MultiTimeframeContext,
    MultiTimeframeContextError,
)
from quantforge.prediction.context import (
    PredictionContextError,
    PredictionContextRequirements,
    PredictionIndicatorOutputCache,
    RejectedPredictionContextError,
)
from quantforge.prediction.contracts import (
    EvaluationValuesT,
    OutcomeValuesT,
    PredictionRecord,
    PredictionRecordT,
    PredictionStudy,
    PredictionValues,
)
from quantforge.prediction.errors import (
    InvalidPredictionConfigurationError,
    InvalidPredictionOutputError,
)
from quantforge.prediction.study import (
    STUDY_ENGINE_VERSION,
    PredictionStudyConfiguration,
    PredictionStudyDatasetSession,
    PredictionStudyResult,
    _capture_study_configuration,  # pyright: ignore[reportPrivateUsage]
    _prediction_record_primitive,  # pyright: ignore[reportPrivateUsage]
    prepare_prediction_study_dataset,
    run_prediction_study_in_session,
)
from quantforge.timeframes import (
    BarCompletion,
    CrossSessionPolicy,
    DevelopingBarExposure,
    IntradayInterval,
    SessionScope,
    Timeframe,
)

PREDICTION_WINDOW_SCHEMA_VERSION = "1"
PREDICTION_WINDOW_ENGINE_VERSION = "4"


def _utc(timestamp: datetime) -> datetime:
    if (
        not isinstance(cast(object, timestamp), datetime)
        or timestamp.utcoffset() is None
    ):
        raise InvalidPredictionConfigurationError(
            "historical decision boundaries must be timezone-aware timestamps"
        )
    return timestamp.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class PredictionDecisionSchedule:
    """Every primary bar end in a closed UTC interval, including missing bars.

    Calendar windows, not observed rows, determine eligibility. Completed leading
    and terminal partial-duration bars count; labels never advance availability.
    """

    primary_timeframe: Timeframe
    start_timestamp: datetime
    end_timestamp: datetime
    decision_timestamps: tuple[datetime, ...] = field(init=False)
    decision_sessions: tuple[date, ...] = field(init=False)

    def __post_init__(self) -> None:
        start, end = _utc(self.start_timestamp), _utc(self.end_timestamp)
        timeframe = self.primary_timeframe
        if (
            not isinstance(cast(object, timeframe), Timeframe)
            or not isinstance(timeframe.interval, IntradayInterval)
            or timeframe.interval.cross_session_policy
            is not CrossSessionPolicy.PROHIBITED
            or timeframe.developing_bar_exposure is not DevelopingBarExposure.EXCLUDE
        ):
            raise InvalidPredictionConfigurationError(
                "historical schedules require a completed intraday primary timeframe "
                "with prohibited cross-session bars"
            )
        if start > end:
            raise InvalidPredictionConfigurationError(
                "historical interval start must not follow its end"
            )
        if calendar_has_intraday_recesses(timeframe.session_policy.calendar_name):
            raise InvalidPredictionConfigurationError(
                "historical schedules do not support calendars with intraday recesses: "
                f"{timeframe.session_policy.calendar_name}"
            )
        timezone = ZoneInfo(timeframe.session_policy.timezone_name)
        sessions = expected_sessions(
            start.astimezone(timezone).date(),
            end.astimezone(timezone).date(),
            timeframe.session_policy.calendar_name,
            include_overnight=(
                timeframe.session_policy.scope is SessionScope.REGULAR_HOURS
            ),
        )
        decisions = tuple(
            (window.end_timestamp, session)
            for session in sessions
            for window in intraday_session_windows(session, timeframe)
            if start <= window.end_timestamp <= end
        )
        object.__setattr__(self, "start_timestamp", start)
        object.__setattr__(self, "end_timestamp", end)
        object.__setattr__(
            self, "decision_timestamps", tuple(timestamp for timestamp, _ in decisions)
        )
        object.__setattr__(
            self, "decision_sessions", tuple(session for _, session in decisions)
        )

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "schema_version": PREDICTION_WINDOW_SCHEMA_VERSION,
            "policy": "every_completed_primary_bar_end_closed_interval",
            "primary_timeframe": self.primary_timeframe.to_primitive(),
            "start_timestamp": self.start_timestamp.isoformat(),
            "end_timestamp": self.end_timestamp.isoformat(),
            "decision_timestamps": [
                item.isoformat() for item in self.decision_timestamps
            ],
        }

    @property
    def schedule_id(self) -> str:
        return configuration_identity(self.to_primitive())


class PredictionWindowContextProvider(Protocol):
    """Resolve a QF-20/QF-21 context independently at the requested UTC instant."""

    def get_context_at(
        self, requirements: PredictionContextRequirements, *, as_of: datetime
    ) -> MultiTimeframeContext: ...


@dataclass(frozen=True, slots=True)
class _DecisionContextProvider:
    provider: PredictionWindowContextProvider
    decision_timestamp: datetime
    dataset_family_fingerprint: str

    def get_context(
        self, requirements: PredictionContextRequirements
    ) -> MultiTimeframeContext:
        context = self.provider.get_context_at(
            requirements, as_of=self.decision_timestamp
        )
        if not isinstance(cast(object, context), MultiTimeframeContext):
            raise PredictionContextError(
                "historical provider returned an invalid context"
            )
        try:
            if context.as_of != self.decision_timestamp:
                raise PredictionContextError(
                    "historical context has the wrong decision timestamp"
                )
            if context.source_consistency.family_id != self.dataset_family_fingerprint:
                raise PredictionContextError(
                    "historical context has the wrong dataset family"
                )
            # QF-28 allows an older primary bar in general. A scheduled bar-end
            # decision requires this exact bar, even without a freshness limit.
            primary = context.latest_bar_for(requirements.primary.timeframe)
            if (
                primary.end_timestamp != self.decision_timestamp
                or primary.completion is BarCompletion.DEVELOPING
            ):
                raise PredictionContextError(
                    "historical context is missing the scheduled primary bar"
                )
        except (PredictionContextError, MultiTimeframeContextError) as error:
            # The runner has not received this object yet. Carry it into its
            # existing skip manifest/identity path without exposing it to a rule.
            raise RejectedPredictionContextError(
                str(error), source_context=context
            ) from error
        return context


@dataclass(frozen=True, slots=True)
class PredictionWindowDecision[
    PredictionRecordT: PredictionRecord,
    OutcomeValuesT: PredictionValues,
    EvaluationValuesT: PredictionValues,
]:
    """One original QF-11 result, with its timestamp even when context is skipped."""

    decision_timestamp: datetime
    result: PredictionStudyResult[PredictionRecordT, OutcomeValuesT, EvaluationValuesT]
    snapshot: PrimitiveMappingSnapshot = field(init=False, repr=False)

    def __post_init__(self) -> None:
        timestamp = _utc(self.decision_timestamp)
        context_snapshot = self.result.prediction_context_snapshot
        if context_snapshot is None:
            raise InvalidPredictionOutputError(
                "historical results require QF-28 context evidence"
            )
        context = context_snapshot.to_primitive()
        skipped = context.get("status") == "skipped"
        source = context.get("source_context")
        source_context = source if isinstance(source, dict) else context
        # A skipped source may itself have the wrong as-of timestamp. Preserve
        # that rejected snapshot alongside the requested decision timestamp.
        if not skipped and source_context.get("as_of") != timestamp.isoformat():
            raise InvalidPredictionOutputError(
                "historical result decision timestamp changed"
            )
        signal_records = [
            _prediction_record_primitive(signal) for signal in self.result.signals
        ]
        snapshot = PrimitiveMappingSnapshot.capture(
            {
                "decision_timestamp": timestamp.isoformat(),
                "prediction_study_id": self.result.study_id,
                "context_id": source_context.get("context_id"),
                "status": "skipped"
                if skipped
                else "no_prediction"
                if not signal_records
                else "evaluated",
                "prediction_study": self.result.to_primitive(),
                # QF-11's serialized rows omit unlabeled end-of-data signals. Preserve
                # those fixed signals separately without changing the QF-11 schema.
                "generated_signals": cast(list[Primitive], signal_records),
            }
        )
        object.__setattr__(self, "decision_timestamp", timestamp)
        object.__setattr__(self, "snapshot", snapshot)

    @property
    def context_id(self) -> str | None:
        return cast(str | None, self.snapshot.to_primitive()["context_id"])

    def to_primitive(self) -> PrimitiveMapping:
        return self.snapshot.to_primitive()


def _window_record_counts(decisions: list[PrimitiveMapping]) -> PrimitiveMapping:
    """Derive window totals from complete, validated decision snapshots."""
    dispositions: dict[str, int] = {}
    generated_predictions = 0
    unavailable_outcomes = 0
    for decision in decisions:
        signals = cast(list[PrimitiveMapping], decision["generated_signals"])
        study = cast(PrimitiveMapping, decision["prediction_study"])
        rows = cast(list[PrimitiveMapping], study["rows"])
        generated_predictions += len(signals)
        unavailable_outcomes += len(signals) - len(rows)
        for signal in signals:
            prediction = cast(PrimitiveMapping, signal["prediction"])
            values = cast(PrimitiveMapping, prediction["values"])
            disposition = values.get("disposition", "unclassified")
            key = disposition if isinstance(disposition, str) else "unclassified"
            dispositions[key] = dispositions.get(key, 0) + 1
    return {
        "scheduled_decisions": len(decisions),
        "valid_decisions": sum(item["status"] != "skipped" for item in decisions),
        "skipped_decisions": sum(item["status"] == "skipped" for item in decisions),
        "no_prediction_decisions": sum(
            item["status"] == "no_prediction" for item in decisions
        ),
        "generated_predictions": generated_predictions,
        "unavailable_outcomes": unavailable_outcomes,
        "signal_dispositions": dict(sorted(dispositions.items())),
    }


@dataclass(frozen=True, slots=True)
class PredictionWindowResult[
    PredictionRecordT: PredictionRecord,
    OutcomeValuesT: PredictionValues,
    EvaluationValuesT: PredictionValues,
]:
    """Complete ordered historical collection; never a synthetic QF-11 result."""

    schedule: PredictionDecisionSchedule
    identity_snapshot: PrimitiveMappingSnapshot
    decisions: tuple[
        PredictionWindowDecision[PredictionRecordT, OutcomeValuesT, EvaluationValuesT],
        ...,
    ]

    def __post_init__(self) -> None:
        if (
            tuple(item.decision_timestamp for item in self.decisions)
            != self.schedule.decision_timestamps
        ):
            raise InvalidPredictionOutputError(
                "historical window is incomplete or out of order"
            )
        identity = self.identity_snapshot.to_primitive()
        if identity.get("schedule") != self.schedule.to_primitive():
            raise InvalidPredictionOutputError(
                "historical window schedule identity is incompatible"
            )
        if any(
            item.result.configuration.to_primitive() != identity.get("configuration")
            or item.result.market_data.to_primitive() != identity.get("market_data")
            or item.result.engine_version != identity.get("prediction_engine_version")
            for item in self.decisions
        ):
            raise InvalidPredictionOutputError(
                "historical decisions do not match the window configuration or dataset"
            )

    @property
    def window_id(self) -> str:
        """Configuration identity, known before any decision executes."""
        return configuration_identity(self.identity_snapshot.to_primitive())

    @property
    def window_result_id(self) -> str:
        return configuration_identity(
            {
                "window_id": self.window_id,
                "decisions": [item.to_primitive() for item in self.decisions],
            }
        )

    @property
    def results(
        self,
    ) -> tuple[
        PredictionStudyResult[PredictionRecordT, OutcomeValuesT, EvaluationValuesT], ...
    ]:
        return tuple(item.result for item in self.decisions)

    def counts_primitive(self) -> PrimitiveMapping:
        return _window_record_counts([item.to_primitive() for item in self.decisions])

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "manifest": {
                **self.identity_snapshot.to_primitive(),
                "window_id": self.window_id,
                "window_result_id": self.window_result_id,
                "schedule_id": self.schedule.schedule_id,
                "record_counts": self.counts_primitive(),
            },
            "decisions": [item.to_primitive() for item in self.decisions],
        }

    def serialize(self) -> bytes:
        return PrimitiveMappingSnapshot.capture(
            self.to_primitive()
        ).canonical_json.encode()


def run_prediction_window(
    dataset: MarketDataset,
    study: PredictionStudy[PredictionRecordT, OutcomeValuesT, EvaluationValuesT],
    *,
    schedule: PredictionDecisionSchedule,
    context_provider: PredictionWindowContextProvider,
    dataset_family_fingerprint: str,
    context_environment: PrimitiveMapping,
    indicator_backend_environment: PrimitiveMapping | None = None,
    indicator_output_cache: PredictionIndicatorOutputCache | None = None,
) -> PredictionWindowResult[PredictionRecordT, OutcomeValuesT, EvaluationValuesT]:
    """Evaluate the same QF-11 configuration independently at every scheduled end."""
    return run_prediction_window_in_session(
        prepare_prediction_study_dataset(dataset),
        study,
        schedule=schedule,
        context_provider=context_provider,
        dataset_family_fingerprint=dataset_family_fingerprint,
        context_environment=context_environment,
        indicator_backend_environment=indicator_backend_environment,
        indicator_output_cache=indicator_output_cache,
    )


def _capture_window_identity(
    prepared: PredictionStudyDatasetSession,
    configuration: PredictionStudyConfiguration,
    *,
    schedule: PredictionDecisionSchedule,
    dataset_family_fingerprint: str,
    context_environment: PrimitiveMapping,
    indicator_backend_environment: PrimitiveMapping | None,
) -> PrimitiveMappingSnapshot:
    """Share the scientific identity between execution and offline grid validation."""
    return PrimitiveMappingSnapshot.capture(
        {
            "component": "quantforge_prediction_window",
            "schema_version": PREDICTION_WINDOW_SCHEMA_VERSION,
            "engine_version": PREDICTION_WINDOW_ENGINE_VERSION,
            "prediction_engine_version": STUDY_ENGINE_VERSION,
            "schedule": schedule.to_primitive(),
            "market_data": prepared.market_data.to_primitive(),
            "configuration": configuration.to_primitive(),
            "dataset_family_fingerprint": dataset_family_fingerprint,
            "context_environment": context_environment,
            "indicator_backend_environment": indicator_backend_environment,
        }
    )


def run_prediction_window_in_session(
    prepared: PredictionStudyDatasetSession,
    study: PredictionStudy[PredictionRecordT, OutcomeValuesT, EvaluationValuesT],
    *,
    schedule: PredictionDecisionSchedule,
    context_provider: PredictionWindowContextProvider,
    dataset_family_fingerprint: str,
    context_environment: PrimitiveMapping,
    indicator_backend_environment: PrimitiveMapping | None = None,
    indicator_output_cache: PredictionIndicatorOutputCache | None = None,
) -> PredictionWindowResult[PredictionRecordT, OutcomeValuesT, EvaluationValuesT]:
    """Reuse one validated dataset session, with isolated components per decision.

    A FAIL policy aborts without returning a partial window. QF-32 persists the
    candidate failure; an interrupted window is rerun as a whole on resume.
    """
    identity = _capture_window_identity(
        prepared,
        _capture_study_configuration(study),
        schedule=schedule,
        dataset_family_fingerprint=dataset_family_fingerprint,
        context_environment=context_environment,
        indicator_backend_environment=indicator_backend_environment,
    )
    return PredictionWindowResult(
        schedule,
        identity,
        tuple(
            iter_prediction_window_decisions(
                prepared,
                study,
                schedule=schedule,
                context_provider=context_provider,
                dataset_family_fingerprint=dataset_family_fingerprint,
                indicator_output_cache=indicator_output_cache,
            )
        ),
    )


def iter_prediction_window_decisions(
    prepared: PredictionStudyDatasetSession,
    study: PredictionStudy[PredictionRecordT, OutcomeValuesT, EvaluationValuesT],
    *,
    schedule: PredictionDecisionSchedule,
    context_provider: PredictionWindowContextProvider,
    dataset_family_fingerprint: str,
    indicator_output_cache: PredictionIndicatorOutputCache | None = None,
    start_sequence: int = 0,
) -> Iterator[
    PredictionWindowDecision[PredictionRecordT, OutcomeValuesT, EvaluationValuesT]
]:
    """Execute an authoritative schedule suffix, retaining only the current result.

    start_sequence is supplied only after the caller verifies a durable prefix.
    Scientific execution, component isolation and QF-11 validation are shared
    with the legacy materialized API.
    """
    if type(start_sequence) is not int or not 0 <= start_sequence <= len(
        schedule.decision_timestamps
    ):
        raise InvalidPredictionConfigurationError("invalid historical start sequence")
    requirements = getattr(study.strategy, "context_requirements", None)
    if (
        not isinstance(requirements, PredictionContextRequirements)
        or requirements.primary.timeframe != schedule.primary_timeframe
        or not dataset_family_fingerprint
        or not callable(getattr(context_provider, "get_context_at", None))
    ):
        raise InvalidPredictionConfigurationError(
            "historical execution requires matching QF-28 primary semantics, "
            "a dataset family, and a timestamp-aware context provider"
        )
    configuration = _capture_study_configuration(study)
    # Freeze a pristine template before any future-bearing labeler callback. A
    # previous decision's component state must never influence the next decision.
    template = deepcopy(study)
    if (
        template.strategy is study.strategy
        or template.outcome_labeler is study.outcome_labeler
        or template.evaluator is study.evaluator
    ):
        raise InvalidPredictionConfigurationError(
            "historical study components must support independent copies"
        )
    for sequence in range(start_sequence, len(schedule.decision_timestamps)):
        timestamp = schedule.decision_timestamps[sequence]
        decision_study = deepcopy(template)
        if (
            decision_study.strategy is template.strategy
            or decision_study.outcome_labeler is template.outcome_labeler
            or decision_study.evaluator is template.evaluator
            or _capture_study_configuration(decision_study) != configuration
        ):
            raise InvalidPredictionConfigurationError(
                "historical study copy changed configuration or retained "
                "component state"
            )
        result = run_prediction_study_in_session(
            # Labelers are new component instances per decision. Do not share
            # QF-11's object-id validation memo across their separate lifetimes.
            replace(prepared, validated_labelers=set()),
            decision_study,
            context_provider=_DecisionContextProvider(
                context_provider, timestamp, dataset_family_fingerprint
            ),
            indicator_output_cache=indicator_output_cache,
        )
        if result.configuration != configuration:
            raise InvalidPredictionOutputError(
                "historical decision changed the study configuration"
            )
        yield PredictionWindowDecision(timestamp, result)
        # Drop the generator frame's reference before starting another decision.
        del result, decision_study


__all__ = [
    "PREDICTION_WINDOW_ENGINE_VERSION",
    "PREDICTION_WINDOW_SCHEMA_VERSION",
    "PredictionDecisionSchedule",
    "PredictionWindowContextProvider",
    "PredictionWindowDecision",
    "PredictionWindowResult",
    "iter_prediction_window_decisions",
    "run_prediction_window",
    "run_prediction_window_in_session",
]
