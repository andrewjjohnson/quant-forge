from datetime import date
from pathlib import Path
from typing import cast

import pytest

from quantforge.data import (
    SessionAggregationPolicy,
    TimeframeBarSeries,
    aggregate_session_dataset,
)
from quantforge.prediction import (
    InvalidPredictionConfigurationError,
    NextSessionOpenGapOutcomeLabeler,
    OvernightGapPredictionParameters,
    OvernightGapPredictionStrategy,
    run_prediction_study,
)
from quantforge.timeframes import SessionInterval, Timeframe
from quantforge.validation import (
    DatasetProvenance,
    ExchangeSessionBoundary,
    FinalHoldout,
    IndicatorComponent,
    IndicatorProvenance,
    OutcomeProvenance,
    PartitionRole,
    PurgePolicy,
    ResearchEnvironment,
    ResearchRuleProvenance,
    ResearchStudyType,
    TemporalOffset,
    TimestampBoundary,
    TrainingWindowMode,
    ValidationFold,
    ValidationInterval,
    ValidationPlan,
    ValidationPlanError,
    ValidationWindow,
    select_window_observations,
)
from tests.unit.data.test_multi_timeframe import (
    _family,  # pyright: ignore[reportPrivateUsage]
    _persisted_source_dataset,  # pyright: ignore[reportPrivateUsage]
)
from tests.unit.helpers import make_dataset
from tests.unit.prediction.test_study import (
    _study,  # pyright: ignore[reportPrivateUsage]
)


@pytest.mark.parametrize("horizon", [0, -1, True, False, 1.5, "1"])
def test_session_outcome_capture_rejects_horizons_rejected_by_runner(
    horizon: object,
) -> None:
    events: list[str] = []
    study = _study(events, horizon=cast(int, horizon))
    with pytest.raises(
        InvalidPredictionConfigurationError, match="positive future-session horizon"
    ):
        run_prediction_study(make_dataset(("100", "101", "102")), study)
    with pytest.raises(ValidationPlanError, match="positive integer"):
        OutcomeProvenance.capture_exchange_sessions(study.outcome_labeler)
    assert events == []


@pytest.mark.parametrize("horizon", [1, 2])
def test_session_outcome_capture_preserves_valid_runner_horizon(horizon: int) -> None:
    events: list[str] = []
    study = _study(events, horizon=horizon)
    outcome = OutcomeProvenance.capture_exchange_sessions(study.outcome_labeler)
    assert events == []
    result = run_prediction_study(make_dataset(("100", "101", "102")), study)
    assert outcome.future_horizon == TemporalOffset.sessions(
        result.configuration.required_future_sessions
    )
    assert outcome.configuration_id == study.outcome_labeler.configuration_id


def test_warm_up_consumes_existing_validated_source_and_derived_artifacts(
    tmp_path: Path,
) -> None:
    dataset, cache = _persisted_source_dataset(tmp_path)
    family = _family(source_dataset_id=dataset.metadata.dataset_id)
    series = TimeframeBarSeries.from_source_dataset(dataset, family=family, cache=cache)
    keys = tuple(TimestampBoundary(bar.end_timestamp) for bar in series.bars)
    window = ValidationWindow(
        "intraday_selection",
        PartitionRole.SELECTION,
        ValidationInterval(keys[19], keys[19]),
        19,
    )
    selected = select_window_observations(
        window, keys[:20], source=series, source_timeframe=series.timeframe
    )
    assert selected.warm_up_context == keys[:19]
    assert selected.study_observations == (keys[19],)
    assert selected.source_dataset_id == dataset.metadata.dataset_id
    assert selected.source_family_manifest_id == family.manifest_id
    assert (
        select_window_observations(
            window, keys, source=series, source_timeframe=series.timeframe
        ).selection_id
        == selected.selection_id
    )

    daily = Timeframe.us_equity(SessionInterval())
    derived = aggregate_session_dataset(
        dataset, daily, policy=SessionAggregationPolicy()
    )
    daily_series = TimeframeBarSeries.from_aggregated_session_dataset(derived)
    for boundary in (
        ExchangeSessionBoundary(derived.bars[0].session_dates[-1]),
        TimestampBoundary(derived.bars[0].end_timestamp),
    ):
        daily_window = ValidationWindow(
            "daily_selection",
            PartitionRole.SELECTION,
            ValidationInterval(boundary, boundary),
        )
        daily_selection = select_window_observations(
            daily_window, (boundary,), source=daily_series, source_timeframe=daily
        )
        assert daily_selection.source_dataset_id == derived.metadata.dataset_id
        assert (
            daily_selection.source_timeframe_configuration_id == daily.configuration_id
        )
        assert daily_selection.study_observations == (boundary,)


def _window(
    name: str,
    role: PartitionRole,
    start: date,
    end: date,
    warm_up_observations: int,
) -> ValidationWindow:
    return ValidationWindow(
        name,
        role,
        ValidationInterval(
            ExchangeSessionBoundary(start),
            ExchangeSessionBoundary(end),
        ),
        warm_up_observations,
    )


def test_qf11_prediction_components_bind_directly_to_validation_plan() -> None:
    strategy = OvernightGapPredictionStrategy(OvernightGapPredictionParameters())
    outcome_labeler = NextSessionOpenGapOutcomeLabeler()
    outcome = OutcomeProvenance.capture_exchange_sessions(outcome_labeler)
    warm_up_context = strategy.warm_up_observations - 1
    timeframe = Timeframe.us_equity(SessionInterval())
    dataset = make_dataset(("100", "101"))
    environment = ResearchEnvironment(
        ResearchStudyType.PREDICTION,
        DatasetProvenance.from_market_dataset(dataset),
        (timeframe,),
        ResearchRuleProvenance.capture_prediction(strategy),
        indicators=tuple(
            IndicatorProvenance.capture(cast(IndicatorComponent, indicator), timeframe)
            for indicator in strategy.required_indicators
        ),
        outcomes=(outcome,),
    )
    plan = ValidationPlan(
        "qf11_validation",
        environment,
        (
            ValidationFold(
                "fold_1",
                _window(
                    "development",
                    PartitionRole.DEVELOPMENT,
                    date(2024, 1, 2),
                    date(2024, 1, 5),
                    warm_up_context,
                ),
                _window(
                    "test",
                    PartitionRole.WALK_FORWARD_TEST,
                    date(2024, 1, 9),
                    date(2024, 1, 10),
                    warm_up_context,
                ),
                _window(
                    "selection",
                    PartitionRole.SELECTION,
                    date(2024, 1, 8),
                    date(2024, 1, 8),
                    warm_up_context,
                ),
            ),
        ),
        FinalHoldout(
            _window(
                "holdout",
                PartitionRole.FINAL_HOLDOUT,
                date(2024, 1, 11),
                date(2024, 1, 12),
                warm_up_context,
            ),
            "reserved QF-11 confirmation",
        ),
        PurgePolicy(TemporalOffset.sessions(1), TemporalOffset.sessions(0)),
        TrainingWindowMode.EXPANDING,
    )

    assert plan.environment.research_rule.configuration_id == strategy.configuration_id
    assert plan.environment.research_rule.warm_up_observations == (
        strategy.warm_up_observations
    )
    assert set(plan.environment.research_rule.required_indicator_configuration_ids) == {
        indicator.configuration_id for indicator in strategy.required_indicators
    }
    assert plan.environment.outcomes[0].configuration.configuration_id == (
        outcome_labeler.configuration_id
    )
    assert plan.environment.outcomes[0].future_horizon == TemporalOffset.sessions(1)
    assert {item.name for item in plan.environment.indicators} == {
        item.name for item in strategy.required_indicators
    }
    assert all(item.legacy_native_configuration for item in plan.environment.indicators)
