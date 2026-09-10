from datetime import date
from typing import cast

from quantforge.prediction import (
    NextSessionOpenGapOutcomeLabeler,
    OvernightGapPredictionParameters,
    OvernightGapPredictionStrategy,
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
    TrainingWindowMode,
    ValidationFold,
    ValidationInterval,
    ValidationPlan,
    ValidationWindow,
)


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
    environment = ResearchEnvironment(
        ResearchStudyType.PREDICTION,
        DatasetProvenance("a" * 64, ("qf11-fixture-dataset",)),
        (Timeframe.us_equity(SessionInterval()),),
        ResearchRuleProvenance.capture_prediction(strategy),
        indicators=tuple(
            IndicatorProvenance.capture(cast(IndicatorComponent, indicator))
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
