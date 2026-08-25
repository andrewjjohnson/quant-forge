from datetime import date
from typing import cast

from quantforge.prediction import (
    NextSessionOpenGapOutcomeLabeler,
    OvernightGapPredictionParameters,
    OvernightGapPredictionStrategy,
)
from quantforge.timeframes import SessionInterval, Timeframe
from quantforge.validation import (
    ConfigurationReference,
    ConfiguredComponent,
    DatasetProvenance,
    ExchangeSessionBoundary,
    FinalHoldout,
    IndicatorProvenance,
    OutcomeProvenance,
    PartitionRole,
    PurgePolicy,
    ResearchEnvironment,
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
) -> ValidationWindow:
    return ValidationWindow(
        name,
        role,
        ValidationInterval(
            ExchangeSessionBoundary(start),
            ExchangeSessionBoundary(end),
        ),
    )


def test_qf11_prediction_components_bind_directly_to_validation_plan() -> None:
    strategy = OvernightGapPredictionStrategy(OvernightGapPredictionParameters())
    outcome_labeler = NextSessionOpenGapOutcomeLabeler()
    outcome = OutcomeProvenance.capture_exchange_sessions(outcome_labeler)
    environment = ResearchEnvironment(
        ResearchStudyType.PREDICTION,
        DatasetProvenance("a" * 64, ("qf11-fixture-dataset",)),
        (Timeframe.us_equity(SessionInterval()),),
        ConfigurationReference.capture_component("prediction_rule", strategy),
        indicators=tuple(
            IndicatorProvenance.capture(cast(ConfiguredComponent, indicator))
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
                ),
                _window(
                    "test",
                    PartitionRole.WALK_FORWARD_TEST,
                    date(2024, 1, 9),
                    date(2024, 1, 10),
                ),
                _window(
                    "selection",
                    PartitionRole.SELECTION,
                    date(2024, 1, 8),
                    date(2024, 1, 8),
                ),
            ),
        ),
        FinalHoldout(
            _window(
                "holdout",
                PartitionRole.FINAL_HOLDOUT,
                date(2024, 1, 11),
                date(2024, 1, 12),
            ),
            "reserved QF-11 confirmation",
        ),
        PurgePolicy(TemporalOffset.sessions(1), TemporalOffset.sessions(0)),
        TrainingWindowMode.EXPANDING,
    )

    assert plan.environment.research_rule.configuration_id == strategy.configuration_id
    assert plan.environment.outcomes[0].configuration.configuration_id == (
        outcome_labeler.configuration_id
    )
    assert plan.environment.outcomes[0].future_horizon == TemporalOffset.sessions(1)
    assert {item.name for item in plan.environment.indicators} == {
        item.name for item in strategy.required_indicators
    }
    assert all(item.legacy_native_configuration for item in plan.environment.indicators)
