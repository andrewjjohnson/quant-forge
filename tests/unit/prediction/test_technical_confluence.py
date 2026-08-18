import json
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import TypedDict, cast

import pytest

from quantforge.configuration import PrimitiveMapping
from quantforge.data import (
    ContextCompletionPolicy,
    FeedScope,
    MarketDataset,
    MultiTimeframeContext,
)
from quantforge.indicators import (
    SIMPLE_MOVING_AVERAGE_OUTPUT,
    TALIB_INDICATOR_BACKEND,
    VOLUME_MOVING_AVERAGE_OUTPUT,
    MarketField,
    RelativeVolume,
    RelativeVolumeParameters,
    SimpleMovingAverage,
    SimpleMovingAverageParameters,
    TimeframeNeutralIndicator,
    VolumeMovingAverage,
    VolumeMovingAverageParameters,
)
from quantforge.prediction import (
    ForwardReturnValues,
    InvalidPredictionDataError,
    PredictionContextRequirements,
    PredictionDirection,
    PredictionIndicatorRequirement,
    PredictionStudy,
    PredictionTimeframeRequirement,
    SignalDisposition,
    SignalFeatureCandidate,
    TechnicalCondition,
    TechnicalConditionOperand,
    TechnicalConditionOperator,
    TechnicalConditionStatus,
    TechnicalConfluenceOutcome,
    TechnicalConfluenceParameters,
    TechnicalConfluencePredictionRule,
    build_prediction_rule_context,
    build_signal_feature_dataset,
    create_reference_technical_confluence_rule,
    forward_return_outcome,
    run_prediction_study,
)
from quantforge.timeframes import Timeframe
from tests.unit.indicators import test_timeframe_evaluation as timeframe_fixtures
from tests.unit.prediction import test_multi_timeframe_study as study_fixtures

from ..helpers import make_dataset


class _FixtureCase(TypedDict):
    name: str
    weekly_closes: list[str]
    daily_closes: list[str]
    four_hour_closes: list[str]
    expected_outcome: str
    expected_disposition: str


_FIXTURE_PATH = (
    Path(__file__).parents[2]
    / "fixtures"
    / "prediction"
    / "technical_confluence_cases.json"
)


def _cases() -> tuple[_FixtureCase, ...]:
    values = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    return cast(tuple[_FixtureCase, ...], tuple(values))


def _dataset() -> MarketDataset:
    return make_dataset(
        ("10", "10", "10", "10", "11"),
        sessions=(
            date(2024, 7, 1),
            date(2024, 7, 2),
            date(2024, 7, 3),
            date(2024, 7, 5),
            date(2024, 7, 8),
        ),
    )


def _context(case: _FixtureCase) -> MultiTimeframeContext:
    five_minute, four_hour, daily, weekly = timeframe_fixtures._timeframes()  # pyright: ignore[reportPrivateUsage]
    return timeframe_fixtures._context(  # pyright: ignore[reportPrivateUsage]
        {
            five_minute.configuration_id: timeframe_fixtures._intraday_bars(  # pyright: ignore[reportPrivateUsage]
                five_minute,
                closes=("10", "10", "10"),
                sessions=timeframe_fixtures.SESSIONS[:3],
            ),
            four_hour.configuration_id: timeframe_fixtures._intraday_bars(  # pyright: ignore[reportPrivateUsage]
                four_hour,
                closes=tuple(case["four_hour_closes"]),
                sessions=timeframe_fixtures.SESSIONS[:2],
            ),
            daily.configuration_id: timeframe_fixtures._session_bars(  # pyright: ignore[reportPrivateUsage]
                daily,
                closes=tuple(case["daily_closes"]),
            ),
            weekly.configuration_id: timeframe_fixtures._session_bars(  # pyright: ignore[reportPrivateUsage]
                weekly,
                closes=tuple(case["weekly_closes"]),
            ),
        }
    )


def _rule(
    *,
    threshold: Decimal = Decimal("10"),
    backend_id: str = TALIB_INDICATOR_BACKEND,
    disable_weekly_up: bool = False,
) -> TechnicalConfluencePredictionRule:
    five_minute, four_hour, daily, weekly = timeframe_fixtures._timeframes()  # pyright: ignore[reportPrivateUsage]
    feed_scope = FeedScope.consolidated()

    def requirement(timeframe: Timeframe) -> PredictionTimeframeRequirement:
        return PredictionTimeframeRequirement(
            timeframe,
            feed_scope,
            (
                PredictionIndicatorRequirement(
                    "trend",
                    SimpleMovingAverage(
                        SimpleMovingAverageParameters(2),
                        backend_id=backend_id,
                    ),
                ),
            ),
        )

    requirements = PredictionContextRequirements(
        PredictionTimeframeRequirement(five_minute, feed_scope),
        (requirement(four_hour), requirement(daily), requirement(weekly)),
    )
    trend = TechnicalConditionOperand.indicator(
        "trend", SIMPLE_MOVING_AVERAGE_OUTPUT, "price_per_share"
    )

    def condition(
        name: str,
        timeframe_name: str,
        timeframe: Timeframe,
        operator: TechnicalConditionOperator,
        *,
        enabled: bool = True,
    ) -> TechnicalCondition:
        return TechnicalCondition(
            name,
            timeframe_name,
            timeframe,
            trend,
            operator,
            threshold,
            enabled,
        )

    up = (
        condition(
            "up_weekly_trend",
            "weekly",
            weekly,
            TechnicalConditionOperator.GREATER_THAN,
            enabled=not disable_weekly_up,
        ),
        condition(
            "up_daily_trend",
            "daily",
            daily,
            TechnicalConditionOperator.GREATER_THAN,
        ),
        condition(
            "up_four_hour_trend",
            "four_hour",
            four_hour,
            TechnicalConditionOperator.GREATER_THAN,
        ),
    )
    down = (
        condition(
            "down_weekly_trend",
            "weekly",
            weekly,
            TechnicalConditionOperator.LESS_THAN,
        ),
        condition(
            "down_daily_trend",
            "daily",
            daily,
            TechnicalConditionOperator.LESS_THAN,
        ),
        condition(
            "down_four_hour_trend",
            "four_hour",
            four_hour,
            TechnicalConditionOperator.LESS_THAN,
        ),
    )
    return TechnicalConfluencePredictionRule(
        TechnicalConfluenceParameters(up, down, "fixture_v1"),
        requirements,
    )


def _study(rule: TechnicalConfluencePredictionRule):
    outcome = forward_return_outcome(1)
    return PredictionStudy[
        SignalFeatureCandidate, ForwardReturnValues, ForwardReturnValues
    ].create(rule, outcome.labeler, outcome.evaluator), outcome


@pytest.mark.parametrize("case", _cases(), ids=lambda case: case["name"])
def test_deterministic_cases_record_acceptance_direction_conditions_and_timestamps(
    case: _FixtureCase,
) -> None:
    rule = _rule()
    study, _ = _study(rule)

    result = run_prediction_study(
        _dataset(),
        study,
        context_provider=study_fixtures.FixtureContextProvider(_context(case)),
    )

    assert len(result.signals) == 1
    candidate = result.signals[0]
    expected_outcome = TechnicalConfluenceOutcome(case["expected_outcome"])
    assert candidate.disposition is SignalDisposition(case["expected_disposition"])
    assert candidate.direction is (
        PredictionDirection.UP
        if expected_outcome is TechnicalConfluenceOutcome.UP
        else PredictionDirection.DOWN
        if expected_outcome is TechnicalConfluenceOutcome.DOWN
        else None
    )
    features = candidate.features_primitive()
    assert features["prediction_outcome"] == expected_outcome.value
    assert features["condition_up_weekly_trend_status"] in {
        TechnicalConditionStatus.PASSED.value,
        TechnicalConditionStatus.FAILED.value,
    }
    assert features["condition_down_weekly_trend_status"] in {
        TechnicalConditionStatus.PASSED.value,
        TechnicalConditionStatus.FAILED.value,
    }
    for timeframe_name in ("weekly", "daily", "four_hour"):
        timestamp = features[f"timeframe_{timeframe_name}_latest_source_timestamp"]
        assert isinstance(timestamp, str)
        manifest_context = cast(
            PrimitiveMapping,
            result.manifest_primitive()["prediction_context"],
        )
        source_context = cast(PrimitiveMapping, manifest_context["source_context"])
        assert timestamp <= cast(str, source_context["as_of"])


def test_rule_identity_binds_threshold_enabled_state_backend_and_requirements() -> None:
    baseline = _rule()
    variants = (
        _rule(threshold=Decimal("10.1")),
        _rule(disable_weekly_up=True),
        _rule(backend_id="native_v1"),
    )

    assert (
        len({baseline.configuration_id, *(item.configuration_id for item in variants)})
        == 4
    )
    assert all(baseline.configuration_id != item.configuration_id for item in variants)
    assert all(
        item.backend_identity is not None
        for requirement in baseline.context_requirements.contextual
        for item in requirement.indicators
    )


@pytest.mark.parametrize(
    ("indicator_value", "output_name"),
    [
        (
            RelativeVolume(RelativeVolumeParameters(2, FeedScope.consolidated())),
            "relative_volume",
        ),
        (
            VolumeMovingAverage(
                VolumeMovingAverageParameters(2, FeedScope.consolidated())
            ),
            VOLUME_MOVING_AVERAGE_OUTPUT,
        ),
    ],
)
def test_volume_indicator_is_supported_without_inventing_backend_provenance(
    indicator_value: TimeframeNeutralIndicator,
    output_name: str,
) -> None:
    _, _, daily, _ = timeframe_fixtures._timeframes()  # pyright: ignore[reportPrivateUsage]
    requirement = PredictionIndicatorRequirement(
        "relative_volume",
        indicator_value,
    )

    assert requirement.backend_identity is None
    primitive = requirement.to_primitive()
    indicator = cast(PrimitiveMapping, primitive["indicator"])
    assert indicator["backend"] is None
    assert (
        requirement.evaluate(
            timeframe_fixtures._all_completed_context(),  # pyright: ignore[reportPrivateUsage]
            daily,
            ContextCompletionPolicy.COMPLETED_BARS_ONLY,
        ).values_for(output_name)[-1]
        is not None
    )


def test_reference_rule_declares_full_suite_and_explicit_developing_policy() -> None:
    five_minute, four_hour, daily, weekly = timeframe_fixtures._timeframes()  # pyright: ignore[reportPrivateUsage]
    rule = create_reference_technical_confluence_rule(
        primary_timeframe=five_minute,
        four_hour_timeframe=four_hour,
        daily_timeframe=daily,
        weekly_timeframe=weekly,
        feed_scope=FeedScope.consolidated(),
        completion_policy=ContextCompletionPolicy.DEVELOPING_BAR_AS_OF,
    )

    aliases = {
        item.alias
        for requirement in rule.context_requirements.contextual
        for item in requirement.indicators
    }
    assert {
        "atr_14",
        "bollinger_20_2",
        "directional_14",
        "ema_9",
        "ema_10",
        "ema_21",
        "ema_50",
        "macd",
        "relative_volume_20",
        "rsi_14",
        "sma_20",
        "stochastic",
        "volume_average_20",
    } <= aliases
    for requirement in rule.context_requirements.contextual:
        for indicator in requirement.indicators:
            backend = indicator.backend_identity
            if backend is not None:
                assert backend.backend_id == TALIB_INDICATOR_BACKEND
    assert (
        rule.context_requirements.context_completion_policy
        is ContextCompletionPolicy.DEVELOPING_BAR_AS_OF
    )
    assert (
        rule.context_requirements.primary.completion_policy
        is ContextCompletionPolicy.COMPLETED_BARS_ONLY
    )
    assert len(rule.multi_timeframe_feature_requests) >= 12


def test_qf7_qf29_capture_condition_values_and_normalized_backend_provenance(
    tmp_path: Path,
) -> None:
    case = _cases()[0]
    rule = _rule()
    study, outcome = _study(rule)

    result = build_signal_feature_dataset(
        dataset=_dataset(),
        prediction_study=study,
        contextual_features=(),
        multi_timeframe_features=rule.multi_timeframe_feature_requests,
        context_provider=study_fixtures.FixtureContextProvider(_context(case)),
        outcomes=(outcome,),
        output_root=tmp_path,
    )

    assert len(result.rows) == 1
    row = result.rows[0].to_primitive()
    assert row["feature_prediction_outcome"] == TechnicalConfluenceOutcome.UP.value
    assert row["feature_condition_up_weekly_trend_passed"] is True
    for request in rule.multi_timeframe_feature_requests:
        assert row[f"feature_{request.name}"] is not None
        metadata = cast(
            PrimitiveMapping,
            row[f"feature_{request.metadata_name}"],
        )
        backend = cast(PrimitiveMapping, metadata["indicator_backend"])
        assert backend["backend_id"] == TALIB_INDICATOR_BACKEND
        assert cast(str, metadata["source_bar_observed_through_timestamp"]) <= cast(
            str, metadata["context_as_of"]
        )


def test_future_context_bar_is_rejected_before_rule_evaluation() -> None:
    rule = _rule()
    study, _ = _study(rule)
    future_exposing_context = timeframe_fixtures._all_completed_context()  # pyright: ignore[reportPrivateUsage]

    with pytest.raises(InvalidPredictionDataError, match="decision boundary"):
        run_prediction_study(
            _dataset(),
            study,
            context_provider=study_fixtures.FixtureContextProvider(
                future_exposing_context
            ),
        )


def test_crossover_boundaries_are_strict_at_current_and_inclusive_at_previous() -> None:
    case = _cases()[0]
    base_rule = _rule()
    _, four_hour, daily, weekly = timeframe_fixtures._timeframes()  # pyright: ignore[reportPrivateUsage]
    close = TechnicalConditionOperand.bar(MarketField.CLOSE, "price_per_share")
    up_cross = TechnicalCondition(
        "up_cross",
        "four_hour",
        four_hour,
        close,
        TechnicalConditionOperator.CROSSES_ABOVE,
        Decimal("10"),
    )
    down_guard = TechnicalCondition(
        "down_guard",
        "daily",
        daily,
        close,
        TechnicalConditionOperator.LESS_THAN,
        Decimal("0"),
    )
    weekly_guard = TechnicalCondition(
        "up_weekly_guard",
        "weekly",
        weekly,
        close,
        TechnicalConditionOperator.GREATER_THAN,
        Decimal("0"),
    )
    parameters = TechnicalConfluenceParameters(
        (up_cross, weekly_guard), (down_guard,), "cross"
    )
    rule = TechnicalConfluencePredictionRule(parameters, base_rule.context_requirements)
    crossing_case = case.copy()
    crossing_case["four_hour_closes"] = ["10", "11"]

    evaluation = rule.evaluate(
        build_prediction_rule_context(
            rule.context_requirements,
            _context(crossing_case),
            prediction_dataset_id=_dataset().metadata.dataset_id,
            symbol="SPY",
            prediction_adjustment_basis=timeframe_fixtures._adjustment_basis(),  # pyright: ignore[reportPrivateUsage]
        )
    )

    result = next(
        item
        for item in evaluation.condition_results
        if item.condition.name == "up_cross"
    )
    assert result.status is TechnicalConditionStatus.PASSED
    assert result.previous_left_value == Decimal("10")
    assert result.left_value == Decimal("11")
    assert weekly in tuple(
        item.timeframe for item in rule.context_requirements.contextual
    )
