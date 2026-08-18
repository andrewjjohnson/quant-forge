import ast
from dataclasses import dataclass, replace
from datetime import date, timedelta
from pathlib import Path
from typing import cast

import pytest

from quantforge.configuration import Primitive, PrimitiveMapping, configuration_identity
from quantforge.data import (
    AdjustmentBasis,
    AdjustmentMode,
    ContextAvailability,
    ContextCompletionPolicy,
    ContextTimeframeRequirement,
    FeedScope,
    IntradayBar,
    MultiTimeframeContext,
    TimeframeContext,
)
from quantforge.indicators import (
    NATIVE_INDICATOR_BACKEND,
    SIMPLE_MOVING_AVERAGE_OUTPUT,
    TALIB_INDICATOR_BACKEND,
    Indicator,
    SimpleMovingAverage,
    SimpleMovingAverageParameters,
)
from quantforge.prediction import (
    InvalidPredictionDataError,
    InvalidPredictionOutputError,
    NextSessionOpenGapOutcomeLabeler,
    NextSessionOpenGapValues,
    OvernightGapDirectionEvaluationValues,
    OvernightGapDirectionEvaluator,
    PredictionContextAccessError,
    PredictionContextFailurePolicy,
    PredictionContextRequirements,
    PredictionDirection,
    PredictionFeature,
    PredictionIndicatorRequirement,
    PredictionParameter,
    PredictionRuleContext,
    PredictionSignal,
    PredictionStrategyOutput,
    PredictionStudy,
    PredictionTimeframeRequirement,
    run_prediction_study,
)
from quantforge.timeframes import IntradayInterval, Timeframe
from tests.unit.indicators import test_timeframe_evaluation as timeframe_fixtures

from ..helpers import make_dataset


@dataclass(frozen=True, slots=True)
class FixtureParameters:
    mode: str = "fixture_multi_timeframe"

    def to_primitive(self) -> PrimitiveMapping:
        return {"mode": self.mode}


class FixtureContextProvider:
    def __init__(self, context: MultiTimeframeContext) -> None:
        self.context = context
        self.requests: list[PredictionContextRequirements] = []

    def get_context(
        self, requirements: PredictionContextRequirements
    ) -> MultiTimeframeContext:
        self.requests.append(requirements)
        return self.context


class FixtureMultiTimeframeRule:
    name = "fixture_multi_timeframe_rule"
    implementation_version = "1"
    warm_up_observations = 2

    def __init__(self, requirements: PredictionContextRequirements) -> None:
        self.context_requirements = requirements
        self._parameters = FixtureParameters()
        self.calls = 0
        self.declared_timeframes_seen: tuple[str, ...] = ()
        self.undeclared_access_blocked = False
        self.latest_bar_ends = ()

    @property
    def parameters(self) -> FixtureParameters:
        return self._parameters

    @property
    def required_indicators(self) -> tuple[Indicator, ...]:
        return tuple(
            indicator.indicator
            for timeframe in self.context_requirements.all_timeframes
            for indicator in timeframe.indicators
        )

    def configuration(self) -> PrimitiveMapping:
        return {
            "component_name": self.name,
            "component_type": "prediction_strategy",
            "contract_version": "2",
            "implementation_version": self.implementation_version,
            "parameters": self.parameters.to_primitive(),
            "required_indicators": [
                item.to_primitive()
                for timeframe in self.context_requirements.all_timeframes
                for item in timeframe.indicators
            ],
            "context_requirements": self.context_requirements.to_primitive(),
            "warm_up_observations": self.warm_up_observations,
        }

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    def generate_with_context(
        self, context: PredictionRuleContext
    ) -> PredictionStrategyOutput:
        self.calls += 1
        declared = self.context_requirements.all_timeframes
        self.declared_timeframes_seen = tuple(
            item.timeframe.configuration_id for item in declared
        )
        self.latest_bar_ends = tuple(
            context.latest_bar_for(item.timeframe).end_timestamp for item in declared
        )
        for item in declared:
            output = context.indicator_for(item.timeframe, "trend")
            assert output.backend_identity is not None
            assert output.values_for(SIMPLE_MOVING_AVERAGE_OUTPUT)[-1] is not None

        undeclared = Timeframe.us_equity(IntradayInterval(timedelta(minutes=15)))
        with pytest.raises(PredictionContextAccessError, match="not declared"):
            context.bars_for(undeclared)
        self.undeclared_access_blocked = True

        primary = self.context_requirements.primary.timeframe
        signal_bar = context.latest_bar_for(primary)
        daily = timeframe_fixtures._timeframes()[2]  # pyright: ignore[reportPrivateUsage]
        daily_trend = context.indicator_for(daily, "trend").values_for(
            SIMPLE_MOVING_AVERAGE_OUTPUT
        )[-1]
        assert daily_trend is not None
        signal = PredictionSignal(
            symbol=context.symbol,
            signal_session=signal_bar.end_timestamp.date(),
            direction=PredictionDirection.UP,
            strategy_id=self.name,
            strategy_implementation_version=self.implementation_version,
            strategy_configuration_id=self.configuration_id,
            strategy_parameters=(PredictionParameter("mode", self.parameters.mode),),
            reason="fixture_normalized_multi_timeframe_context",
            feature_values=(PredictionFeature("daily_trend", daily_trend),),
        )
        return PredictionStrategyOutput(
            self.name,
            self.configuration_id,
            context.prediction_dataset_id,
            (signal,),
        )


class BackdatedFixtureMultiTimeframeRule(FixtureMultiTimeframeRule):
    def generate_with_context(
        self, context: PredictionRuleContext
    ) -> PredictionStrategyOutput:
        output = super().generate_with_context(context)
        backdated_signal = replace(output.signals[0], signal_session=date(2024, 7, 10))
        return replace(output, signals=(backdated_signal,))


def _requirements(
    *,
    backend_id: str = NATIVE_INDICATOR_BACKEND,
    window: int = 2,
    feed_scope: FeedScope | None = None,
    daily_maximum_age: timedelta | None = None,
    failure_policy: PredictionContextFailurePolicy = (
        PredictionContextFailurePolicy.FAIL
    ),
) -> PredictionContextRequirements:
    five_minute, four_hour, daily, weekly = timeframe_fixtures._timeframes()  # pyright: ignore[reportPrivateUsage]
    selected_feed = feed_scope or FeedScope.consolidated()

    def declared(timeframe: Timeframe) -> PredictionTimeframeRequirement:
        return PredictionTimeframeRequirement(
            timeframe,
            selected_feed,
            (
                PredictionIndicatorRequirement(
                    "trend",
                    SimpleMovingAverage(
                        SimpleMovingAverageParameters(window),
                        backend_id=backend_id,
                    ),
                ),
            ),
            maximum_age=(daily_maximum_age if timeframe == daily else None),
        )

    return PredictionContextRequirements(
        declared(five_minute),
        (declared(weekly), declared(daily), declared(four_hour)),
        failure_policy,
    )


def _study(
    rule: FixtureMultiTimeframeRule,
) -> PredictionStudy[
    PredictionSignal,
    NextSessionOpenGapValues,
    OvernightGapDirectionEvaluationValues,
]:
    return PredictionStudy[
        PredictionSignal,
        NextSessionOpenGapValues,
        OvernightGapDirectionEvaluationValues,
    ].create(
        rule,
        NextSessionOpenGapOutcomeLabeler(),
        OvernightGapDirectionEvaluator(),
    )


def _prediction_dataset(
    adjustment_mode: AdjustmentMode = AdjustmentMode.UNADJUSTED,
):
    return make_dataset(
        ("100", "101", "102"),
        sessions=(date(2024, 7, 10), date(2024, 7, 11), date(2024, 7, 12)),
        opens=("99", "100", "103"),
        highs=("101", "102", "104"),
        lows=("98", "99", "101"),
        adjustment_mode=adjustment_mode,
    )


def _context_with_provenance(
    source: MultiTimeframeContext,
    *,
    symbol: str = "SPY",
    adjustment_basis: AdjustmentBasis | None = None,
) -> MultiTimeframeContext:
    aligned: list[TimeframeContext] = []
    for item in source.timeframes:
        bars = tuple(
            replace(
                bar,
                symbol=symbol,
                **(
                    {
                        "provenance": replace(
                            bar.provenance,
                            adjustment_basis=adjustment_basis,
                        )
                    }
                    if isinstance(bar, IntradayBar) and adjustment_basis is not None
                    else {}
                ),
            )
            for bar in item.bars
        )
        aligned.append(
            TimeframeContext._from_aligned_series(  # pyright: ignore[reportPrivateUsage]
                requirement=item.requirement,
                dataset_reference=item.dataset_reference,
                availability=item.availability,
                bars=bars,
                latest_completed_bar_timestamp=item.latest_completed_bar_timestamp,
                age=item.age,
            )
        )
    return MultiTimeframeContext._from_aligned_timeframes(  # pyright: ignore[reportPrivateUsage]
        as_of=source.as_of,
        primary_timeframe=source.primary_timeframe,
        required_timeframes=source.required_timeframes,
        completion_policy=source.completion_policy,
        source_consistency=source.source_consistency,
        timeframes=tuple(aligned),
    )


def _context_with_daily_state(
    requirements: PredictionContextRequirements,
    availability: ContextAvailability,
) -> MultiTimeframeContext:
    source = (
        timeframe_fixtures._all_completed_context()  # pyright: ignore[reportPrivateUsage]
    )
    daily = timeframe_fixtures._timeframes()[2]  # pyright: ignore[reportPrivateUsage]
    contextual_requirements = requirements.context_timeframe_requirements()
    by_id = {item.timeframe.configuration_id: item for item in contextual_requirements}
    aligned: list[TimeframeContext] = []
    for item in source.timeframes:
        requirement = (
            ContextTimeframeRequirement(requirements.primary.timeframe)
            if item.timeframe == requirements.primary.timeframe
            else by_id[item.timeframe.configuration_id]
        )
        missing = (
            item.timeframe == daily and availability is ContextAvailability.MISSING
        )
        selected_availability = (
            availability if item.timeframe == daily else item.availability
        )
        aligned.append(
            TimeframeContext._from_aligned_series(  # pyright: ignore[reportPrivateUsage]
                requirement=requirement,
                dataset_reference=item.dataset_reference,
                availability=selected_availability,
                bars=() if missing else item.bars,
                latest_completed_bar_timestamp=(
                    None if missing else item.latest_completed_bar_timestamp
                ),
                age=None if missing else item.age,
            )
        )
    return MultiTimeframeContext._from_aligned_timeframes(  # pyright: ignore[reportPrivateUsage]
        as_of=source.as_of,
        primary_timeframe=requirements.primary.timeframe,
        required_timeframes=contextual_requirements,
        completion_policy=requirements.context_completion_policy,
        source_consistency=source.source_consistency,
        timeframes=tuple(aligned),
    )


def test_fixture_rule_runs_through_generic_study_with_declared_normalized_inputs() -> (
    None
):
    context = (
        timeframe_fixtures._all_completed_context()  # pyright: ignore[reportPrivateUsage]
    )
    requirements = _requirements()
    provider = FixtureContextProvider(context)
    rule = FixtureMultiTimeframeRule(requirements)

    result = run_prediction_study(
        _prediction_dataset(), _study(rule), context_provider=provider
    )

    assert provider.requests == [requirements]
    assert rule.calls == 1
    assert len(rule.declared_timeframes_seen) == 4
    assert rule.undeclared_access_blocked is True
    assert all(timestamp <= context.as_of for timestamp in rule.latest_bar_ends)
    assert len(result.rows) == 1
    assert result.rows[0].signal.feature_values[0].name == "daily_trend"
    manifest = result.manifest_primitive()
    prediction_context = cast(PrimitiveMapping, manifest["prediction_context"])
    configuration = cast(PrimitiveMapping, manifest["configuration"])
    assert prediction_context["status"] == "available"
    assert prediction_context["decision_session"] == "2024-07-11"
    primary_bar = context.bars_for(requirements.primary.timeframe)[0]
    assert isinstance(primary_bar, IntradayBar)
    assert prediction_context["adjustment_basis"] == (
        primary_bar.provenance.adjustment_basis.to_primitive()
    )
    assert (
        configuration["prediction_context_requirements"] == requirements.to_primitive()
    )
    configured_backend = requirements.primary.indicators[0].backend_identity
    assert configured_backend.backend_id == NATIVE_INDICATOR_BACKEND
    timeframe_manifests = cast(list[Primitive], prediction_context["timeframes"])
    primary_manifest = cast(PrimitiveMapping, timeframe_manifests[0])
    indicator_manifests = cast(list[Primitive], primary_manifest["indicators"])
    primary_indicator = cast(PrimitiveMapping, indicator_manifests[0])
    assert primary_indicator["backend"] == configured_backend.to_primitive()


def test_context_symbol_must_match_prediction_dataset_before_rule_execution() -> None:
    context = _context_with_provenance(
        timeframe_fixtures._all_completed_context(),  # pyright: ignore[reportPrivateUsage]
        symbol="QQQ",
    )
    rule = FixtureMultiTimeframeRule(_requirements())

    with pytest.raises(InvalidPredictionDataError, match="context symbol"):
        run_prediction_study(
            _prediction_dataset(),
            _study(rule),
            context_provider=FixtureContextProvider(context),
        )

    assert rule.calls == 0


def test_context_adjustment_basis_must_match_before_rule_execution() -> None:
    context = (
        timeframe_fixtures._all_completed_context()  # pyright: ignore[reportPrivateUsage]
    )
    rule = FixtureMultiTimeframeRule(_requirements())

    with pytest.raises(InvalidPredictionDataError, match="adjustment basis"):
        run_prediction_study(
            _prediction_dataset(AdjustmentMode.SPLIT_ADJUSTED),
            _study(rule),
            context_provider=FixtureContextProvider(context),
        )

    assert rule.calls == 0


def test_contextual_signal_must_use_the_context_decision_session() -> None:
    context = (
        timeframe_fixtures._all_completed_context()  # pyright: ignore[reportPrivateUsage]
    )
    rule = BackdatedFixtureMultiTimeframeRule(_requirements())

    with pytest.raises(InvalidPredictionOutputError, match="context decision session"):
        run_prediction_study(
            _prediction_dataset(),
            _study(rule),
            context_provider=FixtureContextProvider(context),
        )

    assert rule.calls == 1


def test_incompatible_context_fails_or_skips_by_explicit_policy() -> None:
    context = (
        timeframe_fixtures._all_completed_context()  # pyright: ignore[reportPrivateUsage]
    )
    failing_rule = FixtureMultiTimeframeRule(
        _requirements(feed_scope=FeedScope.iex_only())
    )

    with pytest.raises(InvalidPredictionDataError, match="feed scope"):
        run_prediction_study(
            _prediction_dataset(),
            _study(failing_rule),
            context_provider=FixtureContextProvider(context),
        )

    skipping_rule = FixtureMultiTimeframeRule(
        _requirements(
            feed_scope=FeedScope.iex_only(),
            failure_policy=PredictionContextFailurePolicy.SKIP,
        )
    )
    skipped = run_prediction_study(
        _prediction_dataset(),
        _study(skipping_rule),
        context_provider=FixtureContextProvider(context),
    )

    assert skipping_rule.calls == 0
    assert skipped.generated_prediction_count == 0
    assert skipped.rows == ()
    skipped_context = cast(
        PrimitiveMapping,
        skipped.manifest_primitive()["prediction_context"],
    )
    assert skipped_context["status"] == "skipped"
    assert skipped_context["source_context"] == context.to_primitive()


@pytest.mark.parametrize(
    ("availability", "maximum_age", "message"),
    [
        (ContextAvailability.MISSING, None, "missing declared timeframe"),
        (ContextAvailability.STALE, timedelta(seconds=1), "stale for timeframe"),
    ],
)
def test_missing_and_stale_context_follow_explicit_failure_policy(
    availability: ContextAvailability,
    maximum_age: timedelta | None,
    message: str,
) -> None:
    failing_requirements = _requirements(daily_maximum_age=maximum_age)
    failing_context = _context_with_daily_state(failing_requirements, availability)
    with pytest.raises(InvalidPredictionDataError, match=message):
        run_prediction_study(
            _prediction_dataset(),
            _study(FixtureMultiTimeframeRule(failing_requirements)),
            context_provider=FixtureContextProvider(failing_context),
        )

    skipping_requirements = _requirements(
        daily_maximum_age=maximum_age,
        failure_policy=PredictionContextFailurePolicy.SKIP,
    )
    skipping_context = _context_with_daily_state(skipping_requirements, availability)
    rule = FixtureMultiTimeframeRule(skipping_requirements)
    result = run_prediction_study(
        _prediction_dataset(),
        _study(rule),
        context_provider=FixtureContextProvider(skipping_context),
    )

    assert rule.calls == 0
    assert result.generated_prediction_count == 0


def test_requirement_identity_binds_timeframe_indicator_backend_and_completion() -> (
    None
):
    baseline = _requirements(failure_policy=PredictionContextFailurePolicy.SKIP)
    changed_indicator = _requirements(
        window=3,
        failure_policy=PredictionContextFailurePolicy.SKIP,
    )
    changed_backend = _requirements(
        backend_id=TALIB_INDICATOR_BACKEND,
        failure_policy=PredictionContextFailurePolicy.SKIP,
    )
    _, four_hour, daily, _ = timeframe_fixtures._timeframes()  # pyright: ignore[reportPrivateUsage]
    one_hour = Timeframe.us_equity(IntradayInterval(timedelta(hours=1)))
    changed_timeframe = PredictionContextRequirements(
        baseline.primary,
        (
            PredictionTimeframeRequirement(
                one_hour,
                FeedScope.consolidated(),
                baseline.contextual[0].indicators,
            ),
            *tuple(
                item
                for item in baseline.contextual
                if item.timeframe not in {four_hour}
            ),
        ),
        PredictionContextFailurePolicy.SKIP,
    )
    developing_daily = PredictionTimeframeRequirement(
        daily,
        FeedScope.consolidated(),
        baseline.contextual[
            next(
                index
                for index, item in enumerate(baseline.contextual)
                if item.timeframe == daily
            )
        ].indicators,
        completion_policy=ContextCompletionPolicy.DEVELOPING_BAR_AS_OF,
    )
    changed_completion = PredictionContextRequirements(
        baseline.primary,
        tuple(
            developing_daily if item.timeframe == daily else item
            for item in baseline.contextual
        ),
        PredictionContextFailurePolicy.SKIP,
    )

    identities = {
        configuration_identity(item.to_primitive())
        for item in (
            baseline,
            changed_indicator,
            changed_backend,
            changed_timeframe,
            changed_completion,
        )
    }
    assert len(identities) == 5

    context = (
        timeframe_fixtures._all_completed_context()  # pyright: ignore[reportPrivateUsage]
    )
    results = tuple(
        run_prediction_study(
            _prediction_dataset(),
            _study(FixtureMultiTimeframeRule(requirements)),
            context_provider=FixtureContextProvider(context),
        )
        for requirements in (
            baseline,
            changed_indicator,
            changed_backend,
            changed_timeframe,
            changed_completion,
        )
    )
    assert len({result.study_id for result in results}) == 5


def test_generic_prediction_context_orchestration_has_no_talib_or_rule_branches() -> (
    None
):
    module_path = (
        Path(__file__).parents[3] / "src" / "quantforge" / "prediction" / "context.py"
    )
    parsed = ast.parse(module_path.read_text(encoding="utf-8"))
    imported_modules = {
        node.module
        for node in ast.walk(parsed)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert not any("talib" in module.lower() for module in imported_modules)
    source = module_path.read_text(encoding="utf-8")
    assert "simple_moving_average" not in source
    assert "overnight_gap" not in source
