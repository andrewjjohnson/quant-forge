"""Small shared offline fixtures; use the real QF-5/QF-11/QF-32 engines."""

from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

from quantforge.configuration import PrimitiveMapping
from quantforge.data import (
    DatasetFamily,
    DatasetLineage,
    IntradayBar,
    TimeframeBarSeries,
)
from quantforge.data.intraday_aggregation import intraday_session_windows
from quantforge.indicators import (
    NATIVE_INDICATOR_BACKEND,
    SimpleMovingAverage,
    SimpleMovingAverageParameters,
)
from quantforge.optimization import (
    IntegerValues,
    MovingAverageCrossoverFactory,
    ParameterSearchSpace,
)
from quantforge.prediction import (
    PredictionContextRequirements,
    PredictionDirection,
    PredictionGridConfig,
    PredictionIndicatorRequirement,
    PredictionParameter,
    PredictionRankingConfig,
    PredictionRuleContext,
    PredictionSignal,
    PredictionStrategyOutput,
    PredictionStudy,
    PredictionTimeframeRequirement,
)
from quantforge.timeframes import IntradayInterval, SessionInterval, Timeframe
from quantforge.validation import (
    BacktestProvenance,
    ConfigurationReference,
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
    TimeframeWarmUpRequirement,
    TrainingWindowMode,
    ValidationFold,
    ValidationInterval,
    ValidationPlan,
    ValidationWindow,
)
from quantforge.walk_forward import (
    BacktestEvaluator,
    PredictionEvaluator,
    WalkForwardConfig,
)
from tests.unit.data.test_multi_timeframe import (  # pyright: ignore[reportPrivateUsage]
    _family,  # pyright: ignore[reportPrivateUsage]
    _intraday_bar,  # pyright: ignore[reportPrivateUsage]
    _session_bar,  # pyright: ignore[reportPrivateUsage]
)
from tests.unit.helpers import SESSIONS, make_dataset
from tests.unit.indicators.test_timeframe_evaluation import (
    _adjustment_basis,  # pyright: ignore[reportPrivateUsage]
)
from tests.unit.optimization.test_study import (
    _study_config,  # pyright: ignore[reportPrivateUsage]
)
from tests.unit.prediction.test_multi_timeframe_study import (  # pyright: ignore[reportPrivateUsage]
    FixtureMultiTimeframeRule,
    _study,  # pyright: ignore[reportPrivateUsage]
)
from tests.unit.prediction.test_prediction_grid import (  # pyright: ignore[reportPrivateUsage]
    FixtureStudyFactory,
    _backend_environment,  # pyright: ignore[reportPrivateUsage]
)
from tests.unit.prediction.test_prediction_window import WindowAnalyzer

DAILY = Timeframe.us_equity(SessionInterval())
PRIMARY = Timeframe.us_equity(IntradayInterval(timedelta(hours=4)))


def folds(
    environment: ResearchEnvironment,
    *,
    mode: TrainingWindowMode,
    selection: bool = False,
    horizon: int = 0,
    embargo: int = 0,
) -> ValidationPlan:
    daily = environment.outcome_dataset.standalone_timeframe
    assert daily is not None

    def window(
        name: str, role: PartitionRole, start: int, end: int
    ) -> ValidationWindow:
        if environment.prediction_dataset is not None and start == 3:
            start = 4
        interval = ValidationInterval(
            ExchangeSessionBoundary(SESSIONS[start]),
            ExchangeSessionBoundary(SESSIONS[end]),
        )
        if environment.prediction_dataset is None:
            return ValidationWindow(name, role, interval, warm_up_observations=3)
        return ValidationWindow(
            name,
            role,
            interval,
            warm_up_by_timeframe=(
                TimeframeWarmUpRequirement(PRIMARY, 3),
                TimeframeWarmUpRequirement(daily, 3),
            ),
        )

    first = ValidationFold(
        "fold-1",
        window("dev-1", PartitionRole.DEVELOPMENT, 3, 4 if selection else 6),
        window("test-1", PartitionRole.WALK_FORWARD_TEST, 7, 9),
        window("select-1", PartitionRole.SELECTION, 5, 6) if selection else None,
    )
    second = ValidationFold(
        "fold-2",
        window(
            "dev-2",
            PartitionRole.DEVELOPMENT,
            3 if mode is TrainingWindowMode.EXPANDING else 5,
            6 if selection else 9,
        ),
        window("test-2", PartitionRole.WALK_FORWARD_TEST, 10, 12),
        window("select-2", PartitionRole.SELECTION, 7, 9) if selection else None,
    )
    return ValidationPlan(
        "QF-39 fixture",
        environment,
        (first, second),
        FinalHoldout(
            window("reserved", PartitionRole.FINAL_HOLDOUT, 13, 14), "untouched"
        ),
        PurgePolicy(TemporalOffset.sessions(horizon), TemporalOffset.sessions(embargo)),
        mode,
    )


def backtest_fixture(
    tmp_path: Path,
    *,
    mode: TrainingWindowMode = TrainingWindowMode.ROLLING,
    selection: bool = False,
    embargo: int = 0,
    retry_failed: bool = False,
) -> tuple[WalkForwardConfig, BacktestEvaluator]:
    dataset = make_dataset(
        (
            "100",
            "99",
            "98",
            "99",
            "101",
            "103",
            "102",
            "99",
            "97",
            "100",
            "101",
            "102",
            "101",
            "100",
            "99",
        )
    )
    grid = _study_config(tmp_path / "unused")
    factory = MovingAverageCrossoverFactory()
    adapter = BacktestEvaluator(dataset, factory, grid)
    provenance = DatasetProvenance.from_market_dataset(dataset)
    timeframe = provenance.standalone_timeframe
    assert timeframe is not None
    indicators: dict[str, IndicatorProvenance] = {}
    for candidate in adapter.universe.candidates:
        for indicator in factory.build(
            candidate.parameters.to_primitive()
        ).required_indicators:
            indicators[indicator.configuration_id] = IndicatorProvenance.capture(
                cast(IndicatorComponent, indicator), timeframe
            )
    rule = factory.build(adapter.universe.candidates[0].parameters.to_primitive())
    environment = ResearchEnvironment(
        ResearchStudyType.TRADING_BACKTEST,
        provenance,
        (timeframe,),
        ResearchRuleProvenance.capture_trading(rule),
        indicators=tuple(indicators.values()),
        execution=BacktestProvenance.capture(grid.backtest),
    )
    plan = folds(environment, mode=mode, selection=selection, embargo=embargo)
    return WalkForwardConfig(
        "backtest fixture", plan, retry_failed=retry_failed
    ), adapter


class StudyRule(FixtureMultiTimeframeRule):
    def generate_with_context(
        self, context: PredictionRuleContext
    ) -> PredictionStrategyOutput:
        primary = self.context_requirements.primary.timeframe
        bar = context.latest_bar_for(primary)
        signal = PredictionSignal(
            symbol=context.symbol,
            signal_session=bar.end_timestamp.date(),
            direction=PredictionDirection.UP,
            strategy_id=self.name,
            strategy_implementation_version=self.implementation_version,
            strategy_configuration_id=self.configuration_id,
            strategy_parameters=(PredictionParameter("mode", self.parameters.mode),),
            reason="qf39 fixture causal prediction",
            feature_values=(),
        )
        return PredictionStrategyOutput(
            self.name, self.configuration_id, context.prediction_dataset_id, (signal,)
        )


class StudyFactory(FixtureStudyFactory):
    def build(self, parameters: PrimitiveMapping) -> PredictionStudy[Any, Any, Any]:
        window = cast(int, parameters["window"])
        requirements = PredictionContextRequirements(
            PredictionTimeframeRequirement(
                PRIMARY,
                _family().feed_scope,
                (
                    PredictionIndicatorRequirement(
                        "trend",
                        SimpleMovingAverage(
                            SimpleMovingAverageParameters(window),
                            backend_id=NATIVE_INDICATOR_BACKEND,
                        ),
                    ),
                ),
            ),
            (
                PredictionTimeframeRequirement(
                    DAILY,
                    _family().feed_scope,
                    (
                        PredictionIndicatorRequirement(
                            "trend",
                            SimpleMovingAverage(
                                SimpleMovingAverageParameters(4),
                                backend_id=NATIVE_INDICATOR_BACKEND,
                            ),
                        ),
                    ),
                ),
            ),
        )
        return _study(StudyRule(requirements))


def prediction_fixture(
    tmp_path: Path,
    *,
    mode: TrainingWindowMode = TrainingWindowMode.ROLLING,
    embargo: int = 0,
    retry_failed: bool = False,
    factory: StudyFactory | None = None,
    analyzer: WindowAnalyzer | None = None,
) -> tuple[WalkForwardConfig, PredictionEvaluator]:
    dataset = make_dataset(
        tuple(str(100 + index % 5) for index in range(len(SESSIONS)))
    )
    baseline = _family()
    family = DatasetFamily(
        canonical_symbol="SPY",
        provider_name=baseline.provider_name,
        feed_scope=baseline.feed_scope,
        adjustment_basis=_adjustment_basis(),
        aggregation_policy=baseline.aggregation_policy,
        canonical_source_snapshot_id="source-5m",
        datasets=(
            DatasetLineage("source-5m", PRIMARY, "source-5m", None, ("derived-daily",)),
            DatasetLineage("derived-daily", DAILY, "source-5m", "source-5m"),
        ),
    )
    bars: list[IntradayBar] = []
    for index, session in enumerate(SESSIONS):
        for interval in intraday_session_windows(session, PRIMARY):
            bar = _intraday_bar(
                PRIMARY,
                interval.start_timestamp,
                interval.end_timestamp,
                interval.completion,
            )
            bars.append(
                replace(
                    bar,
                    close=Decimal(100 + index % 2),
                    provenance=replace(
                        bar.provenance, adjustment_basis=family.adjustment_basis
                    ),
                )
            )
    series = TimeframeBarSeries._from_validated_artifact(  # pyright: ignore[reportPrivateUsage]
        family.reference("source-5m"),
        PRIMARY,
        tuple(bars),
        dataset_family_manifest_id=family.manifest_id,
    )
    daily_series = TimeframeBarSeries._from_validated_artifact(  # pyright: ignore[reportPrivateUsage]
        family.reference("derived-daily"),
        DAILY,
        tuple(_session_bar(DAILY, session) for session in SESSIONS),
        dataset_family_manifest_id=family.manifest_id,
    )
    factory = factory or StudyFactory()
    grid = PredictionGridConfig(
        "QF-39 prediction grid",
        ParameterSearchSpace({"window": IntegerValues([2, 3])}),
        PredictionRankingConfig("accuracy", "always_up"),
        tmp_path / "unused",
    )
    adapter = PredictionEvaluator(
        dataset=dataset,
        series=(series, daily_series),
        primary_timeframe=PRIMARY,
        study_factory=factory,
        analyzer=analyzer or WindowAnalyzer(),
        indicator_backend=_backend_environment(),
        grid_config=grid,
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
    environment = ResearchEnvironment(
        ResearchStudyType.PREDICTION,
        DatasetProvenance.from_dataset_family(family, ("source-5m", "derived-daily")),
        (PRIMARY, DAILY),
        ResearchRuleProvenance.capture_prediction(studies[0].strategy),
        indicators=tuple(indicators.values()),
        aggregation_policies=(
            ConfigurationReference.capture_aggregation_policy(
                family.aggregation_policy
            ),
        ),
        outcomes=(
            OutcomeProvenance.capture_exchange_sessions(studies[0].outcome_labeler),
        ),
        prediction_dataset=DatasetProvenance.from_market_dataset(dataset),
    )
    plan = folds(environment, mode=mode, horizon=1, embargo=embargo)
    return WalkForwardConfig(
        "prediction fixture", plan, retry_failed=retry_failed
    ), adapter
