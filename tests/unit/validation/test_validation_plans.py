import json
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import cast

import pytest

from quantforge.backtesting import (
    BacktestConfig,
    BasisPointSlippage,
    DividendPolicy,
    ExplicitZeroFees,
    FixedCommission,
)
from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.data import (
    AdjustmentBasis,
    AdjustmentMode,
    AggregationPolicy,
    DatasetFamily,
    DatasetLineage,
    FeedScope,
    MarketDataset,
)
from quantforge.data import (
    ValidationError as MarketDataValidationError,
)
from quantforge.data.identity import canonical_json_bytes
from quantforge.indicators import (
    NATIVE_INDICATOR_BACKEND,
    Indicator,
    SimpleMovingAverage,
    SimpleMovingAverageParameters,
)
from quantforge.prediction import (
    ForwardReturnOutcomeLabeler,
    PredictionContextRequirements,
    PredictionTimeframeRequirement,
)
from quantforge.timeframes import (
    IntradayInterval,
    SessionInterval,
    Timeframe,
    TradingWeekInterval,
)
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
    TimestampBoundary,
    TrainingWindowMode,
    ValidationBoundary,
    ValidationFold,
    ValidationInterval,
    ValidationPlan,
    ValidationPlanError,
    ValidationPlanIdentityError,
    ValidationWindow,
    purge_development_observations,
    purge_partition_observations,
    select_window_observations,
    serialize_validation_plan,
    validate_validation_plan_manifest,
)

from ..helpers import make_dataset
from ..prediction.test_multi_timeframe_study import FixtureMultiTimeframeRule

SOURCE_ID = "qf8-source-1m"
DAILY_ID = "qf8-derived-daily"
WEEKLY_ID = "qf8-derived-weekly"
FIXTURE_ROOT = Path(__file__).parents[2] / "fixtures" / "validation"


def _component(
    component_type: str,
    name: str,
    version: str = "1",
    **configuration: str | int,
) -> ConfigurationReference:
    primitive: PrimitiveMapping = {
        "component": name,
        "implementation_version": version,
        **configuration,
    }
    return ConfigurationReference.capture(component_type, name, version, primitive)


class _FixtureRule:
    implementation_version = "1"

    def __init__(
        self,
        name: str,
        canonical_component_type: str,
        required_indicators: tuple[Indicator, ...],
        warm_up_observations: int,
        configuration: PrimitiveMapping,
        primary_timeframe: Timeframe | None = None,
        context_indicator_bindings: tuple[tuple[Timeframe, Indicator], ...] = (),
    ) -> None:
        self.name = name
        self._canonical_component_type = canonical_component_type
        self.required_indicators = required_indicators
        self.warm_up_observations = warm_up_observations
        self._configuration = configuration
        self._primary_timeframe = primary_timeframe
        self._context_indicator_bindings = context_indicator_bindings

    def configuration(self) -> PrimitiveMapping:
        primitive: PrimitiveMapping = {
            "component_type": self._canonical_component_type,
            "component_name": self.name,
            "implementation_version": self.implementation_version,
            "parameters": dict(self._configuration),
            "required_indicators": [
                indicator.configuration() for indicator in self.required_indicators
            ],
            "warm_up_observations": self.warm_up_observations,
        }
        if self._primary_timeframe is not None:
            bindings = self._context_indicator_bindings or tuple(
                (self._primary_timeframe, indicator)
                for indicator in self.required_indicators
            )
            indicators_by_timeframe: dict[str, list[Indicator]] = {}
            timeframes_by_id: dict[str, Timeframe] = {}
            for timeframe, indicator in bindings:
                timeframe_id = timeframe.configuration_id
                timeframes_by_id[timeframe_id] = timeframe
                indicators_by_timeframe.setdefault(timeframe_id, []).append(indicator)

            def requirement(timeframe: Timeframe) -> PrimitiveMapping:
                return {
                    "feed_scope": FeedScope.consolidated().to_primitive(),
                    "timeframe": {
                        "configuration_id": timeframe.configuration_id,
                        "configuration": timeframe.to_primitive(),
                    },
                    "indicators": [
                        {
                            "indicator": {
                                "configuration_id": indicator.configuration_id,
                                "configuration": indicator.configuration(),
                            }
                        }
                        for indicator in indicators_by_timeframe.get(
                            timeframe.configuration_id, []
                        )
                    ],
                }

            primitive["context_requirements"] = {
                "primary": requirement(self._primary_timeframe),
                "contextual": [
                    requirement(timeframes_by_id[timeframe_id])
                    for timeframe_id in sorted(timeframes_by_id)
                    if timeframe_id != self._primary_timeframe.configuration_id
                ],
            }
        return primitive

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())


class _IncompleteBacktestConfiguration:
    engine_version = "4"
    result_schema_version = "3"

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "engine_version": self.engine_version,
            "result_schema_version": self.result_schema_version,
        }


def _rule(
    component_type: str,
    name: str,
    required_indicators: tuple[Indicator, ...],
    *,
    warm_up_observations: int | None = None,
    primary_timeframe: Timeframe | None = None,
    context_indicator_bindings: tuple[tuple[Timeframe, Indicator], ...] = (),
    **configuration: str | int,
) -> ResearchRuleProvenance:
    required_warm_up = max(
        (indicator.warm_up_observations for indicator in required_indicators),
        default=1,
    )
    canonical_component_type = (
        "prediction_strategy" if component_type == "prediction_rule" else "strategy"
    )
    component = _FixtureRule(
        name,
        canonical_component_type,
        required_indicators,
        required_warm_up if warm_up_observations is None else warm_up_observations,
        cast(PrimitiveMapping, dict(configuration)),
        primary_timeframe,
        context_indicator_bindings,
    )
    if component_type == "prediction_rule":
        return ResearchRuleProvenance.capture_prediction(component)
    if component_type == "trading_strategy":
        return ResearchRuleProvenance.capture_trading(component)
    raise AssertionError(f"unsupported fixture research rule type: {component_type}")


def _backtest_provenance(
    slippage_bps: str = "5",
    *,
    dividend_policy: DividendPolicy = DividendPolicy.REJECT_IF_DIVIDENDS,
) -> BacktestProvenance:
    return BacktestProvenance.capture(
        BacktestConfig(
            Decimal("100000"),
            FixedCommission(Decimal(0)),
            ExplicitZeroFees(),
            BasisPointSlippage(Decimal(slippage_bps)),
            dividend_policy=dividend_policy,
        )
    )


def _family(
    target_timeframe: Timeframe | None = None,
    *,
    missing_constituents: str = "reject",
) -> DatasetFamily:
    source_timeframe = Timeframe.us_equity(IntradayInterval(timedelta(minutes=1)))
    derived_timeframe = target_timeframe or Timeframe.us_equity(SessionInterval())
    return DatasetFamily(
        canonical_symbol="SPY",
        provider_name="fixture",
        feed_scope=FeedScope.consolidated(),
        adjustment_basis=AdjustmentBasis(
            AdjustmentMode.UNADJUSTED,
            "raw_provider",
            "raw_provider",
            "separate_provider_reported_cash_dividends_and_splits",
            False,
        ),
        aggregation_policy=AggregationPolicy(
            "quantforge_session_ohlcv",
            "1",
            {"missing_constituents": missing_constituents},
        ),
        canonical_source_snapshot_id=SOURCE_ID,
        datasets=(
            DatasetLineage(
                SOURCE_ID,
                source_timeframe,
                SOURCE_ID,
                None,
                (DAILY_ID,),
            ),
            DatasetLineage(
                DAILY_ID,
                derived_timeframe,
                SOURCE_ID,
                SOURCE_ID,
            ),
        ),
    )


def _multi_timeframe_family(
    daily_timeframe: Timeframe,
    weekly_timeframe: Timeframe,
) -> DatasetFamily:
    source_timeframe = Timeframe.us_equity(IntradayInterval(timedelta(minutes=1)))
    baseline = _family(daily_timeframe)
    return replace(
        baseline,
        datasets=(
            DatasetLineage(
                SOURCE_ID,
                source_timeframe,
                SOURCE_ID,
                None,
                (DAILY_ID, WEEKLY_ID),
            ),
            DatasetLineage(
                DAILY_ID,
                daily_timeframe,
                SOURCE_ID,
                SOURCE_ID,
            ),
            DatasetLineage(
                WEEKLY_ID,
                weekly_timeframe,
                SOURCE_ID,
                SOURCE_ID,
            ),
        ),
    )


def _environment(
    study_type: ResearchStudyType = ResearchStudyType.PREDICTION,
    *,
    indicator: Indicator | None = None,
    rule: ResearchRuleProvenance | None = None,
    outcome: OutcomeProvenance | None = None,
    execution: BacktestProvenance | None = None,
    timeframe: Timeframe | None = None,
    aggregation_missing: str = "reject",
) -> ResearchEnvironment:
    selected_timeframe = timeframe or Timeframe.us_equity(SessionInterval())
    family = _family(
        selected_timeframe,
        missing_constituents=aggregation_missing,
    )
    selected_indicator_component = indicator or SimpleMovingAverage(
        SimpleMovingAverageParameters(3)
    )
    selected_indicator = IndicatorProvenance.capture(
        cast(IndicatorComponent, selected_indicator_component),
        selected_timeframe,
    )
    selected_outcome = outcome or _session_outcome(1)
    selected_execution = execution
    if study_type is ResearchStudyType.TRADING_BACKTEST and selected_execution is None:
        selected_execution = _backtest_provenance()
    selected_rule = rule or _rule(
        (
            "trading_strategy"
            if study_type is ResearchStudyType.TRADING_BACKTEST
            else "prediction_rule"
        ),
        (
            "fixture_strategy"
            if study_type is ResearchStudyType.TRADING_BACKTEST
            else "fixture_rule"
        ),
        (selected_indicator_component,),
        period=3,
    )
    return ResearchEnvironment(
        study_type=study_type,
        dataset=DatasetProvenance.from_dataset_family(family, (DAILY_ID,)),
        timeframes=(selected_timeframe,),
        aggregation_policies=(
            ConfigurationReference.capture_aggregation_policy(
                family.aggregation_policy
            ),
        ),
        indicators=(selected_indicator,),
        research_rule=selected_rule,
        outcomes=(selected_outcome,)
        if study_type is ResearchStudyType.PREDICTION
        else (),
        execution=selected_execution,
    )


def _session_outcome(horizon_sessions: int) -> OutcomeProvenance:
    return OutcomeProvenance.capture_exchange_sessions(
        ForwardReturnOutcomeLabeler(horizon_sessions)
    )


class _TimestampOutcome:
    name = "intraday_forward_return"
    implementation_version = "1"

    def __init__(self, required_future_duration: timedelta) -> None:
        self.required_future_duration = required_future_duration

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    def configuration(self) -> PrimitiveMapping:
        return {
            "component": self.name,
            "implementation_version": self.implementation_version,
            "horizon_microseconds": int(
                self.required_future_duration.total_seconds() * 1_000_000
            ),
        }


def _timestamp_outcome(horizon: timedelta) -> OutcomeProvenance:
    return OutcomeProvenance.capture_timestamp(_TimestampOutcome(horizon))


def _session(value: str) -> ExchangeSessionBoundary:
    return ExchangeSessionBoundary(date.fromisoformat(value))


def _session_window(
    name: str,
    role: PartitionRole,
    start: str,
    end: str | None = None,
    *,
    warm_up: int = 2,
) -> ValidationWindow:
    return ValidationWindow(
        name,
        role,
        ValidationInterval(_session(start), _session(end or start)),
        warm_up,
    )


def _session_plan(
    *,
    embargo_sessions: int = 0,
    horizon_sessions: int = 1,
    environment: ResearchEnvironment | None = None,
) -> ValidationPlan:
    fold = ValidationFold(
        "fold_1",
        _session_window(
            "development_1", PartitionRole.DEVELOPMENT, "2024-01-02", "2024-01-08"
        ),
        _session_window("test_1", PartitionRole.WALK_FORWARD_TEST, "2024-01-10"),
        _session_window(
            "selection_1", PartitionRole.SELECTION, "2024-01-09", warm_up=2
        ),
    )
    return ValidationPlan(
        "prediction_validation",
        environment or _environment(outcome=_session_outcome(horizon_sessions)),
        (fold,),
        FinalHoldout(
            _session_window(
                "reserved_holdout",
                PartitionRole.FINAL_HOLDOUT,
                "2024-01-11",
                "2024-01-12",
                warm_up=2,
            ),
            "one untouched final confirmation",
        ),
        PurgePolicy(
            TemporalOffset.sessions(horizon_sessions),
            TemporalOffset.sessions(embargo_sessions),
        ),
        TrainingWindowMode.EXPANDING,
    )


def _timestamp(value: str) -> TimestampBoundary:
    return TimestampBoundary(datetime.fromisoformat(value))


def _fixture(name: str) -> dict[str, object]:
    loaded = json.loads((FIXTURE_ROOT / name).read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return cast(dict[str, object], loaded)


def _session_dates(boundaries: tuple[ValidationBoundary, ...]) -> tuple[date, ...]:
    assert all(isinstance(item, ExchangeSessionBoundary) for item in boundaries)
    return tuple(
        cast(ExchangeSessionBoundary, item).session_date for item in boundaries
    )


def _timestamps(
    boundaries: tuple[ValidationBoundary, ...],
) -> tuple[datetime, ...]:
    assert all(isinstance(item, TimestampBoundary) for item in boundaries)
    return tuple(cast(TimestampBoundary, item).timestamp for item in boundaries)


def test_session_plan_serialization_is_stable_and_records_fixed_provenance() -> None:
    plan = _session_plan()

    first = serialize_validation_plan(plan)
    second = serialize_validation_plan(_session_plan())
    manifest = validate_validation_plan_manifest(plan, first)

    assert first == second
    assert first == canonical_json_bytes(plan.to_manifest())
    assert manifest["plan_id"] == plan.plan_id
    assert manifest["training_window_mode"] == "expanding"
    holdout_primitive = plan.final_holdout.to_primitive()
    assert holdout_primitive["reservation"] == {
        "reserved": True,
        "purpose": "one untouched final confirmation",
        "consumption": "outside_qf8_scope",
    }
    assert len(plan.environment.dataset.dataset_fingerprint) == 64
    assert plan.environment.dataset.family_references[0].family_id in first.decode()
    assert Timeframe.us_equity(SessionInterval()).configuration_id in first.decode()
    assert '"backend_id":"native_v1"' in first.decode()
    assert plan.environment.research_rule.implementation_version == "1"
    assert '"future_sessions":1' in first.decode()


def test_manifest_validation_rejects_identity_mismatch_and_noncanonical_cache() -> None:
    plan = _session_plan()
    altered = json.loads(serialize_validation_plan(plan))
    altered["environment"]["dataset"]["dataset_fingerprint"] = "b" * 64

    with pytest.raises(ValidationPlanIdentityError, match="fixed plan identity"):
        validate_validation_plan_manifest(plan, canonical_json_bytes(altered))
    with pytest.raises(ValidationPlanIdentityError, match="canonical"):
        validate_validation_plan_manifest(
            plan, json.dumps(plan.to_manifest(), indent=2).encode()
        )
    with pytest.raises(ValidationPlanIdentityError, match="fixed plan identity"):
        validate_validation_plan_manifest(
            replace(plan, name="another_plan"), serialize_validation_plan(plan)
        )


def test_environment_identity_binds_every_required_scientific_input() -> None:
    baseline = _environment()
    explicit_native = SimpleMovingAverage(
        SimpleMovingAverageParameters(3),
        backend_id=NATIVE_INDICATOR_BACKEND,
    )
    rule_indicator = cast(
        Indicator, SimpleMovingAverage(SimpleMovingAverageParameters(3))
    )
    changes = (
        replace(
            baseline,
            dataset=DatasetProvenance.from_dataset_family(
                replace(_family(), provider_name="different-provider"),
                (DAILY_ID,),
            ),
        ),
        _environment(timeframe=Timeframe.us_equity(SessionInterval(2))),
        _environment(aggregation_missing="diagnostic"),
        _environment(indicator=explicit_native),
        _environment(
            rule=_rule(
                "prediction_rule",
                "fixture_rule",
                (rule_indicator,),
                period=4,
            )
        ),
        _environment(outcome=_session_outcome(2)),
        replace(
            baseline,
            execution=_backtest_provenance("10"),
        ),
    )

    assert all(item.environment_id != baseline.environment_id for item in changes)


def test_environment_rejects_timeframe_that_mismatches_family_reference() -> None:
    environment = _environment()

    with pytest.raises(ValidationPlanError, match="exactly match dataset provenance"):
        replace(
            environment,
            timeframes=(Timeframe.us_equity(SessionInterval(2)),),
        )


@pytest.mark.parametrize("study_type", list(ResearchStudyType))
@pytest.mark.parametrize("dataset_ids", [(DAILY_ID, WEEKLY_ID), (WEEKLY_ID, DAILY_ID)])
def test_environment_rejects_multiple_selected_datasets_on_one_timeframe(
    study_type: ResearchStudyType,
    dataset_ids: tuple[str, ...],
) -> None:
    daily = Timeframe.us_equity(SessionInterval())
    # Distinct lineage IDs may legitimately record the same timeframe.
    family = _multi_timeframe_family(daily, daily)
    provenance = DatasetProvenance.from_dataset_family(family, dataset_ids)
    assert len(provenance.family_references) == 2
    with pytest.raises(ValidationPlanError, match="one dataset per timeframe"):
        replace(_environment(study_type), dataset=provenance)


@pytest.mark.parametrize("study_type", list(ResearchStudyType))
def test_environment_selects_one_of_same_timeframe_family_members(
    study_type: ResearchStudyType,
) -> None:
    daily = Timeframe.us_equity(SessionInterval())
    family = _multi_timeframe_family(daily, daily)
    baseline = _environment(study_type)
    plans = tuple(
        _session_plan(
            environment=replace(
                baseline,
                dataset=DatasetProvenance.from_dataset_family(family, (dataset_id,)),
            ),
            horizon_sessions=1 if study_type is ResearchStudyType.PREDICTION else 0,
        )
        for dataset_id in (DAILY_ID, WEEKLY_ID)
    )
    for plan, dataset_id in zip(plans, (DAILY_ID, WEEKLY_ID), strict=True):
        assert plan.environment.dataset.dataset_ids == (dataset_id,)
        assert plan.environment.dataset.family_manifest_id == family.manifest_id
        content = serialize_validation_plan(plan)
        assert validate_validation_plan_manifest(plan, content) == plan.to_manifest()
    assert plans[0].plan_id != plans[1].plan_id
    with pytest.raises(ValidationPlanIdentityError):
        validate_validation_plan_manifest(plans[1], serialize_validation_plan(plans[0]))


def test_environment_binds_aggregation_to_selected_dataset_lineage() -> None:
    environment = _environment()
    mismatched_policy = AggregationPolicy(
        "quantforge_session_ohlcv",
        "1",
        {"missing_constituents": "diagnostic"},
    )

    with pytest.raises(ValidationPlanError, match="selected dataset lineage"):
        replace(
            environment,
            aggregation_policies=(
                ConfigurationReference.capture_aggregation_policy(mismatched_policy),
            ),
        )
    with pytest.raises(ValidationPlanError, match="selected dataset lineage"):
        replace(
            environment,
            aggregation_policies=(
                _component("prediction_rule", "not_an_aggregation_policy"),
            ),
        )


def test_standalone_dataset_binds_its_canonical_daily_timeframe() -> None:
    dataset = make_dataset(("100", "101"))
    provenance = DatasetProvenance.from_market_dataset(dataset)
    daily = Timeframe.us_equity(SessionInterval())
    baseline = replace(
        _environment(),
        dataset=provenance,
        aggregation_policies=(),
    )

    assert provenance.dataset_fingerprint == dataset.metadata.data_sha256
    assert provenance.dataset_ids == (dataset.metadata.dataset_id,)
    assert provenance.standalone_timeframe == daily
    assert baseline.timeframes == (daily,)
    with pytest.raises(ValidationPlanError, match="exactly match dataset provenance"):
        replace(
            baseline,
            timeframes=(Timeframe.us_equity(IntradayInterval(timedelta(minutes=5))),),
        )
    with pytest.raises(ValidationPlanError, match="exactly match dataset provenance"):
        replace(
            baseline,
            timeframes=(daily, Timeframe.us_equity(SessionInterval(2))),
        )


def test_standalone_dataset_uses_exchange_not_provider_timezone() -> None:
    dataset = make_dataset(("100", "101"), calendar="XLON")

    provenance = DatasetProvenance.from_market_dataset(dataset)

    assert dataset.metadata.provider_timezone == "America/New_York"
    assert provenance.standalone_timeframe is not None
    assert provenance.standalone_timeframe.session_policy.calendar_name == "XLON"
    assert (
        provenance.standalone_timeframe.session_policy.timezone_name == "Europe/London"
    )


@pytest.mark.parametrize(
    ("build_dataset", "error_message"),
    [
        (
            lambda: make_dataset(
                ("100", "101"), adjustment_mode=AdjustmentMode.SPLIT_ADJUSTED
            ),
            "adjusted market data",
        ),
        (
            lambda: make_dataset(("100", "101"), corporate_actions_complete=False),
            "complete explicit corporate actions",
        ),
        (
            lambda: make_dataset(("100", "101"), adjusted_fields_used=True),
            "consistent raw provider OHLCV",
        ),
        (
            lambda: make_dataset(
                ("100", "101"),
                sessions=(date(2024, 7, 1), date(2024, 7, 3)),
                missing_sessions=(date(2024, 7, 2),),
            ),
            "missing expected sessions",
        ),
    ],
)
def test_standalone_accounting_compatibility_is_trading_only(
    build_dataset: Callable[[], MarketDataset],
    error_message: str,
) -> None:
    dataset = build_dataset()
    provenance = DatasetProvenance.from_market_dataset(dataset)
    prediction = replace(_environment(), dataset=provenance, aggregation_policies=())
    assert prediction.dataset.market_data_metadata == dataset.metadata
    with pytest.raises(ValidationPlanError, match=error_message):
        replace(
            _environment(ResearchStudyType.TRADING_BACKTEST),
            dataset=provenance,
            aggregation_policies=(),
        )


@pytest.mark.parametrize("dividend_policy", list(DividendPolicy))
def test_standalone_trading_binds_dividend_policy_to_validated_action_metadata(
    dividend_policy: DividendPolicy,
) -> None:
    dataset = make_dataset(("100", "101"), dividends=((date(2024, 7, 2), "1"),))
    environment = _environment(
        ResearchStudyType.TRADING_BACKTEST,
        execution=_backtest_provenance(dividend_policy=dividend_policy),
    )
    provenance = DatasetProvenance.from_market_dataset(dataset)
    if dividend_policy is DividendPolicy.REJECT_IF_DIVIDENDS:
        with pytest.raises(
            ValidationPlanError, match="dataset contains cash dividends"
        ):
            replace(environment, dataset=provenance, aggregation_policies=())
    else:
        accepted = replace(environment, dataset=provenance, aggregation_policies=())
        plan = _session_plan(environment=accepted, horizon_sessions=0)
        manifest = serialize_validation_plan(plan)
        assert dataset.metadata.corporate_action_snapshot_id.encode() in manifest
        assert dividend_policy.value.encode() in manifest
        assert validate_validation_plan_manifest(plan, manifest) == plan.to_manifest()


def test_standalone_metadata_preserves_basis_actions_and_cache_identity() -> None:
    plain = make_dataset(("100", "101"))
    actions = make_dataset(("100", "101"), splits=((date(2024, 7, 2), "2"),))
    baseline = replace(
        _environment(ResearchStudyType.TRADING_BACKTEST),
        dataset=DatasetProvenance.from_market_dataset(plain),
        aggregation_policies=(),
    )
    changed = replace(baseline, dataset=DatasetProvenance.from_market_dataset(actions))
    assert baseline.dataset.dataset_fingerprint == changed.dataset.dataset_fingerprint
    assert baseline.environment_id != changed.environment_id
    original = _session_plan(environment=baseline, horizon_sessions=0)
    plan = _session_plan(environment=changed, horizon_sessions=0)
    content = serialize_validation_plan(plan)
    metadata = cast(
        PrimitiveMapping, changed.dataset.to_primitive()["market_data_metadata"]
    )
    assert metadata["adjustment_mode"] == "unadjusted"
    assert metadata["ohlc_basis"] == "raw_provider"
    assert metadata["volume_basis"] == "raw_provider"
    assert metadata["split_count"] == 1
    assert metadata["corporate_actions_complete"] is True
    assert (
        metadata["corporate_action_snapshot_id"]
        == actions.metadata.corporate_action_snapshot_id
    )
    with pytest.raises(ValidationPlanIdentityError):
        validate_validation_plan_manifest(plan, serialize_validation_plan(original))
    metadata["split_count"] = 0
    assert serialize_validation_plan(plan) == content


def test_standalone_capture_rejects_forged_action_metadata() -> None:
    dataset = make_dataset(("100", "101"), dividends=((date(2024, 7, 2), "1"),))
    forged = replace(dataset, metadata=replace(dataset.metadata, dividend_count=0))
    with pytest.raises(MarketDataValidationError, match="counts or sessions"):
        DatasetProvenance.from_market_dataset(forged)


def test_dataset_provenance_requires_typed_factory_capture() -> None:
    with pytest.raises(TypeError, match="captured from a MarketDataset"):
        DatasetProvenance()


@pytest.mark.parametrize(
    "feed_scope",
    [
        FeedScope.consolidated(),
        FeedScope.iex_only(),
        FeedScope.single_venue("XNYS"),
        FeedScope.provider_defined("fixture_feed"),
    ],
)
@pytest.mark.parametrize("requirement_role", ["primary", "contextual"])
def test_context_feed_scope_matches_selected_family_and_survives_capture(
    feed_scope: FeedScope,
    requirement_role: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    family = replace(_family(), feed_scope=feed_scope)
    primary = family.source_timeframe
    daily = Timeframe.us_equity(SessionInterval())
    requirements = PredictionContextRequirements(
        PredictionTimeframeRequirement(primary, feed_scope),
        (PredictionTimeframeRequirement(daily, feed_scope),),
    )
    component = FixtureMultiTimeframeRule(requirements)
    provenance = ResearchRuleProvenance.capture_prediction(component)
    # Every requirement is indicator-free: feed validation must not depend on
    # the presence of an IndicatorTimeframeBinding.
    assert provenance.required_indicator_bindings == ()
    environment = ResearchEnvironment(
        ResearchStudyType.PREDICTION,
        DatasetProvenance.from_dataset_family(family, (SOURCE_ID, DAILY_ID)),
        (primary, daily),
        provenance,
        aggregation_policies=(
            ConfigurationReference.capture_aggregation_policy(
                family.aggregation_policy
            ),
        ),
        outcomes=(_session_outcome(1),),
    )
    mismatched_scope = (
        FeedScope.iex_only()
        if feed_scope == FeedScope.consolidated()
        else FeedScope.consolidated()
    )
    with pytest.raises(ValidationPlanError, match="context feed scope"):
        replace(
            environment,
            dataset=DatasetProvenance.from_dataset_family(
                replace(family, feed_scope=mismatched_scope), (SOURCE_ID, DAILY_ID)
            ),
        )

    original_id = environment.environment_id
    original_manifest = environment.to_primitive()
    component.context_requirements = PredictionContextRequirements(
        PredictionTimeframeRequirement(primary, mismatched_scope),
        (PredictionTimeframeRequirement(daily, mismatched_scope),),
    )
    assert replace(environment).to_primitive() == original_manifest
    assert environment.environment_id == original_id
    with pytest.raises(ValidationPlanError, match="context feed scope"):
        replace(
            environment,
            research_rule=ResearchRuleProvenance.capture_prediction(component),
        )

    # A custom component must not bypass the check for an indicator-free
    # contextual input merely because its primary input has a matching scope.
    component.context_requirements = requirements
    configuration = component.configuration()
    context = cast(PrimitiveMapping, configuration["context_requirements"])
    requirement = (
        cast(PrimitiveMapping, context["primary"])
        if requirement_role == "primary"
        else cast(list[PrimitiveMapping], context["contextual"])[0]
    )
    requirement["feed_scope"] = mismatched_scope.to_primitive()
    monkeypatch.setattr(component, "configuration", lambda: configuration)
    expected_timeframe = primary if requirement_role == "primary" else daily
    with pytest.raises(ValidationPlanError, match=expected_timeframe.configuration_id):
        replace(
            environment,
            research_rule=ResearchRuleProvenance.capture_prediction(component),
        )


def test_dataset_provenance_rejects_unknown_family_member() -> None:
    family = _family()

    with pytest.raises(ValidationPlanError, match="recorded in the supplied"):
        DatasetProvenance.from_dataset_family(
            family,
            ("not-in-family",),
        )


def test_dataset_provenance_embeds_verified_complete_family_manifest() -> None:
    family = _family()
    provenance = DatasetProvenance.from_dataset_family(family, (DAILY_ID,))
    primitive = provenance.to_primitive()
    dataset_family = cast(PrimitiveMapping, primitive["dataset_family"])
    expected_fingerprint = configuration_identity(
        {
            "dataset_family_manifest_id": family.manifest_id,
            "selected_dataset_references": [
                family.reference(DAILY_ID).to_primitive(include_feed_scope=True)
            ],
        }
    )

    assert provenance.dataset_fingerprint == expected_fingerprint
    assert provenance.family_manifest_id == family.manifest_id
    assert dataset_family["manifest_id"] == family.manifest_id
    assert dataset_family["manifest"] == family.to_manifest()


def test_dataset_family_fingerprint_changes_with_immutable_family_evidence() -> None:
    family = _family()
    changed_family = replace(family, provider_name="different-provider")

    baseline = DatasetProvenance.from_dataset_family(family, (DAILY_ID,))
    changed = DatasetProvenance.from_dataset_family(changed_family, (DAILY_ID,))

    assert baseline.dataset_ids == changed.dataset_ids
    assert baseline.dataset_fingerprint != changed.dataset_fingerprint


def test_historical_native_indicator_is_not_silently_migrated() -> None:
    legacy_component = SimpleMovingAverage(SimpleMovingAverageParameters(3))
    explicit_component = SimpleMovingAverage(
        SimpleMovingAverageParameters(3),
        backend_id=NATIVE_INDICATOR_BACKEND,
    )
    timeframe = Timeframe.us_equity(SessionInterval())
    legacy = IndicatorProvenance.capture(legacy_component, timeframe)
    explicit = IndicatorProvenance.capture(explicit_component, timeframe)

    assert legacy.backend_identity == explicit.backend_identity
    assert legacy.legacy_native_configuration is True
    assert explicit.legacy_native_configuration is False
    assert legacy.configuration_id != explicit.configuration_id
    assert "backend" not in legacy.configuration_snapshot.to_primitive()
    explicit_configuration = explicit.configuration_snapshot.to_primitive()
    explicit_backend = cast(PrimitiveMapping, explicit_configuration["backend"])
    assert explicit_backend["backend_id"] == NATIVE_INDICATOR_BACKEND
    assert _session_plan(
        environment=_environment(indicator=legacy_component)
    ).plan_id != (
        _session_plan(environment=_environment(indicator=explicit_component)).plan_id
    )


def test_prediction_fixture_purges_future_labels_at_protected_boundary() -> None:
    plan = _session_plan(horizon_sessions=1)
    fixture = _fixture("prediction_session_boundaries.json")
    observation_values = cast(list[str], fixture["observations"])
    observations = tuple(_session(value) for value in observation_values)

    result = purge_development_observations(plan, 0, observations)

    assert _session_dates(result.retained) == tuple(
        date.fromisoformat(value)
        for value in cast(list[str], fixture["expected_retained_development"])
    )
    assert _session_dates(result.purged) == tuple(
        date.fromisoformat(value)
        for value in cast(list[str], fixture["expected_purged_development"])
    )
    selection = plan.folds[0].selection
    assert selection is not None
    assert result.protected_window_id == selection.window_id


def test_explicit_session_embargo_adds_separation_beyond_label_horizon() -> None:
    observations = tuple(
        _session(value)
        for value in (
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
            "2024-01-09",
        )
    )

    without_embargo = purge_development_observations(
        _session_plan(horizon_sessions=1), 0, observations
    )
    with_embargo = purge_development_observations(
        _session_plan(horizon_sessions=1, embargo_sessions=1), 0, observations
    )

    assert _session_dates(without_embargo.purged) == (date(2024, 1, 8),)
    assert _session_dates(with_embargo.purged) == (
        date(2024, 1, 5),
        date(2024, 1, 8),
    )
    assert with_embargo.result_id != without_embargo.result_id


def test_exchange_session_purge_uses_observed_label_horizon() -> None:
    plan = _session_plan(horizon_sessions=1)
    observations = tuple(
        _session(value)
        for value in (
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-09",
        )
    )

    result = purge_development_observations(plan, 0, observations)

    assert _session_dates(result.purged) == (date(2024, 1, 5),)


def test_exchange_session_purge_rejects_missing_protected_chronology() -> None:
    observations = tuple(
        _session(value)
        for value in (
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
        )
    )

    with pytest.raises(ValidationPlanError, match="protected validation window"):
        purge_development_observations(
            _session_plan(horizon_sessions=1), 0, observations
        )


def test_zero_session_separation_does_not_require_protected_observations() -> None:
    observations = tuple(
        _session(value)
        for value in (
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
        )
    )

    result = purge_development_observations(
        _session_plan(
            horizon_sessions=0,
            environment=_environment(ResearchStudyType.TRADING_BACKTEST),
        ),
        0,
        observations,
    )

    assert result.retained == observations
    assert result.purged == ()


def test_selection_and_test_labels_cannot_cross_test_or_holdout_boundaries() -> None:
    plan = _session_plan(horizon_sessions=1)
    observations = (
        _session("2024-01-09"),
        _session("2024-01-10"),
        _session("2024-01-11"),
    )

    selection = purge_partition_observations(
        plan,
        0,
        PartitionRole.SELECTION,
        observations,
    )
    test = purge_partition_observations(
        plan,
        0,
        PartitionRole.WALK_FORWARD_TEST,
        observations,
    )

    assert selection.retained == ()
    assert _session_dates(selection.purged) == (date(2024, 1, 9),)
    assert selection.protected_window_id == plan.folds[0].test.window_id
    assert test.retained == ()
    assert _session_dates(test.purged) == (date(2024, 1, 10),)
    assert test.protected_window_id == plan.final_holdout.window.window_id


def test_warm_up_context_precedes_and_is_excluded_from_protected_membership() -> None:
    plan = _session_plan()
    observations = tuple(
        _session(value)
        for value in (
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
            "2024-01-09",
            "2024-01-10",
        )
    )

    selection = plan.folds[0].selection
    assert selection is not None
    selected = select_window_observations(selection, observations)

    assert _session_dates(selected.warm_up_context) == (
        date(2024, 1, 5),
        date(2024, 1, 8),
    )
    assert _session_dates(selected.study_observations) == (date(2024, 1, 9),)
    assert selected.to_primitive()["warm_up_eligible_for_selection"] is False
    assert set(selected.warm_up_context).isdisjoint(selected.study_observations)


def test_multi_timeframe_warm_up_is_validated_and_selected_per_source() -> None:
    daily = Timeframe.us_equity(SessionInterval())
    weekly = Timeframe.us_equity(TradingWeekInterval())
    daily_indicator = cast(
        Indicator, SimpleMovingAverage(SimpleMovingAverageParameters(3))
    )
    weekly_indicator = cast(
        Indicator, SimpleMovingAverage(SimpleMovingAverageParameters(5))
    )
    rule = _rule(
        "prediction_rule",
        "multi_timeframe_rule",
        (daily_indicator, weekly_indicator),
        warm_up_observations=3,
        primary_timeframe=daily,
        context_indicator_bindings=(
            (daily, daily_indicator),
            (weekly, weekly_indicator),
        ),
    )
    family = _multi_timeframe_family(daily, weekly)
    environment = ResearchEnvironment(
        ResearchStudyType.PREDICTION,
        DatasetProvenance.from_dataset_family(
            family,
            (DAILY_ID, WEEKLY_ID),
        ),
        (daily, weekly),
        rule,
        aggregation_policies=(
            ConfigurationReference.capture_aggregation_policy(
                family.aggregation_policy
            ),
        ),
        indicators=(
            IndicatorProvenance.capture(
                cast(IndicatorComponent, daily_indicator), daily
            ),
            IndicatorProvenance.capture(
                cast(IndicatorComponent, weekly_indicator), weekly
            ),
        ),
        outcomes=(_session_outcome(1),),
    )
    scalar_fold = ValidationFold(
        "fold_1",
        _session_window(
            "development_1",
            PartitionRole.DEVELOPMENT,
            "2024-01-02",
            "2024-01-08",
        ),
        _session_window(
            "test_1",
            PartitionRole.WALK_FORWARD_TEST,
            "2024-01-10",
        ),
        _session_window(
            "selection_1",
            PartitionRole.SELECTION,
            "2024-01-09",
        ),
    )
    scalar_holdout = FinalHoldout(
        _session_window(
            "reserved_holdout",
            PartitionRole.FINAL_HOLDOUT,
            "2024-01-11",
            "2024-01-12",
        ),
        "untouched",
    )
    with pytest.raises(ValidationPlanError, match="every configured source"):
        ValidationPlan(
            "unsafe_scalar_warm_up",
            environment,
            (scalar_fold,),
            scalar_holdout,
            PurgePolicy(TemporalOffset.sessions(1), TemporalOffset.sessions(0)),
            TrainingWindowMode.EXPANDING,
        )

    requirements = (
        TimeframeWarmUpRequirement(daily, 2),
        TimeframeWarmUpRequirement(weekly, 4),
    )

    def bind_warm_up(window: ValidationWindow) -> ValidationWindow:
        return replace(
            window,
            warm_up_observations=0,
            warm_up_by_timeframe=requirements,
        )

    fold = replace(
        scalar_fold,
        development=bind_warm_up(scalar_fold.development),
        selection=bind_warm_up(cast(ValidationWindow, scalar_fold.selection)),
        test=bind_warm_up(scalar_fold.test),
    )
    plan = ValidationPlan(
        "source_specific_warm_up",
        environment,
        (fold,),
        replace(
            scalar_holdout,
            window=bind_warm_up(scalar_holdout.window),
        ),
        PurgePolicy(TemporalOffset.sessions(1), TemporalOffset.sessions(0)),
        TrainingWindowMode.EXPANDING,
    )
    selection = cast(ValidationWindow, plan.folds[0].selection)
    daily_chronology = tuple(
        _session(item)
        for item in (
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
            "2024-01-09",
        )
    )
    weekly_chronology = tuple(
        _session(item)
        for item in (
            "2023-12-11",
            "2023-12-18",
            "2023-12-26",
            "2024-01-02",
            "2024-01-09",
        )
    )

    selected_daily = select_window_observations(
        selection,
        daily_chronology,
        source_timeframe=daily,
    )
    selected_weekly = select_window_observations(
        selection,
        weekly_chronology,
        source_timeframe=weekly,
    )

    assert len(selected_daily.warm_up_context) == 2
    assert len(selected_weekly.warm_up_context) == 4
    assert selected_daily.source_timeframe_configuration_id == daily.configuration_id
    assert selected_weekly.source_timeframe_configuration_id == weekly.configuration_id
    with pytest.raises(ValidationPlanError, match="source timeframe is required"):
        select_window_observations(selection, daily_chronology)


def test_rule_requires_duplicate_indicator_configuration_on_each_source() -> None:
    daily = Timeframe.us_equity(SessionInterval())
    weekly = Timeframe.us_equity(TradingWeekInterval())
    shared_indicator = cast(
        Indicator, SimpleMovingAverage(SimpleMovingAverageParameters(5))
    )
    rule = _rule(
        "prediction_rule",
        "shared_multi_timeframe_rule",
        (shared_indicator,),
        warm_up_observations=1,
        primary_timeframe=daily,
        context_indicator_bindings=(
            (daily, shared_indicator),
            (weekly, shared_indicator),
        ),
    )
    family = _multi_timeframe_family(daily, weekly)

    with pytest.raises(ValidationPlanError, match="source-timeframe indicator"):
        ResearchEnvironment(
            ResearchStudyType.PREDICTION,
            DatasetProvenance.from_dataset_family(
                family,
                (DAILY_ID, WEEKLY_ID),
            ),
            (daily, weekly),
            rule,
            aggregation_policies=(
                ConfigurationReference.capture_aggregation_policy(
                    family.aggregation_policy
                ),
            ),
            indicators=(
                IndicatorProvenance.capture(
                    cast(IndicatorComponent, shared_indicator), daily
                ),
            ),
            outcomes=(_session_outcome(1),),
        )


def test_appending_future_data_cannot_change_historical_membership() -> None:
    plan = _session_plan()
    history = tuple(
        _session(value)
        for value in (
            "2024-01-02",
            "2024-01-03",
            "2024-01-04",
            "2024-01-05",
            "2024-01-08",
            "2024-01-09",
        )
    )
    future = (_session("2024-01-10"), _session("2024-01-11"))

    original = purge_development_observations(plan, 0, history)
    appended = purge_development_observations(plan, 0, (*history, *future))

    assert appended.retained == original.retained
    assert appended.purged == original.purged
    assert appended.result_id == original.result_id


@pytest.mark.parametrize("study_type", list(ResearchStudyType))
@pytest.mark.parametrize("include_daily", [False, True])
def test_session_plan_rejects_selected_intraday_sources(
    study_type: ResearchStudyType,
    include_daily: bool,
) -> None:
    intraday = Timeframe.us_equity(IntradayInterval(timedelta(minutes=5)))
    indicator = cast(Indicator, SimpleMovingAverage(SimpleMovingAverageParameters(1)))
    environment = _environment(study_type, timeframe=intraday, indicator=indicator)
    if include_daily:
        daily = Timeframe.us_equity(SessionInterval())
        family = _multi_timeframe_family(intraday, daily)
        environment = replace(
            environment,
            dataset=DatasetProvenance.from_dataset_family(
                family, (DAILY_ID, WEEKLY_ID)
            ),
            timeframes=(intraday, daily),
            research_rule=_rule(
                environment.research_rule.component_type,
                "intraday_rule",
                (indicator,),
                primary_timeframe=intraday,
            ),
        )
    template = _session_plan()

    def without_warm_up(window: ValidationWindow) -> ValidationWindow:
        return replace(
            window,
            warm_up_observations=0,
            warm_up_by_timeframe=tuple(
                TimeframeWarmUpRequirement(timeframe, 0)
                for timeframe in environment.timeframes
            )
            if include_daily
            else (),
        )

    fold = template.folds[0]
    with pytest.raises(ValidationPlanError, match="require timestamp"):
        replace(
            template,
            environment=environment,
            folds=(
                replace(
                    fold,
                    development=without_warm_up(fold.development),
                    selection=without_warm_up(cast(ValidationWindow, fold.selection)),
                    test=without_warm_up(fold.test),
                ),
            ),
            final_holdout=replace(
                template.final_holdout,
                window=without_warm_up(template.final_holdout.window),
            ),
            purge_policy=PurgePolicy(
                TemporalOffset.sessions(
                    1 if study_type is ResearchStudyType.PREDICTION else 0
                ),
                TemporalOffset.sessions(0),
            ),
        )


@pytest.mark.parametrize("source_specific", [False, True])
def test_session_window_cannot_select_intraday_bars_as_session_keys(
    source_specific: bool,
) -> None:
    intraday = Timeframe.us_equity(IntradayInterval(timedelta(minutes=5)))
    window = _session_window(
        "selection", PartitionRole.SELECTION, "2024-01-09", warm_up=0
    )
    if source_specific:
        window = replace(
            window,
            warm_up_by_timeframe=(TimeframeWarmUpRequirement(intraday, 0),),
        )
    with pytest.raises(ValidationPlanError, match="require timestamp"):
        select_window_observations(
            window, (_session("2024-01-09"),), source_timeframe=intraday
        )


def test_intraday_timestamp_horizon_and_embargo_are_exact() -> None:
    timeframe = Timeframe.us_equity(IntradayInterval(timedelta(minutes=5)))
    environment = _environment(
        timeframe=timeframe,
        outcome=_timestamp_outcome(timedelta(minutes=10)),
    )
    development = ValidationWindow(
        "intraday_development",
        PartitionRole.DEVELOPMENT,
        ValidationInterval(
            _timestamp("2024-01-02T14:30:00+00:00"),
            _timestamp("2024-01-02T15:00:00+00:00"),
        ),
        2,
    )
    selection = ValidationWindow(
        "intraday_selection",
        PartitionRole.SELECTION,
        ValidationInterval(
            _timestamp("2024-01-02T15:15:00+00:00"),
            _timestamp("2024-01-02T15:30:00+00:00"),
        ),
        2,
    )
    test = ValidationWindow(
        "intraday_test",
        PartitionRole.WALK_FORWARD_TEST,
        ValidationInterval(
            _timestamp("2024-01-02T15:45:00+00:00"),
            _timestamp("2024-01-02T16:00:00+00:00"),
        ),
        2,
    )
    plan = ValidationPlan(
        "intraday_validation",
        environment,
        (ValidationFold("intraday_fold", development, test, selection),),
        FinalHoldout(
            ValidationWindow(
                "intraday_holdout",
                PartitionRole.FINAL_HOLDOUT,
                ValidationInterval(
                    _timestamp("2024-01-02T16:15:00+00:00"),
                    _timestamp("2024-01-02T16:30:00+00:00"),
                ),
                2,
            ),
            "reserved intraday interval",
        ),
        PurgePolicy(
            TemporalOffset.duration(timedelta(minutes=10)),
            TemporalOffset.duration(timedelta(minutes=5)),
        ),
        TrainingWindowMode.ROLLING,
    )
    observations = tuple(
        _timestamp(value)
        for value in (
            "2024-01-02T09:30:00-05:00",
            "2024-01-02T09:45:00-05:00",
            "2024-01-02T10:00:00-05:00",
            "2024-01-02T10:15:00-05:00",
        )
    )

    purged = purge_development_observations(plan, 0, observations)

    assert _timestamps(purged.retained) == (
        datetime(2024, 1, 2, 14, 30, tzinfo=UTC),
        datetime(2024, 1, 2, 14, 45, tzinfo=UTC),
    )
    assert _timestamps(purged.purged) == (datetime(2024, 1, 2, 15, 0, tzinfo=UTC),)

    # Distinct intraday keys in the same session supply exact prior-bar context.
    source_observations = tuple(
        _timestamp(f"2024-01-02T{clock_time}:00+00:00")
        for clock_time in ("15:05", "15:10", "15:15", "15:20")
    )
    source_window = replace(
        selection,
        warm_up_observations=0,
        warm_up_by_timeframe=(TimeframeWarmUpRequirement(timeframe, 2),),
    )
    selected = select_window_observations(
        source_window, source_observations, source_timeframe=timeframe
    )
    assert selected.warm_up_context == source_observations[:2]
    assert selected.study_observations == source_observations[2:]
    assert selected.source_timeframe_configuration_id == timeframe.configuration_id
    appended = select_window_observations(
        source_window,
        (*source_observations, _timestamp("2024-01-02T16:00:00+00:00")),
        source_timeframe=timeframe,
    )
    assert appended.selection_id == selected.selection_id


def test_backtest_fixture_uses_same_contract_without_prediction_results() -> None:
    environment = _environment(ResearchStudyType.TRADING_BACKTEST)
    fixture = _fixture("backtest_session_boundaries.json")
    plan = ValidationPlan(
        "backtest_validation",
        environment,
        (
            ValidationFold(
                "backtest_fold",
                _session_window(
                    "backtest_development",
                    PartitionRole.DEVELOPMENT,
                    "2024-01-02",
                    "2024-01-05",
                ),
                _session_window(
                    "backtest_test",
                    PartitionRole.WALK_FORWARD_TEST,
                    "2024-01-08",
                    "2024-01-09",
                ),
            ),
        ),
        FinalHoldout(
            _session_window(
                "backtest_holdout",
                PartitionRole.FINAL_HOLDOUT,
                "2024-01-10",
                "2024-01-11",
            ),
            "reserved backtest confirmation",
        ),
        PurgePolicy(TemporalOffset.sessions(0), TemporalOffset.sessions(0)),
        TrainingWindowMode.ROLLING,
    )
    observations = tuple(
        _session(value) for value in cast(list[str], fixture["observations"])
    )

    result = purge_development_observations(plan, 0, observations)

    assert result.purged == ()
    assert _session_dates(result.retained) == tuple(
        date.fromisoformat(value)
        for value in cast(list[str], fixture["expected_retained_development"])
    )
    assert plan.environment.outcomes == ()
    assert plan.environment.execution is not None
    assert "equity" not in serialize_validation_plan(plan).decode()
    assert "prediction_metric" not in serialize_validation_plan(plan).decode()


@pytest.mark.parametrize("study_type", list(ResearchStudyType))
def test_plan_rejects_caller_owned_fold_list(study_type: ResearchStudyType) -> None:
    plan = _session_plan(
        environment=_environment(study_type),
        horizon_sessions=1 if study_type is ResearchStudyType.PREDICTION else 0,
    )
    caller_folds = list(plan.folds)
    with pytest.raises(ValidationPlanError, match="folds must be a tuple"):
        replace(plan, folds=cast(tuple[ValidationFold, ...], caller_folds))


@pytest.mark.parametrize("study_type", list(ResearchStudyType))
def test_explicit_fold_tuple_detaches_mutable_input(
    study_type: ResearchStudyType,
) -> None:
    baseline = _session_plan(
        environment=_environment(study_type),
        horizon_sessions=1 if study_type is ResearchStudyType.PREDICTION else 0,
    )
    caller_folds = list(baseline.folds)
    plan = replace(baseline, folds=tuple(caller_folds))
    content = serialize_validation_plan(plan)
    overlapping_fold = replace(
        baseline.folds[0],
        test=_session_window(
            "overlapping_test", PartitionRole.WALK_FORWARD_TEST, "2024-01-11"
        ),
    )
    caller_folds.append(overlapping_fold)
    caller_folds.pop(0)
    # The same replacement would overlap the reserved holdout if accepted.
    with pytest.raises(ValidationPlanError, match="final holdout"):
        replace(plan, folds=tuple(caller_folds))
    assert plan.folds == baseline.folds
    assert plan.axis is baseline.axis
    assert plan.plan_id == baseline.plan_id
    assert serialize_validation_plan(plan) == content
    caller_folds.clear()
    assert plan.folds == baseline.folds
    assert validate_validation_plan_manifest(plan, content) == plan.to_manifest()


@pytest.mark.parametrize("folds", [(), (object(),)])
def test_plan_rejects_empty_or_invalid_fold_tuple(folds: tuple[object, ...]) -> None:
    with pytest.raises(ValidationPlanError, match="requires explicit folds"):
        replace(_session_plan(), folds=cast(tuple[ValidationFold, ...], folds))


def test_explicit_folds_represent_expanding_and_rolling_progression() -> None:
    first = ValidationFold(
        "fold_1",
        _session_window(
            "development_1", PartitionRole.DEVELOPMENT, "2024-01-02", "2024-01-05"
        ),
        _session_window("test_1", PartitionRole.WALK_FORWARD_TEST, "2024-01-09"),
        _session_window("selection_1", PartitionRole.SELECTION, "2024-01-08"),
    )
    expanding_second = ValidationFold(
        "fold_2",
        _session_window(
            "development_2", PartitionRole.DEVELOPMENT, "2024-01-02", "2024-01-09"
        ),
        _session_window("test_2", PartitionRole.WALK_FORWARD_TEST, "2024-01-11"),
        _session_window("selection_2", PartitionRole.SELECTION, "2024-01-10"),
    )
    rolling_second = replace(
        expanding_second,
        development=_session_window(
            "development_2", PartitionRole.DEVELOPMENT, "2024-01-03", "2024-01-09"
        ),
    )
    holdout = FinalHoldout(
        _session_window(
            "holdout", PartitionRole.FINAL_HOLDOUT, "2024-01-12", "2024-01-16"
        ),
        "reserved",
    )
    purge = PurgePolicy(TemporalOffset.sessions(1), TemporalOffset.sessions(0))

    expanding = ValidationPlan(
        "expanding",
        _environment(),
        (first, expanding_second),
        holdout,
        purge,
        TrainingWindowMode.EXPANDING,
    )
    rolling = ValidationPlan(
        "rolling",
        _environment(),
        (first, rolling_second),
        holdout,
        purge,
        TrainingWindowMode.ROLLING,
    )

    assert len(expanding.folds) == 2
    assert len(rolling.folds) == 2
    assert expanding.plan_id != rolling.plan_id


@pytest.mark.parametrize(
    "build",
    [
        lambda: ValidationFold(
            "overlap",
            _session_window(
                "development",
                PartitionRole.DEVELOPMENT,
                "2024-01-02",
                "2024-01-09",
            ),
            _session_window("test", PartitionRole.WALK_FORWARD_TEST, "2024-01-10"),
            _session_window("selection", PartitionRole.SELECTION, "2024-01-09"),
        ),
        lambda: ValidationInterval(
            _session("2024-01-02"),
            _timestamp("2024-01-02T21:00:00+00:00"),
        ),
        lambda: ExchangeSessionBoundary(date(2024, 1, 6)),
        lambda: TimestampBoundary(datetime(2024, 1, 2, 15, 0)),
    ],
)
def test_invalid_or_leaking_boundaries_are_rejected(
    build: Callable[[], object],
) -> None:
    with pytest.raises(ValidationPlanError):
        build()


def test_holdout_overlap_is_rejected() -> None:
    plan = _session_plan()
    overlapping = FinalHoldout(
        _session_window(
            "overlapping_holdout", PartitionRole.FINAL_HOLDOUT, "2024-01-10"
        ),
        "invalid overlap",
    )

    with pytest.raises(ValidationPlanError, match="final holdout"):
        replace(plan, final_holdout=overlapping)


def test_warm_up_fails_closed_when_history_is_insufficient() -> None:
    window = _session_window(
        "selection", PartitionRole.SELECTION, "2024-01-09", warm_up=2
    )

    with pytest.raises(ValidationPlanError, match="insufficient historical"):
        select_window_observations(
            window,
            (_session("2024-01-08"), _session("2024-01-09")),
        )


def test_component_reference_rejects_declared_identity_mismatch() -> None:
    configuration: PrimitiveMapping = {"component": "rule", "version": "1"}

    with pytest.raises(ValidationPlanError, match="declared ID"):
        ConfigurationReference.capture(
            "prediction_rule",
            "rule",
            "1",
            configuration,
            configuration_id="f" * 64,
        )


def test_plan_rejects_outcome_horizon_that_exceeds_purge_horizon() -> None:
    environment = _environment(outcome=_session_outcome(2))

    with pytest.raises(ValidationPlanError, match="maximum configured outcome"):
        _session_plan(horizon_sessions=1, environment=environment)


def test_outcome_provenance_requires_typed_component_capture() -> None:
    with pytest.raises(TypeError, match="typed outcome component"):
        OutcomeProvenance()

    session_outcome = OutcomeProvenance.capture_exchange_sessions(
        ForwardReturnOutcomeLabeler(2)
    )
    timestamp_outcome = OutcomeProvenance.capture_timestamp(
        _TimestampOutcome(timedelta(minutes=10))
    )

    assert session_outcome.future_horizon == TemporalOffset.sessions(2)
    assert timestamp_outcome.future_horizon == TemporalOffset.duration(
        timedelta(minutes=10)
    )


def test_timestamp_outcome_rejects_invalid_component_horizon() -> None:
    with pytest.raises(ValidationPlanError, match="non-negative timedelta"):
        OutcomeProvenance.capture_timestamp(_TimestampOutcome(timedelta(minutes=-1)))


def test_trading_environment_requires_execution_provenance() -> None:
    environment = _environment(ResearchStudyType.TRADING_BACKTEST)

    with pytest.raises(ValidationPlanError, match="requires execution provenance"):
        replace(environment, execution=None)


def test_research_environment_requires_rule_type_for_study_type() -> None:
    prediction = _environment()
    trading = _environment(ResearchStudyType.TRADING_BACKTEST)
    indicator = cast(Indicator, SimpleMovingAverage(SimpleMovingAverageParameters(3)))

    with pytest.raises(ValidationPlanError, match="prediction_rule"):
        replace(
            prediction,
            research_rule=_rule(
                "trading_strategy",
                "wrong_strategy",
                (indicator,),
            ),
        )
    with pytest.raises(ValidationPlanError, match="trading_strategy"):
        replace(
            trading,
            research_rule=_rule("prediction_rule", "wrong_rule", (indicator,)),
        )


def test_plan_rejects_undersized_indicator_warm_up_context() -> None:
    plan = _session_plan()
    indicator = plan.environment.indicators[0]
    assert indicator.warm_up_observations == 3
    assert indicator.required_context_observations == 2
    assert indicator.to_primitive()["warm_up"] == {
        "observations_required_for_first_result": 3,
        "required_pre_window_context": 2,
    }

    undersized_test = replace(plan.folds[0].test, warm_up_observations=1)
    with pytest.raises(ValidationPlanError, match="warm-up observations for timeframe"):
        replace(
            plan,
            folds=(replace(plan.folds[0], test=undersized_test),),
        )

    undersized_holdout = replace(
        plan.final_holdout.window,
        warm_up_observations=1,
    )
    with pytest.raises(ValidationPlanError, match="warm-up observations for timeframe"):
        replace(
            plan,
            final_holdout=replace(plan.final_holdout, window=undersized_holdout),
        )


def test_indicator_provenance_requires_typed_component_capture() -> None:
    with pytest.raises(TypeError, match="typed indicator component"):
        IndicatorProvenance()


def test_plan_binds_rule_warm_up_and_required_indicators() -> None:
    required_indicator = cast(
        Indicator, SimpleMovingAverage(SimpleMovingAverageParameters(3))
    )
    rule = _rule(
        "prediction_rule",
        "history_dependent_rule",
        (required_indicator,),
        warm_up_observations=4,
    )
    environment = _environment(rule=rule)

    assert rule.required_context_observations == 3
    assert rule.to_primitive()["warm_up"] == {
        "observations_required_for_first_result": 4,
        "required_pre_window_context": 3,
        "source_timeframe_configuration_id": None,
    }
    with pytest.raises(ValidationPlanError, match="3 warm-up observations"):
        _session_plan(environment=environment)

    other_indicator = cast(
        Indicator, SimpleMovingAverage(SimpleMovingAverageParameters(4))
    )
    mismatched_rule = _rule(
        "prediction_rule",
        "mismatched_rule",
        (other_indicator,),
    )
    with pytest.raises(ValidationPlanError, match="indicator binding required"):
        _environment(rule=mismatched_rule)


def test_rule_and_backtest_provenance_require_typed_capture() -> None:
    with pytest.raises(TypeError, match="typed rule or strategy"):
        ResearchRuleProvenance()
    with pytest.raises(TypeError, match="typed backtest configuration"):
        BacktestProvenance()


def test_rule_provenance_derives_semantic_type_from_component() -> None:
    indicator = cast(Indicator, SimpleMovingAverage(SimpleMovingAverageParameters(3)))
    prediction_rule = _FixtureRule(
        "prediction_rule",
        "prediction_strategy",
        (indicator,),
        indicator.warm_up_observations,
        {},
    )

    with pytest.raises(ValidationPlanError, match="canonical configuration type"):
        ResearchRuleProvenance.capture_trading(prediction_rule)


def test_trading_environment_rejects_generic_execution_reference() -> None:
    environment = _environment(ResearchStudyType.TRADING_BACKTEST)
    generic_reference = _component(
        "prediction_rule",
        "not_execution_provenance",
    )

    with pytest.raises(ValidationPlanError, match="complete backtest configuration"):
        replace(
            environment,
            execution=cast(BacktestProvenance, generic_reference),
        )


def test_backtest_provenance_captures_complete_execution_configuration() -> None:
    provenance = _backtest_provenance()
    snapshot = provenance.configuration.configuration_snapshot.to_primitive()

    assert provenance.configuration.component_type == "backtest_configuration"
    assert {
        "commission",
        "execution",
        "fees",
        "sizing",
        "slippage",
        "split_policy",
    }.issubset(snapshot)
    with pytest.raises(ValidationPlanError, match="validated BacktestConfig"):
        BacktestProvenance.capture(
            cast(BacktestConfig, _IncompleteBacktestConfiguration())
        )


def test_component_configuration_is_detached_from_caller_mutation() -> None:
    configuration: PrimitiveMapping = {
        "component": "rule",
        "implementation_version": "1",
        "period": 3,
    }
    reference = ConfigurationReference.capture(
        "prediction_rule",
        "rule",
        "1",
        configuration,
    )
    expected = reference.to_primitive()

    configuration["period"] = 999

    assert reference.to_primitive() == expected


def test_timestamp_boundary_normalizes_offsets_to_utc() -> None:
    boundary = TimestampBoundary(
        datetime(2024, 1, 2, 9, 30, tzinfo=timezone(timedelta(hours=-5)))
    )

    assert boundary.timestamp == datetime(2024, 1, 2, 14, 30, tzinfo=UTC)
    assert boundary.to_primitive()["timestamp"] == "2024-01-02T14:30:00+00:00"
