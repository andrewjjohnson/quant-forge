import json
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta, timezone
from pathlib import Path
from typing import cast

import pytest

import quantforge.validation.partitioning as validation_partitioning
from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.data import (
    AdjustmentBasis,
    AdjustmentMode,
    AggregationPolicy,
    DatasetFamily,
    DatasetLineage,
    FeedScope,
)
from quantforge.data.identity import canonical_json_bytes
from quantforge.indicators import (
    NATIVE_INDICATOR_BACKEND,
    SimpleMovingAverage,
    SimpleMovingAverageParameters,
)
from quantforge.prediction import ForwardReturnOutcomeLabeler
from quantforge.timeframes import (
    ExchangeSession,
    ExchangeSessionPolicy,
    IntradayInterval,
    SessionInterval,
    Timeframe,
)
from quantforge.validation import (
    ConfigurationReference,
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

FINGERPRINT = "a" * 64
SOURCE_ID = "qf8-source-1m"
DAILY_ID = "qf8-derived-daily"
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


def _family(target_timeframe: Timeframe | None = None) -> DatasetFamily:
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
            {"missing_constituents": "reject"},
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


def _environment(
    study_type: ResearchStudyType = ResearchStudyType.PREDICTION,
    *,
    indicator: IndicatorProvenance | None = None,
    rule: ConfigurationReference | None = None,
    outcome: OutcomeProvenance | None = None,
    execution: ConfigurationReference | None = None,
    timeframe: Timeframe | None = None,
) -> ResearchEnvironment:
    selected_timeframe = timeframe or Timeframe.us_equity(SessionInterval())
    family = _family(selected_timeframe)
    selected_indicator = indicator or IndicatorProvenance.capture(
        SimpleMovingAverage(SimpleMovingAverageParameters(3))
    )
    selected_outcome = outcome or _session_outcome(1)
    selected_execution = execution
    if study_type is ResearchStudyType.TRADING_BACKTEST and selected_execution is None:
        selected_execution = _component(
            "execution_cost", "next_open_execution", slippage_bps=5
        )
    selected_rule = rule or _component(
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
        period=3,
    )
    return ResearchEnvironment(
        study_type=study_type,
        dataset=DatasetProvenance.from_dataset_family(FINGERPRINT, family, (DAILY_ID,)),
        timeframes=(selected_timeframe,),
        aggregation_policies=(
            _component(
                "aggregation_policy", "quantforge_session_ohlcv", missing="reject"
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
    assert plan.environment.dataset.dataset_fingerprint == FINGERPRINT
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
    explicit_native = IndicatorProvenance.capture(
        SimpleMovingAverage(
            SimpleMovingAverageParameters(3),
            backend_id=NATIVE_INDICATOR_BACKEND,
        )
    )
    changes = (
        replace(
            baseline,
            dataset=replace(baseline.dataset, dataset_fingerprint="b" * 64),
        ),
        _environment(timeframe=Timeframe.us_equity(SessionInterval(2))),
        replace(
            baseline,
            aggregation_policies=(
                _component(
                    "aggregation_policy",
                    "quantforge_session_ohlcv",
                    missing="diagnostic",
                ),
            ),
        ),
        _environment(indicator=explicit_native),
        _environment(rule=_component("prediction_rule", "fixture_rule", period=4)),
        _environment(outcome=_session_outcome(2)),
        replace(
            baseline,
            execution=_component(
                "execution_cost", "hypothetical_cost_overlay", slippage_bps=5
            ),
        ),
    )

    assert all(item.environment_id != baseline.environment_id for item in changes)


def test_environment_rejects_timeframe_that_mismatches_family_reference() -> None:
    environment = _environment()

    with pytest.raises(ValidationPlanError, match="family references"):
        replace(
            environment,
            timeframes=(Timeframe.us_equity(SessionInterval(2)),),
        )


def test_dataset_provenance_requires_manifest_bound_family_references() -> None:
    family = _family()
    references = (family.reference(DAILY_ID),)

    with pytest.raises(ValidationPlanError, match="complete dataset family"):
        DatasetProvenance(FINGERPRINT, (DAILY_ID,), references)


def test_dataset_provenance_rejects_unrelated_family_manifest() -> None:
    family = _family()
    unrelated_family = replace(family, provider_name="unrelated-provider")

    with pytest.raises(ValidationPlanError, match="supplied family manifest"):
        DatasetProvenance(
            FINGERPRINT,
            (DAILY_ID,),
            (family.reference(DAILY_ID),),
            unrelated_family,
        )


def test_dataset_provenance_embeds_verified_complete_family_manifest() -> None:
    family = _family()
    provenance = DatasetProvenance.from_dataset_family(FINGERPRINT, family, (DAILY_ID,))
    primitive = provenance.to_primitive()
    dataset_family = cast(PrimitiveMapping, primitive["dataset_family"])

    assert provenance.family_manifest_id == family.manifest_id
    assert dataset_family["manifest_id"] == family.manifest_id
    assert dataset_family["manifest"] == family.to_manifest()


def test_historical_native_indicator_is_not_silently_migrated() -> None:
    legacy = IndicatorProvenance.capture(
        SimpleMovingAverage(SimpleMovingAverageParameters(3))
    )
    explicit = IndicatorProvenance.capture(
        SimpleMovingAverage(
            SimpleMovingAverageParameters(3),
            backend_id=NATIVE_INDICATOR_BACKEND,
        )
    )

    assert legacy.backend_identity == explicit.backend_identity
    assert legacy.legacy_native_configuration is True
    assert explicit.legacy_native_configuration is False
    assert legacy.configuration_id != explicit.configuration_id
    assert "backend" not in legacy.configuration_snapshot.to_primitive()
    explicit_configuration = explicit.configuration_snapshot.to_primitive()
    explicit_backend = cast(PrimitiveMapping, explicit_configuration["backend"])
    assert explicit_backend["backend_id"] == NATIVE_INDICATOR_BACKEND
    assert _session_plan(environment=_environment(indicator=legacy)).plan_id != (
        _session_plan(environment=_environment(indicator=explicit)).plan_id
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


def test_exchange_session_purge_computes_one_calendar_cutoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _session_plan(horizon_sessions=1)
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
    original_resolver = validation_partitioning.resolve_exchange_session
    calendar_lookups = 0

    def counting_resolver(
        session_date: date,
        policy: ExchangeSessionPolicy,
    ) -> ExchangeSession:
        nonlocal calendar_lookups
        calendar_lookups += 1
        return original_resolver(session_date, policy)

    monkeypatch.setattr(
        validation_partitioning,
        "resolve_exchange_session",
        counting_resolver,
    )

    result = purge_development_observations(plan, 0, observations)

    assert _session_dates(result.purged) == (date(2024, 1, 8),)
    assert calendar_lookups == 1


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

    with pytest.raises(ValidationPlanError, match="prediction_rule"):
        replace(
            prediction,
            research_rule=_component("trading_strategy", "wrong_strategy"),
        )
    with pytest.raises(ValidationPlanError, match="trading_strategy"):
        replace(
            trading,
            research_rule=_component("prediction_rule", "wrong_rule"),
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
    with pytest.raises(ValidationPlanError, match="indicator warm-up observations"):
        replace(
            plan,
            folds=(replace(plan.folds[0], test=undersized_test),),
        )

    undersized_holdout = replace(
        plan.final_holdout.window,
        warm_up_observations=1,
    )
    with pytest.raises(ValidationPlanError, match="indicator warm-up observations"):
        replace(
            plan,
            final_holdout=replace(plan.final_holdout, window=undersized_holdout),
        )


def test_indicator_provenance_requires_typed_component_capture() -> None:
    with pytest.raises(TypeError, match="typed indicator component"):
        IndicatorProvenance()


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
