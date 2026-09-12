"""Exercise the real QF-31/QF-28/QF-11 flow through QF-8 contracts."""

import json
from dataclasses import dataclass, replace
from datetime import date, timedelta
from typing import cast

import pytest

from quantforge.data import (
    AdjustmentMode,
    AggregatedSessionBar,
    DatasetFamily,
    IntradayBar,
    IntradayContractValidationError,
    MarketDataset,
    TimeframeBarSeries,
    build_multi_timeframe_context,
)
from quantforge.prediction import (
    TechnicalConfluencePredictionRule,
    run_prediction_study,
)
from quantforge.timeframes import BarCompletion, IntradayInterval, SessionInterval
from quantforge.validation import (
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
    ValidationFold,
    ValidationInterval,
    ValidationPlan,
    ValidationPlanError,
    ValidationPlanIdentityError,
    ValidationWindow,
    purge_development_observations,
    purge_partition_observations,
    select_prediction_context_observations,
    select_window_observations,
    serialize_validation_plan,
    validate_validation_plan_manifest,
)
from tests.unit.data.test_multi_timeframe import (
    _family,  # pyright: ignore[reportPrivateUsage]
)
from tests.unit.helpers import make_dataset
from tests.unit.indicators.test_timeframe_evaluation import (
    _adjustment_basis,  # pyright: ignore[reportPrivateUsage]
)
from tests.unit.prediction.test_multi_timeframe_study import FixtureContextProvider
from tests.unit.prediction.test_technical_confluence import (
    _cases,  # pyright: ignore[reportPrivateUsage]
    _context,  # pyright: ignore[reportPrivateUsage]
    _dataset,  # pyright: ignore[reportPrivateUsage]
    _rule,  # pyright: ignore[reportPrivateUsage]
    _study,  # pyright: ignore[reportPrivateUsage]
)


@dataclass(frozen=True)
class PredictionCase:
    dataset: MarketDataset
    family: DatasetFamily
    series: tuple[TimeframeBarSeries, ...]
    rule: TechnicalConfluencePredictionRule
    plan: ValidationPlan
    as_of: TimestampBoundary


@pytest.fixture
def prediction_case() -> PredictionCase:
    rule = _rule()
    source_context = _context(_cases()[0])
    family = replace(_family(), adjustment_basis=_adjustment_basis())
    # Reuse the fixed, validated bar fixtures. Public QF-20 context construction
    # and the actual production rule/study runner are exercised below.
    series = tuple(
        TimeframeBarSeries._from_validated_artifact(  # pyright: ignore[reportPrivateUsage]
            family.reference(lineage.dataset_id),
            lineage.timeframe,
            tuple(
                bar
                for bar in source_context.bars_for(lineage.timeframe)
                if isinstance(bar, (IntradayBar, AggregatedSessionBar))
            ),
            dataset_family_manifest_id=family.manifest_id,
        )
        for lineage in family.datasets
    )
    dataset = _dataset()
    study, _ = _study(rule)
    requirements = rule.context_requirements.all_timeframes
    environment = ResearchEnvironment(
        ResearchStudyType.PREDICTION,
        DatasetProvenance.from_dataset_family(
            family, tuple(item.dataset_id for item in family.datasets)
        ),
        tuple(item.timeframe for item in requirements),
        ResearchRuleProvenance.capture_prediction(rule),
        prediction_dataset=DatasetProvenance.from_market_dataset(dataset),
        aggregation_policies=(
            ConfigurationReference.capture_aggregation_policy(
                family.aggregation_policy
            ),
        ),
        indicators=tuple(
            IndicatorProvenance.capture(
                cast(IndicatorComponent, indicator.indicator), item.timeframe
            )
            for item in requirements
            for indicator in item.indicators
        ),
        outcomes=(OutcomeProvenance.capture_exchange_sessions(study.outcome_labeler),),
    )
    counts = tuple(
        TimeframeWarmUpRequirement(
            item.timeframe,
            max(
                (
                    indicator.indicator.warm_up_observations - 1
                    for indicator in item.indicators
                ),
                default=0,
            ),
        )
        for item in requirements
    )

    def window(
        name: str, role: PartitionRole, start: int, end: int
    ) -> ValidationWindow:
        return ValidationWindow(
            name,
            role,
            ValidationInterval(
                ExchangeSessionBoundary(date(2024, 7, start)),
                ExchangeSessionBoundary(date(2024, 7, end)),
            ),
            warm_up_by_timeframe=counts,
        )

    plan = ValidationPlan(
        "production_multi_timeframe_prediction",
        environment,
        (
            ValidationFold(
                "fold",
                window("development", PartitionRole.DEVELOPMENT, 1, 2),
                window("test", PartitionRole.WALK_FORWARD_TEST, 5, 5),
                window("selection", PartitionRole.SELECTION, 3, 3),
            ),
        ),
        FinalHoldout(window("holdout", PartitionRole.FINAL_HOLDOUT, 8, 8), "reserved"),
        PurgePolicy(TemporalOffset.sessions(1), TemporalOffset.sessions(0)),
        TrainingWindowMode.EXPANDING,
    )
    as_of = TimestampBoundary(
        source_context.latest_bar_for(
            rule.context_requirements.primary.timeframe
        ).end_timestamp
    )
    return PredictionCase(dataset, family, series, rule, plan, as_of)


def test_real_prediction_flow_binds_both_inputs_and_session_outcomes(
    prediction_case: PredictionCase,
) -> None:
    case = prediction_case
    context = build_multi_timeframe_context(
        as_of=case.as_of.timestamp,
        primary_timeframe=case.rule.context_requirements.primary.timeframe,
        required_timeframes=case.rule.context_requirements.context_timeframe_requirements(),
        series=case.series,
    )
    study, _ = _study(case.rule)
    result = run_prediction_study(
        case.dataset, study, context_provider=FixtureContextProvider(context)
    )
    assert len(result.rows) == 1
    assert result.rows[0].outcome.outcome_session == date(2024, 7, 8)
    assert result.signals[0].signal_session == date(2024, 7, 5)
    keys = tuple(ExchangeSessionBoundary(bar.session_date) for bar in case.dataset.bars)
    purged = purge_partition_observations(
        case.plan, 0, PartitionRole.WALK_FORWARD_TEST, keys, source=case.dataset
    )
    assert purged.retained == ()
    assert purged.purged == (ExchangeSessionBoundary(date(2024, 7, 5)),)
    assert purged.source_dataset_id == case.dataset.metadata.dataset_id
    daily = case.plan.environment.prediction_dataset
    assert daily is not None
    selected = select_window_observations(
        case.plan.folds[0].test,
        keys,
        source=case.dataset,
        source_timeframe=daily.standalone_timeframe,
    )
    assert selected.study_observations == (ExchangeSessionBoundary(date(2024, 7, 5)),)
    assert selected.warm_up_context == (ExchangeSessionBoundary(date(2024, 7, 3)),)
    manifest = serialize_validation_plan(case.plan)
    assert (
        validate_validation_plan_manifest(case.plan, manifest)
        == case.plan.to_manifest()
    )
    assert case.dataset.metadata.dataset_id.encode() in manifest
    assert case.family.manifest_id.encode() in manifest


def test_context_selection_keeps_source_timestamps_and_prior_weekly_anchor(
    prediction_case: PredictionCase,
) -> None:
    case = prediction_case
    for source in case.series:
        result = select_prediction_context_observations(
            case.plan, case.plan.folds[0].test, source=source, as_of=case.as_of
        )
        selected = result.source_selection
        keys = (*selected.warm_up_context, *selected.study_observations)
        assert all(
            isinstance(key, TimestampBoundary) and key.timestamp <= case.as_of.timestamp
            for key in keys
        )
        assert selected.source_dataset_id == source.dataset_reference.dataset_id
        assert result.to_primitive()["eligible_for_outcome_selection"] is False
        if not isinstance(source.timeframe.interval, IntradayInterval):
            assert selected.study_observations == ()
            assert len(selected.warm_up_context) == 2


@pytest.mark.parametrize("changed_input", ["prediction_bars", "context_family"])
def test_either_input_changes_plan_identity_and_rejects_cache(
    prediction_case: PredictionCase, changed_input: str
) -> None:
    case = prediction_case
    if changed_input == "prediction_bars":
        dataset = make_dataset(
            ("10", "10", "10", "10", "15"),
            sessions=tuple(bar.session_date for bar in case.dataset.bars),
        )
        environment = replace(
            case.plan.environment,
            prediction_dataset=DatasetProvenance.from_market_dataset(dataset),
        )
    else:
        family = replace(case.family, provider_name="other-provider")
        environment = replace(
            case.plan.environment,
            dataset=DatasetProvenance.from_dataset_family(
                family, case.plan.environment.dataset.dataset_ids
            ),
        )
    changed = replace(case.plan, environment=environment)
    assert changed.environment.environment_id != case.plan.environment.environment_id
    assert changed.plan_id != case.plan.plan_id
    with pytest.raises(ValidationPlanIdentityError):
        validate_validation_plan_manifest(changed, serialize_validation_plan(case.plan))


def test_missing_or_forged_prediction_input_fails_closed(
    prediction_case: PredictionCase,
) -> None:
    case = prediction_case
    with pytest.raises(ValidationPlanError, match="separate prediction dataset"):
        replace(case.plan.environment, prediction_dataset=None)
    with pytest.raises(ValidationPlanError, match="standalone provenance"):
        replace(case.plan.environment, prediction_dataset=case.plan.environment.dataset)
    adjusted = make_dataset(("10", "11"), adjustment_mode=AdjustmentMode.SPLIT_ADJUSTED)
    with pytest.raises(ValidationPlanError, match="adjustment basis"):
        replace(
            case.plan.environment,
            prediction_dataset=DatasetProvenance.from_market_dataset(adjusted),
        )
    persisted = json.loads(serialize_validation_plan(case.plan))
    del persisted["environment"]["prediction_dataset"]
    with pytest.raises(ValidationPlanIdentityError):
        validate_validation_plan_manifest(case.plan, json.dumps(persisted).encode())


def test_purge_rejects_context_artifact_or_different_daily_data(
    prediction_case: PredictionCase,
) -> None:
    case = prediction_case
    daily = next(
        item
        for item in case.series
        if isinstance(item.timeframe.interval, SessionInterval)
    )
    with pytest.raises(ValidationPlanError, match="does not match plan dataset"):
        purge_development_observations(case.plan, 0, (), source=daily)
    changed = make_dataset(
        ("10", "10", "10", "10", "15"),
        sessions=tuple(bar.session_date for bar in case.dataset.bars),
    )
    with pytest.raises(ValidationPlanError, match="does not match plan dataset"):
        purge_development_observations(case.plan, 0, (), source=changed)


def test_context_rejects_foreign_windows_and_artifacts(
    prediction_case: PredictionCase,
) -> None:
    case = prediction_case
    with pytest.raises(ValidationPlanError, match="belong"):
        select_prediction_context_observations(
            case.plan,
            replace(case.plan.folds[0].test, name="foreign"),
            source=case.series[0],
            as_of=case.as_of,
        )
    with pytest.raises(ValidationPlanError, match="belong"):
        select_prediction_context_observations(
            case.plan,
            case.plan.folds[0].test,
            source=case.series[0],
            as_of=TimestampBoundary(case.as_of.timestamp + timedelta(days=3)),
        )
    other_family = replace(case.family, provider_name="foreign")
    other = TimeframeBarSeries._from_validated_artifact(  # pyright: ignore[reportPrivateUsage]
        other_family.reference(case.series[0].dataset_reference.dataset_id),
        case.series[0].timeframe,
        case.series[0].bars,
        dataset_family_manifest_id=other_family.manifest_id,
    )
    with pytest.raises(ValidationPlanError, match="family artifact"):
        select_prediction_context_observations(
            case.plan, case.plan.folds[0].test, source=other, as_of=case.as_of
        )


def test_observed_session_purge_handles_missing_session_and_embargo(
    prediction_case: PredictionCase,
) -> None:
    case = prediction_case
    dataset = make_dataset(
        ("10", "10", "10", "10", "11"),
        sessions=tuple(date(2024, 7, day) for day in (1, 2, 5, 8, 9)),
        missing_sessions=(date(2024, 7, 3),),
    )
    original = case.plan.folds[0]
    assert original.selection is not None

    def on_day(window: ValidationWindow, day: int) -> ValidationWindow:
        key = ExchangeSessionBoundary(date(2024, 7, day))
        return replace(window, interval=ValidationInterval(key, key))

    plan = replace(
        case.plan,
        environment=replace(
            case.plan.environment,
            prediction_dataset=DatasetProvenance.from_market_dataset(dataset),
        ),
        folds=(
            replace(
                original,
                selection=on_day(original.selection, 5),
                test=on_day(original.test, 8),
            ),
        ),
        final_holdout=replace(
            case.plan.final_holdout, window=on_day(case.plan.final_holdout.window, 9)
        ),
    )
    keys = tuple(ExchangeSessionBoundary(bar.session_date) for bar in dataset.bars)
    result = purge_development_observations(plan, 0, keys[:3], source=dataset)
    assert result.retained == keys[:1]
    assert result.purged == keys[1:2]
    assert purge_development_observations(plan, 0, keys, source=dataset) == result
    embargoed = replace(
        plan,
        purge_policy=PurgePolicy(
            TemporalOffset.sessions(1), TemporalOffset.sessions(1)
        ),
    )
    assert (
        purge_development_observations(embargoed, 0, keys, source=dataset).purged
        == keys[:2]
    )
    with pytest.raises(ValidationPlanError, match="exact prefix"):
        purge_development_observations(
            plan,
            0,
            (*keys[:2], ExchangeSessionBoundary(date(2024, 7, 3)), *keys[2:]),
            source=dataset,
        )


def test_context_ignores_future_bars_and_rejects_developing_input(
    prediction_case: PredictionCase,
) -> None:
    case = prediction_case
    source = next(
        item
        for item in case.series
        if item.timeframe == case.rule.context_requirements.primary.timeframe
    )
    baseline = select_prediction_context_observations(
        case.plan, case.plan.folds[0].test, source=source, as_of=case.as_of
    )
    last = cast(IntradayBar, source.bars[-1])
    future = replace(
        last,
        start_timestamp=last.end_timestamp,
        end_timestamp=last.end_timestamp + timedelta(minutes=5),
    )
    appended = TimeframeBarSeries._from_validated_artifact(  # pyright: ignore[reportPrivateUsage]
        source.dataset_reference,
        source.timeframe,
        (*source.bars, future),
        dataset_family_manifest_id=source.dataset_family_manifest_id,
    )
    # Same immutable fixture snapshot, represented by shorter/longer views.
    # Future keys cannot change a historical context selection or identity.
    assert (
        select_prediction_context_observations(
            case.plan, case.plan.folds[0].test, source=appended, as_of=case.as_of
        )
        == baseline
    )
    # The production timeframe itself rejects developing bars before a series
    # can claim them as completed context. The shared source validator also has
    # explicit developing-exposure coverage in the partition unit suite.
    with pytest.raises(IntradayContractValidationError, match="include-developing"):
        replace(future, completion=BarCompletion.DEVELOPING)
    short = TimeframeBarSeries._from_validated_artifact(  # pyright: ignore[reportPrivateUsage]
        case.series[-1].dataset_reference,
        case.series[-1].timeframe,
        (),
        dataset_family_manifest_id=case.family.manifest_id,
    )
    with pytest.raises(ValidationPlanError, match="insufficient historical"):
        select_prediction_context_observations(
            case.plan, case.plan.folds[0].test, source=short, as_of=case.as_of
        )


def test_early_close_context_cutoff_and_direct_result_guards(
    prediction_case: PredictionCase,
) -> None:
    case = prediction_case
    selection = case.plan.folds[0].selection
    assert selection is not None
    from datetime import UTC, datetime

    for timestamp in (
        datetime(2024, 7, 3, 13, 0, tzinfo=UTC),
        datetime(2024, 7, 3, 18, 0, tzinfo=UTC),
    ):
        with pytest.raises(ValidationPlanError, match="within the exchange session"):
            select_prediction_context_observations(
                case.plan,
                selection,
                source=case.series[0],
                as_of=TimestampBoundary(timestamp),
            )
    # July 3 is an early close at 17:00 UTC, not the usual 20:00 UTC.
    accepted = select_prediction_context_observations(
        case.plan,
        selection,
        source=case.series[0],
        as_of=TimestampBoundary(datetime(2024, 7, 3, 17, 0, tzinfo=UTC)),
    )
    with pytest.raises(ValidationPlanError, match="available by as_of"):
        replace(
            accepted, as_of=TimestampBoundary(datetime(2024, 7, 1, 12, 0, tzinfo=UTC))
        )
    with pytest.raises(ValidationPlanError, match="provenance is invalid"):
        replace(accepted, plan_id="forged")


@pytest.mark.parametrize(
    "mismatch", ["symbol", "calendar", "raw_split", "incomplete_actions"]
)
def test_dual_input_metadata_and_outcome_compatibility(
    prediction_case: PredictionCase, mismatch: str
) -> None:
    case = prediction_case
    if mismatch == "symbol":
        family = replace(case.family, canonical_symbol="QQQ")
        with pytest.raises(ValidationPlanError, match="symbol"):
            replace(
                case.plan.environment,
                dataset=DatasetProvenance.from_dataset_family(
                    family, case.plan.environment.dataset.dataset_ids
                ),
            )
        return
    dataset = (
        make_dataset(("100", "101"), calendar="XLON")
        if mismatch == "calendar"
        else make_dataset(("100", "50"), splits=((date(2024, 7, 2), "2"),))
        if mismatch == "raw_split"
        else make_dataset(("100", "101"), corporate_actions_complete=False)
    )
    with pytest.raises(ValidationPlanError, match=r"session policy|raw unadjusted"):
        replace(
            case.plan.environment,
            prediction_dataset=DatasetProvenance.from_market_dataset(dataset),
        )


def test_context_decision_cannot_use_a_missing_prediction_session(
    prediction_case: PredictionCase,
) -> None:
    case = prediction_case
    dataset = make_dataset(
        ("10", "10", "10", "11"),
        sessions=tuple(date(2024, 7, day) for day in (1, 2, 3, 8)),
        missing_sessions=(date(2024, 7, 5),),
    )
    plan = replace(
        case.plan,
        environment=replace(
            case.plan.environment,
            prediction_dataset=DatasetProvenance.from_market_dataset(dataset),
        ),
    )
    with pytest.raises(ValidationPlanError, match="observed in the prediction dataset"):
        select_prediction_context_observations(
            plan, plan.folds[0].test, source=case.series[0], as_of=case.as_of
        )
