from dataclasses import replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from quantforge.configuration import PrimitiveMapping
from quantforge.data import (
    SessionAggregationPolicy,
    TimeframeBarSeries,
    aggregate_session_dataset,
)
from quantforge.data.identity import canonical_json_bytes
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
    SessionOutcomeComponent,
    TemporalOffset,
    TimestampBoundary,
    TrainingWindowMode,
    ValidationFold,
    ValidationInterval,
    ValidationPlan,
    ValidationPlanError,
    ValidationPlanIdentityError,
    ValidationWindow,
    select_window_observations,
    serialize_validation_plan,
    validate_validation_plan_manifest,
)
from tests.unit.data.test_multi_timeframe import (
    _family,  # pyright: ignore[reportPrivateUsage]
    _persisted_source_dataset,  # pyright: ignore[reportPrivateUsage]
)
from tests.unit.helpers import make_dataset
from tests.unit.prediction.test_study import (
    FutureCloseOutcomeLabeler,
    _study,  # pyright: ignore[reportPrivateUsage]
)
from tests.unit.validation.test_validation_plans import (
    _environment,  # pyright: ignore[reportPrivateUsage]
    _session_plan,  # pyright: ignore[reportPrivateUsage]
)


class _SeparatelyConfiguredOutcome(FutureCloseOutcomeLabeler):
    """QF-11 does not require contract properties inside configuration()."""

    def configuration(self) -> PrimitiveMapping:
        return {
            "component_name": self.name,
            "component_type": "prediction_outcome_labeler",
            "contract_version": "1",
            "implementation_version": self.implementation_version,
            "parameters": {"future_sessions": self.required_future_sessions},
        }


@pytest.mark.parametrize(
    "attribute", ["required_market_fields", "result_schema_version"]
)
def test_session_outcome_contract_fields_change_plan_identity(attribute: str) -> None:
    events: list[str] = []
    labeler = _SeparatelyConfiguredOutcome(events)
    study = replace(_study(events), outcome_labeler=labeler)
    dataset = make_dataset(("100", "101", "102"))
    original = OutcomeProvenance.capture_exchange_sessions(labeler)
    assert original.required_market_fields == ("close",)
    assert original.result_schema_version == "1"
    plan = _session_plan(environment=_environment(outcome=original))
    content = serialize_validation_plan(plan)
    first_result = run_prediction_study(dataset, study)
    events.clear()

    setattr(
        labeler,
        attribute,
        ("close", "open") if attribute == "required_market_fields" else "2",
    )
    changed = OutcomeProvenance.capture_exchange_sessions(labeler)
    assert events == []
    changed_plan = _session_plan(environment=_environment(outcome=changed))
    second_result = run_prediction_study(dataset, study)
    assert original.configuration_id == changed.configuration_id
    assert changed.required_market_fields == labeler.required_market_fields
    assert changed.result_schema_version == labeler.result_schema_version
    assert (
        first_result.configuration.to_primitive()
        != second_result.configuration.to_primitive()
    )
    assert original.to_primitive() != changed.to_primitive()
    assert plan.environment.environment_id != changed_plan.environment.environment_id
    assert plan.plan_id != changed_plan.plan_id
    assert serialize_validation_plan(plan) == content
    assert validate_validation_plan_manifest(plan, content) == plan.to_manifest()
    with pytest.raises(ValidationPlanIdentityError):
        validate_validation_plan_manifest(changed_plan, content)
    for field_name in ("required_market_fields", "result_schema_version"):
        corrupted = plan.to_manifest()
        # Canonical bytes cannot make omitted outcome identity safe to reuse.
        outcome_mapping = cast(
            list[PrimitiveMapping],
            cast(PrimitiveMapping, corrupted["environment"])["outcomes"],
        )[0]
        outcome_mapping.pop(field_name)
        with pytest.raises(ValidationPlanIdentityError):
            validate_validation_plan_manifest(plan, canonical_json_bytes(corrupted))


@pytest.mark.parametrize(
    ("attribute", "invalid", "message"),
    [
        ("required_market_fields", (), "required market fields"),
        ("required_market_fields", ["close"], "required market fields"),
        ("required_market_fields", ("open", "close"), "required market fields"),
        ("required_market_fields", ("close", "close"), "required market fields"),
        ("required_market_fields", ("",), "required market fields"),
        ("required_market_fields", (1,), "required market fields"),
        ("required_market_fields", None, "required market fields"),
        ("result_schema_version", "", "result schema version"),
        ("result_schema_version", None, "result schema version"),
    ],
)
def test_session_outcome_capture_rejects_invalid_contract_fields(
    attribute: str,
    invalid: object,
    message: str,
) -> None:
    events: list[str] = []
    labeler = _SeparatelyConfiguredOutcome(events)
    setattr(labeler, attribute, invalid)
    study = replace(_study(events), outcome_labeler=labeler)
    with pytest.raises(InvalidPredictionConfigurationError, match=message):
        run_prediction_study(make_dataset(("100", "101", "102")), study)
    with pytest.raises(ValidationPlanError, match=message):
        OutcomeProvenance.capture_exchange_sessions(labeler)
    assert events == []


@pytest.mark.parametrize(
    "attribute", ["required_market_fields", "result_schema_version"]
)
def test_session_outcome_capture_rejects_missing_contract_fields(
    attribute: str,
) -> None:
    labeler = _SeparatelyConfiguredOutcome([])
    component = SimpleNamespace(
        name=labeler.name,
        implementation_version=labeler.implementation_version,
        configuration_id=labeler.configuration_id,
        configuration=labeler.configuration,
        required_future_sessions=labeler.required_future_sessions,
        required_market_fields=labeler.required_market_fields,
        result_schema_version=labeler.result_schema_version,
    )
    delattr(component, attribute)
    with pytest.raises(ValidationPlanError, match=attribute.replace("_", " ")):
        OutcomeProvenance.capture_exchange_sessions(
            cast(SessionOutcomeComponent, component)
        )


@pytest.mark.parametrize("version", [1, True, " "])
def test_session_outcome_schema_requires_nonblank_string(version: object) -> None:
    labeler = _SeparatelyConfiguredOutcome([])
    labeler.result_schema_version = cast(str, version)
    with pytest.raises(ValidationPlanError, match="result schema version"):
        OutcomeProvenance.capture_exchange_sessions(labeler)


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
    assert result.configuration.required_future_sessions is not None
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
