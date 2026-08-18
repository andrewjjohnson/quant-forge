import json
from dataclasses import dataclass, replace
from datetime import date, timedelta
from pathlib import Path
from typing import cast

import pytest

import quantforge.prediction.feature_dataset as feature_dataset_module
from quantforge.configuration import (
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.data import (
    ContextCompletionPolicy,
    FeedScope,
    MarketDataset,
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
    MULTI_TIMEFRAME_FEATURE_DATASET_ENGINE_VERSION,
    ForwardReturnValues,
    MultiTimeframeFeatureRequest,
    PredictionContextFailurePolicy,
    PredictionContextRequirements,
    PredictionDirection,
    PredictionIndicatorRequirement,
    PredictionRuleContext,
    PredictionStudy,
    PredictionTimeframeRequirement,
    SchemaField,
    SchemaFieldCategory,
    SignalDisposition,
    SignalFeatureCandidate,
    SignalFeatureCandidateOutput,
    SignalFeatureDatasetError,
    SignalFeatureValue,
    build_signal_feature_dataset,
    forward_return_outcome,
)
from tests.unit.indicators import test_timeframe_evaluation as timeframe_fixtures
from tests.unit.prediction import test_multi_timeframe_study as study_fixtures

from ..helpers import make_dataset

_SOURCE_RULE_CONFIGURATION_ID = configuration_identity(
    {
        "component_name": "fixture_multi_timeframe_source_rule",
        "implementation_version": "1",
    }
)


@dataclass(frozen=True, slots=True)
class _FixtureParameters:
    mode: str = "multi_timeframe_feature_capture"

    def to_primitive(self) -> PrimitiveMapping:
        return {"mode": self.mode}


class _FixtureCandidateRule:
    name = "fixture_multi_timeframe_candidates"
    implementation_version = "1"
    warm_up_observations = 1

    def __init__(self, requirements: PredictionContextRequirements) -> None:
        self.context_requirements = requirements
        self._parameters = _FixtureParameters()
        self.generate_calls = 0

    @property
    def parameters(self) -> _FixtureParameters:
        return self._parameters

    @property
    def required_indicators(self) -> tuple[Indicator, ...]:
        return tuple(
            item.indicator
            for timeframe in self.context_requirements.all_timeframes
            for item in timeframe.indicators
        )

    @property
    def strategy_feature_definitions(self) -> tuple[SchemaField, ...]:
        return (
            SchemaField(
                "decision_close",
                SchemaFieldCategory.CONTEMPORANEOUS_FEATURE,
                "decimal",
                "price_per_share",
                False,
                "completed primary decision-bar close",
                "available at the primary decision timestamp",
            ),
        )

    def configuration(self) -> PrimitiveMapping:
        return {
            "component_name": self.name,
            "component_type": "prediction_strategy",
            "context_requirements": self.context_requirements.to_primitive(),
            "contract_version": "2",
            "implementation_version": self.implementation_version,
            "parameters": self.parameters.to_primitive(),
            "required_indicators": [
                item.to_primitive()
                for timeframe in self.context_requirements.all_timeframes
                for item in timeframe.indicators
            ],
            "warm_up_observations": self.warm_up_observations,
        }

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    def generate_with_context(
        self, context: PredictionRuleContext
    ) -> SignalFeatureCandidateOutput:
        self.generate_calls += 1
        primary_bar = context.latest_bar_for(
            self.context_requirements.primary.timeframe
        )
        candidate = SignalFeatureCandidate(
            symbol=context.symbol,
            signal_session=context.decision_session,
            strategy_id=self.name,
            strategy_implementation_version=self.implementation_version,
            strategy_configuration_id=self.configuration_id,
            source_rule_id="fixture_multi_timeframe_source_rule",
            source_rule_implementation_version="1",
            source_rule_configuration_id=_SOURCE_RULE_CONFIGURATION_ID,
            strategy_parameters=PrimitiveMappingSnapshot.capture(
                self.parameters.to_primitive()
            ),
            disposition=SignalDisposition.ACCEPTED,
            reason_codes=("fixture_context_available",),
            explanation="fixture multi-timeframe candidate",
            direction=PredictionDirection.UP,
            selected_rule_reason="fixture_up",
            matched_rule_reasons=("fixture_up",),
            strategy_features=(
                SignalFeatureValue("decision_close", primary_bar.close),
            ),
        )
        return SignalFeatureCandidateOutput(
            self.name,
            self.configuration_id,
            context.prediction_dataset_id,
            (candidate,),
        )


def _dataset(*, append_future: bool = False) -> MarketDataset:
    sessions = (
        date(2024, 7, 10),
        date(2024, 7, 11),
        date(2024, 7, 12),
        date(2024, 7, 15),
    )
    closes = ("100", "101", "102", "103")
    count = 4 if append_future else 3
    return make_dataset(closes[:count], sessions=sessions[:count])


def _requests(
    _: PredictionContextRequirements,
) -> tuple[MultiTimeframeFeatureRequest, ...]:
    five_minute, four_hour, daily, weekly = timeframe_fixtures._timeframes()  # pyright: ignore[reportPrivateUsage]
    return (
        MultiTimeframeFeatureRequest(
            "weekly", weekly, "trend", SIMPLE_MOVING_AVERAGE_OUTPUT, "price_per_share"
        ),
        MultiTimeframeFeatureRequest(
            "daily", daily, "trend", SIMPLE_MOVING_AVERAGE_OUTPUT, "price_per_share"
        ),
        MultiTimeframeFeatureRequest(
            "four_hour",
            four_hour,
            "trend",
            SIMPLE_MOVING_AVERAGE_OUTPUT,
            "price_per_share",
        ),
        MultiTimeframeFeatureRequest(
            "five_minute",
            five_minute,
            "trend",
            SIMPLE_MOVING_AVERAGE_OUTPUT,
            "price_per_share",
        ),
    )


def _build(
    output_root: Path,
    *,
    dataset: MarketDataset | None = None,
    context: MultiTimeframeContext | None = None,
    requirements: PredictionContextRequirements | None = None,
    requests: tuple[MultiTimeframeFeatureRequest, ...] | None = None,
):
    selected_requirements = requirements or study_fixtures._requirements()  # pyright: ignore[reportPrivateUsage]
    selected_context = context or study_fixtures._prediction_context()  # pyright: ignore[reportPrivateUsage]
    rule = _FixtureCandidateRule(selected_requirements)
    primary = forward_return_outcome(1)
    study = PredictionStudy[
        SignalFeatureCandidate, ForwardReturnValues, ForwardReturnValues
    ].create(rule, primary.labeler, primary.evaluator)
    provider = study_fixtures.FixtureContextProvider(selected_context)
    result = build_signal_feature_dataset(
        dataset=dataset or _dataset(),
        prediction_study=study,
        contextual_features=(),
        multi_timeframe_features=(
            _requests(selected_requirements) if requests is None else requests
        ),
        context_provider=provider,
        outcomes=(primary,),
        output_root=output_root,
    )
    return result, rule, provider


def _shifted_context(source: MultiTimeframeContext) -> MultiTimeframeContext:
    shift = timedelta(seconds=1)
    aligned = tuple(
        TimeframeContext._from_aligned_series(  # pyright: ignore[reportPrivateUsage]
            requirement=item.requirement,
            dataset_reference=item.dataset_reference,
            availability=item.availability,
            bars=item.bars,
            latest_completed_bar_timestamp=item.latest_completed_bar_timestamp,
            age=None if item.age is None else item.age + shift,
        )
        for item in source.timeframes
    )
    return MultiTimeframeContext._from_aligned_timeframes(  # pyright: ignore[reportPrivateUsage]
        as_of=source.as_of + shift,
        primary_timeframe=source.primary_timeframe,
        required_timeframes=source.required_timeframes,
        completion_policy=source.completion_policy,
        source_consistency=source.source_consistency,
        timeframes=aligned,
    )


def test_one_row_flattens_four_timeframes_with_complete_provenance_and_parquet(
    tmp_path: Path,
) -> None:
    result, rule, provider = _build(tmp_path)
    row = result.rows[0].to_primitive()
    value_names = tuple(
        request.name for request in _requests(rule.context_requirements)
    )

    assert rule.generate_calls == 1
    assert provider.requests == [rule.context_requirements]
    assert all(f"feature_{name}" in row for name in value_names)
    assert all(f"feature_{name}__metadata" in row for name in value_names)
    weekly_metadata = cast(
        PrimitiveMapping,
        row["feature_weekly_trend_simple_moving_average__metadata"],
    )
    assert weekly_metadata["completion_state"] == "completed"
    assert cast(int, weekly_metadata["staleness_microseconds"]) >= 0
    assert weekly_metadata["dataset_family_id"] == timeframe_fixtures.FAMILY_ID
    assert weekly_metadata["normalized_output_name"] == (SIMPLE_MOVING_AVERAGE_OUTPUT)
    backend = cast(PrimitiveMapping, weekly_metadata["indicator_backend"])
    assert backend["backend_id"] == NATIVE_INDICATOR_BACKEND
    assert backend["library_version"]
    assert backend["function_name"]

    schema = json.loads(
        (tmp_path / result.dataset_id / "schema.json").read_text(encoding="utf-8")
    )
    fields = {item["field_name"]: item for item in schema["fields"]}
    value_field = fields["feature_weekly_trend_simple_moving_average"]
    assert value_field["unit"] == "price_per_share"
    assert value_field["temporal_availability"].startswith("available at the signal")
    assert value_field["provenance"] == weekly_metadata
    parquet = (tmp_path / result.dataset_id / "features.parquet").read_bytes()
    assert parquet[:4] == b"PAR1"
    assert parquet[-4:] == b"PAR1"


def test_future_prediction_bars_do_not_change_historical_context_features(
    tmp_path: Path,
) -> None:
    baseline, _, _ = _build(tmp_path / "baseline")
    extended, _, _ = _build(
        tmp_path / "extended",
        dataset=_dataset(append_future=True),
    )
    baseline_row = baseline.rows[0].to_primitive()
    extended_row = extended.rows[0].to_primitive()

    baseline_features = {
        name: value
        for name, value in baseline_row.items()
        if name.startswith("feature_")
    }
    extended_features = {
        name: value
        for name, value in extended_row.items()
        if name.startswith("feature_")
    }
    assert extended_features == baseline_features


def test_backend_and_indicator_configuration_change_dataset_identity(
    tmp_path: Path,
) -> None:
    native, _, _ = _build(tmp_path / "native")
    talib_requirements = study_fixtures._requirements(  # pyright: ignore[reportPrivateUsage]
        backend_id=TALIB_INDICATOR_BACKEND
    )
    talib, _, _ = _build(
        tmp_path / "talib",
        requirements=talib_requirements,
    )
    longer_requirements = study_fixtures._requirements(window=3)  # pyright: ignore[reportPrivateUsage]
    longer, _, _ = _build(
        tmp_path / "longer",
        requirements=longer_requirements,
    )

    assert len({native.dataset_id, talib.dataset_id, longer.dataset_id}) == 3
    talib_metadata = cast(
        PrimitiveMapping,
        talib.rows[0].to_primitive()[
            "feature_weekly_trend_simple_moving_average__metadata"
        ],
    )
    assert (
        cast(PrimitiveMapping, talib_metadata["indicator_backend"])["backend_id"]
        == TALIB_INDICATOR_BACKEND
    )


def test_developing_feature_is_structurally_distinguishable(tmp_path: Path) -> None:
    five_minute, four_hour, _, _ = timeframe_fixtures._timeframes()  # pyright: ignore[reportPrivateUsage]
    indicator = PredictionIndicatorRequirement(
        "trend", SimpleMovingAverage(SimpleMovingAverageParameters(2))
    )
    requirements = PredictionContextRequirements(
        PredictionTimeframeRequirement(
            five_minute,
            FeedScope.consolidated(),
            (indicator,),
        ),
        (
            PredictionTimeframeRequirement(
                four_hour,
                FeedScope.consolidated(),
                (indicator,),
                completion_policy=ContextCompletionPolicy.DEVELOPING_BAR_AS_OF,
            ),
        ),
    )
    context = timeframe_fixtures._developing_context()  # pyright: ignore[reportPrivateUsage]
    dataset = make_dataset(
        ("99", "100", "101"),
        sessions=(date(2024, 7, 8), date(2024, 7, 9), date(2024, 7, 10)),
    )
    request = MultiTimeframeFeatureRequest(
        "four_hour",
        four_hour,
        "trend",
        SIMPLE_MOVING_AVERAGE_OUTPUT,
        "price_per_share",
    )

    result, _, _ = _build(
        tmp_path,
        dataset=dataset,
        context=context,
        requirements=requirements,
        requests=(request,),
    )
    metadata = cast(
        PrimitiveMapping,
        result.rows[0].to_primitive()[
            "feature_four_hour_trend_simple_moving_average__metadata"
        ],
    )
    assert metadata["completion_policy"] == "developing_bar_as_of"
    assert metadata["completion_state"] == "developing"
    assert (
        metadata["source_bar_completion_timestamp"]
        != (metadata["source_bar_observed_through_timestamp"])
    )


def test_mixed_feed_is_rejected_before_candidate_generation(tmp_path: Path) -> None:
    requirements = study_fixtures._requirements(  # pyright: ignore[reportPrivateUsage]
        feed_scope=FeedScope.iex_only()
    )
    rule = _FixtureCandidateRule(requirements)
    primary = forward_return_outcome(1)
    study = PredictionStudy[
        SignalFeatureCandidate, ForwardReturnValues, ForwardReturnValues
    ].create(rule, primary.labeler, primary.evaluator)

    with pytest.raises(SignalFeatureDatasetError, match="feed scope"):
        build_signal_feature_dataset(
            dataset=_dataset(),
            prediction_study=study,
            contextual_features=(),
            multi_timeframe_features=_requests(requirements),
            context_provider=study_fixtures.FixtureContextProvider(  # pyright: ignore[reportPrivateUsage]
                study_fixtures._prediction_context()  # pyright: ignore[reportPrivateUsage]
            ),
            outcomes=(primary,),
            output_root=tmp_path,
        )

    assert rule.generate_calls == 0


def test_complete_resume_keeps_one_aligned_row_and_exact_parquet(
    tmp_path: Path,
) -> None:
    first, _, _ = _build(tmp_path)
    parquet_path = tmp_path / first.dataset_id / "features.parquet"
    parquet_before = parquet_path.read_bytes()

    resumed, resumed_rule, _ = _build(tmp_path)

    assert resumed_rule.generate_calls == 0
    assert resumed.dataset_id == first.dataset_id
    assert len(resumed.rows) == 1
    assert resumed.rows[0].row_id == first.rows[0].row_id
    assert parquet_path.read_bytes() == parquet_before


def test_context_identity_is_bound_without_requested_feature_columns(
    tmp_path: Path,
) -> None:
    source_context = study_fixtures._prediction_context()  # pyright: ignore[reportPrivateUsage]
    first, first_rule, first_provider = _build(
        tmp_path,
        context=source_context,
        requests=(),
    )
    shifted_context = _shifted_context(source_context)
    second, second_rule, second_provider = _build(
        tmp_path,
        context=shifted_context,
        requests=(),
    )

    assert first.dataset_id != second.dataset_id
    first_prediction_context = cast(
        PrimitiveMapping, first.configuration["prediction_context"]
    )
    second_prediction_context = cast(
        PrimitiveMapping, second.configuration["prediction_context"]
    )
    assert first_prediction_context["status"] == "available"
    assert second_prediction_context["status"] == "available"
    assert (
        cast(PrimitiveMapping, first_prediction_context["source_context"])["context_id"]
        == source_context.context_id
    )
    assert (
        cast(PrimitiveMapping, second_prediction_context["source_context"])[
            "context_id"
        ]
        == shifted_context.context_id
    )
    assert first.engine_version == MULTI_TIMEFRAME_FEATURE_DATASET_ENGINE_VERSION
    assert second.engine_version == MULTI_TIMEFRAME_FEATURE_DATASET_ENGINE_VERSION
    assert first_rule.generate_calls == second_rule.generate_calls == 1
    assert first_provider.requests == [first_rule.context_requirements]
    assert second_provider.requests == [second_rule.context_requirements]


def test_skip_policy_produces_context_bound_empty_dataset_without_feature_requests(
    tmp_path: Path,
) -> None:
    requirements = study_fixtures._requirements(  # pyright: ignore[reportPrivateUsage]
        feed_scope=FeedScope.iex_only(),
        failure_policy=PredictionContextFailurePolicy.SKIP,
    )
    source_context = study_fixtures._prediction_context()  # pyright: ignore[reportPrivateUsage]
    first, first_rule, first_provider = _build(
        tmp_path,
        context=source_context,
        requirements=requirements,
        requests=(),
    )
    shifted_context = _shifted_context(source_context)
    second, second_rule, second_provider = _build(
        tmp_path,
        context=shifted_context,
        requirements=requirements,
        requests=(),
    )

    assert first.rows == second.rows == ()
    assert first.dataset_id != second.dataset_id
    assert first_rule.generate_calls == second_rule.generate_calls == 0
    assert first_provider.requests == [requirements]
    assert second_provider.requests == [requirements]
    first_prediction_context = cast(
        PrimitiveMapping, first.configuration["prediction_context"]
    )
    second_prediction_context = cast(
        PrimitiveMapping, second.configuration["prediction_context"]
    )
    assert first_prediction_context["status"] == "skipped"
    assert second_prediction_context["status"] == "skipped"
    assert first_prediction_context["source_context"] == source_context.to_primitive()
    assert second_prediction_context["source_context"] == (
        shifted_context.to_primitive()
    )


def test_progress_manifest_uses_selected_multi_timeframe_engine_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def interrupt_candidate_generation(
        *_args: object, **_kwargs: object
    ) -> tuple[SignalFeatureCandidate, ...]:
        raise RuntimeError("fixture interruption")

    monkeypatch.setattr(
        feature_dataset_module,
        "_generate_candidate_population",
        interrupt_candidate_generation,
    )

    with pytest.raises(RuntimeError, match="fixture interruption"):
        _build(tmp_path)

    destination = next(tmp_path.iterdir())
    manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["engine_version"] == (
        MULTI_TIMEFRAME_FEATURE_DATASET_ENGINE_VERSION
    )
    assert manifest["engine_version"] == manifest["configuration"]["engine_version"]


def test_timeframe_and_completion_policy_change_dataset_identity(
    tmp_path: Path,
) -> None:
    baseline_requirements = study_fixtures._requirements()  # pyright: ignore[reportPrivateUsage]
    _, four_hour, daily, _ = timeframe_fixtures._timeframes()  # pyright: ignore[reportPrivateUsage]
    baseline_request = MultiTimeframeFeatureRequest(
        "context",
        four_hour,
        "trend",
        SIMPLE_MOVING_AVERAGE_OUTPUT,
        "price_per_share",
    )
    baseline, _, _ = _build(
        tmp_path / "baseline",
        requirements=baseline_requirements,
        requests=(baseline_request,),
    )
    daily_request = replace(baseline_request, timeframe=daily)
    daily_result, _, _ = _build(
        tmp_path / "daily",
        requirements=baseline_requirements,
        requests=(daily_request,),
    )

    developing_daily = replace(
        next(
            item for item in baseline_requirements.contextual if item.timeframe == daily
        ),
        completion_policy=ContextCompletionPolicy.DEVELOPING_BAR_AS_OF,
    )
    developing_requirements = PredictionContextRequirements(
        baseline_requirements.primary,
        tuple(
            developing_daily if item.timeframe == daily else item
            for item in baseline_requirements.contextual
        ),
    )
    source_context = study_fixtures._prediction_context()  # pyright: ignore[reportPrivateUsage]
    developing_context = MultiTimeframeContext._from_aligned_timeframes(  # pyright: ignore[reportPrivateUsage]
        as_of=source_context.as_of,
        primary_timeframe=source_context.primary_timeframe,
        required_timeframes=source_context.required_timeframes,
        completion_policy=ContextCompletionPolicy.DEVELOPING_BAR_AS_OF,
        source_consistency=source_context.source_consistency,
        timeframes=source_context.timeframes,
    )
    developing_result, _, _ = _build(
        tmp_path / "developing",
        context=developing_context,
        requirements=developing_requirements,
        requests=(daily_request,),
    )

    assert (
        len(
            {baseline.dataset_id, daily_result.dataset_id, developing_result.dataset_id}
        )
        == 3
    )
