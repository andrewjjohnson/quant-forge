"""Original QF-42 anchors survive QF-7 replay and QF-29 export without price math."""

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import cast

import pytest

from quantforge.configuration import (
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.data import MarketDataset
from quantforge.prediction import (
    PredictionContextRequirements,
    PredictionDecisionSchedule,
    PredictionStudy,
    PredictionStudyOutcome,
    SchemaField,
    SchemaFieldCategory,
    SignalFeatureCandidate,
    build_signal_feature_dataset,
    run_prediction_study,
)
from quantforge.prediction.contracts import PredictionOutcome
from quantforge.prediction.feature_dataset import (
    _fixed_candidate_population_id,  # pyright: ignore[reportPrivateUsage]
    _FixedCandidateRule,  # pyright: ignore[reportPrivateUsage]
    _SignalFeatureRule,  # pyright: ignore[reportPrivateUsage]
)
from quantforge.validation import PartitionRole
from quantforge.walk_forward.partitions import partition
from quantforge.walk_forward.prediction import (
    _PermittedContextProvider,  # pyright: ignore[reportPrivateUsage]
)
from tests.unit.prediction.test_multi_timeframe_feature_dataset import (
    _FixtureCandidateRule,  # pyright: ignore[reportPrivateUsage]
)
from tests.unit.prediction.test_multi_timeframe_study import FixtureContextProvider
from tests.unit.walk_forward.timestamp_fixtures import (
    PRIMARY,
    AvailabilityValues,
    FixtureEvaluation,
    MetadataEvaluator,
    MetadataLabeler,
    instant,
    timestamp_fixture,
)


@dataclass(frozen=True)
class ReplayValues:
    decision_timestamp: str

    def to_primitive(self) -> PrimitiveMapping:
        return {"decision_timestamp": self.decision_timestamp}


class ReplayEvaluator:
    name = "qf48_replay_fixture"
    implementation_version = "1"
    result_schema_version = "1"

    def configuration(self) -> PrimitiveMapping:
        return {"component_name": self.name, "implementation_version": "1"}

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    def evaluate(
        self,
        signal: SignalFeatureCandidate,
        outcome: PredictionOutcome[AvailabilityValues],
    ) -> ReplayValues:
        assert signal.decision_timestamp is not None
        assert outcome.temporal_resolution is not None
        resolution = outcome.temporal_resolution.to_primitive()
        request = cast(PrimitiveMapping, resolution["request"])
        assert (
            cast(PrimitiveMapping, request["anchor"])["decision_timestamp"]
            == signal.decision_timestamp.isoformat()
        )
        return ReplayValues(signal.decision_timestamp.isoformat())


class ReplayRule(_FixedCandidateRule):
    @property
    def strategy_feature_definitions(self) -> tuple[SchemaField, ...]:
        return cast(_FixtureCandidateRule, self._source).strategy_feature_definitions  # pyright: ignore[reportPrivateUsage]


def population(
    tmp_path: Path,
) -> tuple[
    MarketDataset,
    ReplayRule,
    MetadataLabeler,
    PredictionStudy[SignalFeatureCandidate, AvailabilityValues, ReplayValues],
]:
    config, adapter = timestamp_fixture(tmp_path)
    template = adapter.factory.build({"window": 2})
    requirements = cast(
        PredictionContextRequirements,
        getattr(template.strategy, "context_requirements"),
    )
    rule = _FixtureCandidateRule(requirements)
    source = template.outcome_source
    assert source is not None
    labeler = cast(MetadataLabeler, template.outcome_labeler)
    study = PredictionStudy[
        SignalFeatureCandidate, AvailabilityValues, FixtureEvaluation
    ].create(rule, labeler, MetadataEvaluator(), outcome_source=source)
    permitted = partition(
        adapter.dataset,
        config.plan,
        0,
        PartitionRole.DEVELOPMENT,
        minimum_observations=1,
    )
    schedule = PredictionDecisionSchedule(
        PRIMARY, instant(4, "10:00"), instant(4, "10:05")
    )
    provider = _PermittedContextProvider(
        config.plan, permitted, adapter.series, schedule
    )
    signals: list[SignalFeatureCandidate] = []
    for timestamp in schedule.decision_timestamps:
        context = provider.get_context_at(requirements, as_of=timestamp)
        result = run_prediction_study(
            permitted.dataset, study, context_provider=FixtureContextProvider(context)
        )
        assert result.signals[0].decision_timestamp == timestamp
        signals.extend(result.signals)
    fixed = tuple(signals)
    replay = ReplayRule(
        cast(_SignalFeatureRule, rule),
        PrimitiveMappingSnapshot.capture(rule.configuration()),
        fixed,
        _fixed_candidate_population_id(fixed),
        len(fixed),
    )
    replay_study = PredictionStudy[
        SignalFeatureCandidate, AvailabilityValues, ReplayValues
    ].create(replay, labeler, ReplayEvaluator(), outcome_source=source)
    # Direct replay uses daily membership as its warm-up evidence. Contextual
    # export separately verifies replay over a metadata-only projection.
    return adapter.dataset, replay, labeler, replay_study


def test_two_same_session_candidates_keep_original_anchors_through_replay(
    tmp_path: Path,
) -> None:
    dataset, _, _, study = population(tmp_path)
    result = run_prediction_study(dataset, study)
    expected = (instant(4, "10:00"), instant(4, "10:05"))
    assert tuple(row.signal.decision_timestamp for row in result.rows) == expected
    assert len({row.row_id for row in result.rows}) == 2
    assert len({row.outcome.outcome_id for row in result.rows}) == 2
    assert (
        tuple(
            datetime.fromisoformat(row.evaluation.values.decision_timestamp)
            for row in result.rows
        )
        == expected
    )
    assert len({s.signal_session for s in result.signals}) == 1


def test_contextual_elapsed_export_reuses_validated_primary_history(
    tmp_path: Path,
) -> None:
    from tests.unit.experiments.test_elapsed_prediction_integrity import (
        contextual_study,
    )

    dataset, template, provider = contextual_study(tmp_path)
    requirements = cast(
        PredictionContextRequirements,
        getattr(template.strategy, "context_requirements"),
    )
    rule = _FixtureCandidateRule(requirements)
    rule.warm_up_observations = 4
    study = PredictionStudy[
        SignalFeatureCandidate, AvailabilityValues, ReplayValues
    ].create(
        rule,
        template.outcome_labeler,
        ReplayEvaluator(),
        outcome_source=template.outcome_source,
    )
    outcome = PredictionStudyOutcome[AvailabilityValues, ReplayValues].create(
        "anchor",
        template.outcome_labeler,
        ReplayEvaluator(),
        (
            SchemaField(
                "decision_timestamp",
                SchemaFieldCategory.FUTURE_OUTCOME,
                "string",
                "UTC_timestamp",
                True,
                "QF-46 anchor",
                "future metadata",
            ),
        ),
        unavailable_values={"decision_timestamp": None},
        outcome_source=template.outcome_source,
    )
    result = build_signal_feature_dataset(
        dataset=dataset,
        prediction_study=study,
        contextual_features=(),
        outcomes=(outcome,),
        output_root=tmp_path / "contextual-export",
        context_provider=provider,
    )
    assert len(result.rows) == 1
    assert dataset.bars[-1].session_date < result.rows[0].signal_session
    assert result == build_signal_feature_dataset(
        dataset=dataset,
        prediction_study=study,
        contextual_features=(),
        outcomes=(outcome,),
        output_root=tmp_path / "contextual-export",
        context_provider=provider,
    )


def test_export_and_chunked_resume_do_not_collapse_same_session_candidates(
    tmp_path: Path,
) -> None:
    dataset, _, labeler, study = population(tmp_path)
    outcome = PredictionStudyOutcome[AvailabilityValues, ReplayValues].create(
        "anchor",
        labeler,
        ReplayEvaluator(),
        (
            SchemaField(
                "decision_timestamp",
                SchemaFieldCategory.FUTURE_OUTCOME,
                "string",
                "UTC_timestamp",
                True,
                "QF-46 anchor",
                "future metadata",
            ),
        ),
        unavailable_values={"decision_timestamp": None},
        outcome_source=study.outcome_source,
    )
    first = build_signal_feature_dataset(
        dataset=dataset,
        prediction_study=study,
        contextual_features=(),
        outcomes=(outcome,),
        output_root=tmp_path / "exports",
        chunk_size=1,
    )
    resumed = build_signal_feature_dataset(
        dataset=dataset,
        prediction_study=study,
        contextual_features=(),
        outcomes=(outcome,),
        output_root=tmp_path / "exports",
        chunk_size=10,
    )
    assert resumed == first
    assert len(first.rows) == 2
    for row, expected in zip(
        first.rows, (instant(4, "10:00"), instant(4, "10:05")), strict=True
    ):
        values = row.to_primitive()
        assert values["decision_timestamp"] == expected.isoformat()
        assert values["outcome_anchor_decision_timestamp"] == expected.isoformat()
    assert len({r.candidate_id for r in first.rows}) == 2
    from quantforge.experiments._feature_row_integrity import (
        validate_feature_row_provenance,
    )

    rows = [row.to_primitive() for row in first.rows]
    validate_feature_row_provenance(
        first.manifest_primitive(), {"schema": first.schema.to_primitive()}, rows
    )
    rows[0]["decision_timestamp"] = None
    with pytest.raises(ValueError, match="original exact anchor"):
        validate_feature_row_provenance(
            first.manifest_primitive(), {"schema": first.schema.to_primitive()}, rows
        )


def test_replay_cannot_reconstruct_missing_anchor_from_session(tmp_path: Path) -> None:
    dataset, replay, labeler, study = population(tmp_path)
    candidates = tuple(
        replace(c, decision_timestamp=None)
        for c in replay.generate(dataset).signals[:1]
    )
    missing = ReplayRule(
        replay,
        PrimitiveMappingSnapshot.capture(replay.configuration()),
        candidates,
        _fixed_candidate_population_id(candidates),
        1,
    )
    with pytest.raises(ValueError, match="timestamp"):
        run_prediction_study(
            dataset, replace(study, strategy=missing, outcome_labeler=labeler)
        )


def test_shifted_anchor_changes_candidate_and_outcome_identity(tmp_path: Path) -> None:
    dataset, replay, _, study = population(tmp_path)
    original = run_prediction_study(dataset, study)
    fixed = tuple(
        replace(c, decision_timestamp=c.decision_timestamp + timedelta(minutes=5))
        for c in replay.generate(dataset).signals
        if c.decision_timestamp is not None
    )
    shifted = ReplayRule(
        replay,
        PrimitiveMappingSnapshot.capture(replay.configuration()),
        fixed,
        _fixed_candidate_population_id(fixed),
        len(fixed),
    )
    result = run_prediction_study(dataset, replace(study, strategy=shifted))
    assert tuple(r.outcome.outcome_id for r in result.rows) != tuple(
        r.outcome.outcome_id for r in original.rows
    )
    assert configuration_identity(result.to_primitive()) != configuration_identity(
        original.to_primitive()
    )


def test_ordinary_session_candidate_study_keeps_legacy_output() -> None:
    from quantforge.prediction import ForwardReturnValues, forward_return_outcome
    from tests.unit.prediction import test_multi_timeframe_study as fixtures
    from tests.unit.prediction.test_multi_timeframe_feature_dataset import (
        _dataset,  # pyright: ignore[reportPrivateUsage]
    )

    rule = _FixtureCandidateRule(fixtures._requirements())  # pyright: ignore[reportPrivateUsage]
    outcome = forward_return_outcome(1)
    study = PredictionStudy[
        SignalFeatureCandidate, ForwardReturnValues, ForwardReturnValues
    ].create(rule, outcome.labeler, outcome.evaluator)
    result = run_prediction_study(
        _dataset(),
        study,
        context_provider=FixtureContextProvider(
            fixtures._prediction_context()  # pyright: ignore[reportPrivateUsage]
        ),
    )
    assert result.signals[0].decision_timestamp is None
    assert "decision_timestamp" not in result.signals[0].prediction_primitive()
    assert result.configuration.required_future_sessions == 1


def test_elapsed_unavailable_outcome_keeps_explicit_resolution(tmp_path: Path) -> None:
    dataset, replay, _, study = population(tmp_path)
    fixed = tuple(
        replace(c, decision_timestamp=instant(4, "15:55"))
        for c in replay.generate(dataset).signals[:1]
    )
    late = ReplayRule(
        replay,
        PrimitiveMappingSnapshot.capture(replay.configuration()),
        fixed,
        _fixed_candidate_population_id(fixed),
        1,
    )
    result = run_prediction_study(dataset, replace(study, strategy=late))
    resolution = result.rows[0].outcome.temporal_resolution
    assert resolution is not None
    assert resolution.to_primitive()["available"] is False
    assert resolution.to_primitive()["status"] == "session_overflow"
    assert resolution.to_primitive()["resolved_observation_timestamp"] is None


@pytest.mark.parametrize(
    "change", ["anchor", "reach", "source", "target", "availability"]
)
def test_resolution_reader_rejects_contradictory_elapsed_metadata(
    tmp_path: Path, change: str
) -> None:
    from quantforge.oos._records import mapping
    from quantforge.prediction.window_validation import (
        _validate_timestamp_resolution,  # pyright: ignore[reportPrivateUsage]
    )

    dataset, _, labeler, study = population(tmp_path)
    result = run_prediction_study(dataset, study)
    row = result.rows[0].to_primitive()
    outcome = mapping(row["outcome"])
    resolution = mapping(outcome["temporal_resolution"])
    request = mapping(resolution["request"])
    if change == "anchor":
        mapping(request["anchor"])["decision_timestamp"] = instant(
            4, "10:05"
        ).isoformat()
    elif change == "reach":
        mapping(mapping(request["temporal_configuration"])["horizon"])[
            "duration_microseconds"
        ] = 1
    elif change == "source":
        mapping(request["source_reference"])["dataset_id"] = "foreign"
    elif change == "target":
        resolution["requested_target_timestamp"] = instant(4, "11:00").isoformat()
    else:
        resolution["available"] = False
    with pytest.raises(ValueError, match="elapsed outcome"):
        _validate_timestamp_resolution(
            outcome,
            mapping(row["prediction"]),
            mapping(result.configuration.to_primitive()["outcome_labeler"]),
            labeler.temporal,
            instant(4, "10:00").isoformat(),
        )
