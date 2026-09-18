"""Elapsed studies retain causal warm-up and complete-source availability checks."""

from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from quantforge.configuration import PrimitiveMappingSnapshot
from quantforge.data import TimeframeBarSeries
from quantforge.prediction import PredictionContextRequirements, run_prediction_study
from quantforge.prediction.errors import InvalidPredictionOutputError
from quantforge.prediction.feature_dataset import (
    _fixed_candidate_population_id,  # pyright: ignore[reportPrivateUsage]
    _SignalFeatureRule,  # pyright: ignore[reportPrivateUsage]
)
from quantforge.validation import PartitionRole
from quantforge.walk_forward.partitions import partition
from quantforge.walk_forward.prediction import (
    _PermittedContextProvider,  # pyright: ignore[reportPrivateUsage]
)
from tests.unit.helpers import SESSIONS, make_dataset
from tests.unit.prediction.test_multi_timeframe_feature_dataset import (
    _FixtureCandidateRule,  # pyright: ignore[reportPrivateUsage]
)
from tests.unit.prediction.test_multi_timeframe_study import FixtureContextProvider
from tests.unit.prediction.test_timestamp_replay import ReplayRule, population
from tests.unit.walk_forward.timestamp_fixtures import instant, timestamp_fixture


@pytest.mark.parametrize(
    ("day", "warm_up", "accepted"), [(0, 20, False), (4, 6, False), (4, 5, True)]
)
def test_direct_elapsed_study_enforces_dataset_warm_up(
    tmp_path: Path, day: int, warm_up: int, accepted: bool
) -> None:
    _, replay, _, study = population(tmp_path)
    dataset = make_dataset(("100",) * 7)
    candidates = (
        replace(
            replay.generate(dataset).signals[0],
            signal_session=SESSIONS[day],
            decision_timestamp=instant(day, "10:00"),
        ),
    )
    source_rule = cast(_FixtureCandidateRule, replay._source)  # pyright: ignore[reportPrivateUsage]
    source_rule.warm_up_observations = warm_up
    direct_rule = ReplayRule(
        cast(_SignalFeatureRule, source_rule),
        PrimitiveMappingSnapshot.capture(source_rule.configuration()),
        candidates,
        _fixed_candidate_population_id(candidates),
        1,
    )
    direct_study = replace(study, strategy=direct_rule)
    if accepted:
        assert len(run_prediction_study(dataset, direct_study).rows) == 1
    else:
        with pytest.raises(InvalidPredictionOutputError, match="warm-up completed"):
            run_prediction_study(dataset, direct_study)


@pytest.mark.parametrize("sufficient_context", [False, True])
def test_elapsed_study_uses_validated_primary_history_for_warm_up(
    tmp_path: Path, sufficient_context: bool
) -> None:
    config, adapter = timestamp_fixture(tmp_path)
    study = adapter.factory.build({"window": 2})
    requirements = cast(
        PredictionContextRequirements, getattr(study.strategy, "context_requirements")
    )
    permitted = partition(
        adapter.dataset,
        config.plan,
        0,
        PartitionRole.DEVELOPMENT,
        minimum_observations=1,
    )
    membership = config.plan.prediction_membership
    assert membership is not None
    provider = _PermittedContextProvider(
        config.plan, permitted, adapter.series, membership.schedule
    )
    context = provider.get_context_at(requirements, as_of=instant(4, "10:00"))
    observed_count = len(context.bars_for(requirements.primary.timeframe))
    assert observed_count > 1
    rule = _FixtureCandidateRule(requirements)
    rule.warm_up_observations = observed_count + (0 if sufficient_context else 1)
    contextual_study = replace(study, strategy=rule)
    # The signal is the first daily row; only validated primary context can supply
    # its declared history. Merely having a context must not waive the requirement.
    dataset = make_dataset(("100",), sessions=(SESSIONS[4],))
    if sufficient_context:
        assert (
            len(
                run_prediction_study(
                    dataset,
                    contextual_study,
                    context_provider=FixtureContextProvider(context),
                ).rows
            )
            == 1
        )
    else:
        with pytest.raises(InvalidPredictionOutputError, match="warm-up completed"):
            run_prediction_study(
                dataset,
                contextual_study,
                context_provider=FixtureContextProvider(context),
            )


@pytest.mark.parametrize(
    ("decision", "expected", "truncate", "status"),
    [
        ("10:01", "10:35", False, "missing_required_observation"),
        ("15:30", "16:00", False, "missing_required_observation"),
        ("10:01", "10:35", True, "dataset_end"),
        ("15:30", "16:00", True, "dataset_end"),
    ],
)
def test_persisted_availability_distinguishes_gaps_from_dataset_end(
    tmp_path: Path, decision: str, expected: str, truncate: bool, status: str
) -> None:
    dataset, replay, _, study = population(tmp_path)
    complete = study.outcome_source
    assert complete is not None
    endpoint = instant(4, expected)
    bars = tuple(
        bar
        for bar in complete.bars
        if (bar.end_timestamp < endpoint if truncate else bar.end_timestamp != endpoint)
    )
    source = TimeframeBarSeries._from_validated_artifact(  # pyright: ignore[reportPrivateUsage]
        complete.dataset_reference,
        complete.timeframe,
        bars,
        dataset_family_manifest_id=complete.dataset_family_manifest_id,
    )
    candidates = (
        replace(
            replay.generate(dataset).signals[0], decision_timestamp=instant(4, decision)
        ),
    )
    rule = ReplayRule(
        replay,
        PrimitiveMappingSnapshot.capture(replay.configuration()),
        candidates,
        _fixed_candidate_population_id(candidates),
        1,
    )
    result = run_prediction_study(
        dataset, replace(study, strategy=rule, outcome_source=source)
    )
    resolution = result.rows[0].outcome.temporal_resolution
    assert resolution is not None
    metadata = resolution.to_primitive()
    assert metadata["status"] == status
    assert metadata["available"] is False
    assert metadata["expected_observation_timestamp"] == endpoint.isoformat()
    assert metadata["resolved_observation_timestamp"] is None
    assert result.rows[0].outcome.values.available is False
