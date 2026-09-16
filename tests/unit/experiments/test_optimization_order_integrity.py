from dataclasses import replace
from decimal import Decimal, localcontext
from pathlib import Path
from typing import cast

import pytest

from quantforge.configuration import PrimitiveMapping, PrimitiveMappingSnapshot
from quantforge.experiments import ManifestError, StudyType, inspect_study
from quantforge.experiments._ranking_integrity import validate_optimization_summaries
from quantforge.optimization import (
    GridSearchStudy,
    IntegerValues,
    MetricName,
    MetricTieBreaker,
    MovingAverageCrossoverFactory,
    ParameterSearchSpace,
    RankingConfig,
    RankingDirection,
    StabilityConfig,
    TrialRecord,
    analyze_stability,
    rank_trials,
)
from quantforge.optimization.models import StudyResult
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_contracts import write_json
from tests.unit.experiments.test_grid_integrity import read_record
from tests.unit.helpers import make_dataset
from tests.unit.optimization.test_ranking_and_stability import (
    _surface_trials,  # pyright: ignore[reportPrivateUsage]
)
from tests.unit.optimization.test_study import (
    _dataset,  # pyright: ignore[reportPrivateUsage]
    _study_config,  # pyright: ignore[reportPrivateUsage]
)

type Documents = tuple[
    PrimitiveMapping,
    list[PrimitiveMapping],
    PrimitiveMapping,
    dict[str, PrimitiveMapping],
]


def producer_documents(
    scenario: str = "objective",
    direction: RankingDirection = RankingDirection.MAXIMIZE,
    fraction: str = "0",
) -> Documents:
    """Capture the real QF-6 ranking/stability producers over fixed metrics."""
    trials, candidates, space = _surface_trials(spike=False)
    captured: list[TrialRecord] = []
    for index, trial in enumerate(trials):
        metrics: PrimitiveMapping = {
            "total_return": str(index + 1) if scenario == "objective" else "1",
            "annualized_volatility": str(index + 1)
            if scenario == "first_tie"
            else "0.1",
            "trade_count": index + 1 if scenario == "second_tie" else 1,
            "maximum_drawdown": "-0.1",
        }
        if scenario == "undefined_tie" and index % 2:
            metrics["annualized_volatility"] = None
        if scenario == "missing_tie" and index % 2:
            del metrics["annualized_volatility"]
        captured.append(
            replace(trial, metrics_snapshot=PrimitiveMappingSnapshot.capture(metrics))
        )
    trials = tuple(captured)
    opposite_direction = (
        RankingDirection.MINIMIZE
        if direction is RankingDirection.MAXIMIZE
        else RankingDirection.MAXIMIZE
    )
    ranking_config = RankingConfig(
        MetricName.TOTAL_RETURN,
        direction=direction,
        tie_breakers=(
            MetricTieBreaker(MetricName.ANNUALIZED_VOLATILITY, opposite_direction),
            MetricTieBreaker(MetricName.TRADE_COUNT, direction),
        ),
    )
    stability_config = StabilityConfig(
        minimum_eligible_neighbors=1,
        stable_maximum_relative_dispersion=Decimal(100),
        isolated_peak_top_fraction=Decimal(0),
        robust_recommendation_top_fraction=Decimal(fraction),
    )
    ranking = rank_trials(trials, ranking_config)
    stability = analyze_stability(
        trials,
        candidates,
        ranking,
        space,
        MovingAverageCrossoverFactory(),
        ranking_config,
        stability_config,
    )
    configuration: PrimitiveMapping = {
        "study_schema_version": "1",
        "ranking_configuration": ranking_config.to_primitive(),
        "stability_configuration": stability_config.to_primitive(),
    }
    derived: dict[str, PrimitiveMapping] = {
        "ranking.json": {
            "configuration": ranking_config.to_primitive(),
            "eligible_rankings": [item.to_primitive() for item in ranking.rankings],
            "ineligible_trials": [
                item.to_primitive() for item in ranking.ineligible_trials
            ],
        },
        "stability.json": {
            "configuration": stability_config.to_primitive(),
            "summaries": [item.to_primitive() for item in stability.summaries],
        },
    }
    summary: PrimitiveMapping = {
        **StudyResult(
            study_id=trials[0].study_id,
            schema_version="1",
            total_combinations=len(candidates),
            trials=trials,
            rankings=ranking.rankings,
            ineligible_trials=ranking.ineligible_trials,
            stability=stability.summaries,
            parameter_summaries=(),
            best_objective_trial_id=None,
            best_stability_trial_id=None,
            recommended_robust_trial_id=stability.recommended_robust_trial_id,
            warnings=(),
            limitations=(),
        ).summary_primitive(),
        **configuration,
        "parameter_summaries": [],
    }
    documents = (
        configuration,
        [item.to_primitive() for item in trials],
        summary,
        derived,
    )
    refresh_projections(documents, sort_stability=False)
    return documents


def refresh_projections(documents: Documents, *, sort_stability: bool) -> None:
    """Keep dependent references consistent while deliberately changing order."""
    _, _, summary, derived = documents
    rankings = cast(
        list[PrimitiveMapping], derived["ranking.json"]["eligible_rankings"]
    )
    stable = cast(list[PrimitiveMapping], derived["stability.json"]["summaries"])
    ranks = {}
    for rank, record in enumerate(rankings, 1):
        record["rank"] = rank
        ranks[cast(str, record["trial_id"])] = rank
    for record in stable:
        record["objective_rank"] = ranks[cast(str, record["trial_id"])]
    if sort_stability:
        stable.sort(
            key=lambda record: (
                -Decimal(cast(str, record["stability_score"])),
                cast(int, record["objective_rank"]),
                cast(str, record["combination_id"]),
            )
        )
    for rank, record in enumerate(stable, 1):
        record["stability_rank"] = rank
    summary["top_objective_trials"] = [row for row in rankings[:10]]
    summary["top_stability_trials"] = [row for row in stable[:10]]
    summary["best_objective_trial_id"] = rankings[0]["trial_id"] if rankings else None
    summary["best_stability_trial_id"] = stable[0]["trial_id"] if stable else None


@pytest.mark.parametrize("direction", list(RankingDirection))
@pytest.mark.parametrize(
    "scenario",
    [
        "objective",
        "first_tie",
        "second_tie",
        "combination_id",
        "undefined_tie",
        "missing_tie",
    ],
)
def test_optimization_order_matches_producer_and_rejects_consistent_reordering(
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
    direction: RankingDirection,
) -> None:
    documents = producer_documents(scenario, direction)
    block_research(monkeypatch)
    validate_optimization_summaries(*documents)
    rankings = cast(
        list[PrimitiveMapping], documents[3]["ranking.json"]["eligible_rankings"]
    )
    rankings[0], rankings[-1] = rankings[-1], rankings[0]
    refresh_projections(documents, sort_stability=True)
    with pytest.raises(ManifestError, match="optimization ranking order"):
        validate_optimization_summaries(*documents)


@pytest.mark.parametrize("scenario", ["objective", "combination_id"])
def test_stability_order_uses_saved_score_then_objective_rank(
    monkeypatch: pytest.MonkeyPatch,
    scenario: str,
) -> None:
    documents = producer_documents(scenario, fraction="1")
    block_research(monkeypatch)
    validate_optimization_summaries(*documents)
    stable = cast(list[PrimitiveMapping], documents[3]["stability.json"]["summaries"])
    stable[0], stable[-1] = stable[-1], stable[0]
    refresh_projections(documents, sort_stability=False)
    with pytest.raises(ManifestError, match="optimization stability order"):
        validate_optimization_summaries(*documents)


@pytest.mark.parametrize(
    "fraction", ["0", "0.0001", "0.1111111111111111111111111111111111", "0.5", "1"]
)
@pytest.mark.parametrize("scenario", ["objective", "combination_id"])
@pytest.mark.parametrize("direction", list(RankingDirection))
def test_recommendation_matches_producer_cutoff_and_preserves_no_choice(
    monkeypatch: pytest.MonkeyPatch,
    fraction: str,
    scenario: str,
    direction: RankingDirection,
) -> None:
    documents = producer_documents(scenario, direction, fraction=fraction)
    block_research(monkeypatch)
    with localcontext() as context:
        context.prec = 4
        validate_optimization_summaries(*documents)
    if fraction == "0":
        assert documents[2]["recommended_robust_trial_id"] is None
        rankings = cast(
            list[PrimitiveMapping], documents[3]["ranking.json"]["eligible_rankings"]
        )
        documents[2]["recommended_robust_trial_id"] = rankings[0]["trial_id"]
    else:
        assert documents[2]["recommended_robust_trial_id"] is not None
        if (
            direction is RankingDirection.MINIMIZE
            and scenario == "objective"
            and fraction == "0.0001"
        ):
            assert (
                documents[2]["recommended_robust_trial_id"]
                != documents[2]["best_stability_trial_id"]
            )
        documents[2]["recommended_robust_trial_id"] = None
    with pytest.raises(ManifestError, match="optimization recommendation"):
        validate_optimization_summaries(*documents)


@pytest.mark.parametrize("direction", list(RankingDirection))
def test_export_rejects_reordered_objectives_with_updated_summary_and_stability(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    direction: RankingDirection,
) -> None:
    study = GridSearchStudy(
        _dataset(),
        MovingAverageCrossoverFactory(),
        _study_config(
            tmp_path / "optimization",
            ranking=RankingConfig(MetricName.TOTAL_RETURN, direction=direction),
            stability=StabilityConfig(robust_recommendation_top_fraction=Decimal(0)),
        ),
    )
    study.export(study.run())
    block_research(monkeypatch)
    inspect_study(StudyType.OPTIMIZATION, study.study_path, artifact_root=tmp_path)
    derived = {
        name: read_record(study.study_path / name)
        for name in ("ranking.json", "stability.json")
    }
    summary = read_record(study.study_path / "summary.json")
    rankings = cast(
        list[PrimitiveMapping], derived["ranking.json"]["eligible_rankings"]
    )
    assert rankings[0]["objective_value"] != rankings[-1]["objective_value"]
    rankings[0], rankings[-1] = rankings[-1], rankings[0]
    refresh_projections(({}, [], summary, derived), sort_stability=True)
    for name, document in {**derived, "summary.json": summary}.items():
        write_json(study.study_path / name, document)
    with pytest.raises(ManifestError, match="optimization ranking order"):
        inspect_study(StudyType.OPTIMIZATION, study.study_path, artifact_root=tmp_path)


@pytest.mark.parametrize("replacement", ["later_inside", "outside_cutoff", "missing"])
def test_export_requires_first_qualifying_robust_recommendation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement: str,
) -> None:
    config = replace(
        _study_config(tmp_path / "optimization"),
        search_space=ParameterSearchSpace(
            {
                "fast_window": IntegerValues([1, 2, 3, 4]),
                "slow_window": IntegerValues([5]),
            }
        ),
        stability=StabilityConfig(
            minimum_eligible_neighbors=1,
            isolated_peak_top_fraction=Decimal(0),
            robust_recommendation_top_fraction=Decimal("0.5"),
        ),
    )
    study = GridSearchStudy(
        make_dataset(("100",) * 9), MovingAverageCrossoverFactory(), config
    )
    study.export(study.run())
    block_research(monkeypatch)
    inspect_study(StudyType.OPTIMIZATION, study.study_path, artifact_root=tmp_path)
    stable = cast(
        list[PrimitiveMapping],
        read_record(study.study_path / "stability.json")["summaries"],
    )
    assert len(stable) == 4
    assert all(
        row["classification"] == "stable" and row["is_isolated_peak"] is False
        for row in stable
    )
    summary = read_record(study.study_path / "summary.json")
    assert summary["recommended_robust_trial_id"] == stable[0]["trial_id"]
    summary["recommended_robust_trial_id"] = (
        None
        if replacement == "missing"
        else stable[1 if replacement == "later_inside" else -1]["trial_id"]
    )
    write_json(study.study_path / "summary.json", summary)
    with pytest.raises(ManifestError, match="optimization recommendation"):
        inspect_study(StudyType.OPTIMIZATION, study.study_path, artifact_root=tmp_path)
