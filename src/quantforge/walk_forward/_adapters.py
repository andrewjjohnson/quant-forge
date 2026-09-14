"""Small shared selection checks; no metric or optimizer implementation."""

from dataclasses import replace
from typing import cast

from quantforge.configuration import PrimitiveMapping, PrimitiveMappingSnapshot
from quantforge.data import MarketDataset
from quantforge.optimization.models import StabilityClassification, StudyResult
from quantforge.prediction.grid import PredictionGridResult
from quantforge.validation import PartitionRole, ResearchRuleProvenance, ValidationPlan
from quantforge.walk_forward.models import (
    CandidateConfiguration,
    CandidateUniverse,
    FrozenSelection,
    SelectionEvidence,
    SelectionPolicy,
    WalkForwardConfig,
    WalkForwardError,
)
from quantforge.walk_forward.partitions import partition


def validate_rule(plan: ValidationPlan, rule: ResearchRuleProvenance) -> None:
    expected = plan.environment.research_rule
    if (
        rule.configuration.name != expected.configuration.name
        or rule.implementation_version != expected.implementation_version
    ):
        raise WalkForwardError("candidate changed the declared rule family/version")
    # QF-8 validates exact indicator bindings and warm-up requirements for EVERY
    # candidate against the permitted indicator union. The original plan stays fixed.
    replace(plan, environment=replace(plan.environment, research_rule=rule))


def validate_fixed_backends(plan: ValidationPlan) -> None:
    identities = set[str]()
    for indicator in plan.environment.indicators:
        backend = indicator.backend_identity
        if backend is None:
            identities.add("legacy_native")
        else:
            identity = backend.to_primitive()
            identity.pop("function_name", None)
            identities.add(PrimitiveMappingSnapshot.capture(identity).canonical_json)
    if len(identities) > 1:
        raise WalkForwardError(
            "indicator backend choice must be fixed across the universe"
        )


def membership(
    dataset: MarketDataset, config: WalkForwardConfig, fold_index: int
) -> PrimitiveMapping:
    fold = config.plan.folds[fold_index]
    development = partition(
        dataset,
        config.plan,
        fold_index,
        PartitionRole.DEVELOPMENT,
        minimum_observations=config.minimum_training_observations,
    )
    selection = (
        None
        if fold.selection is None
        else partition(
            dataset,
            config.plan,
            fold_index,
            PartitionRole.SELECTION,
            minimum_observations=config.minimum_training_observations,
        )
    )
    test = partition(
        dataset,
        config.plan,
        fold_index,
        PartitionRole.WALK_FORWARD_TEST,
        minimum_observations=config.minimum_test_observations,
    )
    return {
        "development": development.to_primitive(),
        "selection": None if selection is None else selection.to_primitive(),
        "test": test.to_primitive(),
    }


def choose(
    universe: CandidateUniverse,
    result: StudyResult | PredictionGridResult,
    policy: SelectionPolicy,
) -> SelectionEvidence:
    stable = {
        item.trial_id
        for item in result.stability
        if item.classification is StabilityClassification.STABLE
        and not item.is_isolated_peak
    }
    ranked = [
        item
        for item in result.rankings
        if policy is SelectionPolicy.BEST_ELIGIBLE or item.trial_id in stable
    ]
    if not ranked:
        raise WalkForwardError(
            "no eligible candidate under the declared selection policy"
        )
    selected = ranked[0]
    candidate = universe.candidate(selected.combination_id)
    trial = next(item for item in result.trials if item.trial_id == selected.trial_id)
    if trial.parameters != candidate.parameters.to_primitive():
        raise WalkForwardError("ranked trial parameters escaped the candidate universe")
    evidence: PrimitiveMapping = {
        "rankings": [item.to_primitive() for item in result.rankings],
        "stability": [item.to_primitive() for item in result.stability],
        "trial_statuses": [
            {
                "trial_id": item.trial_id,
                "combination_id": item.combination_id,
                "status": item.status.value,
            }
            for item in result.trials
        ],
    }
    return SelectionEvidence(
        candidate,
        selected.trial_id,
        result.study_id,
        PrimitiveMappingSnapshot.capture(evidence),
    )


def frozen_candidate(
    universe: CandidateUniverse, selection: FrozenSelection
) -> CandidateConfiguration:
    record = selection.snapshot.to_primitive()["candidate"]
    if not isinstance(record, dict) or not isinstance(
        record.get("combination_id"), str
    ):
        raise WalkForwardError("invalid frozen candidate")
    candidate = universe.candidate(cast(str, record["combination_id"]))
    if record != candidate.to_primitive():
        raise WalkForwardError(
            "frozen configuration differs from the declared universe"
        )
    return candidate


def validate_search_parameters(names: frozenset[str]) -> None:
    if any("backend" in name.lower() for name in names):
        raise WalkForwardError("indicator backend cannot be a search parameter")
