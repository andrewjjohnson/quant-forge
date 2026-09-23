"""Offline ancestry must authenticate the permitted session range independently."""

from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from shutil import copytree
from typing import Any, cast

import pytest

from quantforge.configuration import Primitive, PrimitiveMapping, configuration_identity
from quantforge.data import (
    DatasetFamily,
    FeedScope,
    MarketDataset,
    validate_market_dataset,
)
from quantforge.data.models import IntradayPredictionProvenance
from quantforge.data.prediction_views import validate_bounded_prediction_ancestry
from quantforge.experiments import ManifestError
from quantforge.experiments._holdout_integrity import validate_holdout_artifact
from quantforge.experiments._window_integrity import validate_window_snapshot
from quantforge.oos import (
    HoldoutEvaluation,
    HoldoutLedger,
    OOSIntegrityError,
    load_oos_source,
)
from quantforge.oos._records import mapping, records, text
from quantforge.oos.models import OOSSource
from quantforge.prediction import (
    PredictionContextRequirements,
    PredictionStudy,
    PredictionTimeframeRequirement,
)
from quantforge.prediction.models import PredictionMarketData
from quantforge.validation import (
    ConfigurationReference,
    DatasetProvenance,
    ExchangeSessionBoundary,
    ResearchRuleProvenance,
    TimeframeWarmUpRequirement,
    ValidationWindow,
)
from quantforge.walk_forward import FoldStatus, PredictionEvaluator, WalkForwardStudy
from quantforge.walk_forward.partitions import project_dataset
from quantforge.walk_forward.persistence import read_record, write_record
from tests.integration.test_intraday_prediction_provenance import DAILY, cached_fixture
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_window_session_integrity import refresh_window_manifest
from tests.unit.helpers import SESSIONS
from tests.unit.prediction.test_prediction_window import (
    _rewrite_decision_identities,  # pyright: ignore[reportPrivateUsage]
)
from tests.unit.walk_forward.fixtures import (
    PRIMARY,
    StudyFactory,
    StudyRule,
    prediction_fixture,
)


class SessionFactory(StudyFactory):
    def __init__(self, feed_scope: FeedScope) -> None:
        self.feed_scope = feed_scope

    def build(self, parameters: PrimitiveMapping) -> PredictionStudy[Any, Any, Any]:
        original = super().build(parameters)
        requirements = PredictionContextRequirements(
            PredictionTimeframeRequirement(PRIMARY, self.feed_scope),
            (PredictionTimeframeRequirement(DAILY, self.feed_scope),),
        )
        return PredictionStudy[Any, Any, Any].create(
            StudyRule(requirements), original.outcome_labeler, original.evaluator
        )


@dataclass(frozen=True)
class CapturedSessionStudy:
    dataset: MarketDataset
    source: OOSSource
    study_path: Path
    consumption: PrimitiveMapping
    holdout: PrimitiveMapping


@pytest.fixture(scope="module")
def captured(tmp_path_factory: pytest.TempPathFactory) -> CapturedSessionStudy:
    root = tmp_path_factory.mktemp("bounded-session-integrity")
    fixture = cached_fixture(
        root / "cache", session_dates=SESSIONS[:10], primary_timeframe=PRIMARY
    )
    base, legacy = prediction_fixture(root)
    factory = SessionFactory(fixture.primary.dataset_reference.feed_scope)
    adapter = PredictionEvaluator(
        dataset=fixture.dataset,
        series=(fixture.primary, fixture.daily),
        primary_timeframe=PRIMARY,
        study_factory=factory,
        analyzer=legacy.analyzer,
        indicator_backend=legacy.backend,
        grid_config=legacy.grid_config,
    )
    original = fixture.dataset.metadata.intraday_provenance
    assert isinstance(original, IntradayPredictionProvenance)
    family = DatasetFamily.from_manifest(original.family_manifest.to_primitive())
    environment = replace(
        base.plan.environment,
        dataset=DatasetProvenance.from_dataset_family(
            family,
            (
                fixture.primary.dataset_reference.dataset_id,
                fixture.daily.dataset_reference.dataset_id,
            ),
        ),
        prediction_dataset=DatasetProvenance.from_market_dataset(fixture.dataset),
        timeframes=(PRIMARY, DAILY),
        research_rule=ResearchRuleProvenance.capture_prediction(
            factory.build({"window": 2}).strategy
        ),
        indicators=(),
        aggregation_policies=(
            ConfigurationReference.capture_aggregation_policy(
                family.aggregation_policy
            ),
        ),
    )

    def window(original: ValidationWindow, first: int, last: int) -> ValidationWindow:
        return replace(
            original,
            interval=replace(
                original.interval,
                start=ExchangeSessionBoundary(SESSIONS[first]),
                end=ExchangeSessionBoundary(SESSIONS[last]),
            ),
            warm_up_by_timeframe=(
                TimeframeWarmUpRequirement(PRIMARY, 2),
                TimeframeWarmUpRequirement(DAILY, 2),
            ),
        )

    fold = base.plan.folds[0]
    plan = replace(
        base.plan,
        environment=environment,
        folds=(
            replace(
                fold,
                development=window(fold.development, 2, 3),
                test=window(fold.test, 5, 5),
            ),
        ),
        final_holdout=replace(
            base.plan.final_holdout,
            window=window(base.plan.final_holdout.window, 7, 8),
        ),
    )
    study = WalkForwardStudy(replace(base, plan=plan), adapter, root / "study")
    result = study.run()
    if result.folds[0].status is not FoldStatus.COMPLETED:
        pytest.fail(
            str([failure.to_primitive() for failure in result.folds[0].failures])
        )
    source = load_oos_source(plan, study.study_path)
    ledger = HoldoutLedger.create(root / "ledger")
    ledger.reserve(source)
    evaluation = HoldoutEvaluation.prepare(
        source, adapter, selection_fold_id=source.folds[0].fold_id
    )
    consumed = ledger.consume(evaluation, run_id="bounded-session-regression")
    assert consumed.consumption is not None
    return CapturedSessionStudy(
        fixture.dataset,
        source,
        study.study_path,
        consumed.consumption.to_primitive(),
        ledger.result(evaluation).to_primitive(),
    )


def test_session_producers_have_independently_valid_ancestry(
    captured: CapturedSessionStudy,
) -> None:
    assert validate_market_dataset(captured.dataset) == ()
    assert load_oos_source(captured.source.plan, captured.study_path) is not None
    validate_holdout_artifact(captured.source, captured.consumption, captured.holdout)


def substitute_view(
    payload: PrimitiveMapping, canonical: MarketDataset, change: str
) -> tuple[PrimitiveMapping, PrimitiveMapping]:
    """Rehash every affected record around a valid but unauthorized canonical subset."""
    manifest = mapping(payload["manifest"])
    original = mapping(manifest["market_data"])
    sessions = tuple(bar.session_date for bar in canonical.bars)
    first = sessions.index(date.fromisoformat(text(original["actual_first_session"])))
    last = sessions.index(date.fromisoformat(text(original["actual_last_session"])))
    if change == "future_end":
        last += 1
    else:
        first += -1 if change == "earlier_start" else 1
    view = project_dataset(canonical, sessions[first], sessions[last])
    validate_bounded_prediction_ancestry(view, canonical)
    replacement = PredictionMarketData.from_qf3(view.metadata).to_primitive()

    def rewrite(value: Primitive) -> Primitive:
        if isinstance(value, dict):
            return {
                key: deepcopy(replacement) if key == "market_data" else rewrite(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if value == original["dataset_id"]:
            return replacement["dataset_id"]
        if value == original["bars_fingerprint"]:
            return replacement["bars_fingerprint"]
        return value

    changed = mapping(rewrite(payload))
    for decision in records(changed["decisions"]):
        for row in records(mapping(decision["prediction_study"])["rows"]):
            outcome = mapping(row["outcome"])
            outcome["outcome_id"] = configuration_identity(
                {
                    **{
                        key: item
                        for key, item in outcome.items()
                        if key != "outcome_id"
                    },
                    "record_type": "prediction_outcome",
                }
            )
            evaluation = mapping(row["evaluation"])
            evaluation["outcome_id"] = outcome["outcome_id"]
            evaluation["evaluation_id"] = configuration_identity(
                {
                    **{
                        key: item
                        for key, item in evaluation.items()
                        if key != "evaluation_id"
                    },
                    "record_type": "prediction_evaluation",
                    "prediction": mapping(row["prediction"])["values"],
                }
            )
        _rewrite_decision_identities(cast(dict[str, Any], decision))
    refresh_window_manifest(changed)
    changed_manifest = mapping(changed["manifest"])
    changed_manifest["window_result_id"] = configuration_identity(
        {
            "window_id": changed_manifest["window_id"],
            "decisions": changed["decisions"],
        }
    )
    validate_window_snapshot(changed)
    part = mapping(
        mapping(mapping(changed_manifest["context_environment"])["configuration"])[
            "partition"
        ]
    )
    return changed, part


@pytest.mark.parametrize("change", ["future_end", "earlier_start", "later_start"])
def test_oos_rejects_rehashed_session_bounds(
    captured: CapturedSessionStudy,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    path = tmp_path / captured.study_path.name
    copytree(captured.study_path, path)
    fold = path / "folds" / captured.source.folds[0].fold_id
    artifact = read_record(fold / "oos.json")
    payload, part = substitute_view(
        mapping(artifact["result"]), captured.dataset, change
    )
    selection = read_record(fold / "selection.json")
    mapping(selection["membership"])["test"] = part
    selection["selection_id"] = configuration_identity(
        {key: item for key, item in selection.items() if key != "selection_id"}
    )
    artifact.update(
        result=payload,
        selection_id=selection["selection_id"],
        result_id=mapping(payload["manifest"])["window_result_id"],
    )
    state = read_record(fold / "state.json")
    state.update(
        selection_id=selection["selection_id"],
        artifact_id=configuration_identity(artifact),
    )
    for name, record in (("selection", selection), ("oos", artifact), ("state", state)):
        write_record(fold / f"{name}.json", record)
    block_research(monkeypatch)
    with pytest.raises(OOSIntegrityError, match="canonical plan ancestry or cutoff"):
        load_oos_source(captured.source.plan, path)


@pytest.mark.parametrize("change", ["future_end", "earlier_start", "later_start"])
def test_holdout_rejects_rehashed_session_bounds(
    captured: CapturedSessionStudy, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    consumption, result = deepcopy(captured.consumption), deepcopy(captured.holdout)
    artifact = mapping(result["artifact"])
    payload, part = substitute_view(
        mapping(artifact["result"]), captured.dataset, change
    )
    request = mapping(consumption["request"])
    request["evaluation_membership"] = part
    consumption["request_id"] = configuration_identity(request)
    artifact.update(
        result=payload, result_id=mapping(payload["manifest"])["window_result_id"]
    )
    mapping(artifact["holdout_summary"])["window_result_id"] = artifact["result_id"]
    result["artifact_sha256"] = configuration_identity(artifact)
    block_research(monkeypatch)
    with pytest.raises(ManifestError, match="canonical plan ancestry or cutoff"):
        validate_holdout_artifact(captured.source, consumption, result)
