"""QF-69 study lifecycle: frozen specification, stage order, floors and gate.

Synthetic QF-67 datasets on the QF-8 timestamp plan fixture (contract
fixtures, never research evidence). Only the QF-67 build from a QF-39 study is
replaced (``ConditionalStudy._build``); every QF-68 call, artifact and reader is
the real one. The QF-39 path is exercised in
``tests/integration/test_conditional_ml_study.py``.
"""

import json
from collections.abc import Sequence
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest

from quantforge.configuration import (
    Primitive,
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
)
from quantforge.ml import (
    DispositionPolicy,
    EventDataset,
    ForwardReturnBinaryTarget,
    read_event_dataset,
)
from quantforge.ml.modeling import MODEL_FILE, ModelConfiguration, ScoreStatus
from quantforge.ml.study import (
    INSUFFICIENT_CLASS_OBSERVATIONS,
    ConditionalStudy,
    ConditionalStudyError,
    ConditionalStudySpecification,
    schema_leakage_audit,
)
from quantforge.validation import PartitionRole, ValidationPlan
from quantforge.walk_forward.models import CandidateConfiguration
from tests.unit.ml.model_fixtures import (
    SCHEMA,
    Event,
    build_dataset,
    change,
    configuration,
    plan_fixture,
    standard_events,
)

STUDY_ID = "a" * 64
PASSED_AUDIT: PrimitiveMapping = {"status": "passed", "detail": "fixture"}
VALID_ARTIFACTS: PrimitiveMapping = {"status": "passed", "detail": "fixture"}


def part(value: Primitive) -> PrimitiveMapping:
    assert isinstance(value, dict)
    return value


def entries(value: Primitive) -> list[PrimitiveMapping]:
    assert isinstance(value, list)
    return [part(item) for item in value]


def specification(
    plan: ValidationPlan,
    *,
    model: ModelConfiguration | None = None,
    floor: int = 2,
    buckets: int = 5,
) -> ConditionalStudySpecification:
    return ConditionalStudySpecification(
        name="qf69_fixture",
        version="1",
        plan=plan,
        fold_id=plan.folds[0].fold_id,
        qf39_study_id=STUDY_ID,
        candidate=CandidateConfiguration(
            "qf68-population-a",
            PrimitiveMappingSnapshot.capture({"fixture": True}),
            PrimitiveMappingSnapshot.capture({"fixture": "definition"}),
        ),
        feature_schema=SCHEMA,
        target=ForwardReturnBinaryTarget(),
        disposition_policy=DispositionPolicy.ACCEPTED_ONLY,
        model=model or configuration(calibration_bins=buckets),
        minimum_training_class_observations=floor,
        score_buckets=buckets,
        design=PrimitiveMappingSnapshot.capture({"fixture": "qf69 unit"}),
    )


@pytest.fixture
def plan(tmp_path: Path) -> ValidationPlan:
    return plan_fixture(tmp_path / "plan")


def study_for(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    plan: ValidationPlan,
    events: Sequence[Event] | None = None,
    **options: object,
) -> ConditionalStudy:
    dataset = build_dataset(plan, standard_events() if events is None else events)

    def build(self: ConditionalStudy) -> EventDataset:
        del self
        return dataset

    monkeypatch.setattr(ConditionalStudy, "_build", build)
    return ConditionalStudy(
        specification(plan, **options),  # pyright: ignore[reportArgumentType]
        root=tmp_path / "study",
        study_path=tmp_path / STUDY_ID,
        workspace=tmp_path,
    )


def run_lifecycle(study: ConditionalStudy) -> PrimitiveMapping:
    study.freeze_specification()
    study.build_dataset()
    assert study.train_and_freeze().frozen
    study.evaluate_out_of_sample()
    study.verify_reproduction()
    return study.pre_holdout_gate(
        leakage_audit=PASSED_AUDIT,
        artifact_validation=VALID_ARTIFACTS,
        holdout_state="reserved_unconsumed",
    )


def stage_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.name: path.read_bytes() for path in sorted((root / "stages").iterdir())
    }


def test_specification_refuses_filters_unknown_folds_and_bad_floors(
    plan: ValidationPlan,
) -> None:
    with pytest.raises(ConditionalStudyError, match="probability-only"):
        specification(plan, model=configuration(class_threshold=Decimal("0.5")))
    with pytest.raises(ConditionalStudyError, match="not in the validation plan"):
        replace(specification(plan), fold_id="missing")
    for floor in (0, -1):
        with pytest.raises(ConditionalStudyError, match="positive integer"):
            specification(plan, floor=floor)
    first, second = specification(plan), specification(plan)
    assert first.specification_id == second.specification_id
    assert specification(plan, floor=3).specification_id != first.specification_id


def test_research_stages_require_the_frozen_specification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, plan: ValidationPlan
) -> None:
    study = study_for(tmp_path, monkeypatch, plan)
    with pytest.raises(ConditionalStudyError, match="must be frozen before"):
        study.build_dataset()
    study.freeze_specification()
    study.build_dataset()
    # A changed choice (here the class floor) cannot reuse the frozen study.
    changed = ConditionalStudy(
        specification(plan, floor=3),
        root=study.root,
        study_path=study.study_path,
        workspace=tmp_path,
    )
    with pytest.raises(ConditionalStudyError, match="cannot change"):
        changed.freeze_specification()
    with pytest.raises(ConditionalStudyError, match="differs from this design"):
        changed.build_dataset()
    with pytest.raises(ConditionalStudyError, match="differs from this design"):
        changed.train_and_freeze()


def test_lifecycle_trains_on_development_only_and_freezes_before_oos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, plan: ValidationPlan
) -> None:
    study = study_for(tmp_path, monkeypatch, plan)
    gate = run_lifecycle(study)
    assert gate["outcome"] == "PRE_HOLDOUT_COMPLETE"
    dataset = read_event_dataset(study.dataset_path())
    fold = plan.folds[0].fold_id
    training = part(study.stage("training"))
    freeze = part(study.stage("freeze"))
    oos = part(study.stage("out_of_sample"))
    membership = part(training["membership"])
    assert membership["fitting_observations"] == len(
        dataset.rows_for(PartitionRole.DEVELOPMENT, fold_id=fold)
    )
    # Stages bind each other: OOS uses the frozen model of this training.
    assert training["model_id"] == freeze["model_id"] == oos["model_id"]
    model = study.frozen_model()
    assert {identity for identity, _ in model.state.membership.fitting} == {
        dataset.rows[index].source_observation_id
        for index in dataset.rows_for(PartitionRole.DEVELOPMENT, fold_id=fold)
    }
    checks = part(part(study.stage("reproduction"))["checks"])
    assert all(value is True for value in checks.values())
    out_of_sample = part(study.report()["out_of_sample"])
    buckets = entries(out_of_sample["score_buckets"])
    assert len(buckets) == 5
    assert sum(int(str(item["scored"])) for item in buckets) == len(
        dataset.rows_for(PartitionRole.WALK_FORWARD_TEST, fold_id=fold)
    )
    coverage = part(out_of_sample["coverage"])
    assert coverage["metric_rows"] == coverage["partition_rows"]


def test_out_of_sample_and_reproduction_require_the_persisted_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, plan: ValidationPlan
) -> None:
    study = study_for(tmp_path, monkeypatch, plan)
    study.freeze_specification()
    study.build_dataset()
    for stage in (study.evaluate_out_of_sample, study.verify_reproduction):
        with pytest.raises(ConditionalStudyError, match="freeze stage has not"):
            stage()
    assert not (study.root / "models").exists()


def test_class_floor_ends_training_without_model_or_oos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, plan: ValidationPlan
) -> None:
    study = study_for(tmp_path, monkeypatch, plan, floor=1000)
    study.freeze_specification()
    study.build_dataset()
    outcome = study.train_and_freeze()
    assert (outcome.status, outcome.model_path) == (
        INSUFFICIENT_CLASS_OBSERVATIONS,
        None,
    )
    assert not (study.root / "models").exists()
    assert not (study.root / "predictions").exists()
    report = study.report()
    assert report["out_of_sample"] == {
        "status": "not_evaluated",
        "reason": f"training ended with {INSUFFICIENT_CLASS_OBSERVATIONS}",
    }
    assert "walk_forward_test" not in part(report["events"])
    gate = study.pre_holdout_gate(
        leakage_audit=PASSED_AUDIT,
        artifact_validation=VALID_ARTIFACTS,
        holdout_state="reserved_unconsumed",
    )
    assert gate["outcome"] == "PRE_HOLDOUT_COMPLETE_INSUFFICIENT_TRAINING"
    items = part(gate["items"])
    assert part(items["persisted_frozen_model"])["status"] == "not_applicable"


def test_insufficient_training_rows_are_recorded_not_relaxed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, plan: ValidationPlan
) -> None:
    model = configuration(minimum_training_observations=1000, calibration_bins=5)
    study = study_for(tmp_path, monkeypatch, plan, model=model)
    study.freeze_specification()
    study.build_dataset()
    outcome = study.train_and_freeze()
    assert outcome.status == "insufficient_training_observations"
    assert part(study.stage("training"))["model_id"] is None


def test_missing_labels_stay_unavailable_in_training_and_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, plan: ValidationPlan
) -> None:
    items = standard_events()
    # Unavailable labels: the first two development and test events.
    marked = {
        PartitionRole.DEVELOPMENT: 0,
        PartitionRole.WALK_FORWARD_TEST: 0,
    }
    events: list[Event] = []
    for item in items:
        count = marked.get(item.role)
        if count is not None and count < 2:
            marked[item.role] = count + 1
            events.append(replace(item, label=None, status="session_overflow"))
        else:
            events.append(item)
    study = study_for(tmp_path, monkeypatch, plan, events)
    run_lifecycle(study)
    membership = part(part(study.stage("training"))["membership"])
    assert membership["excluded_by_reason"] == {"target_unavailable": 2}
    out_of_sample = part(study.report()["out_of_sample"])
    coverage = part(out_of_sample["coverage"])
    assert coverage["excluded_unlabeled"] == 2
    assert coverage["metric_rows"] == int(str(coverage["partition_rows"])) - 2
    summary = part(out_of_sample["summary"])
    assert summary["unlabeled_scored_by_status"] == {"session_overflow": 2}


def test_null_features_are_reported_unscored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, plan: ValidationPlan
) -> None:
    events = change(
        standard_events(),
        PartitionRole.WALK_FORWARD_TEST,
        features=(None, "0.25", 1, False),
    )
    study = study_for(tmp_path, monkeypatch, plan, events)
    run_lifecycle(study)
    out_of_sample = part(study.report()["out_of_sample"])
    coverage = part(out_of_sample["coverage"])
    assert coverage["metric_rows"] == 0
    assert coverage["excluded_unscored"] == coverage["partition_rows"]
    metrics = part(out_of_sample["model_metrics"])
    assert metrics["log_loss"] == {
        "status": "undefined",
        "reason": "no_labeled_observations",
    }
    assert all(item["scored"] == 0 for item in entries(out_of_sample["score_buckets"]))
    distribution = part(out_of_sample["score_distribution"])
    assert distribution["count"] == 0
    assert distribution["median"] == {"status": "undefined", "reason": "no_scored_rows"}
    oos = part(study.stage("out_of_sample"))
    records = (study.root / str(oos["path"]) / "predictions.jsonl").read_text()
    assert {json.loads(line)["score_status"] for line in records.splitlines()} == {
        ScoreStatus.UNSCORED_NULL_FEATURE.value
    }


def test_tampered_artifacts_and_foreign_stages_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, plan: ValidationPlan
) -> None:
    study = study_for(tmp_path, monkeypatch, plan)
    run_lifecycle(study)
    freeze = part(study.stage("freeze"))
    envelope = study.root / str(freeze["model_path"]) / MODEL_FILE
    original = envelope.read_bytes()
    envelope.write_bytes(original.replace(b'"iterations":', b'"iterations": '))
    with pytest.raises(ConditionalStudyError, match="changed after freezing"):
        study.evaluate_out_of_sample()
    envelope.write_bytes(original)
    stage = study.root / "stages" / "02-training.json"
    stage.write_text(stage.read_text().replace('"fitted"', '"forged"'))
    with pytest.raises(ConditionalStudyError, match="corrupt"):
        study.stage("training")
    # Stage records of another specification are refused.
    other = tmp_path / "other"
    (other / "stages").mkdir(parents=True)
    (other / "stages" / "01-dataset.json").write_bytes(
        (study.root / "stages" / "01-dataset.json").read_bytes()
    )
    foreign = ConditionalStudy(
        specification(plan, floor=3),
        root=other,
        study_path=study.study_path,
        workspace=tmp_path,
    )
    with pytest.raises(ConditionalStudyError, match="another study"):
        foreign.stage("dataset")


def test_repeated_lifecycle_reproduces_every_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, plan: ValidationPlan
) -> None:
    study = study_for(tmp_path, monkeypatch, plan)
    first = run_lifecycle(study)
    before = stage_bytes(study.root)
    second = run_lifecycle(study)
    assert first == second
    assert stage_bytes(study.root) == before


def test_gate_blocks_on_failed_audit_or_holdout_not_reserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, plan: ValidationPlan
) -> None:
    study = study_for(tmp_path, monkeypatch, plan)
    run_lifecycle(study)
    failed: PrimitiveMapping = {"status": "failed"}
    for audit, state in (
        (failed, "reserved_unconsumed"),
        (PASSED_AUDIT, "consumed"),
    ):
        gate = study.pre_holdout_gate(
            leakage_audit=audit,
            artifact_validation=VALID_ARTIFACTS,
            holdout_state=state,
        )
        assert gate["outcome"] == "PRE_HOLDOUT_BLOCKED"
        assert part(gate["holdout"])["consumed_by_this_study"] is False


def test_schema_leakage_audit_requires_declared_causal_sources(
    tmp_path: Path, plan: ValidationPlan
) -> None:
    dataset = build_dataset(plan, standard_events())
    declared = frozenset(item.source_field for item in SCHEMA.features)

    def passed(item: EventDataset) -> PrimitiveMapping:
        del item
        return {"status": "passed"}

    def failed(item: EventDataset) -> PrimitiveMapping:
        del item
        return {"status": "failed"}

    assert (
        schema_leakage_audit(
            dataset, declared_contemporaneous=declared, row_audit=passed
        )["status"]
        == "passed"
    )
    assert (
        schema_leakage_audit(
            dataset,
            declared_contemporaneous=declared - {"momentum"},
            row_audit=passed,
        )["status"]
        == "failed"
    )
    assert (
        schema_leakage_audit(
            dataset, declared_contemporaneous=declared, row_audit=failed
        )["status"]
        == "failed"
    )
    del tmp_path
