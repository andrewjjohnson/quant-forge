"""QF-69 on real QF-39/QF-40/QF-67/QF-68 contracts with synthetic prices, offline.

``conditional_study_fixtures`` runs the actual QF-69 runner: a three-role fold,
so QF-39 runs and freezes its trial on selection and the frozen candidate's
development evidence is executed separately, bound and verified. Nothing here
is research evidence and no real ledger or cache is touched.
"""

import math
import shutil
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from quantforge.configuration import PrimitiveMapping
from quantforge.data.prepared_canonical import canonical_preparation
from quantforge.examples.spy_ema_ml_study import prepare_ml_walk_forward
from quantforge.ml import (
    EventDataset,
    EventDatasetIntegrityError,
    build_event_dataset,
    read_event_dataset,
)
from quantforge.ml.modeling import read_event_model
from quantforge.ml.study import STAGES
from quantforge.oos import (
    OOSIntegrityError,
    load_oos_source,
    load_prediction_development_window,
    load_prediction_trial_window,
)
from quantforge.validation import PartitionRole
from quantforge.walk_forward import (
    DevelopmentEvidence,
    WalkForwardError,
    WalkForwardStudy,
)
from quantforge.walk_forward.models import (
    DEVELOPMENT_EVIDENCE_DIRECTORY,
    DEVELOPMENT_EVIDENCE_FILE,
)
from quantforge.walk_forward.persistence import read_record, write_record
from tests.integration.conditional_study_fixtures import (
    WINDOWS,
    StudyRun,
    fixture_design,
    run_fixture_study,
)
from tests.integration.event_dataset_fixtures import EventStudy, run_event_study
from tests.integration.event_model_fixtures import development_inputs


@pytest.fixture(scope="module", autouse=True)
def prepared() -> Iterator[None]:
    """One QF-65 load session for direct source loads (operational only)."""
    with canonical_preparation():
        yield


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory, prepared: None) -> StudyRun:
    del prepared
    root = tmp_path_factory.mktemp("qf69-conditional-study")
    return run_fixture_study(root, development_inputs(root))


def study_of(run: StudyRun) -> WalkForwardStudy:
    with fixture_design():
        config, adapter = prepare_ml_walk_forward(
            run.inputs, run.output / "walk-forward"
        )
    return WalkForwardStudy(config, adapter, run.output / "walk-forward")


def part(value: object) -> PrimitiveMapping:
    assert isinstance(value, dict)
    return cast(PrimitiveMapping, value)


def dataset_of(run: StudyRun) -> EventDataset:
    stage = part(read_record(run.output / "stages" / "01-dataset.json"))
    return read_event_dataset(run.output / str(stage["path"]))


def observations(dataset: EventDataset, role: PartitionRole) -> list[str]:
    return [dataset.rows[i].source_observation_id for i in dataset.rows_for(role)]


def test_runner_stops_at_the_gate_with_the_holdout_reserved(run: StudyRun) -> None:
    assert run.gate["outcome"] == "PRE_HOLDOUT_COMPLETE"
    assert {str(part(item)["status"]) for item in part(run.gate["items"]).values()} == {
        "passed"
    }
    holdout = part(run.gate["holdout"])
    assert holdout["state"] == "reserved_unconsumed"
    assert holdout["consumed_by_this_study"] is False
    ledger = run.ledger.root
    assert not any((ledger / "exposures").iterdir())
    (lineage,) = (ledger / "lineages").iterdir()
    assert [path.name for path in lineage.iterdir()] == ["reservation.json"]
    # The specification precedes the reservation, which precedes research; OOS
    # inference and OOS publication follow the persisted freeze.
    stages = [line.split(" ", 2)[1] for line in run.log if line.startswith("STAGE")]
    assert stages[:3] == ["specification", "reserved", "development"]
    order = [line for line in run.log if line.startswith("STAGE")]
    freeze = order.index("STAGE training, selection predictions and freeze")
    assert order.index("STAGE out-of-sample predictions from the frozen model") > (
        freeze
    )
    assert run.log[-1].startswith("PRE_HOLDOUT_COMPLETE: holdout RESERVED_UNCONSUMED")
    for name in ("specification.json", "report.json", "report.md"):
        assert (run.output / name).is_file()
    assert [path.name[3:-5] for path in sorted((run.output / "stages").iterdir())] == (
        list(STAGES)
    )
    assert any((run.output / "html").glob("*.html"))


def test_development_evidence_binds_the_frozen_fold_membership(run: StudyRun) -> None:
    study = study_of(run)
    plan = study.config.plan
    fold = plan.folds[0]
    source = load_oos_source(plan, study.study_path)
    frozen = source.folds[0].selection
    assert frozen is not None
    development = load_prediction_development_window(
        source, study.study_path, fold_id=fold.fold_id
    )
    trial = load_prediction_trial_window(
        source,
        study.study_path,
        fold_id=fold.fold_id,
        combination_id=development.candidate.combination_id,
    )
    assert development.role is PartitionRole.DEVELOPMENT
    assert trial.role is PartitionRole.SELECTION
    assert (
        development.candidate.to_primitive()
        == frozen.snapshot.to_primitive()["candidate"]
    )
    membership = part(part(frozen.snapshot.to_primitive()["membership"])["development"])
    assert [
        stamp.isoformat()
        for stamp in development.reader.evidence.schedule.decision_timestamps
    ] == membership["evaluation_timestamps"]
    dataset = dataset_of(run)
    excluded = part(dataset.scientific.to_primitive()["partition_plan"])
    assert excluded["excluded_sources"] == []
    counts = {
        role: [dataset.rows[index] for index in dataset.rows_for(role)]
        for role in (
            PartitionRole.DEVELOPMENT,
            PartitionRole.SELECTION,
            PartitionRole.WALK_FORWARD_TEST,
        )
    }
    assert {role: len(rows) for role, rows in counts.items()} == {
        PartitionRole.DEVELOPMENT: 6,
        PartitionRole.SELECTION: 2,
        PartitionRole.WALK_FORWARD_TEST: 3,
    }
    for role, rows in counts.items():
        start, end = WINDOWS[role]
        assert all(start <= row.decision_timestamp <= end for row in rows)
    labels = [row.label.value for row in counts[PartitionRole.DEVELOPMENT]]
    assert (labels.count(True), labels.count(False)) == (3, 3)
    assert [row.label.status for row in counts[PartitionRole.WALK_FORWARD_TEST]] == [
        "session_overflow",
        "available",
        "available",
    ]


def test_preprocessing_and_fit_use_development_rows_only(run: StudyRun) -> None:
    stage = part(read_record(run.output / "stages" / "03-freeze.json"))
    model = read_event_model(run.output / str(stage["model_path"]))
    dataset = dataset_of(run)
    development = dataset.rows_for(PartitionRole.DEVELOPMENT)
    assert [identity for identity, _ in model.state.membership.fitting] == [
        dataset.rows[index].source_observation_id for index in development
    ]
    numeric = dataset.numeric_feature_rows()
    for column, transform in enumerate(model.state.preprocessing.features):
        values = [cast(float, numeric[index][column]) for index in development]
        assert transform.center == math.fsum(values) / len(values)


def test_rerun_verifies_and_reproduces_every_record(run: StudyRun) -> None:
    def snapshot() -> dict[str, bytes]:
        return {
            str(path.relative_to(run.workspace)): path.read_bytes()
            for path in sorted(run.workspace.rglob("*"))
            if path.is_file() and path.name != ".lock"
        }

    before = snapshot()
    again = run_fixture_study(run.workspace.parent, run.inputs)
    assert again.gate == run.gate
    assert snapshot() == before


def copy_study(run: StudyRun, root: Path) -> Path:
    study = study_of(run)
    copied = root / "walk-forward" / study.study_id
    shutil.copytree(study.study_path, copied)
    return copied


def test_corrupt_or_rebound_development_evidence_fails_closed(
    run: StudyRun, tmp_path: Path
) -> None:
    study = study_of(run)
    plan = study.config.plan
    fold_id = plan.folds[0].fold_id
    copied = copy_study(run, tmp_path)
    record_path = copied / "folds" / fold_id / DEVELOPMENT_EVIDENCE_FILE
    original = read_record(record_path)
    # Rebound to another selection, consistently re-fingerprinted.
    write_record(record_path, {**original, "selection_id": "f" * 64})
    with pytest.raises(OOSIntegrityError, match="differs from its frozen fold"):
        load_prediction_development_window(
            load_oos_source(plan, copied), copied, fold_id=fold_id
        )
    # A forged membership identity is rejected the same way.
    evidence = DevelopmentEvidence.from_primitive(original)
    forged = replace(
        evidence,
        membership=type(evidence.membership).capture(
            {**evidence.membership.to_primitive(), "retained_decision_count": 1}
        ),
    )
    write_record(record_path, forged.to_primitive())
    with pytest.raises(OOSIntegrityError, match="differs from its frozen fold"):
        load_prediction_development_window(
            load_oos_source(plan, copied), copied, fold_id=fold_id
        )
    # A torn envelope, and an interrupted execution without its record.
    record_path.write_text(
        record_path.read_text().replace("development", "Development")
    )
    with pytest.raises(OOSIntegrityError, match="missing or invalid"):
        load_prediction_development_window(
            load_oos_source(plan, copied), copied, fold_id=fold_id
        )
    record_path.unlink()
    assert (copied / "folds" / fold_id / DEVELOPMENT_EVIDENCE_DIRECTORY).is_dir()
    dataset = dataset_of(run)
    with pytest.raises(EventDatasetIntegrityError, match="development evidence"):
        build_event_dataset(
            plan=plan,
            study_path=copied,
            workspace=run.workspace,
            combination_id=evidence.candidate.combination_id,
            feature_schema=dataset.feature_schema,
            target=dataset.target_definition,
        )


def test_evaluate_development_refuses_unknown_folds(run: StudyRun) -> None:
    study = study_of(run)
    with pytest.raises(WalkForwardError, match="not in the validation plan"):
        study.evaluate_development("missing-fold")


@pytest.fixture(scope="module")
def comparison(tmp_path_factory: pytest.TempPathFactory, prepared: None) -> EventStudy:
    del prepared
    root = tmp_path_factory.mktemp("qf69-comparison")
    return run_event_study(root, development_inputs(root), schema="4")


def test_development_rows_belong_only_to_the_frozen_candidate(
    comparison: EventStudy,
) -> None:
    plan = comparison.config.plan
    fold_id = plan.folds[0].fold_id
    study = WalkForwardStudy(
        comparison.config, comparison.adapter, comparison.study_path.parent
    )
    before = comparison.build()
    assert {
        "fold_id": fold_id,
        "role": PartitionRole.DEVELOPMENT.value,
        "reason": "role_not_executed_by_qf39_selection",
    } in cast(
        list[object],
        part(before.scientific.to_primitive()["partition_plan"])["excluded_sources"],
    )
    evidence = study.evaluate_development(fold_id)
    assert study.evaluate_development(fold_id) == evidence  # Verified, not rerun.
    selected = comparison.build()
    rows = selected.rows_for(PartitionRole.DEVELOPMENT)
    assert rows
    assert {selected.rows[index].fold_id for index in rows} == {fold_id}
    other = next(
        pair for pair in ("8/40", "8/48", "12/60") if pair != comparison.selected_pair
    )
    excluded = comparison.build(pair=other)
    assert not excluded.rows_for(PartitionRole.DEVELOPMENT)
    assert {
        "fold_id": fold_id,
        "role": PartitionRole.DEVELOPMENT.value,
        "reason": "frozen_selection_is_another_candidate",
    } in cast(
        list[object],
        part(excluded.scientific.to_primitive()["partition_plan"])["excluded_sources"],
    )
    # Selection and test evidence are unchanged by the development execution.
    for role in (PartitionRole.SELECTION, PartitionRole.WALK_FORWARD_TEST):
        assert observations(before, role) == observations(selected, role)
