"""Rehashed consumed requests cannot redefine their source holdout or lineage."""

from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path

import pytest

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.experiments import (
    ArtifactType,
    ManifestError,
    inspect_validation,
    verify_artifacts,
)
from quantforge.experiments._json import mapping
from quantforge.oos import HoldoutEvaluation, HoldoutLedger
from quantforge.walk_forward.persistence import read_record, write_record
from tests.unit.experiments.test_adapters import block_research
from tests.unit.oos.conftest import CompletedStudy, complete_study


@dataclass(frozen=True)
class CapturedRequest:
    completed: CompletedStudy
    ledger: HoldoutLedger
    marker_path: Path
    marker: PrimitiveMapping
    result_path: Path | None
    result: PrimitiveMapping | None


@pytest.fixture(scope="module", params=[False, True], ids=["backtest", "prediction"])
def prediction(request: pytest.FixtureRequest) -> bool:
    return bool(request.param)


@pytest.fixture(scope="module", params=[False, True], ids=["interrupted", "completed"])
def captured(
    tmp_path_factory: pytest.TempPathFactory,
    request: pytest.FixtureRequest,
    prediction: bool,
) -> CapturedRequest:
    root = tmp_path_factory.mktemp("holdout-request")
    completed = complete_study(root, prediction=prediction)
    source = completed.source
    ledger = HoldoutLedger.create(root / "ledger")
    ledger.reserve(source)
    evaluation = HoldoutEvaluation.prepare(
        source, completed.evaluator, selection_fold_id=source.folds[-1].fold_id
    )
    if request.param:
        ledger.consume(evaluation, run_id="captured-holdout")
        result_path = ledger.root / "lineages" / source.lineage_id / "result.json"
    else:

        def interrupt(*args: object, **kwargs: object) -> None:
            raise RuntimeError("interrupted after permanent consumption")

        with pytest.MonkeyPatch.context() as interruption:
            interruption.setattr(HoldoutEvaluation, "_evaluate", interrupt)
            with pytest.raises(RuntimeError, match="permanent consumption"):
                ledger.consume(evaluation, run_id="interrupted-holdout")
        result_path = None
    marker_path = ledger.root / "exposures" / f"{source.lineage_id}.json"
    return CapturedRequest(
        completed,
        ledger,
        marker_path,
        read_record(marker_path),
        result_path,
        None if result_path is None else read_record(result_path),
    )


@pytest.fixture(autouse=True)
def no_research(captured: CapturedRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    block_research(monkeypatch)

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("request validation attempted holdout execution or partitioning")

    for method in ("prepare", "validate", "_evaluate"):
        monkeypatch.setattr(HoldoutEvaluation, method, forbidden)
    for method in ("reserve", "consume"):
        monkeypatch.setattr(HoldoutLedger, method, forbidden)
    for name in ("select_window_observations", "observation_keys", "project_dataset"):
        monkeypatch.setattr(f"quantforge.oos.holdout_evaluation.{name}", forbidden)


def inspect_request(
    captured: CapturedRequest,
    request: PrimitiveMapping,
    *,
    reject: bool = True,
) -> None:
    source = captured.completed.source
    original_state = captured.ledger.state(source)
    original_bytes = {captured.marker_path: captured.marker_path.read_bytes()}
    marker = deepcopy(captured.marker)
    marker["request"] = request
    marker["request_id"] = configuration_identity(request)
    write_record(captured.marker_path, marker)
    if captured.result_path is not None:
        assert captured.result is not None
        original_bytes[captured.result_path] = captured.result_path.read_bytes()
        result = deepcopy(captured.result)
        result["request_id"] = marker["request_id"]
        result["consumption_sha256"] = configuration_identity(marker)
        write_record(captured.result_path, result)
        assert result["artifact"] == captured.result["artifact"]
    try:
        current = captured.ledger.state(source)
        assert current.state.value == "consumed"
        assert (current.result_reference is None) == (captured.result_path is None)
        before = {
            path: path.read_bytes()
            for path in captured.ledger.root.rglob("*")
            if path.is_file()
        }
        if reject:
            with pytest.raises(ManifestError):
                inspect_validation(
                    source,
                    captured.completed.study.study_path,
                    artifact_root=captured.ledger.root.parent,
                    ledger=captured.ledger,
                )
        else:
            bundle = inspect_validation(
                source,
                captured.completed.study.study_path,
                artifact_root=captured.ledger.root.parent,
                ledger=captured.ledger,
            )
            assert verify_artifacts(bundle.index, captured.ledger.root.parent).valid
            types = [entry.artifact_type for entry in bundle.index.entries]
            assert types.count(ArtifactType.HOLDOUT_CONSUMPTION) == 1
            assert types.count(ArtifactType.HOLDOUT_RESULT) == (
                captured.result_path is not None
            )
        assert {path: path.read_bytes() for path in before} == before
        assert captured.ledger.state(source) == current
    finally:
        for path, content in original_bytes.items():
            path.write_bytes(content)
    assert captured.ledger.state(source) == original_state


@pytest.mark.parametrize(
    "field",
    [
        "schema_version",
        "operation",
        "tail_policy",
        "lineage_id",
        "lineage",
        "study_id",
        "plan_id",
        "study_definition",
        "holdout",
        "frozen_selection",
    ],
)
@pytest.mark.parametrize("change", ["missing", "null", "changed"])
def test_rehashed_request_source_fields_cannot_change(
    captured: CapturedRequest,
    field: str,
    change: str,
) -> None:
    request = deepcopy(mapping(captured.marker["request"]))
    if change == "missing":
        request.pop(field)
    elif change == "null":
        request[field] = None
    elif isinstance(request[field], dict):
        mapping(request[field])["foreign_metadata"] = True
    else:
        request[field] = "foreign"
    inspect_request(captured, request)


def test_rehashed_request_rejects_undeclared_fields(captured: CapturedRequest) -> None:
    request = deepcopy(mapping(captured.marker["request"]))
    request["undeclared"] = None
    inspect_request(captured, request)


def test_rehashed_valid_holdout_boundary_cannot_replace_the_reserved_interval(
    captured: CapturedRequest,
) -> None:
    source = captured.completed.source
    holdout = source.plan.final_holdout
    changed = replace(
        holdout,
        window=replace(holdout.window, interval=source.plan.folds[0].test.interval),
    )
    assert changed.holdout_id != holdout.holdout_id
    request = deepcopy(mapping(captured.marker["request"]))
    request["holdout"] = changed.to_primitive()
    inspect_request(captured, request)


def test_native_completed_and_interrupted_requests_remain_consumed(
    captured: CapturedRequest,
) -> None:
    inspect_request(captured, mapping(captured.marker["request"]), reject=False)
