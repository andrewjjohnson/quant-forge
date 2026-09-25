"""Durable compact ingestion must outlive its caller-owned source file."""

import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from quantforge.oos import (
    HoldoutEvaluation,
    HoldoutLedger,
    HoldoutState,
    load_oos_source,
)
from quantforge.prediction.errors import InvalidPredictionOutputError
from quantforge.walk_forward import PredictionEvaluator, WalkForwardStudy
from tests.unit.walk_forward.test_incremental_prediction import compact_adapter
from tests.unit.walk_forward.timestamp_fixtures import timestamp_fixture


@pytest.fixture(scope="module")
def finalized_evaluation(tmp_path_factory: pytest.TempPathFactory) -> HoldoutEvaluation:
    root = tmp_path_factory.mktemp("finalized-holdout")
    config, original = timestamp_fixture(root)
    adapter = compact_adapter(original)
    study = WalkForwardStudy(config, adapter, root / "study")
    study.run()
    source = load_oos_source(config.plan, study.study_path)
    evaluation = HoldoutEvaluation.prepare(
        source, adapter, selection_fold_id=source.folds[-1].fold_id
    )
    adapter.evaluate_partition(
        config.plan, evaluation.permitted, evaluation.selection, root / "finalized"
    )
    return replace(
        evaluation,
        finalized_prediction_window=root / "finalized" / "prediction-window.jsonl",
    )


@pytest.fixture
def imported_request(
    finalized_evaluation: HoldoutEvaluation,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[HoldoutEvaluation, HoldoutLedger, Path]:
    original = finalized_evaluation.finalized_prediction_window
    assert original is not None
    external = tmp_path / "external" / original.name
    external.parent.mkdir()
    shutil.copyfile(original, external)
    evaluation = replace(finalized_evaluation, finalized_prediction_window=external)
    ledger = HoldoutLedger.create(tmp_path / "ledger")
    ledger.reserve(evaluation.source)

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("finalized holdout must not execute prediction research")

    monkeypatch.setattr(PredictionEvaluator, "evaluate_partition", forbidden)
    return evaluation, ledger, external


@pytest.mark.parametrize("change", ["delete", "move", "corrupt"])
def test_consumed_copy_survives_changes_to_external_source(
    imported_request: tuple[HoldoutEvaluation, HoldoutLedger, Path], change: str
) -> None:
    evaluation, ledger, external = imported_request
    consumed = ledger.consume(evaluation, run_id="first-import")
    expected = ledger.result(evaluation)
    if change == "delete":
        external.unlink()
    elif change == "move":
        external.rename(external.with_name("moved.jsonl"))
    else:
        external.write_bytes(b"corrupt caller-owned source\n")

    reopened = HoldoutLedger(ledger.root)
    assert reopened.result(evaluation) == expected
    assert reopened.consume(evaluation, run_id="retry") == consumed
    assert reopened.consume(evaluation, run_id="reproduce", reproduce=True) == consumed
    assert reopened.state(evaluation.source) == consumed

    # Recovery after loss of the result envelope must also use the durable copy.
    result_path = (
        ledger.root / "lineages" / evaluation.source.lineage_id / "result.json"
    )
    result_path.unlink()
    assert reopened.state(evaluation.source).state is HoldoutState.CONSUMED
    assert reopened.consume(evaluation, run_id="recover-result") == consumed
    assert reopened.result(evaluation) == expected


@pytest.mark.parametrize("change", ["missing", "corrupt", "foreign_window"])
def test_first_import_still_validates_external_source(
    imported_request: tuple[HoldoutEvaluation, HoldoutLedger, Path], change: str
) -> None:
    evaluation, ledger, external = imported_request
    if change == "missing":
        external.unlink()
    elif change == "corrupt":
        external.write_bytes(b"corrupt source\n")
    else:
        foreign = next(
            r.path for r in evaluation.source.prediction_windows if r is not None
        )
        shutil.copyfile(foreign, external)
    with pytest.raises((ValueError, OSError)):
        ledger.consume(evaluation, run_id="invalid-import")
    state = ledger.state(evaluation.source)
    assert state.state is HoldoutState.CONSUMED
    assert state.result_reference is None


def test_durable_copy_corruption_is_not_repaired_from_external_source(
    imported_request: tuple[HoldoutEvaluation, HoldoutLedger, Path],
) -> None:
    evaluation, ledger, _ = imported_request
    consumed = ledger.consume(evaluation, run_id="first-import")
    owned = (
        ledger.root
        / "lineages"
        / evaluation.source.lineage_id
        / "evaluation"
        / "prediction-window.jsonl"
    )
    owned.write_bytes(b"corrupt ledger-owned artifact\n")
    with pytest.raises(InvalidPredictionOutputError):
        ledger.result(evaluation)
    for reproduce in (False, True):
        with pytest.raises(InvalidPredictionOutputError):
            ledger.consume(evaluation, run_id="retry", reproduce=reproduce)
    assert ledger.state(evaluation.source) == consumed
    assert owned.read_bytes() == b"corrupt ledger-owned artifact\n"
