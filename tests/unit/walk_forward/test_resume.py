"""Crash boundaries, immutable freezes, terminal failures, and incompatible caches."""

from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from quantforge.configuration import PrimitiveMapping
from quantforge.validation import ValidationPlan
from quantforge.walk_forward import (
    CandidateUniverse,
    FoldStatus,
    FrozenSelection,
    OOSArtifact,
    SelectionEvidence,
    WalkForwardConfig,
    WalkForwardEvaluator,
    WalkForwardPersistenceError,
    WalkForwardStudy,
)
from quantforge.walk_forward.persistence import read_record, write_record
from tests.unit.walk_forward.fixtures import backtest_fixture, prediction_fixture


class ProbeEvaluator:
    """Observe the real adapter's boundaries, without replacing its execution."""

    def __init__(self, inner: WalkForwardEvaluator) -> None:
        self.inner = inner
        self.selection_calls: list[int] = []
        self.test_calls: list[int] = []
        self.fail_fold: int | None = None
        self.interrupt = False
        self.escape_universe = False
        self.mutate_freeze = False

    @property
    def universe(self) -> CandidateUniverse:
        return self.inner.universe

    def configuration(self) -> PrimitiveMapping:
        return self.inner.configuration()

    def validate(self, plan: ValidationPlan) -> None:
        self.inner.validate(plan)

    def membership(
        self, config: WalkForwardConfig, fold_index: int
    ) -> PrimitiveMapping:
        return self.inner.membership(config, fold_index)

    def select(
        self, config: WalkForwardConfig, fold_index: int, output_root: Path
    ) -> SelectionEvidence:
        self.selection_calls.append(fold_index)
        evidence = self.inner.select(config, fold_index, output_root)
        if self.escape_universe:
            return replace(
                evidence, candidate=replace(evidence.candidate, combination_id="f" * 64)
            )
        return evidence

    def evaluate(
        self,
        config: WalkForwardConfig,
        fold_index: int,
        selection: FrozenSelection,
        output_root: Path,
    ) -> OOSArtifact:
        self.test_calls.append(fold_index)
        # The state and immutable snapshot must exist before ANY test work.
        assert (
            read_record(output_root.parent / "selection.json")
            == selection.to_primitive()
        )
        state = read_record(output_root.parent / "state.json")
        assert state["status"] == FoldStatus.EVALUATING_TEST.value
        assert state["selection_id"] == selection.selection_id
        if fold_index == self.fail_fold:
            if self.interrupt:
                raise KeyboardInterrupt("fixture interruption after freeze")
            raise RuntimeError("api_key=secret must never enter persisted diagnostics")
        result = self.inner.evaluate(config, fold_index, selection, output_root)
        if self.mutate_freeze:
            record = read_record(output_root.parent / "selection.json")
            cast(PrimitiveMapping, record["candidate"])["parameters"] = {"injected": 1}
            write_record(output_root.parent / "selection.json", record)
        return result

    def validate_artifact(
        self,
        config: WalkForwardConfig,
        fold_index: int,
        selection: FrozenSelection,
        artifact: OOSArtifact,
        output_root: Path,
    ) -> None:
        self.inner.validate_artifact(
            config, fold_index, selection, artifact, output_root
        )


@pytest.mark.parametrize("family", ["prediction", "backtest"])
def test_completed_resume_performs_no_evaluation(tmp_path: Path, family: str) -> None:
    config, inner = (
        prediction_fixture(tmp_path)
        if family == "prediction"
        else backtest_fixture(tmp_path)
    )
    probe = ProbeEvaluator(inner)
    study = WalkForwardStudy(config, probe, tmp_path)
    original = study.run()
    assert all(fold.status is FoldStatus.COMPLETED for fold in original.folds)
    assert probe.selection_calls == probe.test_calls == [0, 1]
    assert study.resume() == original
    assert probe.selection_calls == probe.test_calls == [0, 1]


@pytest.mark.parametrize("family", ["prediction", "backtest"])
def test_resume_after_frozen_selection_never_reselects(
    tmp_path: Path, family: str
) -> None:
    config, inner = (
        prediction_fixture(tmp_path)
        if family == "prediction"
        else backtest_fixture(tmp_path)
    )
    probe = ProbeEvaluator(inner)
    probe.fail_fold, probe.interrupt = 0, True
    study = WalkForwardStudy(config, probe, tmp_path)
    with pytest.raises(KeyboardInterrupt):
        study.run()
    root = study.study_path / "folds" / config.plan.folds[0].fold_id
    frozen_before = (root / "selection.json").read_bytes()
    assert (
        read_record(root / "state.json")["status"] == FoldStatus.EVALUATING_TEST.value
    )
    assert not (root / "oos.json").exists()
    probe.fail_fold = None
    result = study.resume()
    assert all(fold.status is FoldStatus.COMPLETED for fold in result.folds)
    assert probe.selection_calls == [0, 1]
    assert probe.test_calls == [0, 0, 1]
    assert (root / "selection.json").read_bytes() == frozen_before


@pytest.mark.parametrize("retry", [False, True])
@pytest.mark.parametrize("family", ["prediction", "backtest"])
def test_failed_test_is_explicit_and_retry_keeps_frozen_selection(
    tmp_path: Path,
    retry: bool,
    family: str,
) -> None:
    config, inner = (
        prediction_fixture(tmp_path, retry_failed=retry)
        if family == "prediction"
        else backtest_fixture(tmp_path, retry_failed=retry)
    )
    probe = ProbeEvaluator(inner)
    probe.fail_fold = 0
    study = WalkForwardStudy(config, probe, tmp_path)
    original = study.run()
    assert [fold.status for fold in original.folds] == [
        FoldStatus.FAILED,
        FoldStatus.COMPLETED,
    ]
    root = study.study_path / "folds" / config.plan.folds[0].fold_id
    assert "secret" not in (root / "state.json").read_text()
    frozen_before = (root / "selection.json").read_bytes()
    probe.fail_fold = None
    resumed = study.resume()
    assert resumed.folds[0].status is (
        FoldStatus.COMPLETED if retry else FoldStatus.FAILED
    )
    assert len(resumed.folds[0].failures) == 1
    assert probe.selection_calls == [0, 1]
    assert probe.test_calls == ([0, 1, 0] if retry else [0, 1])
    assert (root / "selection.json").read_bytes() == frozen_before


def test_outside_universe_is_rejected_before_test(tmp_path: Path) -> None:
    config, inner = backtest_fixture(tmp_path)
    probe = ProbeEvaluator(inner)
    probe.escape_universe = True
    result = WalkForwardStudy(config, probe, tmp_path).run()
    assert [fold.status for fold in result.folds] == [FoldStatus.FAILED] * 2
    assert not probe.test_calls
    assert all(fold.selection is None for fold in result.folds)


def test_test_callback_cannot_change_frozen_config(tmp_path: Path) -> None:
    config, inner = backtest_fixture(tmp_path)
    probe = ProbeEvaluator(inner)
    probe.mutate_freeze = True
    result = WalkForwardStudy(config, probe, tmp_path).run()
    assert all(
        fold.status is FoldStatus.FAILED and fold.artifact is None
        for fold in result.folds
    )


@pytest.mark.parametrize(
    "filename", ["selection.json", "oos.json", "state.json", "manifest.json"]
)
def test_corrupt_artifacts_rejected_on_resume(tmp_path: Path, filename: str) -> None:
    config, inner = backtest_fixture(tmp_path)
    study = WalkForwardStudy(config, inner, tmp_path)
    study.run()
    root = (
        study.study_path
        if filename == "manifest.json"
        else study.study_path / "folds" / config.plan.folds[0].fold_id
    )
    target = root / filename
    target.write_text(
        target.read_text().replace('"fingerprint":"', '"fingerprint":"0', 1)
    )
    with pytest.raises(WalkForwardPersistenceError, match="corrupt"):
        study.resume()


@pytest.mark.parametrize("filename", ["selection.json", "oos.json"])
def test_incompatible_but_rehashed_artifacts_rejected(
    tmp_path: Path, filename: str
) -> None:
    config, inner = backtest_fixture(tmp_path)
    study = WalkForwardStudy(config, inner, tmp_path)
    study.run()
    root = study.study_path / "folds" / config.plan.folds[0].fold_id
    record = read_record(root / filename)
    record["selection_id"] = "f" * 64
    write_record(root / filename, record)
    with pytest.raises(WalkForwardPersistenceError):
        study.resume()


def test_partial_fold_is_not_inferred_complete_from_stale_artifact(
    tmp_path: Path,
) -> None:
    config, inner = backtest_fixture(tmp_path)
    probe = ProbeEvaluator(inner)
    study = WalkForwardStudy(config, probe, tmp_path)
    study.run()
    root = study.study_path / "folds" / config.plan.folds[0].fold_id
    state = read_record(root / "state.json")
    state["status"] = FoldStatus.EVALUATING_TEST.value
    state["artifact_id"] = None
    write_record(root / "state.json", state)
    probe.fail_fold = 0
    result = study.resume()
    assert result.folds[0].status is FoldStatus.FAILED
    assert result.folds[0].artifact is None
    assert probe.selection_calls == [0, 1]
    assert probe.test_calls == [0, 1, 0]
    # A subsequent resume must retain the failure despite the old artifact.
    probe.fail_fold = None
    assert study.resume().folds[0].status is FoldStatus.FAILED


def test_stop_policy_preserves_completed_folds(tmp_path: Path) -> None:
    config, inner = backtest_fixture(tmp_path)
    probe = ProbeEvaluator(inner)
    probe.fail_fold = 0
    result = WalkForwardStudy(
        replace(config, continue_on_failure=False), probe, tmp_path
    ).run()
    assert len(result.folds) == 1
    assert result.folds[0].status is FoldStatus.FAILED
    assert probe.selection_calls == [0]


def test_freeze_persistence_failure_never_evaluates_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from quantforge.walk_forward import study as module

    original = module.write_record

    def fail_freeze(
        path: Path, payload: PrimitiveMapping, *, immutable: bool = False
    ) -> None:
        if path.name == "selection.json":
            raise OSError("disk full")
        original(path, payload, immutable=immutable)

    monkeypatch.setattr(module, "write_record", fail_freeze)
    config, inner = backtest_fixture(tmp_path)
    probe = ProbeEvaluator(inner)
    result = WalkForwardStudy(config, probe, tmp_path).run()
    assert all(fold.status is FoldStatus.FAILED for fold in result.folds)
    assert not probe.test_calls


def test_result_persistence_interruption_requires_test_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from quantforge.walk_forward import study as module

    original = module.write_record
    interrupted = False

    def interrupt_after_artifact(
        path: Path, payload: PrimitiveMapping, *, immutable: bool = False
    ) -> None:
        nonlocal interrupted
        original(path, payload, immutable=immutable)
        if path.name == "oos.json" and not interrupted:
            interrupted = True
            raise KeyboardInterrupt("crash before completion state")

    monkeypatch.setattr(module, "write_record", interrupt_after_artifact)
    config, inner = backtest_fixture(tmp_path)
    probe = ProbeEvaluator(inner)
    study = WalkForwardStudy(config, probe, tmp_path)
    with pytest.raises(KeyboardInterrupt):
        study.run()
    result = study.resume()
    assert all(fold.status is FoldStatus.COMPLETED for fold in result.folds)
    assert probe.test_calls == [0, 0, 1]
    assert probe.selection_calls == [0, 1]
