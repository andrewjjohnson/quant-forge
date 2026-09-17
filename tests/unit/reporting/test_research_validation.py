from pathlib import Path
from typing import cast

import pytest

from quantforge.configuration import PrimitiveMapping
from quantforge.experiments import (
    StudyType,
    create_manifest,
    inspect_study,
    inspect_validation,
    write_manifest,
)
from quantforge.oos import (
    HoldoutConsumptionRecord,
    HoldoutEvaluation,
    HoldoutLedger,
    aggregate_backtest,
    aggregate_prediction,
    export_oos_aggregate,
)
from quantforge.reporting import (
    ReportPhase,
    ResearchReportError,
    build_research_report,
    export_research_report,
)
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_contracts import execution
from tests.unit.oos.conftest import complete_study
from tests.unit.reporting.test_research_reports import codes, values


def block_holdout(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("report attempted holdout mutation/evaluation")

    for name in ("consume", "reserve", "create"):
        monkeypatch.setattr(HoldoutLedger, name, forbidden)
    for name in ("prepare", "_evaluate"):
        monkeypatch.setattr(HoldoutEvaluation, name, forbidden)


@pytest.mark.parametrize("prediction", [True, False], ids=["prediction", "backtest"])
def test_native_validation_oos_and_authoritative_reserved_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prediction: bool,
) -> None:
    completed = complete_study(tmp_path, prediction=prediction)
    source = completed.source
    aggregate = (
        aggregate_prediction(source) if prediction else aggregate_backtest(source)
    )
    aggregate_path = export_oos_aggregate(aggregate, tmp_path / "oos")
    ledger = HoldoutLedger.create(tmp_path / "ledger")
    ledger.reserve(source)
    manifest = create_manifest(
        inspect_validation(
            source,
            completed.study.study_path,
            artifact_root=tmp_path,
            aggregate_path=aggregate_path,
            ledger=ledger,
            study_type=StudyType.OOS_VALIDATION,
        ),
        execution(),
    )
    path = write_manifest(manifest, tmp_path / "experiments", artifact_root=tmp_path)
    block_research(monkeypatch)
    block_holdout(monkeypatch)
    report = build_research_report(
        path, artifact_root=tmp_path, holdout_source=source, holdout_ledger=ledger
    )
    assert (
        values(report, "Final holdout state")[0]
        == values(
            build_research_report(
                path,
                artifact_root=tmp_path,
                holdout_source=source,
                holdout_ledger=ledger,
            ),
            "Final holdout state",
        )[0]
    )
    state = cast(PrimitiveMapping, values(report, "Final holdout state")[0])
    assert state["state"] == "reserved_unconsumed"
    assert state["authority_checked"] is True
    assert len(values(report, "Per-fold OOS result")) == 2
    summary = cast(PrimitiveMapping, values(report, "Walk-forward OOS summary")[0])
    assert summary["summary"] == aggregate.summary.to_primitive()
    assert values(report, "Configuration turnover") == [
        aggregate.stability.to_primitive()
    ]
    assert values(report, "Validation plan and historical reservation definition") == [
        source.plan.to_manifest()
    ]
    html = export_research_report(report, tmp_path / "reports").read_text()
    assert source.lineage_id in html
    assert "WALK-FORWARD OOS" in html
    assert "VALIDATION / SELECTION" in html
    assert "FINAL HOLDOUT" in html
    assert "purge policy" in html
    assert "IN_SAMPLE_ONLY" not in codes(report)
    assert len(values(report, "Persisted fold status")) == 2
    assert "HOLDOUT_ALREADY_CONSUMED" not in codes(report)
    if prediction:
        assert "normalized equity" not in html
    else:
        assert (
            summary["normalized_equity"]
            == aggregate.to_primitive()["normalized_equity"]
        )


def test_stale_reservation_cannot_override_later_consumption_and_missing_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed = complete_study(tmp_path, prediction=True)
    source = completed.source
    ledger = HoldoutLedger.create(tmp_path / "ledger")
    ledger.reserve(source)
    before = write_manifest(
        create_manifest(
            inspect_validation(
                source,
                completed.study.study_path,
                artifact_root=tmp_path,
                ledger=ledger,
            ),
            execution(),
        ),
        tmp_path / "experiments",
        artifact_root=tmp_path,
    )
    untouched = build_research_report(
        before, artifact_root=tmp_path, holdout_source=source, holdout_ledger=ledger
    )
    reserved_record = ledger.state(source)
    # Reservation snapshots alone never assert current unseen status.
    unknown = build_research_report(before, artifact_root=tmp_path)
    assert (
        cast(PrimitiveMapping, values(unknown, "Final holdout state")[0])["state"]
        == "unavailable"
    )
    ledger.consume(
        HoldoutEvaluation.prepare(
            source, completed.evaluator, selection_fold_id=source.folds[-1].fold_id
        ),
        run_id="consumption-001",
    )
    consumed_record = ledger.state(source)
    after = write_manifest(
        create_manifest(
            inspect_validation(
                source,
                completed.study.study_path,
                artifact_root=tmp_path,
                ledger=ledger,
                study_type=StudyType.HOLDOUT_VALIDATION,
            ),
            execution(),
        ),
        tmp_path / "experiments",
        artifact_root=tmp_path,
    )
    block_research(monkeypatch)
    block_holdout(monkeypatch)
    states = iter((reserved_record, consumed_record))

    def changing_state(_: object) -> HoldoutConsumptionRecord:
        return next(states)

    with monkeypatch.context() as race:
        race.setattr(ledger, "state", changing_state)
        with pytest.raises(ResearchReportError, match="changed during rendering"):
            build_research_report(
                before,
                artifact_root=tmp_path,
                holdout_source=source,
                holdout_ledger=ledger,
            )
    stale = build_research_report(
        before, artifact_root=tmp_path, holdout_source=source, holdout_ledger=ledger
    )
    with pytest.raises(ResearchReportError, match="changed before export"):
        export_research_report(untouched, tmp_path / "reports")
    assert stale.report_id != untouched.report_id
    assert (
        cast(PrimitiveMapping, values(stale, "Final holdout state")[0])["state"]
        == "consumed"
    )
    assert values(stale, "Consumed final holdout result") == [None]
    fresh = build_research_report(
        after, artifact_root=tmp_path, holdout_source=source, holdout_ledger=ledger
    )
    assert "HOLDOUT_ALREADY_CONSUMED" in codes(fresh)
    assert values(fresh, "Consumed final holdout result")[0] is not None
    html = export_research_report(fresh, tmp_path / "reports").read_text()
    assert "consumption-001" in html
    assert "consumed" in html
    # A missing result or absent authority cannot erase permanent consumption.
    (ledger.root / "lineages" / source.lineage_id / "result.json").unlink()
    missing = build_research_report(
        after, artifact_root=tmp_path, holdout_source=source, holdout_ledger=ledger
    )
    assert "HOLDOUT_ALREADY_CONSUMED" in codes(missing)
    assert "ARTIFACT_INTEGRITY" in codes(missing)
    assert values(missing, "Consumed final holdout result") == [None]
    historical = build_research_report(after, artifact_root=tmp_path)
    assert "HOLDOUT_ALREADY_CONSUMED" in codes(historical)


def test_holdout_authority_must_match_exact_manifest_and_errors_propagate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed = complete_study(tmp_path, prediction=True)
    ledger = HoldoutLedger.create(tmp_path / "ledger")
    ledger.reserve(completed.source)
    path = write_manifest(
        create_manifest(
            inspect_validation(
                completed.source,
                completed.study.study_path,
                artifact_root=tmp_path,
                ledger=ledger,
            ),
            execution(),
        ),
        tmp_path / "experiments",
        artifact_root=tmp_path,
    )
    with pytest.raises(ResearchReportError, match="paired"):
        build_research_report(path, artifact_root=tmp_path, holdout_ledger=ledger)
    other = complete_study(tmp_path / "other", prediction=False)
    with pytest.raises(ResearchReportError, match="lineage"):
        build_research_report(
            path,
            artifact_root=tmp_path,
            holdout_source=other.source,
            holdout_ledger=ledger,
        )

    def unavailable(*args: object, **kwargs: object) -> None:
        raise ValueError("ledger unavailable")

    monkeypatch.setattr(ledger, "state", unavailable)
    with pytest.raises(ValueError, match="ledger unavailable"):
        build_research_report(
            path,
            artifact_root=tmp_path,
            holdout_source=completed.source,
            holdout_ledger=ledger,
        )


def test_backtest_captured_in_validation_is_labeled_oos_not_in_sample(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed = complete_study(tmp_path, prediction=False)
    validation = inspect_validation(
        completed.source, completed.study.study_path, artifact_root=tmp_path
    )
    fold = completed.source.folds[0]
    folder = next(
        (completed.study.study_path / "folds" / fold.fold_id / "test").iterdir()
    )
    primary = inspect_study(StudyType.BACKTEST, folder, artifact_root=tmp_path)
    manifest = create_manifest(primary, execution(), validation=validation)
    path = write_manifest(manifest, tmp_path / "experiments", artifact_root=tmp_path)
    block_research(monkeypatch)
    report = build_research_report(path, artifact_root=tmp_path)
    assert all(
        section.phase is ReportPhase.OOS
        for section in report.sections
        if section.title in {"Backtest performance and drawdown", "Equity and returns"}
    )
    assert "IN_SAMPLE_ONLY" not in codes(report)
    assert len(values(report, "Persisted fold status")) == 2
