"""Two matching, rehashed copies do not make a malformed holdout summary valid."""

from copy import deepcopy
from pathlib import Path

import pytest

from quantforge.configuration import Primitive, PrimitiveMapping, configuration_identity
from quantforge.experiments import ManifestError, inspect_validation, verify_artifacts
from quantforge.experiments._json import mapping
from quantforge.experiments._prediction_summary_integrity import (
    validate_prediction_summary,
)
from quantforge.oos import HoldoutEvaluation, HoldoutLedger
from quantforge.walk_forward.persistence import read_record, write_record
from tests.unit.experiments.test_adapters import block_research
from tests.unit.oos.conftest import CompletedStudy, complete_study

type Captured = tuple[CompletedStudy, HoldoutLedger, Path, PrimitiveMapping]


@pytest.fixture(scope="module")
def captured(tmp_path_factory: pytest.TempPathFactory) -> Captured:
    root = tmp_path_factory.mktemp("holdout-summary")
    completed = complete_study(root, prediction=True)
    source = completed.source
    ledger = HoldoutLedger.create(root / "ledger")
    ledger.reserve(source)
    ledger.consume(
        HoldoutEvaluation.prepare(
            source, completed.evaluator, selection_fold_id=source.folds[-1].fold_id
        ),
        run_id="summary-schema",
    )
    path = ledger.root / "lineages" / source.lineage_id / "result.json"
    return completed, ledger, path, read_record(path)


@pytest.fixture(autouse=True)
def no_research(captured: Captured, monkeypatch: pytest.MonkeyPatch) -> None:
    block_research(monkeypatch)


def inspect_summary(
    captured: Captured, summary: Primitive, *, reject: bool = True
) -> None:
    completed, ledger, path, original = captured
    result = deepcopy(original)
    artifact = mapping(result["artifact"])
    mapping(artifact["holdout_summary"])["summary"] = summary
    result["summary"] = deepcopy(summary)
    result["artifact_sha256"] = configuration_identity(artifact)
    write_record(path, result)
    before = {
        item: item.read_bytes() for item in ledger.root.rglob("*") if item.is_file()
    }
    state = ledger.state(completed.source)
    assert state.state.value == "consumed"
    if reject:
        with pytest.raises(ManifestError, match="holdout prediction summary"):
            inspect_validation(
                completed.source,
                completed.study.study_path,
                artifact_root=ledger.root.parent,
                ledger=ledger,
            )
    else:
        inspected = inspect_validation(
            completed.source,
            completed.study.study_path,
            artifact_root=ledger.root.parent,
            ledger=ledger,
        )
        assert verify_artifacts(inspected.index, ledger.root.parent).valid
    assert {item: item.read_bytes() for item in before} == before
    assert ledger.state(completed.source) == state


@pytest.mark.parametrize(
    "section",
    [
        None,
        "direction_distribution",
        "accuracy_interval",
        "matched_baseline",
        "signed_outcome",
        "mfe",
        "mae",
        "event_outcomes",
    ],
)
def test_captured_summary_requires_every_producer_field(
    captured: Captured, section: str | None
) -> None:
    original = mapping(captured[3]["summary"])
    target = original if section is None else mapping(original[section])
    for field in (*target, "undeclared"):
        summary = deepcopy(original)
        changed = summary if section is None else mapping(summary[section])
        if field == "undeclared":
            changed[field] = None
        else:
            del changed[field]
        inspect_summary(captured, summary)


@pytest.mark.parametrize("invalid", [{}, None, [], True, "summary"])
def test_matching_malformed_summary_copies_are_rejected(
    captured: Captured, invalid: Primitive
) -> None:
    inspect_summary(captured, invalid)


@pytest.mark.parametrize(
    "field",
    [
        "prediction_count",
        "generated_signal_count",
        "excluded_signal_count",
        "scheduled_decisions",
        "labeled_prediction_count",
        "accuracy_sample_count",
        "accuracy_unavailable_count",
        "direction_distribution",
        "accuracy_interval",
        "matched_baseline",
        "signed_outcome",
        "mfe",
        "mae",
        "event_outcomes",
    ],
)
def test_matching_typed_summary_counts_must_match_stored_decisions(
    captured: Captured, field: str
) -> None:
    summary = deepcopy(mapping(captured[3]["summary"]))
    if field == "direction_distribution":
        mapping(summary[field])["down"] = 7
    elif isinstance(summary[field], dict):
        metric = mapping(summary[field])
        metric[
            "unavailable_count" if "unavailable_count" in metric else "sample_count"
        ] = 7
        if field in {"signed_outcome", "mfe", "mae"} and metric["sample_count"]:
            metric["status"] = "partial"
        if field == "matched_baseline":
            metric.update(status="available", accuracy="0", accuracy_difference="0")
    else:
        count = summary[field]
        assert type(count) is int
        summary[field] = count + 7
    validate_prediction_summary(summary)
    inspect_summary(captured, summary)


def test_native_summary_preserves_captured_values(captured: Captured) -> None:
    inspect_summary(captured, captured[3]["summary"], reject=False)
