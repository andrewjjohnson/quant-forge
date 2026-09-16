"""Rehashed result references cannot certify a malformed permanent marker."""

from copy import deepcopy

import pytest

from quantforge.configuration import Primitive, PrimitiveMapping, configuration_identity
from quantforge.experiments import ManifestError, inspect_validation, verify_artifacts
from quantforge.experiments._holdout_integrity import validate_holdout_consumption
from quantforge.experiments._json import mapping
from quantforge.walk_forward.persistence import write_record
from tests.unit.experiments.test_holdout_request_integrity import CapturedRequest
from tests.unit.experiments.test_holdout_request_integrity import (
    captured as captured,
)
from tests.unit.experiments.test_holdout_request_integrity import (
    no_research as no_research,
)
from tests.unit.experiments.test_holdout_request_integrity import (
    prediction as prediction,
)


def inspect_marker(
    captured: CapturedRequest, marker: PrimitiveMapping, *, reject: bool = True
) -> None:
    source = captured.completed.source
    original_state = captured.ledger.state(source)
    original_bytes = {captured.marker_path: captured.marker_path.read_bytes()}
    write_record(captured.marker_path, marker)
    if captured.result_path is not None:
        assert captured.result is not None
        original_bytes[captured.result_path] = captured.result_path.read_bytes()
        result = deepcopy(captured.result)
        result["consumption_sha256"] = configuration_identity(marker)
        write_record(captured.result_path, result)
        assert result["artifact"] == captured.result["artifact"]
    try:
        state = captured.ledger.state(source)
        assert state.state.value == "consumed"
        assert (state.result_reference is None) == (captured.result_path is None)
        before = {
            path: path.read_bytes()
            for path in captured.ledger.root.rglob("*")
            if path.is_file()
        }
        if reject:
            with pytest.raises(ManifestError, match="holdout consumption"):
                inspect_validation(
                    source,
                    captured.completed.study.study_path,
                    artifact_root=captured.ledger.root.parent,
                    ledger=captured.ledger,
                )
        else:
            inspected = inspect_validation(
                source,
                captured.completed.study.study_path,
                artifact_root=captured.ledger.root.parent,
                ledger=captured.ledger,
            )
            assert verify_artifacts(inspected.index, captured.ledger.root.parent).valid
            observed = mapping(
                inspected.provenance.observations.to_primitive()["holdout"]
            )
            assert observed["consumption_run_id"] == marker["consumption_run_id"]
            assert observed["consumed_at"] == marker["consumed_at"]
        assert {path: path.read_bytes() for path in before} == before
        assert captured.ledger.state(source) == state
    finally:
        for path, content in original_bytes.items():
            path.write_bytes(content)
    assert captured.ledger.state(source) == original_state


@pytest.mark.parametrize(
    "field", ["schema_version", "consumption_run_id", "consumed_at", "transition"]
)
@pytest.mark.parametrize("invalid", [None, "", " ", True, 1, [], {}])
def test_marker_execution_and_contract_fields_reject_invalid_domains(
    captured: CapturedRequest, field: str, invalid: Primitive
) -> None:
    marker = deepcopy(captured.marker)
    marker[field] = invalid
    inspect_marker(captured, marker)


@pytest.mark.parametrize(
    "field", ["schema_version", "consumption_run_id", "consumed_at", "transition"]
)
def test_marker_requires_execution_and_contract_fields(
    captured: CapturedRequest, field: str
) -> None:
    marker = deepcopy(captured.marker)
    del marker[field]
    inspect_marker(captured, marker)


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("schema_version", "2"),
        ("transition", "reversible_after_evaluation"),
        ("consumed_at", "not-a-timestamp"),
        ("consumed_at", "2024-01-02"),
        ("consumed_at", "2024-01-02T03:04:05"),
        ("consumed_at", "2024-01-02T03:04:05-05:00"),
    ],
)
def test_marker_rejects_unsupported_constants_and_timestamps(
    captured: CapturedRequest, field: str, invalid: str
) -> None:
    marker = deepcopy(captured.marker)
    marker[field] = invalid
    inspect_marker(captured, marker)


@pytest.mark.parametrize(
    ("field", "changed"),
    [
        ("symbol", "FOREIGN"),
        ("start", "2024-01-02"),
        ("end", "2025-12-31"),
        ("symbol", None),
        ("start", "not-a-session"),
        ("end", "not-a-session"),
    ],
)
def test_marker_exposure_scope_is_bound_to_the_source_reservation(
    captured: CapturedRequest, field: str, changed: Primitive
) -> None:
    marker = deepcopy(captured.marker)
    mapping(marker["exposure_scope"])[field] = changed
    inspect_marker(captured, marker)


@pytest.mark.parametrize("nested", [False, True], ids=["marker", "scope"])
def test_marker_and_scope_reject_undeclared_fields(
    captured: CapturedRequest, nested: bool
) -> None:
    marker = deepcopy(captured.marker)
    target = mapping(marker["exposure_scope"]) if nested else marker
    target["undeclared"] = None
    inspect_marker(captured, marker)


@pytest.mark.parametrize(
    "timestamp", [None, "2024-01-02T03:04:05+00:00", "2024-01-02T03:04:05Z"]
)
def test_valid_marker_preserves_execution_provenance(
    captured: CapturedRequest,
    timestamp: str | None,
) -> None:
    marker = deepcopy(captured.marker)
    if timestamp is not None:
        marker["consumed_at"] = timestamp
        marker["consumption_run_id"] = " retained execution ID "
    inspect_marker(captured, marker, reject=False)


def test_complete_marker_schema_rejects_every_missing_field(
    captured: CapturedRequest,
) -> None:
    source = captured.completed.source
    reservation = captured.ledger.state(source).reservation.to_primitive()
    for field in captured.marker:
        marker = deepcopy(captured.marker)
        del marker[field]
        before = configuration_identity(marker)
        with pytest.raises(ManifestError, match="holdout consumption"):
            validate_holdout_consumption(source, marker, reservation)
        assert configuration_identity(marker) == before
    for field in mapping(captured.marker["exposure_scope"]):
        marker = deepcopy(captured.marker)
        del mapping(marker["exposure_scope"])[field]
        with pytest.raises(ManifestError, match="holdout consumption"):
            validate_holdout_consumption(source, marker, reservation)
