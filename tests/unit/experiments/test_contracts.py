import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from typing import cast

import pytest

from quantforge.configuration import (
    Primitive,
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
)
from quantforge.experiments import (
    ArtifactFormat,
    ArtifactIndex,
    ArtifactRelationship,
    ArtifactType,
    CodeProvenance,
    ExecutionProvenance,
    ExperimentManifest,
    ManifestError,
    RelationshipType,
    StudyProvenance,
    StudyType,
    index_artifact,
    read_manifest,
    verify_artifacts,
    write_manifest,
)


def execution(run_id: str = "execution-1") -> ExecutionProvenance:
    return ExecutionProvenance(
        run_id,
        datetime(2026, 9, 14, tzinfo=UTC),
        CodeProvenance(
            "0.1.0",
            "a" * 40,
            False,
            "b" * 64,
            "3.13.7",
            PrimitiveMappingSnapshot.capture({"TA-Lib": "0.7.1"}),
        ),
    )


def write_json(path: Path, content: PrimitiveMapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(PrimitiveMappingSnapshot.capture(content).canonical_json + "\n")


def manifest(
    tmp_path: Path, configuration: PrimitiveMapping | None = None
) -> ExperimentManifest:
    path = tmp_path / "result.json"
    write_json(path, {"schema_version": "1", "study_id": "source", "result": [1, 2]})
    entry = index_artifact(
        tmp_path,
        path="result.json",
        artifact_type=ArtifactType.PREDICTION_RESULT,
        schema_version="1",
        producer_study_id="source",
        producer_artifact_id="result",
        bindings={"/study_id": "source", "/schema_version": "1"},
    )
    return ExperimentManifest(
        StudyProvenance(
            StudyType.PREDICTION,
            "source",
            PrimitiveMappingSnapshot.capture(configuration or {"rule": "fixed"}),
        ),
        execution(),
        ArtifactIndex((entry,)),
    )


def test_deterministic_roundtrip_and_execution_identity(tmp_path: Path) -> None:
    original = manifest(tmp_path)
    repeated = replace(original, execution=execution("execution-2"))
    assert original.study_id == repeated.study_id
    assert original.manifest_id != repeated.manifest_id
    assert original.provenance.producer_study_id == "source"
    assert original.serialize() == original.serialize()
    path = write_manifest(original, tmp_path / "manifests", artifact_root=tmp_path)
    loaded = read_manifest(path, artifact_root=tmp_path)
    assert loaded.serialize() == original.serialize()
    assert write_manifest(original, path.parent, artifact_root=tmp_path) == path
    assert json.loads(path.read_bytes())["schema_version"] == "1"


@pytest.mark.parametrize(
    "field",
    [
        "dataset_family_id",
        "data_fingerprint",
        "timeframe",
        "session_policy",
        "aggregation_policy",
        "completion_policy",
        "backend_id",
        "backend_version",
        "rule",
        "strategy",
        "outcome_horizon",
        "target_stop",
        "search_space",
        "validation_plan",
        "validation_lineage",
    ],
)
def test_material_configuration_cannot_alias_even_with_same_producer_id(
    tmp_path: Path, field: str
) -> None:
    first = manifest(tmp_path, {field: "original"})
    second = manifest(tmp_path, {field: "changed"})
    assert first.study_id != second.study_id


def test_provenance_and_manifest_are_detached(tmp_path: Path) -> None:
    original = manifest(tmp_path)
    changed = original.to_primitive()
    cast(PrimitiveMapping, changed["provenance"])["configuration"] = {}
    assert original.provenance.configuration.to_primitive() == {"rule": "fixed"}
    observations = PrimitiveMappingSnapshot.capture({"holdout": {"state": "consumed"}})
    updated = replace(
        original, provenance=replace(original.provenance, observations=observations)
    )
    assert updated.study_id == original.study_id
    assert updated.manifest_id != original.manifest_id


def test_code_and_random_seed_are_material(tmp_path: Path) -> None:
    original = manifest(tmp_path)
    for updated in (
        replace(
            original.execution,
            code=replace(original.execution.code, git_commit="c" * 40),
        ),
        replace(
            original.execution,
            random_seeds=PrimitiveMappingSnapshot.capture({"search": 42}),
        ),
    ):
        assert replace(original, execution=updated).study_id != original.study_id


def test_timestamp_requires_zone_and_normalizes_to_utc() -> None:
    with pytest.raises(ManifestError, match="timezone-aware"):
        ExecutionProvenance("run", datetime(2026, 1, 1))
    local = datetime(2026, 9, 13, 19, tzinfo=timezone(timedelta(hours=-5)))
    assert (
        replace(execution(), created_at=local).to_primitive()
        == execution().to_primitive()
    )


def test_immutable_write_rejects_changed_existing_bytes(tmp_path: Path) -> None:
    original = manifest(tmp_path)
    path = write_manifest(original, tmp_path / "manifests", artifact_root=tmp_path)
    path.write_text("tampered")
    with pytest.raises(ManifestError, match="immutable"):
        write_manifest(original, path.parent, artifact_root=tmp_path)
    assert path.read_text() == "tampered"
    with pytest.raises(ManifestError):
        read_manifest(path)


@pytest.mark.parametrize(
    "mutation",
    [
        "old_version",
        "future_version",
        "omitted_version",
        "changed_id",
        "unknown_field",
        "noncanonical",
        "duplicate_key",
        "artifact_metadata",
        "edge",
    ],
)
def test_strict_reading_rejects_incompatible_metadata(
    tmp_path: Path, mutation: str
) -> None:
    original = manifest(tmp_path)
    path = write_manifest(original, tmp_path / "manifests", artifact_root=tmp_path)
    document = cast(PrimitiveMapping, json.loads(path.read_bytes()))
    if mutation == "old_version":
        document["schema_version"] = "0"
    elif mutation == "future_version":
        document["schema_version"] = "2"
    elif mutation == "omitted_version":
        del document["schema_version"]
    elif mutation == "changed_id":
        document["study_id"] = "x"
    elif mutation == "unknown_field":
        document["guess"] = "not supported"
    elif mutation == "artifact_metadata":
        index = cast(PrimitiveMapping, document["artifact_index"])
        cast(list[PrimitiveMapping], index["entries"])[0]["schema_version"] = "2"
    elif mutation == "edge":
        cast(PrimitiveMapping, document["artifact_index"])["relationships"] = [
            {
                "source_id": "a" * 64,
                "target_id": "b" * 64,
                "relationship": "derived_from",
            }
        ]
    if mutation == "noncanonical":
        path.write_text(json.dumps(document, indent=2))
    elif mutation == "duplicate_key":
        path.write_text('{"schema_version":"0",' + path.read_text()[1:])
    else:
        write_json(path, document)
    with pytest.raises(ManifestError):
        read_manifest(path)


def test_missing_and_tampered_artifacts_do_not_rehash(tmp_path: Path) -> None:
    original = manifest(tmp_path)
    entry = original.artifacts.entries[0]
    (tmp_path / entry.path).write_text('{"valid_json_but_changed":true}')
    report = verify_artifacts(original.artifacts, tmp_path)
    assert [issue.code for issue in report.issues] == ["content_hash_mismatch"]
    assert original.artifacts.entries[0].sha256 == entry.sha256
    with pytest.raises(ManifestError):
        write_manifest(original, tmp_path / "manifests", artifact_root=tmp_path)
    (tmp_path / entry.path).unlink()
    assert (
        verify_artifacts(original.artifacts, tmp_path).issues[0].code
        == "missing_artifact"
    )


def test_schema_bindings_are_verified_even_if_hash_is_current(tmp_path: Path) -> None:
    original = manifest(tmp_path)
    entry = replace(
        original.artifacts.entries[0],
        bindings=PrimitiveMappingSnapshot.capture({"/schema_version": "2"}),
    )
    report = verify_artifacts(ArtifactIndex((entry,)), tmp_path)
    assert report.issues[0].code == "incompatible_metadata"


def test_optional_absence_is_explicit_and_not_silently_adopted(tmp_path: Path) -> None:
    entry = index_artifact(
        tmp_path,
        path="optional.csv",
        artifact_type=ArtifactType.REPORT,
        schema_version="1",
        producer_study_id="source",
        producer_artifact_id="optional",
        required=False,
    )
    report = verify_artifacts(ArtifactIndex((entry,)), tmp_path)
    assert report.valid
    assert report.absent_optional == (entry.artifact_id,)
    (tmp_path / "optional.csv").write_text("new,content\n")
    assert not verify_artifacts(ArtifactIndex((entry,)), tmp_path).valid


def test_artifact_order_relationships_and_duplicate_identity(tmp_path: Path) -> None:
    first = manifest(tmp_path).artifacts.entries[0]
    (tmp_path / "chart.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
    second = index_artifact(
        tmp_path,
        path="chart.svg",
        artifact_type=ArtifactType.CHART,
        schema_version="1",
        producer_study_id="source",
        producer_artifact_id="chart",
    )
    edge = ArtifactRelationship(
        second.artifact_id, RelationshipType.VISUALIZES, first.artifact_id
    )
    index = ArtifactIndex((first, second), (edge,))
    assert (
        index.to_primitive() == ArtifactIndex((second, first), (edge,)).to_primitive()
    )
    assert verify_artifacts(index, tmp_path).valid
    with pytest.raises(ManifestError, match="duplicate"):
        ArtifactIndex((first, first))
    with pytest.raises(ManifestError, match="duplicate"):
        ArtifactIndex((first, replace(first, path="different.json")))
    with pytest.raises(ManifestError, match="references"):
        ArtifactIndex((first,), (edge,))


@pytest.mark.parametrize(
    "filename",
    [
        "../outside.csv",
        "/outside.csv",
        "a/../outside.csv",
        "./result.csv",
        "https://example.org/file.csv",
    ],
)
def test_unsafe_paths_are_rejected(tmp_path: Path, filename: str) -> None:
    with pytest.raises(ManifestError):
        index_artifact(
            tmp_path,
            path=filename,
            artifact_type=ArtifactType.REPORT,
            schema_version="1",
            producer_study_id="source",
            producer_artifact_id="report",
            required=False,
            file_format=ArtifactFormat.CSV,
        )


def test_symlink_escape_rejected(tmp_path: Path) -> None:
    (tmp_path / "escape.json").symlink_to(tmp_path.parent / "external.json")
    with pytest.raises(ManifestError, match="escapes"):
        index_artifact(
            tmp_path,
            path="escape.json",
            artifact_type=ArtifactType.REPORT,
            schema_version="1",
            producer_study_id="source",
            producer_artifact_id="report",
            required=False,
        )


@pytest.mark.parametrize(
    "key",
    [
        "api_key",
        "TIINGO_API_KEY",
        "accessToken",
        "refresh_token",
        "api_token",
        "auth_token",
        "session_token",
        "providerApiToken",
        "AUTH-TOKEN",
        "session.token",
        "password",
        "provider_secret",
        "credentials",
        "connection_string",
        "Authorization",
        "account_id",
        "account_number",
        "broker_account",
        "trading_account",
        "brokerAccountNumber",
        "ACCOUNT-NO",
        "provider.account_identifier",
    ],
)
def test_credentials_rejected_without_echoing_values(tmp_path: Path, key: str) -> None:
    with pytest.raises(ManifestError) as error:
        manifest(tmp_path, {"provider": {key: "sensitive-marker-123"}})
    assert "sensitive-marker-123" not in str(error.value)


@pytest.mark.parametrize(
    "key",
    [
        "api_token",
        "auth_token",
        "session_token",
        "account_number",
        "broker_account",
        "trading_account",
        "brokerAccountNumber",
        "ACCOUNT-NO",
        "provider.account_identifier",
    ],
)
@pytest.mark.parametrize("in_binding", [False, True])
def test_credential_fields_are_rejected_in_json_and_bindings(
    tmp_path: Path, key: str, in_binding: bool
) -> None:
    payload: PrimitiveMapping = {"provider": [{key: "sensitive-marker-123"}]}
    write_json(tmp_path / "artifact.json", {"safe": True} if in_binding else payload)
    if in_binding:
        entry = manifest(tmp_path).artifacts.entries[0]
        with pytest.raises(ManifestError) as error:
            replace(
                entry,
                bindings=PrimitiveMappingSnapshot.capture(
                    {f"/{key}": "sensitive-marker-123"}
                ),
            )
    else:
        with pytest.raises(ManifestError) as error:
            index_artifact(
                tmp_path,
                path="artifact.json",
                artifact_type=ArtifactType.REPORT,
                schema_version="1",
                producer_study_id="study",
                producer_artifact_id="result",
            )
    assert "sensitive-marker-123" not in str(error.value)


@pytest.mark.parametrize("account_label", ["benchmark", "strategy"])
def test_internal_account_labels_are_safe_metadata(
    tmp_path: Path, account_label: str
) -> None:
    record = manifest(tmp_path, {"account_id": account_label})
    path = write_manifest(record, tmp_path / "manifests", artifact_root=tmp_path)
    assert read_manifest(path).serialize() == record.serialize()


@pytest.mark.parametrize(
    "payload",
    [
        {"broker_account_id": "benchmark"},
        {"account_number": "benchmark"},
        {"broker_account": "strategy"},
        {"trading_account": "benchmark"},
        {"account_id": {"account_id": "benchmark"}},
        {"account_id": ["strategy"]},
        {"account_id": "benchmark-sensitive-marker"},
        {"account_id": "strategy", "api_key": "sensitive-marker"},
    ],
)
def test_simulation_label_allowance_does_not_bypass_credential_guard(
    tmp_path: Path, payload: PrimitiveMapping
) -> None:
    with pytest.raises(ManifestError):
        manifest(tmp_path, payload)


def test_accounting_policy_fields_remain_reproducible_metadata(tmp_path: Path) -> None:
    configuration: PrimitiveMapping = {
        "account_initialization": "configured_capital_no_positions",
        "corporate_action_accounting": {"dividends": "pay_date"},
        "dividend_accounting": {"cash_credited": "0"},
    }
    original = manifest(tmp_path, configuration)
    path = write_manifest(original, tmp_path / "manifests", artifact_root=tmp_path)
    assert read_manifest(path).serialize() == original.serialize()


@pytest.mark.parametrize(
    "payload",
    [
        "Bearer sensitive-marker",
        "https://user:password@example.org",
        "https://example.org?api_key=sensitive-marker",
        "-----BEGIN RSA PRIVATE KEY-----",
    ],
)
def test_credential_values_and_authenticated_urls_rejected(
    tmp_path: Path, payload: str
) -> None:
    with pytest.raises(ManifestError):
        manifest(tmp_path, {"provider": payload})


def test_nonfinite_configuration_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="JSON"):
        manifest(tmp_path, {"threshold": cast(Primitive, float("nan"))})
