"""QF-6 manifests retain complete operational provenance outside study identity."""

from copy import deepcopy
from pathlib import Path

import pytest

from quantforge.configuration import Primitive, PrimitiveMapping
from quantforge.experiments import (
    ManifestError,
    StudyType,
    inspect_study,
    verify_artifacts,
)
from quantforge.experiments._json import mapping
from tests.unit.experiments.test_contracts import write_json
from tests.unit.experiments.test_grid_integrity import build_grid_export, read_record


@pytest.fixture(params=[False, True], ids=["resumable", "complete"])
def export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> Path:
    root = build_grid_export(tmp_path, StudyType.OPTIMIZATION, monkeypatch)
    if not request.param:
        (root / "summary.json").unlink()
    return root


def inspect_manifest(
    export: Path, manifest: PrimitiveMapping, *, reject: bool = True
) -> None:
    path = export / "manifest.json"
    write_json(path, manifest)
    before = {item: item.read_bytes() for item in export.rglob("*") if item.is_file()}
    if reject:
        with pytest.raises(ManifestError):
            inspect_study(StudyType.OPTIMIZATION, export, artifact_root=export.parent)
    else:
        bundle = inspect_study(
            StudyType.OPTIMIZATION, export, artifact_root=export.parent
        )
        assert verify_artifacts(bundle.index, export.parent).valid
    assert {item: item.read_bytes() for item in before} == before


@pytest.mark.parametrize(
    "section",
    [
        None,
        "execution_configuration",
        "persistence_configuration",
        "scale_safeguard",
        "combination_counts",
    ],
)
def test_operational_manifest_requires_every_field(
    export: Path, section: str | None
) -> None:
    original = read_record(export / "manifest.json")
    fields = original if section is None else mapping(original[section])
    for field in (*fields, "undeclared"):
        manifest = deepcopy(original)
        target = manifest if section is None else mapping(manifest[section])
        if field == "undeclared":
            target[field] = None
        else:
            del target[field]
        inspect_manifest(export, manifest)


@pytest.mark.parametrize(
    ("section", "field", "invalid"),
    [
        (None, "execution_configuration", {}),
        (None, "persistence_configuration", {}),
        ("execution_configuration", "mode", "threads"),
        ("execution_configuration", "maximum_workers", 0),
        ("execution_configuration", "maximum_workers", True),
        ("execution_configuration", "maximum_workers", 1.0),
        ("execution_configuration", "retry_failed", "true"),
        ("execution_configuration", "fail_fast", 1),
        (
            "execution_configuration",
            "parallelism",
            "bounded_standard_library_process_pool",
        ),
        ("execution_configuration", "stale_running_policy", "skip"),
        ("persistence_configuration", "store", "remote"),
        ("persistence_configuration", "output_root", None),
        ("persistence_configuration", "successful_trial_overwrite", "allowed"),
        ("scale_safeguard", "maximum_combinations", 0),
        ("scale_safeguard", "allow_large_grid", 1),
        ("scale_safeguard", "count_expression", "foreign"),
        ("scale_safeguard", "maximum_combinations", 1),
        (None, "operational_timestamp_policy", "material"),
        (None, "warnings", None),
        (None, "limitations", [False]),
    ],
)
def test_operational_fields_enforce_producer_domains(
    export: Path, section: str | None, field: str, invalid: Primitive
) -> None:
    manifest = read_record(export / "manifest.json")
    target = manifest if section is None else mapping(manifest[section])
    target[field] = invalid
    inspect_manifest(export, manifest)


@pytest.mark.parametrize("process", [False, True])
def test_valid_operational_variants_remain_observational(
    export: Path, process: bool
) -> None:
    manifest = read_record(export / "manifest.json")
    execution = mapping(manifest["execution_configuration"])
    execution.update(
        mode="process" if process else "sequential",
        parallelism="bounded_standard_library_process_pool" if process else "none",
        maximum_workers=3,
        retry_failed=True,
        fail_fast=True,
    )
    mapping(manifest["persistence_configuration"])["output_root"] = (
        "../historical location"
    )
    mapping(manifest["scale_safeguard"]).update(
        maximum_combinations=1, allow_large_grid=True
    )
    manifest["warnings"] = []
    manifest["limitations"] = ["", "original", "original"]
    inspect_manifest(export, manifest, reject=False)
