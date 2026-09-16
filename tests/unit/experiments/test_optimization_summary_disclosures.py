"""QF-6 completed exports require their original disclosure envelope."""

from pathlib import Path

import pytest

from quantforge.configuration import Primitive
from quantforge.experiments import (
    ManifestError,
    StudyType,
    inspect_study,
    verify_artifacts,
)
from tests.unit.experiments.test_contracts import write_json
from tests.unit.experiments.test_grid_integrity import build_grid_export, read_record


@pytest.mark.parametrize("field", ["warnings", "limitations"])
@pytest.mark.parametrize(
    "invalid",
    [None, "text", {}, False, 1, [None], [False], [1], [{}], [[]], ["valid", False]],
)
def test_optimization_disclosures_require_string_arrays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, invalid: Primitive
) -> None:
    root = build_grid_export(tmp_path, StudyType.OPTIMIZATION, monkeypatch)
    path = root / "summary.json"
    summary = read_record(path)
    summary[field] = invalid
    write_json(path, summary)
    before = path.read_bytes()
    with pytest.raises(ManifestError):
        inspect_study(StudyType.OPTIMIZATION, root, artifact_root=tmp_path)
    assert path.read_bytes() == before


def test_optimization_summary_requires_every_exported_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = build_grid_export(tmp_path, StudyType.OPTIMIZATION, monkeypatch)
    path = root / "summary.json"
    original = read_record(path)
    for field in (*original, "undeclared"):
        summary = original.copy()
        if field == "undeclared":
            summary[field] = None
        else:
            del summary[field]
        write_json(path, summary)
        before = path.read_bytes()
        with pytest.raises(ManifestError):
            inspect_study(StudyType.OPTIMIZATION, root, artifact_root=tmp_path)
        assert path.read_bytes() == before


@pytest.mark.parametrize(
    "disclosures", [None, [], [" original\n", "", "duplicate", "duplicate"]]
)
def test_optimization_disclosures_remain_observational(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, disclosures: list[str] | None
) -> None:
    root = build_grid_export(tmp_path, StudyType.OPTIMIZATION, monkeypatch)
    path = root / "summary.json"
    summary = read_record(path)
    if disclosures is not None:
        summary["warnings"] = list(disclosures)
        summary["limitations"] = list(disclosures)
    write_json(path, summary)
    before = {item: item.read_bytes() for item in root.rglob("*") if item.is_file()}
    inspected = inspect_study(StudyType.OPTIMIZATION, root, artifact_root=tmp_path)
    assert verify_artifacts(inspected.index, tmp_path).valid
    assert {item: item.read_bytes() for item in before} == before
