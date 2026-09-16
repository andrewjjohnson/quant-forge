"""Completed QF-32 summaries retain their complete disclosure schema."""

from pathlib import Path

import pytest

from quantforge.configuration import Primitive, PrimitiveMapping
from quantforge.experiments import (
    ManifestError,
    StudyType,
    inspect_study,
    verify_artifacts,
)
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_contracts import write_json
from tests.unit.experiments.test_grid_integrity import build_grid_export, read_record
from tests.unit.prediction.test_prediction_window import WindowProvider, grid


@pytest.fixture(params=[False, True], ids=["prediction", "window"])
def export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> Path:
    if request.param:
        study = grid(tmp_path / "prediction", WindowProvider())
        result = study.run()
        root = tmp_path / "prediction" / result.study_id
        block_research(monkeypatch)
        return root
    return build_grid_export(tmp_path, StudyType.PARAMETER_STUDY, monkeypatch)


def inspect_summary(export: Path, summary: PrimitiveMapping, *, reject: bool) -> None:
    path = export / "summary.json"
    write_json(path, summary)
    before = {item: item.read_bytes() for item in export.rglob("*") if item.is_file()}
    if reject:
        with pytest.raises(ManifestError):
            inspect_study(
                StudyType.PARAMETER_STUDY, export, artifact_root=export.parent
            )
    else:
        inspected = inspect_study(
            StudyType.PARAMETER_STUDY, export, artifact_root=export.parent
        )
        assert verify_artifacts(inspected.index, export.parent).valid
        assert any(
            entry.path == path.relative_to(export.parent).as_posix()
            for entry in inspected.index.entries
        )
    assert {item: item.read_bytes() for item in before} == before


@pytest.mark.parametrize("field", ["warnings", "limitations"])
@pytest.mark.parametrize(
    "invalid",
    [None, "text", {}, False, 1, [None], [False], [1], [{}], [[]], ["valid", False]],
)
def test_disclosures_require_arrays_containing_only_strings(
    export: Path, field: str, invalid: Primitive
) -> None:
    summary = read_record(export / "summary.json")
    summary[field] = invalid
    inspect_summary(export, summary, reject=True)


@pytest.mark.parametrize(
    "field",
    [
        "study_id",
        "schema_version",
        "counts",
        "rankings",
        "ineligible_trials",
        "stability",
        "cache_statistics",
        "warnings",
        "limitations",
        "undeclared",
    ],
)
def test_completed_summary_requires_the_exact_producer_fields(
    export: Path, field: str
) -> None:
    summary = read_record(export / "summary.json")
    if field == "undeclared":
        summary[field] = None
    else:
        del summary[field]
    inspect_summary(export, summary, reject=True)


@pytest.mark.parametrize(
    "disclosures",
    [
        None,
        [],
        [" original disclosure\n", "", "unicode: \u03b1", "original", "original"],
    ],
)
def test_native_empty_and_custom_disclosures_are_preserved(
    export: Path, disclosures: list[str] | None
) -> None:
    summary = read_record(export / "summary.json")
    if disclosures is not None:
        summary["warnings"] = list(disclosures)
        summary["limitations"] = list(disclosures)
    inspect_summary(export, summary, reject=False)
