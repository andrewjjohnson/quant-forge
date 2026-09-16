from pathlib import Path
from typing import cast

import pytest

from quantforge.configuration import Primitive, PrimitiveMapping
from quantforge.experiments import (
    ManifestError,
    StudyType,
    inspect_study,
    verify_artifacts,
)
from quantforge.prediction import SignalDisposition, SignalFeatureDatasetResult
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_contracts import write_json
from tests.unit.experiments.test_grid_integrity import read_record
from tests.unit.helpers import make_dataset
from tests.unit.prediction.test_feature_dataset import (
    FixtureCandidateRule,
    _build_fixture,  # pyright: ignore[reportPrivateUsage]
)
from tests.unit.prediction.test_multi_timeframe_feature_dataset import (
    _build,  # pyright: ignore[reportPrivateUsage]
)


@pytest.fixture(params=["qf7", "qf29"])
def feature_result(
    tmp_path: Path, request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> SignalFeatureDatasetResult:
    if request.param == "qf29":
        result, _, _ = _build(tmp_path / "features")
    else:
        result = _build_fixture(
            make_dataset(("100", "102", "101", "104")),
            FixtureCandidateRule(tuple(SignalDisposition)),
            tmp_path / "features",
        )
    block_research(monkeypatch)
    return result


@pytest.mark.parametrize("directory", [False, True])
def test_feature_configuration_cannot_retain_stale_dataset_identity(
    tmp_path: Path, feature_result: SignalFeatureDatasetResult, directory: bool
) -> None:
    document = feature_result.to_primitive()
    manifest = cast(PrimitiveMapping, document["manifest"])
    cast(PrimitiveMapping, manifest["configuration"])["feature_schema_version"] = (
        "changed"
    )
    path = (
        tmp_path / "features" / feature_result.dataset_id
        if directory
        else tmp_path / "feature.json"
    )
    write_json(
        path / "manifest.json" if directory else path,
        manifest if directory else document,
    )
    with pytest.raises(ManifestError, match="feature dataset identity"):
        inspect_study(StudyType.FEATURE_DATASET, path, artifact_root=tmp_path)


@pytest.mark.parametrize(
    "change",
    [
        "truncate",
        "disposition",
        "candidate_count",
        "accepted_count",
        "rejected_count",
        "blocked_count",
        "overlapping_count",
        "summary",
        "invalid_disposition",
    ],
)
def test_feature_rows_must_match_declared_candidate_and_disposition_counts(
    tmp_path: Path, feature_result: SignalFeatureDatasetResult, change: str
) -> None:
    document = feature_result.to_primitive()
    rows = cast(list[PrimitiveMapping], document["rows"])
    manifest = cast(PrimitiveMapping, document["manifest"])
    if change == "truncate":
        rows.pop()
    elif change in {"disposition", "invalid_disposition"}:
        rows[0]["signal_disposition"] = (
            "unknown" if change == "invalid_disposition" else "overlapping"
        )
    elif change == "summary":
        cast(PrimitiveMapping, document["summary"])["candidate_count"] = 999
    else:
        cast(PrimitiveMapping, manifest["record_counts"])[change] = 999
    path = tmp_path / "feature.json"
    write_json(path, document)
    with pytest.raises(ManifestError, match=r"feature.*(counts|disposition)"):
        inspect_study(StudyType.FEATURE_DATASET, path, artifact_root=tmp_path)


@pytest.mark.parametrize("invalid", [True, -1, "1", None])
def test_feature_counts_require_nonnegative_integers(
    tmp_path: Path, feature_result: SignalFeatureDatasetResult, invalid: Primitive
) -> None:
    document = feature_result.to_primitive()
    manifest = cast(PrimitiveMapping, document["manifest"])
    cast(PrimitiveMapping, manifest["record_counts"])["accepted_count"] = invalid
    path = tmp_path / "feature.json"
    write_json(path, document)
    with pytest.raises(ManifestError, match="feature record counts"):
        inspect_study(StudyType.FEATURE_DATASET, path, artifact_root=tmp_path)


def test_intact_feature_snapshot_preserves_counts_without_research(
    tmp_path: Path, feature_result: SignalFeatureDatasetResult
) -> None:
    path = tmp_path / "feature.json"
    write_json(path, feature_result.to_primitive())
    bundle = inspect_study(StudyType.FEATURE_DATASET, path, artifact_root=tmp_path)
    assert bundle.provenance.producer_study_id == feature_result.dataset_id
    assert (
        bundle.provenance.observations.to_primitive()["record_counts"]
        == feature_result.summary.to_primitive()
    )


def test_empty_feature_snapshot_is_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = _build_fixture(
        make_dataset(("100", "102")), FixtureCandidateRule(()), tmp_path / "features"
    )
    block_research(monkeypatch)
    test_intact_feature_snapshot_preserves_counts_without_research(tmp_path, result)


@pytest.mark.parametrize(
    "change",
    [
        "candidate_count",
        "accepted_count",
        "rejected_count",
        "blocked_count",
        "overlapping_count",
        "missing",
        "extra",
        "redistributed",
        "float",
        "bool",
    ],
)
def test_directory_feature_summary_must_match_manifest_counts(
    tmp_path: Path, feature_result: SignalFeatureDatasetResult, change: str
) -> None:
    root = tmp_path / "features" / feature_result.dataset_id
    summary = read_record(root / "summary.json")
    if change == "missing":
        summary.pop("accepted_count")
    elif change == "extra":
        summary["unexpected_count"] = 0
    elif change == "redistributed":
        nonempty = next(
            key
            for key in summary
            if key != "candidate_count" and cast(int, summary[key]) > 0
        )
        target = "blocked_count" if nonempty != "blocked_count" else "accepted_count"
        summary[nonempty] = cast(int, summary[nonempty]) - 1
        summary[target] = cast(int, summary[target]) + 1
    elif change == "float":
        summary["candidate_count"] = float(cast(int, summary["candidate_count"]))
    elif change == "bool":
        summary["candidate_count"] = True
    else:
        summary[change] = cast(int, summary[change]) + 1
    write_json(root / "summary.json", summary)
    with pytest.raises(ManifestError, match="feature record counts"):
        inspect_study(StudyType.FEATURE_DATASET, root, artifact_root=tmp_path)


def test_intact_directory_summary_retains_count_bindings(
    tmp_path: Path, feature_result: SignalFeatureDatasetResult
) -> None:
    root = tmp_path / "features" / feature_result.dataset_id
    bundle = inspect_study(StudyType.FEATURE_DATASET, root, artifact_root=tmp_path)
    summary = next(
        entry for entry in bundle.index.entries if entry.path.endswith("/summary.json")
    )
    assert summary.bindings.to_primitive() == {
        "/" + key: count for key, count in feature_result.summary.to_primitive().items()
    }
    assert verify_artifacts(bundle.index, tmp_path).valid


@pytest.mark.parametrize(
    "references", [None, "not-an-array", [], ["bad"], "extra", [True]]
)
def test_directory_feature_prediction_references_match_configured_outcomes(
    tmp_path: Path, feature_result: SignalFeatureDatasetResult, references: Primitive
) -> None:
    root = tmp_path / "features" / feature_result.dataset_id
    manifest = read_record(root / "manifest.json")
    manifest["prediction_study_ids"] = (
        [*cast(list[Primitive], manifest["prediction_study_ids"]), "a" * 64]
        if references == "extra"
        else references
    )
    write_json(root / "manifest.json", manifest)
    with pytest.raises(ManifestError):
        inspect_study(StudyType.FEATURE_DATASET, root, artifact_root=tmp_path)
