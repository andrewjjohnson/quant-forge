import ast
import json
import subprocess
import sys
from dataclasses import replace
from decimal import Decimal
from html.parser import HTMLParser
from pathlib import Path
from typing import cast
from urllib.parse import unquote

import pytest

from quantforge.configuration import (
    Primitive,
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
)
from quantforge.experiments import (
    ArtifactFormat,
    ArtifactIndex,
    ArtifactType,
    ManifestError,
    StudyType,
    index_artifact,
    read_manifest,
    write_manifest,
)
from quantforge.reporting import (
    ResearchReportConfig,
    ResearchReportError,
    build_research_report,
    export_research_report,
)
from tests.unit.experiments.test_contracts import write_json
from tests.unit.reporting.research_fixtures import metadata_manifest
from tests.unit.reporting.test_research_reports import codes, values


@pytest.mark.parametrize(
    "kind",
    [
        StudyType.PREDICTION,
        StudyType.FEATURE_DATASET,
        StudyType.PARAMETER_STUDY,
        StudyType.BACKTEST,
        StudyType.OPTIMIZATION,
    ],
)
def test_warnings_are_structured_configured_and_execution_specific(
    tmp_path: Path, kind: StudyType
) -> None:
    path = metadata_manifest(
        tmp_path,
        kind=kind,
        category=ArtifactType.PARAMETER_SUMMARY,
        configuration={"market_data": {"missing_sessions": ["2024-07-03"]}},
        observations={
            "record_counts": {"prediction_count": 3},
            "trial_counts": {"trials": 12},
            "folds": [{"fold_id": "failed-fold", "status": "failed"}],
        },
        payload={
            "summary": {
                "stability": [{"classification": "fragile", "is_isolated_peak": True}]
            }
        },
    )
    report = build_research_report(
        path,
        artifact_root=tmp_path,
        config=ResearchReportConfig(minimum_sample_size=5, high_trial_count=10),
    )
    assert {
        "IN_SAMPLE_ONLY",
        "LOW_SAMPLE_SIZE",
        "HIGH_TRIAL_COUNT",
        "PARAMETER_INSTABILITY",
        "INCOMPLETE_DATA_COVERAGE",
        "FAILED_OR_MISSING_OOS_WINDOWS",
    } <= codes(report)
    assert ("MISSING_TRANSACTION_COST_ASSUMPTIONS" in codes(report)) is (
        kind in {StudyType.BACKTEST, StudyType.OPTIMIZATION}
    )


def test_threshold_boundaries_and_no_prose_heuristics(tmp_path: Path) -> None:
    path = metadata_manifest(
        tmp_path,
        category=ArtifactType.PARAMETER_SUMMARY,
        observations={
            "record_counts": {"prediction_count": 5},
            "trial_counts": {"trials": 10},
        },
        payload={
            "warnings": ["low sample; incomplete data; fragile; high trial count"],
            "stability": {"configuration_change_frequency": "0.5"},
        },
    )
    default = build_research_report(path, artifact_root=tmp_path)
    assert not {
        "LOW_SAMPLE_SIZE",
        "HIGH_TRIAL_COUNT",
        "PARAMETER_INSTABILITY",
        "INCOMPLETE_DATA_COVERAGE",
    } & codes(default)
    equal = build_research_report(
        path,
        artifact_root=tmp_path,
        config=ResearchReportConfig(
            minimum_sample_size=5,
            high_trial_count=10,
            maximum_configuration_change_frequency=Decimal("0.5"),
        ),
    )
    assert "HIGH_TRIAL_COUNT" in codes(equal)
    assert "LOW_SAMPLE_SIZE" not in codes(equal)
    assert "PARAMETER_INSTABILITY" not in codes(equal)
    lower = build_research_report(
        path,
        artifact_root=tmp_path,
        config=ResearchReportConfig(
            maximum_configuration_change_frequency=Decimal("0.49")
        ),
    )
    assert "PARAMETER_INSTABILITY" in codes(lower)
    assert default.report_id != equal.report_id


@pytest.mark.parametrize(
    ("kind", "configuration_key"),
    [
        (StudyType.BACKTEST, "backtest_configuration"),
        (StudyType.OPTIMIZATION, "backtest"),
    ],
)
@pytest.mark.parametrize("cost", [0, 0.0, "0", {"amount": "0"}])
def test_explicit_zero_transaction_costs_are_present(
    tmp_path: Path, kind: StudyType, configuration_key: str, cost: Primitive
) -> None:
    execution: PrimitiveMapping = dict.fromkeys(
        ("commission", "fees", "slippage"), cost
    )
    configuration: PrimitiveMapping = {configuration_key: execution}
    path = metadata_manifest(tmp_path, kind=kind, configuration=configuration)
    report = build_research_report(path, artifact_root=tmp_path)
    assert "MISSING_TRANSACTION_COST_ASSUMPTIONS" not in codes(report)
    assert values(report, "Research configuration and source lineage") == [
        configuration
    ]


@pytest.mark.parametrize(
    ("kind", "configuration_key"),
    [
        (StudyType.BACKTEST, "backtest_configuration"),
        (StudyType.OPTIMIZATION, "backtest"),
    ],
)
@pytest.mark.parametrize("missing_key", ["commission", "fees", "slippage"])
@pytest.mark.parametrize("explicit_null", [False, True])
def test_only_absent_or_null_transaction_costs_are_reported_missing(
    tmp_path: Path,
    kind: StudyType,
    configuration_key: str,
    missing_key: str,
    explicit_null: bool,
) -> None:
    execution: PrimitiveMapping = dict.fromkeys(("commission", "fees", "slippage"), 0)
    if explicit_null:
        execution[missing_key] = None
    else:
        del execution[missing_key]
    path = metadata_manifest(
        tmp_path, kind=kind, configuration={configuration_key: execution}
    )
    report = build_research_report(path, artifact_root=tmp_path)
    warnings = [
        warning
        for warning in report.warnings
        if warning.code == "MISSING_TRANSACTION_COST_ASSUMPTIONS"
    ]
    assert len(warnings) == 1
    assert warnings[0].message == f"Unavailable execution assumptions: {missing_key}."


def test_recorded_minimum_and_incomplete_oos_use_source_fields(tmp_path: Path) -> None:
    path = metadata_manifest(
        tmp_path,
        kind=StudyType.PARAMETER_STUDY,
        configuration={"ranking": {"minimum_prediction_count": 12}},
        payload={
            "analysis": {"prediction_count": 3},
            "summary": {
                "completeness": {
                    "expected_windows": 3,
                    "completed_windows": 1,
                    "complete": False,
                    "failed_windows": ["a"],
                    "missing_windows": ["b"],
                }
            },
        },
    )
    report = build_research_report(path, artifact_root=tmp_path)
    assert "LOW_SAMPLE_SIZE" in codes(report)
    assert "FAILED_OR_MISSING_OOS_WINDOWS" in codes(report)


@pytest.mark.parametrize(
    ("damage", "expected"),
    [
        ("missing", "missing_artifact"),
        ("tampered", "content_hash_mismatch"),
        ("optional", "unavailable_optional"),
    ],
)
def test_artifact_issues_are_visible_no_values_are_trusted_and_no_records_repaired(
    tmp_path: Path,
    damage: str,
    expected: str,
) -> None:
    path = metadata_manifest(tmp_path, payload={"summary": {"accuracy": "0.987"}})
    if damage == "optional":
        manifest = read_manifest(path)
        optional = index_artifact(
            tmp_path,
            path="optional.json",
            artifact_type=ArtifactType.CHART,
            schema_version="1",
            producer_study_id="fixture-study",
            producer_artifact_id="optional",
            required=False,
        )
        path = write_manifest(
            replace(
                manifest,
                artifacts=ArtifactIndex((*manifest.artifacts.entries, optional)),
            ),
            tmp_path / "experiments",
            artifact_root=tmp_path,
        )
    original = path.read_bytes()
    result = tmp_path / "result.json"
    if damage == "missing":
        result.unlink()
    elif damage == "tampered":
        result.write_text('{"summary":{"accuracy":"0.123"}}')
    before = result.read_bytes() if result.exists() else None
    report = build_research_report(path, artifact_root=tmp_path)
    html = export_research_report(report, tmp_path / "reports").read_text()
    assert expected in html
    assert "ARTIFACT_INTEGRITY" in codes(report)
    if damage != "optional":
        assert "0.987" not in html
        assert "0.123" not in html
        assert values(report, "Prediction metrics and comparisons") == [None]
    assert path.read_bytes() == original
    assert (result.read_bytes() if result.exists() else None) == before


def test_json_binding_failure_and_symlink_escape_are_untrusted(tmp_path: Path) -> None:
    path = metadata_manifest(tmp_path, payload={"summary": {"accuracy": "0.75"}})
    manifest = read_manifest(path)
    entry = replace(
        manifest.artifacts.entries[0],
        bindings=PrimitiveMappingSnapshot.capture({"/summary/accuracy": "wrong"}),
    )
    manifest = replace(manifest, artifacts=ArtifactIndex((entry,)))
    # A historical incompatible binding fixture cannot be published by QF-9.
    path = path.parent / f"{manifest.manifest_id}.json"
    path.write_bytes(manifest.serialize())
    report = build_research_report(path, artifact_root=tmp_path)
    assert report.artifacts[0].status == "incompatible_metadata"
    (tmp_path / "result.json").unlink()
    (tmp_path / "result.json").symlink_to(Path(__file__))
    assert (
        build_research_report(path, artifact_root=tmp_path).artifacts[0].status
        == "invalid_artifact"
    )


def test_replacement_between_verification_and_read_cannot_supply_trusted_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = metadata_manifest(tmp_path, payload={"summary": {"accuracy": "0.987"}})
    original = Path.read_bytes
    reads = 0

    def replaced(file: Path) -> bytes:
        nonlocal reads
        if file.name == "result.json":
            reads += 1
            if reads == 2:
                return b'{"summary":{"accuracy":"0.123"}}'
        return original(file)

    monkeypatch.setattr(Path, "read_bytes", replaced)
    report = build_research_report(path, artifact_root=tmp_path)
    assert report.artifacts[0].status == "invalid_or_changed_artifact"
    assert values(report, "Prediction metrics and comparisons") == [None]


class Elements(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[str] = []
        self.links: list[str] = []
        self.attributes: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append(tag)
        self.attributes.extend(key for key, _ in attrs)
        self.links.extend(
            value for key, value in attrs if key == "href" and value is not None
        )


def test_html_injection_is_escaped_and_all_links_are_local_exact_references(
    tmp_path: Path,
) -> None:
    malicious = '<script>alert("x")</script><img src=x onerror=alert(1)>'
    path = metadata_manifest(
        tmp_path,
        configuration={"symbol": malicious},
        payload={"summary": {"label": malicious, malicious: "label"}},
    )
    report = build_research_report(path, artifact_root=tmp_path)
    output = export_research_report(report, tmp_path / "reports with spaces")
    html = output.read_text()
    parsed = Elements()
    parsed.feed(html)
    assert not {"script", "img", "iframe", "object"} & set(parsed.tags)
    assert not any(key.startswith("on") for key in parsed.attributes)
    assert "&lt;script&gt;" in html
    assert "default-src" in html
    targets = [
        (output.parent / unquote(link)).resolve()
        for link in parsed.links
        if not link.startswith("#")
    ]
    assert path.resolve() in targets
    assert (tmp_path / "result.json").resolve() in targets
    assert all(target.is_file() for target in targets)
    assert all(
        not link.startswith(("http", "javascript:", "data:")) for link in parsed.links
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"api_key": "NEVER-RENDER-ME"},
        {"nested": {"access_token": "NEVER-RENDER-ME"}},
        {"account_number": "NEVER-RENDER-ME"},
        {"label": "Bearer NEVER-RENDER-ME"},
    ],
)
def test_secrets_are_rejected_without_echoing_values(
    tmp_path: Path, payload: PrimitiveMapping
) -> None:
    with pytest.raises(ManifestError) as error:
        metadata_manifest(tmp_path, configuration=payload)
    assert "NEVER-RENDER-ME" not in str(error.value)


def test_unsafe_csv_cells_never_enter_html(tmp_path: Path) -> None:
    path = metadata_manifest(tmp_path, kind=StudyType.BACKTEST)
    manifest = read_manifest(path)
    (tmp_path / "equity.csv").write_text(
        "session,api_key\n2024-01-01,NEVER-RENDER-ME\n"
    )
    entry = index_artifact(
        tmp_path,
        path="equity.csv",
        file_format=ArtifactFormat.CSV,
        artifact_type=ArtifactType.BACKTEST_RESULT,
        schema_version="1",
        producer_study_id="fixture-study",
        producer_artifact_id="equity",
    )
    path = write_manifest(
        replace(
            manifest, artifacts=ArtifactIndex((*manifest.artifacts.entries, entry))
        ),
        tmp_path / "experiments",
        artifact_root=tmp_path,
    )
    report = build_research_report(path, artifact_root=tmp_path)
    html = export_research_report(report, tmp_path / "reports").read_text()
    assert "NEVER-RENDER-ME" not in html
    assert "invalid_or_changed_artifact" in html


@pytest.mark.parametrize(
    "csv_text",
    [
        "price,price\nDISCARDED-EVIDENCE,RETAINED-EVIDENCE\n",
        "price,price\n",
        "price,\nDISCARDED-EVIDENCE,RETAINED-EVIDENCE\n",
        '""\nRETAINED-EVIDENCE\n',
        "",
        "\n",
    ],
)
def test_ambiguous_or_missing_csv_headers_cannot_supply_evidence(
    tmp_path: Path, csv_text: str
) -> None:
    path = metadata_manifest(tmp_path, kind=StudyType.BACKTEST)
    manifest = read_manifest(path)
    csv_path = tmp_path / "equity.csv"
    csv_path.write_text(csv_text)
    entry = index_artifact(
        tmp_path,
        path="equity.csv",
        file_format=ArtifactFormat.CSV,
        artifact_type=ArtifactType.BACKTEST_RESULT,
        schema_version="1",
        producer_study_id="fixture-study",
        producer_artifact_id="equity",
    )
    path = write_manifest(
        replace(
            manifest, artifacts=ArtifactIndex((*manifest.artifacts.entries, entry))
        ),
        tmp_path / "experiments",
        artifact_root=tmp_path,
    )
    report = build_research_report(path, artifact_root=tmp_path)
    artifact = next(item for item in report.artifacts if item.entry == entry)
    assert artifact.status == "invalid_or_changed_artifact"
    assert artifact.content is None
    assert "ARTIFACT_INTEGRITY" in codes(report)
    assert values(report, "Equity and returns") == [None]
    output = export_research_report(report, tmp_path / "reports")
    html = output.read_text()
    assert "DISCARDED-EVIDENCE" not in html
    assert "RETAINED-EVIDENCE" not in html
    parsed = Elements()
    parsed.feed(html)
    assert not any(
        (output.parent / unquote(link)).resolve() == csv_path
        for link in parsed.links
        if not link.startswith("#")
    )
    assert csv_path.read_text() == csv_text


def test_bad_manifest_is_rejected_instead_of_rendered(tmp_path: Path) -> None:
    path = metadata_manifest(tmp_path)
    document = cast(PrimitiveMapping, json.loads(path.read_bytes()))
    document["study_id"] = "tampered"
    write_json(path, document)
    with pytest.raises(ManifestError):
        build_research_report(path, artifact_root=tmp_path)


def test_verified_artifact_changed_before_export_requires_rebuild(
    tmp_path: Path,
) -> None:
    path = metadata_manifest(tmp_path, payload={"summary": {"accuracy": "0.57"}})
    report = build_research_report(path, artifact_root=tmp_path)
    (tmp_path / "result.json").write_text('{"summary":{"accuracy":"0.99"}}')
    with pytest.raises(ManifestError, match="integrity verification failed"):
        export_research_report(report, tmp_path / "reports")
    assert not (tmp_path / "reports").exists()


def test_missing_completed_oos_file_is_not_hidden_by_historical_completed_status(
    tmp_path: Path,
) -> None:
    path = metadata_manifest(
        tmp_path,
        kind=StudyType.WALK_FORWARD,
        category=ArtifactType.WALK_FORWARD_WINDOW,
        observations={"folds": [{"fold_id": "one", "status": "completed"}]},
    )
    (tmp_path / "result.json").unlink()
    report = build_research_report(path, artifact_root=tmp_path)
    assert "FAILED_OR_MISSING_OOS_WINDOWS" in codes(report)


@pytest.mark.parametrize("bad", [0, -1, True])
def test_config_rejects_invalid_thresholds(bad: int) -> None:
    with pytest.raises(ResearchReportError):
        ResearchReportConfig(minimum_sample_size=bad)
    with pytest.raises(ResearchReportError):
        ResearchReportConfig(maximum_preview_rows=bad)


@pytest.mark.parametrize(
    "bad", [Decimal("NaN"), Decimal("Infinity"), Decimal("-0.1"), Decimal("1.1")]
)
def test_config_rejects_invalid_turnover(bad: Decimal) -> None:
    with pytest.raises(ResearchReportError):
        ResearchReportConfig(maximum_configuration_change_frequency=bad)


def test_rendering_import_and_runtime_do_not_load_any_research_engine(
    tmp_path: Path,
) -> None:
    path = metadata_manifest(tmp_path, payload={"summary": {"accuracy": "0.57"}})
    script = """
import sys
from pathlib import Path
blocked = ('numpy', 'talib', 'pyarrow', 'quantforge.prediction',
           'quantforge.indicators', 'quantforge.backtesting',
           'quantforge.optimization', 'quantforge.walk_forward', 'quantforge.oos')
class RejectResearch:
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.') for name in blocked):
            raise AssertionError('report imported research engine: ' + fullname)
sys.meta_path.insert(0, RejectResearch())
from quantforge.reporting import build_research_report, export_research_report
root = Path(sys.argv[2])
report = build_research_report(Path(sys.argv[1]), artifact_root=root)
export_research_report(report, root / 'reports')
assert not any(name in sys.modules for name in blocked)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(path), str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    # Inspect dependencies even when the test process already imported producers.
    folder = Path(__file__).parents[3] / "src/quantforge/reporting"
    for source in [folder / "research.py", *folder.glob("_research*.py")]:
        tree = ast.parse(source.read_text())
        calls = [node.func for node in ast.walk(tree) if isinstance(node, ast.Call)]
        names = {node.attr for node in calls if isinstance(node, ast.Attribute)}
        assert not names & {
            "compute",
            "consume",
            "reserve",
            "run",
            "evaluate",
            "aggregate",
            "resume",
        }
