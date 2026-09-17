from decimal import Decimal
from pathlib import Path

import pytest

from quantforge.configuration import PrimitiveMapping
from quantforge.experiments import ArtifactType, StudyType
from quantforge.reporting import ResearchReportConfig, build_research_report
from tests.unit.reporting.research_fixtures import metadata_manifest
from tests.unit.reporting.test_research_reports import codes


@pytest.mark.parametrize("kind", [StudyType.BACKTEST, StudyType.OPTIMIZATION])
@pytest.mark.parametrize(
    "assessment",
    [
        {"classification": "fragile"},
        {"is_isolated_peak": True},
        {"configuration_change_frequency": "0.9"},
    ],
)
def test_strategy_configuration_fields_are_not_stability_assessments(
    tmp_path: Path, kind: StudyType, assessment: PrimitiveMapping
) -> None:
    configuration: PrimitiveMapping = {"strategy": {"parameters": assessment}}
    path = metadata_manifest(
        tmp_path,
        kind=kind,
        configuration=configuration,
        category=ArtifactType.CONFIGURATION,
        payload=configuration,
    )
    report = build_research_report(
        path,
        artifact_root=tmp_path,
        config=ResearchReportConfig(
            maximum_configuration_change_frequency=Decimal("0.5")
        ),
    )
    assert "PARAMETER_INSTABILITY" not in codes(report)


@pytest.mark.parametrize(
    "category", [ArtifactType.PARAMETER_SUMMARY, ArtifactType.CONFIGURATION_STABILITY]
)
def test_stability_artifact_does_not_turn_nested_parameters_into_assessments(
    tmp_path: Path, category: ArtifactType
) -> None:
    parameters: PrimitiveMapping = {
        "classification": "fragile",
        "is_isolated_peak": True,
        "configuration_change_frequency": "0.9",
    }
    path = metadata_manifest(
        tmp_path,
        kind=StudyType.OPTIMIZATION,
        category=category,
        payload={
            "configuration": parameters,
            "rankings": [{"parameters": parameters}],
            "windows": [{"configuration": parameters}],
            "stability": [{"classification": "stable", "parameters": parameters}],
        },
    )
    report = build_research_report(
        path,
        artifact_root=tmp_path,
        config=ResearchReportConfig(
            maximum_configuration_change_frequency=Decimal("0.5")
        ),
    )
    assert "PARAMETER_INSTABILITY" not in codes(report)


@pytest.mark.parametrize(
    ("category", "payload"),
    [
        (ArtifactType.PARAMETER_SUMMARY, {"classification": "fragile"}),
        (
            ArtifactType.PARAMETER_SUMMARY,
            {"stability": [{"classification": "fragile"}]},
        ),
        (ArtifactType.PARAMETER_SUMMARY, {"summaries": [{"is_isolated_peak": True}]}),
        (
            ArtifactType.PARAMETER_SUMMARY,
            {"top_stability_trials": [{"is_isolated_peak": True}]},
        ),
        (
            ArtifactType.CONFIGURATION_STABILITY,
            {"configuration_change_frequency": "0.9"},
        ),
        (
            ArtifactType.OOS_AGGREGATE,
            {"stability": {"configuration_change_frequency": "0.9"}},
        ),
    ],
)
def test_published_stability_records_still_supply_warnings(
    tmp_path: Path, category: ArtifactType, payload: PrimitiveMapping
) -> None:
    path = metadata_manifest(tmp_path, category=category, payload=payload)
    report = build_research_report(
        path,
        artifact_root=tmp_path,
        config=ResearchReportConfig(
            maximum_configuration_change_frequency=Decimal("0.5")
        ),
    )
    assert "PARAMETER_INSTABILITY" in codes(report)
