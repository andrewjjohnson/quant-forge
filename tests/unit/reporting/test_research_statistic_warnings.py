from pathlib import Path

import pytest

from quantforge.configuration import PrimitiveMapping
from quantforge.experiments import ArtifactType, StudyType
from quantforge.reporting import ResearchReportConfig, build_research_report
from tests.unit.reporting.research_fixtures import metadata_manifest
from tests.unit.reporting.test_research_reports import codes


@pytest.mark.parametrize(
    "kind", [StudyType.BACKTEST, StudyType.OPTIMIZATION, StudyType.PARAMETER_STUDY]
)
@pytest.mark.parametrize(
    "category",
    [
        ArtifactType.CONFIGURATION,
        ArtifactType.TRIAL_RESULT,
        ArtifactType.PARAMETER_SUMMARY,
    ],
)
def test_parameter_names_cannot_trigger_statistic_warnings(
    tmp_path: Path, kind: StudyType, category: ArtifactType
) -> None:
    parameters: PrimitiveMapping = {
        "prediction_count": 1,
        "trade_count": 1,
        "trial_count": 100,
        "coverage_complete": False,
        "missing_sessions": ["custom-parameter"],
        "expected_windows": 1,
        "complete": False,
        "summary": {"prediction_count": 1, "coverage_complete": False},
    }
    payload: PrimitiveMapping = {
        "configuration": parameters,
        "parameters": parameters,
        "strategy_parameters": parameters,
        "analysis": {"prediction_count": 20},
        "rankings": [{"parameters": parameters, "analysis": {"prediction_count": 20}}],
        "eligible_rankings": [
            {"parameters": parameters, "metrics": {"trade_count": 20}}
        ],
    }
    path = metadata_manifest(
        tmp_path,
        kind=kind,
        category=category,
        payload=payload,
        configuration={"strategy": {"parameters": parameters}, "factory": parameters},
        observations={"record_counts": {"prediction_count": 20}},
    )
    report = build_research_report(
        path,
        artifact_root=tmp_path,
        config=ResearchReportConfig(minimum_sample_size=5, high_trial_count=10),
    )
    assert not {
        "LOW_SAMPLE_SIZE",
        "HIGH_TRIAL_COUNT",
        "INCOMPLETE_DATA_COVERAGE",
        "FAILED_OR_MISSING_OOS_WINDOWS",
    } & codes(report)


@pytest.mark.parametrize(
    ("category", "payload", "expected"),
    [
        (
            ArtifactType.PREDICTION_RESULT,
            {"analysis": {"prediction_count": 1}},
            "LOW_SAMPLE_SIZE",
        ),
        (ArtifactType.TRIAL_RESULT, {"metrics": {"trade_count": 1}}, "LOW_SAMPLE_SIZE"),
        (
            ArtifactType.PARAMETER_SUMMARY,
            {"counts": {"trials": 100}},
            "HIGH_TRIAL_COUNT",
        ),
        (
            ArtifactType.PARAMETER_SUMMARY,
            {"rankings": [{"analysis": {"prediction_count": 1}}]},
            "LOW_SAMPLE_SIZE",
        ),
        (
            ArtifactType.BACKTEST_RESULT,
            {"manifest": {"performance": {"trade_count": 1}}},
            "LOW_SAMPLE_SIZE",
        ),
        (
            ArtifactType.CONFIGURATION,
            {"record_counts": {"trade_count": 1}},
            "LOW_SAMPLE_SIZE",
        ),
        (
            ArtifactType.SOURCE_DATASET,
            {"missing_sessions": ["2024-07-03"]},
            "INCOMPLETE_DATA_COVERAGE",
        ),
        (
            ArtifactType.DATA_QUALITY,
            {"coverage": {"coverage_complete": False}},
            "INCOMPLETE_DATA_COVERAGE",
        ),
        (
            ArtifactType.OOS_AGGREGATE,
            {"summary": {"completeness": {"expected_windows": 2, "complete": False}}},
            "FAILED_OR_MISSING_OOS_WINDOWS",
        ),
    ],
)
def test_published_statistics_still_supply_warnings(
    tmp_path: Path, category: ArtifactType, payload: PrimitiveMapping, expected: str
) -> None:
    path = metadata_manifest(tmp_path, category=category, payload=payload)
    report = build_research_report(
        path,
        artifact_root=tmp_path,
        config=ResearchReportConfig(minimum_sample_size=5, high_trial_count=10),
    )
    assert expected in codes(report)
