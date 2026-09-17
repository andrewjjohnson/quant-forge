"""Immutable reporting artifacts over already-computed research results."""

from typing import TYPE_CHECKING

from quantforge.reporting.research import build_research_report, export_research_report
from quantforge.reporting.research_models import (
    ReportArtifact,
    ReportPhase,
    ReportSection,
    ResearchReport,
    ResearchReportConfig,
    ResearchReportError,
    ResearchWarning,
)

# Preserve the QF-34 public API without importing its research-context types when
# the lightweight QF-41 renderer is imported. No QF-34 implementation is changed.
if TYPE_CHECKING:
    from quantforge.reporting.study_inspection import (
        STUDY_INSPECTION_ARTIFACT_FILENAMES,
        STUDY_INSPECTION_REPORT_ENGINE_VERSION,
        STUDY_INSPECTION_REPORT_SCHEMA_VERSION,
        FutureOutcomeRegion,
        StudyInspectionExportStatus,
        StudyInspectionReport,
        StudyInspectionReportConfig,
        StudyInspectionReportError,
        StudyInspectionSelection,
        build_study_inspection_report,
        export_study_inspection_report,
    )

__all__ = [
    "STUDY_INSPECTION_ARTIFACT_FILENAMES",
    "STUDY_INSPECTION_REPORT_ENGINE_VERSION",
    "STUDY_INSPECTION_REPORT_SCHEMA_VERSION",
    "FutureOutcomeRegion",
    "ReportArtifact",
    "ReportPhase",
    "ReportSection",
    "ResearchReport",
    "ResearchReportConfig",
    "ResearchReportError",
    "ResearchWarning",
    "StudyInspectionExportStatus",
    "StudyInspectionReport",
    "StudyInspectionReportConfig",
    "StudyInspectionReportError",
    "StudyInspectionSelection",
    "build_research_report",
    "build_study_inspection_report",
    "export_research_report",
    "export_study_inspection_report",
]


def __getattr__(name: str) -> object:
    if name not in __all__:
        raise AttributeError(name)
    from quantforge.reporting import study_inspection

    return getattr(study_inspection, name)
