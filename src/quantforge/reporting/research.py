"""QF-41 presentation-only static research reports over QF-9 artifacts."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from quantforge.configuration import Primitive, configuration_identity
from quantforge.experiments import ArtifactIndex, read_manifest, verify_artifacts
from quantforge.experiments._json import snapshot
from quantforge.reporting._research_inputs import holdout_snapshot, read_artifacts
from quantforge.reporting._research_sections import build_sections
from quantforge.reporting._research_warnings import build_warnings
from quantforge.reporting.research_models import (
    REPORT_VERSION,
    ResearchReport,
    ResearchReportConfig,
    ResearchReportError,
)

if TYPE_CHECKING:
    from quantforge.oos import HoldoutLedger, OOSSource


def build_research_report(
    manifest_path: Path,
    *,
    artifact_root: Path,
    config: ResearchReportConfig = ResearchReportConfig(),
    holdout_source: OOSSource | None = None,
    holdout_ledger: HoldoutLedger | None = None,
) -> ResearchReport:
    """Capture only persisted inputs; incomplete evidence stays explicitly unavailable.

    A supplied QF-40 source/ledger must match this manifest. The state is queried
    twice to detect consumption during rendering. An exported report remains a
    static snapshot; rebuild it to check subsequent ledger changes.
    """
    root, path = artifact_root.resolve(), manifest_path.resolve()
    if not path.is_relative_to(root):
        raise ResearchReportError("manifest must be inside artifact root")
    # Validate manifest identity even for partial reports.
    manifest = read_manifest(path)
    artifacts = read_artifacts(manifest.artifacts, root, config)
    holdout = holdout_snapshot(manifest, artifacts, holdout_source, holdout_ledger)
    sections = build_sections(manifest, artifacts, holdout)
    warnings = build_warnings(manifest, artifacts, holdout, config)
    if holdout != holdout_snapshot(manifest, artifacts, holdout_source, holdout_ledger):
        raise ResearchReportError("holdout state changed during rendering; retry")
    if path.read_bytes() != manifest.serialize():
        raise ResearchReportError("manifest changed during rendering; retry")
    relative = path.relative_to(root).as_posix()
    header = snapshot(
        {
            "report_version": REPORT_VERSION,
            "study_type": manifest.provenance.study_type.value,
            "study_id": manifest.study_id,
            "producer_study_id": manifest.provenance.producer_study_id,
            "run_id": manifest.execution.run_id,
            "manifest_id": manifest.manifest_id,
            "manifest_schema_version": manifest.to_primitive()["schema_version"],
            "artifact_index_id": manifest.artifacts.index_id,
            "holdout_state": holdout["state"],
        }
    )
    evidence: list[Primitive] = [
        {
            "artifact_id": item.entry.artifact_id,
            "status": item.status,
        }
        for item in artifacts
    ]
    report_id = configuration_identity(
        {
            "component": "quantforge_static_research_report",
            "version": REPORT_VERSION,
            "manifest_id": manifest.manifest_id,
            "manifest_path": relative,
            "config": config.to_primitive(),
            "holdout": holdout,
            "evidence": evidence,
        }
    )
    return ResearchReport(
        report_id,
        manifest.manifest_id,
        relative,
        str(root),
        sections,
        warnings,
        artifacts,
        config,
        header,
        holdout_source,
        holdout_ledger,
    )


def export_research_report(report: ResearchReport, output_root: Path) -> Path:
    """Atomically publish <report-id>.html without changing any input artifact.

    Keep the artifact root's layout alongside the HTML to retain relative links.
    No Python process, backend, JavaScript, or remote assets are needed to view it.
    """
    from quantforge.reporting._research_html import render_html

    root = Path(report.artifact_root)
    output = output_root.resolve()
    if not output.is_relative_to(root):
        raise ResearchReportError("report output must be inside artifact root")
    destination = output / f"{report.report_id}.html"
    inputs = {
        root / report.manifest_path,
        *(root / item.entry.path for item in report.artifacts),
    }
    if destination in inputs:
        raise ResearchReportError("report output cannot replace an input")
    content = render_html(report, destination).encode("utf-8")
    manifest = read_manifest(root / report.manifest_path)
    if manifest.manifest_id != report.manifest_id:
        raise ResearchReportError("manifest changed before export; rebuild report")
    verify_artifacts(
        ArtifactIndex(
            tuple(item.entry for item in report.artifacts if item.status == "verified")
        ),
        root,
    ).require_valid()
    current = holdout_snapshot(
        manifest, report.artifacts, report.holdout_source, report.holdout_ledger
    )
    saved = next(
        section.content.to_primitive()["value"]
        for section in report.sections
        if section.title == "Final holdout state"
    )
    if current != saved:
        raise ResearchReportError("holdout state changed before export; rebuild report")
    output.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        descriptor, filename = tempfile.mkstemp(prefix=".research-report-", dir=output)
        temporary = Path(filename)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.is_symlink() or destination.read_bytes() != content:
                raise ResearchReportError(
                    "immutable report differs; refusing overwrite"
                ) from None
        directory = os.open(output, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return destination
    except OSError as error:
        raise ResearchReportError(
            "failed to persist immutable research report"
        ) from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
