"""Index already-validated nested QF-5 exports with their original provenance."""

from pathlib import Path

from quantforge.configuration import PrimitiveMapping
from quantforge.experiments._backtest_provenance import benchmark_configuration
from quantforge.experiments._json import ManifestError, text
from quantforge.experiments.artifacts import ArtifactEntry, ArtifactType, index_artifact


def index_backtest_files(
    root: Path, export: Path, manifest: PrimitiveMapping, logical_prefix: str
) -> tuple[ArtifactEntry, ...]:
    """Keep QF-5 schema/run metadata independent of the enclosing study type."""
    if not export.resolve().is_relative_to(root):
        raise ManifestError("backtest export is outside artifact root")
    benchmark_configuration(manifest)
    run_id = text(manifest.get("run_id"))
    schema = text(manifest.get("result_schema_version"))
    return tuple(
        index_artifact(
            root,
            path=path.relative_to(root).as_posix(),
            artifact_type=ArtifactType.BACKTEST_RESULT,
            schema_version=schema,
            producer_study_id=run_id,
            producer_run_id=run_id,
            producer_artifact_id=logical_prefix + "/" + path.name,
        )
        for path in sorted(export.iterdir())
        if path.is_file() and path.suffix in {".json", ".csv"}
    )
