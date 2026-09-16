"""Index already-validated nested QF-5 exports with their original provenance."""

from pathlib import Path

from quantforge.configuration import PrimitiveMapping
from quantforge.experiments._backtest_provenance import benchmark_configuration
from quantforge.experiments._json import ManifestError, mapping, parse_json, text
from quantforge.experiments._producer_snapshot import ProducerReadSet
from quantforge.experiments.artifacts import ArtifactEntry, ArtifactType, index_artifact


def validate_backtest_files(export: Path, reads: ProducerReadSet) -> str:
    """Pin the sidecar and its claimed file bytes across producer validation."""
    from quantforge.backtesting.export import (
        BACKTEST_ARTIFACT_FILENAMES,
        ResultExportError,
        validate_backtest_result_artifact,
    )

    sidecar = export / "integrity.json"
    try:
        content = sidecar.read_bytes()
    except OSError as error:
        raise ResultExportError("cannot read backtest integrity sidecar") from error
    validate_backtest_result_artifact(export)
    integrity = parse_json(content)
    files = mapping(integrity.get("files"))
    # Validate the captured sidecar itself even if the producer read another
    # version during an atomic replacement. No result tables are regenerated.
    if (
        set(integrity) != {"schema_version", "algorithm", "files"}
        or integrity.get("schema_version") != "1"
        or integrity.get("algorithm") != "sha256"
        or set(files) != set(BACKTEST_ARTIFACT_FILENAMES) - {"integrity.json"}
    ):
        raise ManifestError("invalid captured backtest integrity sidecar")
    reads.expect(sidecar, content)
    for filename, fingerprint in files.items():
        reads.expect_sha256(export / filename, text(fingerprint))
    # Match the producer's read_text() newline handling for captured identities.
    return content.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")


def index_backtest_files(
    root: Path, export: Path, manifest: PrimitiveMapping, logical_prefix: str
) -> tuple[ArtifactEntry, ...]:
    """Keep QF-5 schema/run metadata independent of the enclosing study type."""
    from quantforge.backtesting.export import BACKTEST_ARTIFACT_FILENAMES

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
        for path in (export / name for name in sorted(BACKTEST_ARTIFACT_FILENAMES))
    )
