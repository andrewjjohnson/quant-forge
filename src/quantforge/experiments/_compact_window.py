"""QF-9 bindings to QF-55 finalized JSONL, without embedded-decision expansion."""

from datetime import date
from pathlib import Path
from typing import cast

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.data.calendar import expected_sessions
from quantforge.data.models import DatasetMetadata
from quantforge.experiments._json import (
    ManifestError,
    mapping,
    safe_metadata,
    snapshot,
    text,
)
from quantforge.experiments._producer_snapshot import ProducerReadSet
from quantforge.experiments.artifacts import (
    ArtifactEntry,
    ArtifactFormat,
    ArtifactRelationship,
    ArtifactType,
    RelationshipType,
    file_sha256,
)
from quantforge.prediction.window_compact_validation import (
    validate_prediction_window_reader,
)
from quantforge.prediction.window_reader import PredictionWindowReader


def validate_compact_window(
    reader: PredictionWindowReader,
    *,
    canonical_metadata: DatasetMetadata | None = None,
    strategy_parameters: PrimitiveMapping | None = None,
) -> None:
    """Verify original scientific identities and ancestry once per shared scope."""
    identity = reader.evidence.identity_snapshot.to_primitive()
    market = mapping(identity["market_data"])
    configuration = mapping(identity["configuration"])
    rule = mapping(mapping(configuration["prediction_rule"])["configuration"])
    sessions = tuple(
        s
        for s in expected_sessions(
            date.fromisoformat(text(market["actual_first_session"])),
            date.fromisoformat(text(market["actual_last_session"])),
            text(market["calendar"]),
        )
        if s.isoformat() not in cast(list[str], market["missing_sessions"])
    )
    try:
        validate_prediction_window_reader(
            reader,
            expected_identity=reader.evidence.identity_snapshot,
            schedule=reader.evidence.schedule,
            outcome_sessions=sessions,
            strategy_parameters=strategy_parameters
            if strategy_parameters is not None
            else mapping(rule.get("parameters", {})),
            canonical_metadata=canonical_metadata,
        )
    except (ValueError, KeyError, TypeError) as error:
        raise ManifestError("invalid compact window scientific evidence") from error


def index_compact_window(
    reader: PredictionWindowReader,
    *,
    artifact_root: Path,
    reads: ProducerReadSet,
) -> tuple[tuple[ArtifactEntry, ...], tuple[ArtifactRelationship, ...]]:
    """Index header, configuration and shared scope without copying their bodies."""
    path = reader.path.resolve()
    if not path.is_relative_to(artifact_root.resolve()):
        raise ManifestError("compact window is outside artifact root")
    fingerprint = file_sha256(path)
    safe_metadata(reader.evidence.to_primitive())
    for record in reader.iterate_decisions():
        safe_metadata(record.to_primitive())
    reads.expect_sha256(path, fingerprint)
    header = reader.header()
    scope = reader.evidence.identity_snapshot.to_primitive()
    market = mapping(scope["market_data"])
    configuration = mapping(scope["configuration"])
    provenance = market.get("intraday_provenance")
    ancestry = mapping(provenance) if provenance is not None else {}
    identities: PrimitiveMapping = {
        "source_dataset_id": market["dataset_id"],
        "source_bars_fingerprint": market["bars_fingerprint"],
        "canonical_input_id": ancestry.get("canonical_input_id"),
        "canonical_provenance_id": ancestry.get("canonical_provenance_id"),
        "study_configuration_id": configuration_identity(configuration),
        "outcome_configuration_id": configuration_identity(
            mapping(configuration["outcome_labeler"])
        ),
        "context_environment_id": configuration_identity(
            mapping(scope["context_environment"])
        ),
    }
    entries: list[ArtifactEntry] = []
    for location, category, identity in (
        ("/header", ArtifactType.PREDICTION_RESULT, text(header["window_result_id"])),
        ("/shared_evidence", ArtifactType.SOURCE_DATASET, reader.evidence.evidence_id),
        (
            "/shared_evidence/window_identity/configuration",
            ArtifactType.CONFIGURATION,
            text(identities["study_configuration_id"]),
        ),
    ):
        entry = ArtifactEntry(
            path=path.relative_to(artifact_root.resolve()).as_posix(),
            artifact_type=category,
            file_format=ArtifactFormat.JSONL,
            sha256=fingerprint,
            schema_version="2",
            producer_study_id=text(header["window_id"]),
            producer_artifact_id="window" + location,
            json_pointer=location,
            metadata=snapshot(
                {
                    **identities,
                    "identity": identity,
                    "window_id": header["window_id"],
                    "shared_evidence_id": reader.evidence.evidence_id,
                }
            ),
            bindings=snapshot({"/header": header}),
        )
        entries.append(entry)
    return tuple(entries), (
        ArtifactRelationship(
            entries[0].artifact_id,
            RelationshipType.DERIVED_FROM,
            entries[1].artifact_id,
        ),
        ArtifactRelationship(
            entries[0].artifact_id,
            RelationshipType.CONFIGURED_BY,
            entries[2].artifact_id,
        ),
    )


def compact_provenance(reader: PredictionWindowReader) -> PrimitiveMapping:
    """Configuration plus identity references; large market proof stays in JSONL."""
    identity = reader.evidence.identity_snapshot.to_primitive()
    market = mapping(identity.pop("market_data"))
    return {
        **identity,
        "schema_version": reader.schema_version,
        "market_data": {
            key: market.get(key)
            for key in (
                "dataset_id",
                "bars_fingerprint",
                "provider",
                "canonical_symbol",
                "adjustment",
            )
        },
        "shared_evidence_id": reader.evidence.evidence_id,
        "window_id": reader.header()["window_id"],
        "schedule_id": reader.evidence.schedule.schedule_id,
    }
