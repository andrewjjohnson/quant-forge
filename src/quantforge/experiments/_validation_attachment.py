"""Attach captured validation evidence without inventing result membership."""

from quantforge.configuration import configuration_identity
from quantforge.experiments._json import ManifestError, mapping, text
from quantforge.experiments.artifacts import (
    ArtifactEntry,
    ArtifactIndex,
    ArtifactRelationship,
    ArtifactType,
)
from quantforge.experiments.models import StudyProvenance, StudyType


def captured_validation_result(
    primary: StudyProvenance, validation: StudyProvenance, index: ArtifactIndex
) -> ArtifactEntry:
    if primary.study_type not in {
        StudyType.PREDICTION,
        StudyType.PREDICTION_WINDOW,
        StudyType.BACKTEST,
    }:
        raise ManifestError(
            "validation attachment requires a prediction/backtest primary study"
        )
    configuration = validation.configuration.to_primitive()
    adapter = mapping(mapping(configuration.get("study_definition")).get("adapter"))
    expected_adapter = (
        "qf39_backtest"
        if primary.study_type is StudyType.BACKTEST
        else "qf39_prediction"
    )
    if adapter.get("adapter") != expected_adapter:
        raise ManifestError("validation research family differs from primary study")
    result_id = (
        text(primary.observations.to_primitive().get("window_result_id"))
        if primary.study_type is StudyType.PREDICTION_WINDOW
        else primary.producer_study_id
    )
    observations = validation.observations.to_primitive()
    folds = observations.get("folds")
    if not isinstance(folds, list):
        raise ManifestError("validation has no captured result references")
    holdout = mapping(observations.get("holdout"))
    for entry in index.entries:
        if entry.producer_study_id != validation.producer_study_id:
            continue
        if entry.artifact_type is ArtifactType.WALK_FORWARD_WINDOW:
            if entry.bindings.to_primitive().get(
                entry.json_pointer + "/result_id"
            ) == result_id and any(
                mapping(fold).get("status") == "completed"
                and mapping(fold).get("fold_id") == entry.producer_artifact_id
                and mapping(fold).get("result_id") == result_id
                for fold in folds
            ):
                return entry
        elif (
            entry.artifact_type is ArtifactType.HOLDOUT_RESULT
            and entry.producer_artifact_id == result_id
            and holdout.get("state") == "consumed"
            and holdout.get("result_artifact_id") == entry.artifact_id
        ):
            return entry
    raise ManifestError("primary study does not match a captured result in validation")


def merge_validation_artifacts(
    entries: tuple[ArtifactEntry, ...],
    relationships: tuple[ArtifactRelationship, ...],
    validation: ArtifactIndex,
) -> tuple[tuple[ArtifactEntry, ...], tuple[ArtifactRelationship, ...]]:
    """Reuse the primary's entries for the same captured QF-5 export files.

    The primary gives its manifest a configuration role and stronger bindings;
    the nested validation export uses fold-specific logical names. Only these
    aliases with identical content and provenance may share an index entry.
    """
    merged = list(entries)
    locations = {(entry.path, entry.json_pointer): entry for entry in entries}
    aliases: dict[str, str] = {}
    for entry in validation.entries:
        existing = locations.get((entry.path, entry.json_pointer))
        if existing is None:
            merged.append(entry)
            locations[(entry.path, entry.json_pointer)] = entry
            continue
        if entry != existing:
            omitted = {"artifact_type", "producer_artifact_id", "bindings"}
            if (
                entry.artifact_type is not ArtifactType.BACKTEST_RESULT
                or existing.artifact_type
                not in {ArtifactType.CONFIGURATION, ArtifactType.BACKTEST_RESULT}
                or configuration_identity(
                    {
                        key: value
                        for key, value in entry.to_primitive().items()
                        if key not in omitted
                    }
                )
                != configuration_identity(
                    {
                        key: value
                        for key, value in existing.to_primitive().items()
                        if key not in omitted
                    }
                )
                or any(
                    key not in existing.bindings.to_primitive()
                    or configuration_identity({"binding": value})
                    != configuration_identity(
                        {"binding": existing.bindings.to_primitive()[key]}
                    )
                    for key, value in entry.bindings.to_primitive().items()
                )
            ):
                raise ManifestError(
                    "validation contains an incompatible shared artifact"
                )
        aliases[entry.artifact_id] = existing.artifact_id
    edges = list(relationships)
    known = set(relationships)
    for edge in validation.relationships:
        mapped = ArtifactRelationship(
            aliases.get(edge.source_id, edge.source_id),
            edge.relationship,
            aliases.get(edge.target_id, edge.target_id),
        )
        if mapped not in known:
            edges.append(mapped)
            known.add(mapped)
    return tuple(merged), tuple(edges)
