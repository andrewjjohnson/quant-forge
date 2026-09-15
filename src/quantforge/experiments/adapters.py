"""Observational adapters over producer JSON and existing local exports.

No producer objects, factories, data series, or execution callbacks are accepted.
The caller supplies a completed export or an already-serialized result snapshot.
"""

from dataclasses import dataclass
from pathlib import Path

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.experiments._backtest_artifacts import index_backtest_files
from quantforge.experiments._feature_integrity import (
    validate_feature_rows,
    validate_feature_summary,
)
from quantforge.experiments._feature_row_integrity import (
    feature_prediction_studies,
    validate_feature_schema,
)
from quantforge.experiments._grid_integrity import (
    validate_backtest_trial,
    validate_prediction_trial_coordinates,
    validate_trial_coordinates,
    validate_trial_counts,
    validate_trial_status,
)
from quantforge.experiments._json import ManifestError, mapping, snapshot, text
from quantforge.experiments._prediction_trial_integrity import (
    validate_prediction_trial_result,
)
from quantforge.experiments._producer_integrity import (
    validate_backtest_identity,
    validate_prediction_counts,
    validate_prediction_identity,
    validate_prediction_rows,
)
from quantforge.experiments._ranking_integrity import (
    validate_optimization_summaries,
    validate_prediction_summary,
)
from quantforge.experiments._window_integrity import validate_window_snapshot
from quantforge.experiments.artifacts import (
    ArtifactEntry,
    ArtifactIndex,
    ArtifactRelationship,
    ArtifactType,
    RelationshipType,
    index_artifact,
)
from quantforge.experiments.models import (
    ExecutionProvenance,
    ExperimentManifest,
    StudyProvenance,
    StudyType,
)
from quantforge.experiments.persistence import read_producer_record


@dataclass(frozen=True, slots=True)
class StudyArtifacts:
    provenance: StudyProvenance
    index: ArtifactIndex


def create_manifest(
    study: StudyArtifacts,
    execution: ExecutionProvenance,
    *,
    additional_artifacts: tuple[ArtifactEntry, ...] = (),
    relationships: tuple[ArtifactRelationship, ...] = (),
    validation: StudyArtifacts | None = None,
) -> ExperimentManifest:
    """Assemble snapshots and references; validation is file integrity only."""
    provenance = study.provenance
    entries = study.index.entries + additional_artifacts
    edges = study.index.relationships + relationships
    if validation is not None:
        if validation.provenance.study_type not in {
            StudyType.WALK_FORWARD,
            StudyType.OOS_VALIDATION,
            StudyType.HOLDOUT_VALIDATION,
        }:
            raise ManifestError(
                "validation attachment must contain validation evidence"
            )
        provenance = StudyProvenance(
            provenance.study_type,
            provenance.producer_study_id,
            snapshot(
                {
                    **provenance.configuration.to_primitive(),
                    "validation": validation.provenance.configuration.to_primitive(),
                }
            ),
            snapshot(
                {
                    **provenance.observations.to_primitive(),
                    "validation": validation.provenance.observations.to_primitive(),
                }
            ),
        )
        entries += validation.index.entries
        edges += validation.index.relationships
        plans = [
            entry
            for entry in validation.index.entries
            if entry.artifact_type is ArtifactType.VALIDATION_PLAN
        ]
        configurations = [
            entry
            for entry in study.index.entries
            if entry.artifact_type is ArtifactType.CONFIGURATION
        ]
        edges += tuple(
            ArtifactRelationship(
                configuration.artifact_id,
                RelationshipType.CONFIGURED_BY,
                plan.artifact_id,
            )
            for configuration in configurations
            for plan in plans
        )
    return ExperimentManifest(
        provenance,
        execution,
        ArtifactIndex(entries, edges),
    )


def _pick(document: PrimitiveMapping, keys: tuple[str, ...]) -> PrimitiveMapping:
    # Explicit None represents unavailable producer provenance. No backend,
    # timeframe, cost, or historical execution-environment defaults are applied.
    return {key: document.get(key) for key in keys}


def _description(
    study_type: StudyType, document: PrimitiveMapping
) -> tuple[str, PrimitiveMapping, PrimitiveMapping, str]:
    observations = _pick(document, ("record_counts", "initiated_at"))
    if study_type is StudyType.PREDICTION:
        if document.get("component") != "quantforge_prediction_study":
            raise ManifestError("expected a QF-11 prediction study")
        validate_prediction_identity(document)
        validate_prediction_counts(document)
        configuration = _pick(
            document,
            ("engine_version", "configuration", "market_data", "prediction_context"),
        )
        schema = text(mapping(document["configuration"])["result_schema_version"])
        return text(document["study_id"]), configuration, observations, schema
    if study_type is StudyType.PREDICTION_WINDOW:
        configuration = {
            key: value
            for key, value in document.items()
            if key
            not in {"window_id", "window_result_id", "schedule_id", "record_counts"}
        }
        # QF-42 already separates window configuration and result identities.
        if configuration_identity(configuration) != document.get("window_id"):
            raise ManifestError("incompatible QF-42 window configuration")
        observations["window_result_id"] = document["window_result_id"]
        return (
            text(document["window_id"]),
            configuration,
            observations,
            text(document["schema_version"]),
        )
    if study_type is StudyType.FEATURE_DATASET:
        if (
            document.get("component") != "quantforge_signal_feature_dataset"
            or document.get("status") != "complete"
        ):
            raise ManifestError("expected a completed QF-7/QF-29 dataset")
        if configuration_identity(
            mapping(document.get("configuration"))
        ) != document.get("dataset_id"):
            raise ManifestError("feature dataset identity is inconsistent")
        configuration = _pick(
            document, ("engine_version", "configuration", "market_data")
        )
        feature_prediction_studies(document)
        observations["prediction_study_ids"] = document["prediction_study_ids"]
        return (
            text(document["dataset_id"]),
            configuration,
            observations,
            text(mapping(document["configuration"])["feature_schema_version"]),
        )
    if study_type is StudyType.BACKTEST:
        validate_backtest_identity(document)
        configuration = _pick(
            document,
            (
                "engine_version",
                "result_schema_version",
                "market_data",
                "strategy",
                "backtest_configuration",
            ),
        )
        costs = mapping(document["backtest_configuration"])
        for key in (
            "initial_capital",
            "commission",
            "fees",
            "slippage",
            "execution",
            "dividend_policy",
            "split_policy",
        ):
            if key not in costs:
                raise ManifestError("backtest execution provenance is incomplete")
        configuration["benchmark_configuration"] = mapping(document["benchmark"])[
            "configuration"
        ]
        return (
            text(document["run_id"]),
            configuration,
            observations,
            text(document["result_schema_version"]),
        )
    if study_type is StudyType.OPTIMIZATION:
        configuration = mapping(document["identity_inputs"])
        if (
            configuration.get("component") != "quantforge_grid_search_study"
            or configuration_identity(configuration) != document["study_id"]
        ):
            raise ManifestError("incompatible QF-6 study identity")
        observations.update(
            _pick(
                document,
                (
                    "combination_counts",
                    "execution_configuration",
                    "persistence_configuration",
                ),
            )
        )
        return (
            text(document["study_id"]),
            configuration,
            observations,
            text(document["study_schema_version"]),
        )
    if study_type is StudyType.PARAMETER_STUDY:
        if document.get("component") != "quantforge_prediction_parameter_grid":
            raise ManifestError("expected a QF-32 parameter study")
        configuration = {
            key: value
            for key, value in document.items()
            if key not in {"study_id", "execution", "cache_policy", "interpretation"}
        }
        if configuration_identity(configuration) != document["study_id"]:
            raise ManifestError("incompatible QF-32 study identity")
        observations["execution"] = document["execution"]
        return (
            text(document["study_id"]),
            configuration,
            observations,
            text(document["schema_version"]),
        )
    raise ManifestError("use inspect_validation for walk-forward/OOS/holdout studies")


def inspect_study(
    study_type: StudyType, source: Path, *, artifact_root: Path
) -> StudyArtifacts:
    """Index QF-11/42/7/29/32/5/6 persisted outputs without loading an engine.

    `source` is an export directory, or a JSON result with its existing
    `manifest` member. A QF-5 `manifest.json` selects its complete parent export;
    a detached QF-5 manifest must use another filename and indexes metadata only.
    Relative artifact paths are rooted at `artifact_root`.
    """
    source = source.resolve()
    root = artifact_root.resolve()
    if (
        study_type is StudyType.BACKTEST
        and not source.is_dir()
        and source.name == "manifest.json"
    ):
        source = source.parent
    manifest_path = source / "manifest.json" if source.is_dir() else source
    document, location = read_producer_record(manifest_path)
    container = document
    if "manifest" in document:
        document = mapping(document["manifest"])
        location += "/manifest"
    producer_id, configuration, observations, schema = _description(
        study_type, document
    )
    if study_type is StudyType.PREDICTION_WINDOW:
        validate_window_snapshot(container)
    if study_type is StudyType.PREDICTION and "rows" in container:
        validate_prediction_rows(document, container["rows"])
    if study_type is StudyType.FEATURE_DATASET and "rows" in container:
        validate_feature_rows(document, container)
    entries: list[ArtifactEntry] = []
    edges: list[ArtifactRelationship] = []

    def add(
        path: Path,
        category: ArtifactType,
        logical_id: str,
        *,
        json_pointer: str = "",
        bindings: PrimitiveMapping | None = None,
        version: str = schema,
    ) -> ArtifactEntry:
        if not path.is_relative_to(root):
            raise ManifestError("producer export is outside artifact root")
        entry = index_artifact(
            root,
            path=path.relative_to(root).as_posix(),
            artifact_type=category,
            schema_version=version,
            producer_study_id=producer_id,
            producer_run_id=producer_id if study_type is StudyType.BACKTEST else None,
            producer_artifact_id=logical_id,
            json_pointer=json_pointer,
            bindings=bindings,
        )
        entries.append(entry)
        return entry

    identity_fields = (
        "study_id",
        "window_id",
        "dataset_id",
        "run_id",
        "engine_version",
        "result_schema_version",
        "schema_version",
    )
    config_entry = add(
        manifest_path,
        ArtifactType.CONFIGURATION,
        "producer_manifest",
        json_pointer=location,
        bindings={
            location + "/" + key: document[key]
            for key in identity_fields
            if key in document
        },
    )
    dataset_pointer = (
        "/identity_inputs/dataset"
        if study_type is StudyType.OPTIMIZATION
        else "/dataset"
        if study_type is StudyType.PARAMETER_STUDY
        else "/market_data"
    )
    from quantforge.experiments._json import pointer

    source_metadata = mapping(pointer(document, dataset_pointer))
    dataset_entry = add(
        manifest_path,
        ArtifactType.SOURCE_DATASET,
        "source_dataset",
        json_pointer=location + dataset_pointer,
        version=text(source_metadata.get("schema_version", "unavailable")),
    )
    edges.append(
        ArtifactRelationship(
            config_entry.artifact_id,
            RelationshipType.DERIVED_FROM,
            dataset_entry.artifact_id,
        )
    )
    # Only provenance is copied. Research result rows/metrics stay in their files.
    if study_type is StudyType.FEATURE_DATASET:
        if source.is_dir():
            feature_schema, _ = read_producer_record(source / "schema.json")
            validate_feature_schema(document, feature_schema)
            schema_entry = add(
                source / "schema.json",
                ArtifactType.FEATURE_SCHEMA,
                "feature_schema",
                version=text(feature_schema["feature_schema_version"]),
            )
        else:
            feature_schema = mapping(container["schema"])
            schema_entry = add(
                manifest_path,
                ArtifactType.FEATURE_SCHEMA,
                "feature_schema",
                json_pointer="/schema",
                version=text(feature_schema["feature_schema_version"]),
            )
        configuration["feature_schema"] = feature_schema
        edges.append(
            ArtifactRelationship(
                config_entry.artifact_id,
                RelationshipType.CONFIGURED_BY,
                schema_entry.artifact_id,
            )
        )

    # Index known producer layouts only, not arbitrary neighboring files.
    if source.is_dir():
        optimization_summaries: dict[str, PrimitiveMapping] = {}
        if study_type is StudyType.OPTIMIZATION and (source / "summary.json").is_file():
            for name in ("ranking", "stability"):
                if not (source / f"{name}.json").is_file():
                    raise ManifestError(
                        f"completed optimization is missing its {name} artifact"
                    )
        if study_type is StudyType.BACKTEST:
            from quantforge.backtesting.export import validate_backtest_result_artifact

            validate_backtest_result_artifact(source)
        if study_type is StudyType.FEATURE_DATASET:
            required_names = {"features.csv", "schema.json", "summary.json"}
            if document["engine_version"] == "35":
                required_names.add("features.parquet")
            if any(not (source / name).is_file() for name in required_names):
                raise ManifestError("completed feature dataset is missing an artifact")
        for path in sorted(source.iterdir()):
            if not path.is_file() or path.name in {"manifest.json", "schema.json"}:
                continue
            category = _export_category(study_type, path.name)
            if category is not None:
                bindings: PrimitiveMapping | None = None
                if (
                    study_type is StudyType.FEATURE_DATASET
                    and path.name == "summary.json"
                ):
                    feature_summary, summary_base = read_producer_record(path)
                    validate_feature_summary(document, feature_summary)
                    bindings = {
                        summary_base + "/" + key: count
                        for key, count in feature_summary.items()
                    }
                if study_type is StudyType.OPTIMIZATION and path.name in {
                    "ranking.json",
                    "stability.json",
                }:
                    owned_summary, summary_base = read_producer_record(path)
                    if owned_summary.get("study_id") != producer_id:
                        raise ManifestError(
                            "optimization summary belongs to another study"
                        )
                    bindings = {summary_base + "/study_id": producer_id}
                    optimization_summaries[path.name] = owned_summary
                entry = add(path, category, path.name, bindings=bindings)
                edges.append(
                    ArtifactRelationship(
                        entry.artifact_id,
                        RelationshipType.CONFIGURED_BY,
                        config_entry.artifact_id,
                    )
                )
        if study_type in {StudyType.PARAMETER_STUDY, StudyType.OPTIMIZATION}:
            summary_path = source / "summary.json"
            summary: PrimitiveMapping | None = None
            if summary_path.is_file():
                summary, _ = read_producer_record(summary_path)
                if summary.get("study_id") != producer_id:
                    raise ManifestError("parameter summary belongs to another study")
                observations["trial_counts"] = summary.get(
                    "counts", summary.get("trial_counts")
                )
            trials: list[str] = []
            trial_records: list[PrimitiveMapping] = []
            statuses: list[str] = []
            for path in sorted((source / "trials").glob("*.json")):
                trial, _ = read_producer_record(path)
                trial_id = text(trial["trial_id"])
                if path.stem != trial_id or trial["study_id"] != producer_id:
                    raise ManifestError("trial identity is incompatible with study")
                validate_trial_status(study_type, trial)
                trials.append(trial_id)
                trial_records.append(trial)
                statuses.append(text(trial.get("status")))
                trial_entry = add(
                    path,
                    ArtifactType.TRIAL_RESULT,
                    trial_id,
                    bindings={"/trial_id": trial_id, "/study_id": producer_id},
                )
                edges.append(
                    ArtifactRelationship(
                        trial_entry.artifact_id,
                        RelationshipType.CONFIGURED_BY,
                        config_entry.artifact_id,
                    )
                )
                relative = trial.get("artifact_location")
                if relative is None and trial.get("status") == "succeeded":
                    raise ManifestError(
                        "successful trial is missing its result artifact"
                    )
                if relative is not None:
                    from quantforge.experiments.artifacts import local_path

                    artifact_path = local_path(source, text(relative))
                    if study_type is StudyType.OPTIMIZATION:
                        from quantforge.backtesting.export import (
                            validate_backtest_result_artifact,
                        )

                        validate_backtest_result_artifact(artifact_path)
                        backtest_manifest, _ = read_producer_record(
                            artifact_path / "manifest.json"
                        )
                        validate_backtest_trial(trial, backtest_manifest, configuration)
                        for entry in index_backtest_files(
                            root, artifact_path, backtest_manifest, trial_id
                        ):
                            entries.append(entry)
                            edges.append(
                                ArtifactRelationship(
                                    entry.artifact_id,
                                    RelationshipType.DERIVED_FROM,
                                    trial_entry.artifact_id,
                                )
                            )
                    else:
                        artifact, _ = read_producer_record(artifact_path)
                        if (
                            artifact.get("grid_study_id") != producer_id
                            or artifact.get("trial_id") != trial_id
                        ):
                            raise ManifestError(
                                "prediction trial artifact belongs to another study"
                            )
                        claimed = artifact.get("artifact_fingerprint")
                        if claimed != trial.get(
                            "artifact_fingerprint"
                        ) or claimed != configuration_identity(
                            {
                                key: value
                                for key, value in artifact.items()
                                if key != "artifact_fingerprint"
                            }
                        ):
                            raise ManifestError(
                                "prediction trial artifact fingerprint mismatch"
                            )
                        if artifact.get("analysis") != trial.get(
                            "analysis"
                        ) or artifact.get("schema_version") != trial.get(
                            "schema_version"
                        ):
                            raise ManifestError(
                                "prediction trial artifact metadata does not match "
                                "its record"
                            )
                        if configuration.get("decision_schedule") is not None:
                            validate_window_snapshot(
                                mapping(artifact.get("prediction_window"))
                            )
                        else:
                            prediction = mapping(artifact.get("prediction_study"))
                            prediction_manifest = mapping(prediction.get("manifest"))
                            validate_prediction_identity(prediction_manifest)
                            validate_prediction_rows(
                                prediction_manifest, prediction.get("rows")
                            )
                        validate_prediction_trial_coordinates(trial, configuration)
                        validate_prediction_trial_result(trial, artifact, configuration)
                        entry = add(
                            artifact_path,
                            ArtifactType.PREDICTION_RESULT,
                            trial_id + "/result",
                        )
                        edges.append(
                            ArtifactRelationship(
                                entry.artifact_id,
                                RelationshipType.DERIVED_FROM,
                                trial_entry.artifact_id,
                            )
                        )
            observations["trial_ids"] = list(trials)
            if summary is not None:
                validate_trial_counts(study_type, summary, statuses)
            for trial in trial_records:
                if study_type is StudyType.OPTIMIZATION:
                    validate_trial_coordinates(trial, configuration)
                elif trial.get("status") != "succeeded":
                    validate_prediction_trial_coordinates(trial, configuration)
            if study_type is StudyType.OPTIMIZATION and optimization_summaries:
                validate_optimization_summaries(
                    configuration, trial_records, summary, optimization_summaries
                )
            elif study_type is StudyType.PARAMETER_STUDY and summary is not None:
                validate_prediction_summary(configuration, trial_records, summary)
    elif "rows" in container or "decisions" in container:
        key = "rows" if "rows" in container else "decisions"
        category = (
            ArtifactType.FEATURE_DATASET
            if study_type is StudyType.FEATURE_DATASET
            else ArtifactType.PREDICTION_RESULT
        )
        result_entry = add(source, category, "result", json_pointer="/" + key)
        edges.append(
            ArtifactRelationship(
                result_entry.artifact_id,
                RelationshipType.CONFIGURED_BY,
                config_entry.artifact_id,
            )
        )
    if read_producer_record(manifest_path)[0] != container:
        raise ManifestError("producer metadata changed during indexing")
    return StudyArtifacts(
        StudyProvenance(
            study_type, producer_id, snapshot(configuration), snapshot(observations)
        ),
        ArtifactIndex(tuple(entries), tuple(edges)),
    )


def _export_category(study_type: StudyType, name: str) -> ArtifactType | None:
    if study_type is StudyType.FEATURE_DATASET:
        return {
            "features.csv": ArtifactType.FEATURE_DATASET,
            "features.parquet": ArtifactType.FEATURE_DATASET,
            "summary.json": ArtifactType.FEATURE_DATASET,
        }.get(name)
    if study_type is StudyType.BACKTEST:
        return (
            ArtifactType.BACKTEST_RESULT
            if name.endswith(".csv") or name == "integrity.json"
            else None
        )
    if study_type in {StudyType.PARAMETER_STUDY, StudyType.OPTIMIZATION}:
        return (
            ArtifactType.PARAMETER_SUMMARY
            if name.endswith(".csv")
            or name in {"summary.json", "result.json", "stability.json"}
            or (study_type is StudyType.OPTIMIZATION and name == "ranking.json")
            else None
        )
    return None
