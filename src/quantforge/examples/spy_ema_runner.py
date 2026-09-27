"""QF-45 pre-holdout orchestration over existing research artifacts only."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from quantforge.configuration import PrimitiveMapping
from quantforge.examples.spy_ema_events import export_candidate_features
from quantforge.examples.spy_ema_inputs import SmokeInputs
from quantforge.examples.spy_ema_plan import prepare_walk_forward
from quantforge.experiments import (
    ArtifactEntry,
    ArtifactRelationship,
    ArtifactType,
    ExecutionProvenance,
    RelationshipType,
    StudyType,
    create_manifest,
    inspect_study,
    inspect_validation,
    verify_artifacts,
    write_manifest,
)
from quantforge.oos import (
    HoldoutLedger,
    aggregate_prediction,
    export_oos_aggregate,
    load_oos_source,
)
from quantforge.oos.prediction import PredictionMetricFields
from quantforge.optimization import TrialStatus
from quantforge.prediction.grid import PredictionGridTrialRecord
from quantforge.prediction.window_encoding import mapping
from quantforge.prediction.window_reader import PredictionWindowReader
from quantforge.reporting import build_research_report, export_research_report
from quantforge.validation import PartitionRole
from quantforge.walk_forward import (
    FoldStatus,
    PredictionEvaluator,
    WalkForwardConfig,
    WalkForwardError,
    WalkForwardStudy,
)
from quantforge.walk_forward.partitions import partition
from quantforge.walk_forward.persistence import read_record, write_record


def publish_pre_holdout(
    config: WalkForwardConfig,
    study_path: Path,
    output_root: Path,
    ledger: HoldoutLedger,
    execution: ExecutionProvenance,
    *,
    additional_entries: tuple[ArtifactEntry, ...] = (),
    additional_relationships: tuple[ArtifactRelationship, ...] = (),
) -> PrimitiveMapping:
    """QF-40 -> QF-9 -> QF-41, with the permanent ledger as authority."""
    source = load_oos_source(config.plan, study_path)
    state = ledger.reserve(source)
    if state.state != "reserved_unconsumed":
        raise ValueError("pre-holdout execution requires an unconsumed reservation")
    aggregate = aggregate_prediction(
        source, fields=PredictionMetricFields(signed_outcome="raw_return")
    )
    aggregate_path = export_oos_aggregate(aggregate, output_root / "oos")
    artifacts = inspect_validation(
        source,
        study_path,
        artifact_root=output_root.parent,
        ledger=ledger,
        aggregate_path=aggregate_path,
    )
    existing = {entry.artifact_id for entry in artifacts.index.entries}
    manifest = create_manifest(
        artifacts,
        execution,
        additional_artifacts=tuple(
            entry for entry in additional_entries if entry.artifact_id not in existing
        ),
        relationships=tuple(
            edge
            for edge in additional_relationships
            if edge not in artifacts.index.relationships
        ),
    )
    verify_artifacts(manifest.artifacts, output_root.parent).require_valid()
    manifest_path = write_manifest(
        manifest, output_root / "manifests", artifact_root=output_root.parent
    )
    report = build_research_report(
        manifest_path,
        artifact_root=output_root.parent,
        holdout_source=source,
        holdout_ledger=ledger,
    )
    html = export_research_report(report, output_root / "html")
    if report.header.to_primitive()["holdout_state"] != "reserved_unconsumed":
        raise ValueError("pre-holdout report lost the authoritative reservation")
    return {
        "study_path": str(study_path),
        "aggregate_path": str(aggregate_path),
        "manifest_path": str(manifest_path),
        "report_path": str(html),
        "holdout_state": state.state,
        "manual_audit": (
            "PENDING: inspect at least 10 fixed candidates, or all if fewer"
        ),
    }


def inspect_phase(
    inputs: SmokeInputs,
    config: WalkForwardConfig,
    adapter: PredictionEvaluator,
    phase_root: Path,
    output_root: Path,
    execution: ExecutionProvenance,
    *,
    role: PartitionRole = PartitionRole.SELECTION,
) -> tuple[tuple[ArtifactEntry, ...], tuple[ArtifactRelationship, ...]]:
    """Validate completed windows and export candidate-only native QF-7 evidence."""
    entries: dict[str, ArtifactEntry] = {}
    edges: dict[tuple[str, str, str], ArtifactRelationship] = {}
    permitted = partition(inputs.dataset, config.plan, 0, role, minimum_observations=1)
    paths = sorted(phase_root.rglob("prediction-window.jsonl"))
    if role is PartitionRole.WALK_FORWARD_TEST:
        paths = [p for p in paths if p.parent.name == "test"]
    if not paths:
        raise ValueError("phase has no finalized prediction windows")
    for path in paths:
        reader = PredictionWindowReader.open(path)
        # Bind the stored executable rule parameters back to the frozen universe.
        inspected = inspect_study(
            StudyType.PREDICTION_WINDOW,
            path,
            artifact_root=output_root.parent,
            canonical_metadata=inputs.dataset.metadata,
        )
        verify_artifacts(inspected.index, output_root.parent).require_valid()
        configuration = mapping(reader.manifest()["configuration"])
        strategy_id = mapping(configuration["prediction_rule"])["configuration_id"]
        matching = [
            candidate
            for candidate in adapter.universe.candidates
            if adapter.factory.build(
                candidate.parameters.to_primitive()
            ).strategy.configuration_id
            == strategy_id
        ]
        if len(matching) != 1:
            raise ValueError("window rule does not match the frozen candidate universe")
        parameters = matching[0].parameters.to_primitive()
        feature_paths = export_candidate_features(
            inputs,
            config,
            permitted,
            reader,
            parameters,
            output_root / "features",
        )
        for feature_path in feature_paths:
            feature = inspect_study(
                StudyType.FEATURE_DATASET,
                feature_path,
                artifact_root=output_root.parent,
            )
            verify_artifacts(feature.index, output_root.parent).require_valid()
            for entry in feature.index.entries:
                entries[entry.artifact_id] = entry
            for edge in feature.index.relationships:
                edges[(edge.source_id, edge.relationship, edge.target_id)] = edge
            window_entry = next(
                entry
                for entry in inspected.index.entries
                if entry.artifact_type is ArtifactType.PREDICTION_RESULT
            )
            for entry in feature.index.entries:
                if entry.artifact_type is ArtifactType.FEATURE_DATASET:
                    edge = ArtifactRelationship(
                        entry.artifact_id,
                        RelationshipType.DERIVED_FROM,
                        window_entry.artifact_id,
                    )
                    edges[(edge.source_id, edge.relationship, edge.target_id)] = edge
        for entry in inspected.index.entries:
            entries[entry.artifact_id] = entry
        for edge in inspected.index.relationships:
            edges[(edge.source_id, edge.relationship, edge.target_id)] = edge
    for grid_manifest in (
        sorted(phase_root.rglob("manifest.json"))
        if role is PartitionRole.SELECTION
        else ()
    ):
        # Grid manifests have sibling summary.json; QF-7 and QF-39 do not.
        if not (grid_manifest.parent / "summary.json").exists():
            continue
        parameter = inspect_study(
            StudyType.PARAMETER_STUDY,
            grid_manifest.parent,
            artifact_root=output_root.parent,
            canonical_metadata=inputs.dataset.metadata,
        )
        verify_artifacts(parameter.index, output_root.parent).require_valid()
        manifest_path = write_manifest(
            create_manifest(parameter, execution),
            output_root / "manifests",
            artifact_root=output_root.parent,
        )
        export_research_report(
            build_research_report(manifest_path, artifact_root=output_root.parent),
            output_root / "html",
        )
        for entry in parameter.index.entries:
            entries[entry.artifact_id] = entry
        for edge in parameter.index.relationships:
            edges[(edge.source_id, edge.relationship, edge.target_id)] = edge
    return tuple(entries.values()), tuple(edges.values())


def run_pre_holdout(
    inputs: SmokeInputs,
    output_root: Path,
    ledger: HoldoutLedger,
    execution: ExecutionProvenance,
) -> PrimitiveMapping:
    """Always stop for manual audit/review; no holdout evaluator is imported."""
    config, adapter = prepare_walk_forward(inputs, output_root / "walk-forward")
    study = WalkForwardStudy(config, adapter, output_root / "walk-forward")
    # Persist QF-39's exact producer definition before research so the permanent
    # QF-40 exposure ledger can reject a consumed/overlapping holdout up front.
    write_record(
        study.study_path / "manifest.json",
        study._identity.to_primitive(),  # pyright: ignore[reportPrivateUsage]
        immutable=True,
    )
    if (
        ledger.reserve(load_oos_source(config.plan, study.study_path)).state
        != "reserved_unconsumed"
    ):
        raise ValueError("holdout already consumed; pre-holdout research is closed")
    # The shortened plan must have its own identities; never discover/copy old work.
    print(f"Comparison plan {config.plan.plan_id}; study {study.study_id}", flush=True)
    fixed_config, fixed_adapter = prepare_walk_forward(
        inputs, output_root / "fixed", fixed=True
    )
    print("STAGE fixed: 8/48/daily50 on the selection window", flush=True)
    fixed_adapter.validate(fixed_config.plan)
    try:
        fixed_adapter.select(fixed_config, 0, output_root / "fixed")
    except WalkForwardError as error:
        # QF-39 uses this error for both failed and unrankable QF-32 trials.
        # Only a succeeded fixed trial may proceed to the normal QF-9 checks;
        # a finalized compact window alone does not prove analysis succeeded.
        if str(error) != "no eligible candidate under the declared selection policy":
            raise
        trial_paths = tuple((output_root / "fixed").glob("*/trials/*.json"))
        if len(trial_paths) != 1:
            raise
        trial = PredictionGridTrialRecord.from_primitive(
            mapping(json.loads(trial_paths[0].read_text(encoding="utf-8")))
        )
        if trial.status is not TrialStatus.SUCCEEDED or trial.analysis is None:
            raise
    fixed_entries, fixed_edges = inspect_phase(
        inputs,
        fixed_config,
        fixed_adapter,
        output_root / "fixed",
        output_root,
        execution,
    )
    print(
        "STAGE fixed verified: QF-9 integrity and QF-7/29 candidate replay", flush=True
    )
    print("STAGE comparison: three trials, freeze, then OOS", flush=True)
    result = (
        study.resume() if (study.study_path / "manifest.json").exists() else study.run()
    )
    if len(result.folds) != 1 or result.folds[0].status is not FoldStatus.COMPLETED:
        raise ValueError("QF-45 comparison did not complete; preserve failure evidence")
    selection_entries, selection_edges = inspect_phase(
        inputs,
        config,
        adapter,
        study.study_path / "folds" / result.folds[0].fold_id / "selection",
        output_root,
        execution,
    )
    oos_entries, oos_edges = inspect_phase(
        inputs,
        config,
        adapter,
        study.study_path,
        output_root,
        execution,
        role=PartitionRole.WALK_FORWARD_TEST,
    )
    all_entries = {
        e.artifact_id: e for e in (*fixed_entries, *selection_entries, *oos_entries)
    }
    all_edges = tuple(dict.fromkeys((*fixed_edges, *selection_edges, *oos_edges)))
    print("STAGE publish: OOS aggregate, manifest, reserved report", flush=True)
    result_paths = publish_pre_holdout(
        config,
        study.study_path,
        output_root,
        ledger,
        execution,
        additional_entries=tuple(all_entries.values()),
        additional_relationships=all_edges,
    )
    write_record(
        output_root / "pre-holdout-complete.json", result_paths, immutable=True
    )
    print(
        "PRE_HOLDOUT_COMPLETE: RESERVED / UNCONSUMED; manual gate review required",
        flush=True,
    )
    return result_paths


def execution_for_run(output_root: Path, repository: Path) -> ExecutionProvenance:
    """Capture clean code once and reject mixed-code resume, using QF-9 provenance."""
    from quantforge.experiments import capture_code_provenance

    path = output_root / "execution.json"
    code = capture_code_provenance(repository)
    if path.exists():
        saved = read_record(path)
        if saved["code"] != code.to_primitive():
            raise ValueError(
                "resume requires the original clean code/dependency environment"
            )
        return ExecutionProvenance(
            cast(str, saved["run_id"]),
            datetime.fromisoformat(cast(str, saved["created_at"])),
            code,
            datetime.fromisoformat(cast(str, saved["execution_started_at"])),
        )
    now = datetime.now(UTC)
    execution = ExecutionProvenance(f"qf45-{now.isoformat()}", now, code, now)
    write_record(path, execution.to_primitive(), immutable=True)
    return execution
