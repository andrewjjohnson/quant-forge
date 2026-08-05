"""Deterministic study-level JSON and CSV exports."""

import csv
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from quantforge.configuration import Primitive, PrimitiveMapping
from quantforge.optimization.errors import StudyPersistenceError
from quantforge.optimization.models import StudyResult, TrialRecord


def _json_text(value: Primitive) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise StudyPersistenceError(
            "optimization export contains a nonfinite or unsupported value"
        ) from error


def _atomic_text(path: Path, content: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=str(path.parent), text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise StudyPersistenceError(
            f"failed to write optimization export: {path}"
        ) from error


def _atomic_json(path: Path, value: PrimitiveMapping) -> None:
    content = (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    _atomic_text(path, content)


def _csv_text(rows: Sequence[PrimitiveMapping], fieldnames: tuple[str, ...]) -> str:
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                field: (
                    _json_text(value)
                    if isinstance((value := row.get(field)), (dict, list))
                    else value
                )
                for field in fieldnames
            }
        )
    return stream.getvalue()


def _write_csv(
    path: Path, rows: Sequence[PrimitiveMapping], fieldnames: tuple[str, ...]
) -> None:
    _atomic_text(path, _csv_text(rows, fieldnames))


def _trial_export_row(trial: TrialRecord) -> PrimitiveMapping:
    dataset = trial.dataset
    return {
        "combination_index": trial.combination_index,
        "trial_id": trial.trial_id,
        "combination_id": trial.combination_id,
        "status": trial.status.value,
        "parameters": trial.parameters,
        "strategy_parameters": trial.strategy_parameters,
        "strategy_name": trial.strategy_name,
        "strategy_version": trial.strategy_version,
        "strategy_configuration_id": trial.strategy_configuration_id,
        "dataset_id": dataset.get("dataset_id"),
        "data_fingerprint": dataset.get("bars_fingerprint", dataset.get("data_sha256")),
        "canonical_symbol": dataset.get("canonical_symbol"),
        "provider_name": dataset.get("provider_name"),
        "requested_start": dataset.get("requested_start"),
        "requested_end": dataset.get("requested_end"),
        "adjustment_mode": dataset.get("adjustment_mode"),
        "backtest_configuration": trial.backtest_configuration,
        "metrics": trial.metrics,
        "qf5_run_id": trial.qf5_run_id,
        "artifact_location": trial.artifact_location,
        "failure_category": trial.failure_category,
        "failure_type": trial.failure_type,
        "failure_message": trial.failure_message,
        "exclusion_code": trial.exclusion_code,
        "exclusion_reason": trial.exclusion_reason,
        "started_at": trial.started_at,
        "finished_at": trial.finished_at,
        "trial_schema_version": trial.schema_version,
    }


_TRIAL_FIELDS = (
    "combination_index",
    "trial_id",
    "combination_id",
    "status",
    "parameters",
    "strategy_parameters",
    "strategy_name",
    "strategy_version",
    "strategy_configuration_id",
    "dataset_id",
    "data_fingerprint",
    "canonical_symbol",
    "provider_name",
    "requested_start",
    "requested_end",
    "adjustment_mode",
    "backtest_configuration",
    "metrics",
    "qf5_run_id",
    "artifact_location",
    "failure_category",
    "failure_type",
    "failure_message",
    "exclusion_code",
    "exclusion_reason",
    "started_at",
    "finished_at",
    "trial_schema_version",
)


def export_study_result(
    result: StudyResult,
    study_path: Path,
    expected_manifest: PrimitiveMapping,
    ranking_configuration: PrimitiveMapping,
    stability_configuration: PrimitiveMapping,
) -> Path:
    """Regenerate complete study exports after verifying manifest compatibility."""
    manifest_path = study_path / "manifest.json"
    try:
        existing: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StudyPersistenceError(
            "cannot export without a valid study manifest"
        ) from error
    if existing != expected_manifest or result.study_id != expected_manifest.get(
        "study_id"
    ):
        raise StudyPersistenceError(
            "refusing to overwrite exports for an incompatible study manifest"
        )

    trials_by_id = {trial.trial_id: trial for trial in result.trials}
    trial_rows = [_trial_export_row(trial) for trial in result.trials]
    ranking_rows: list[PrimitiveMapping] = []
    for ranked in result.rankings:
        trial = trials_by_id[ranked.trial_id]
        ranking_rows.append(
            {
                **ranked.to_primitive(),
                "parameters": trial.parameters,
                "strategy_parameters": trial.strategy_parameters,
                "metrics": trial.metrics,
                "qf5_run_id": trial.qf5_run_id,
            }
        )
    ineligible_rows: list[PrimitiveMapping] = []
    for item in sorted(result.ineligible_trials, key=lambda row: row.combination_id):
        trial = trials_by_id[item.trial_id]
        ineligible_rows.append(
            {
                **item.to_primitive(),
                "parameters": trial.parameters,
                "metrics": trial.metrics,
            }
        )
    failure_rows = [_trial_export_row(trial) for trial in result.failed_trials]
    exclusion_rows = [_trial_export_row(trial) for trial in result.excluded_trials]
    stability_rows: list[PrimitiveMapping] = []
    for summary in sorted(
        result.stability,
        key=lambda item: (item.stability_rank or 0, item.combination_id),
    ):
        trial = trials_by_id[summary.trial_id]
        stability_rows.append(
            {**summary.to_primitive(), "parameters": trial.parameters}
        )
    parameter_rows = [item.to_primitive() for item in result.parameter_summaries]

    _write_csv(study_path / "trials.csv", trial_rows, _TRIAL_FIELDS)
    _write_csv(
        study_path / "eligible_rankings.csv",
        ranking_rows,
        (
            "rank",
            "trial_id",
            "combination_id",
            "objective_metric",
            "objective_value",
            "parameters",
            "strategy_parameters",
            "metrics",
            "qf5_run_id",
        ),
    )
    _write_csv(
        study_path / "ineligible_trials.csv",
        ineligible_rows,
        ("trial_id", "combination_id", "reasons", "parameters", "metrics"),
    )
    _write_csv(study_path / "failures.csv", failure_rows, _TRIAL_FIELDS)
    _write_csv(study_path / "exclusions.csv", exclusion_rows, _TRIAL_FIELDS)
    stability_fields = (
        tuple(result.stability[0].to_primitive())
        if result.stability
        else (
            "trial_id",
            "combination_id",
            "objective_rank",
            "objective_value",
            "valid_neighbor_count",
            "excluded_neighbor_count",
            "successful_eligible_neighbor_count",
            "neighbor_objective_values",
            "mean_neighbor_objective",
            "median_neighbor_objective",
            "worst_neighbor_objective",
            "objective_standard_deviation",
            "constraint_pass_fraction",
            "center_to_neighbor_difference",
            "relative_center_to_neighbor_difference",
            "is_boundary",
            "stability_score",
            "classification",
            "is_isolated_peak",
            "isolation_reason",
            "stability_rank",
        )
    )
    _write_csv(
        study_path / "stability.csv",
        stability_rows,
        (*stability_fields, "parameters"),
    )
    _write_csv(
        study_path / "parameter_summary.csv",
        parameter_rows,
        (
            "parameter_name",
            "parameter_value",
            "successful_count",
            "eligible_count",
            "constraint_pass_fraction",
            "mean_objective",
            "median_objective",
            "best_objective",
        ),
    )
    _atomic_json(
        study_path / "ranking.json",
        {
            "study_id": result.study_id,
            "configuration": ranking_configuration,
            "eligible_rankings": [item.to_primitive() for item in result.rankings],
            "ineligible_trials": [
                item.to_primitive() for item in result.ineligible_trials
            ],
        },
    )
    _atomic_json(
        study_path / "stability.json",
        {
            "study_id": result.study_id,
            "configuration": stability_configuration,
            "summaries": [item.to_primitive() for item in result.stability],
        },
    )
    _atomic_json(
        study_path / "summary.json",
        {
            **result.summary_primitive(),
            "ranking_configuration": ranking_configuration,
            "stability_configuration": stability_configuration,
            "top_objective_trials": [
                item.to_primitive() for item in result.rankings[:10]
            ],
            "top_stability_trials": [
                item.to_primitive() for item in result.stability[:10]
            ],
            "parameter_summaries": cast(list[Primitive], parameter_rows),
        },
    )
    return study_path
