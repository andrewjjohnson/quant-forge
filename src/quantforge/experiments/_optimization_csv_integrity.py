"""Reconcile native QF-6 CSV bytes with validated stored JSON, without research."""

from dataclasses import fields
from hashlib import sha256
from pathlib import Path
from typing import cast

from quantforge.configuration import PrimitiveMapping
from quantforge.experiments._json import ManifestError, mapping, text
from quantforge.experiments._producer_snapshot import ProducerReadSet
from quantforge.experiments._stability_integrity import validate_parameter_summaries
from quantforge.experiments.artifacts import ArtifactEntry, local_path
from quantforge.optimization.export import (
    _TRIAL_FIELDS,  # pyright: ignore[reportPrivateUsage]
    _csv_text,  # pyright: ignore[reportPrivateUsage]
    _trial_export_row,  # pyright: ignore[reportPrivateUsage]
)
from quantforge.optimization.models import (
    IneligibleTrial,
    ParameterSummary,
    RankedTrial,
    StabilitySummary,
    TrialRecord,
)

OPTIMIZATION_CSV_NAMES = frozenset(
    {
        "trials.csv",
        "failures.csv",
        "exclusions.csv",
        "eligible_rankings.csv",
        "ineligible_trials.csv",
        "stability.csv",
        "parameter_summary.csv",
    }
)


def validate_optimization_csv(
    source: Path,
    root: Path,
    entries: list[ArtifactEntry],
    trials: list[PrimitiveMapping],
    summary: PrimitiveMapping,
    summaries: dict[str, PrimitiveMapping],
    reads: ProducerReadSet,
) -> None:
    """Project saved records with QF-6's serializer; never rank or compute metrics."""
    trial_rows = [
        _trial_export_row(TrialRecord.from_primitive(trial))
        for trial in sorted(
            trials, key=lambda item: cast(int, item["combination_index"])
        )
    ]
    by_id = {text(trial["trial_id"]): trial for trial in trial_rows}

    def rows_with_trials(
        records: object, columns: tuple[str, ...]
    ) -> list[PrimitiveMapping]:
        return [
            {
                **(row := mapping(record)),
                **{key: by_id[text(row["trial_id"])][key] for key in columns},
            }
            for record in cast(list[object], records)
        ]

    ranking_rows = rows_with_trials(
        summaries["ranking.json"]["eligible_rankings"],
        ("parameters", "strategy_parameters", "metrics", "qf5_run_id"),
    )
    ineligible_rows = rows_with_trials(
        summaries["ranking.json"]["ineligible_trials"], ("parameters", "metrics")
    )
    ineligible_rows.sort(key=lambda row: text(row["combination_id"]))
    stability_rows = rows_with_trials(
        summaries["stability.json"]["summaries"], ("parameters",)
    )
    stability_rows.sort(
        key=lambda row: (
            cast(int, row["stability_rank"] or 0),
            text(row["combination_id"]),
        )
    )
    tables = {
        "trials.csv": (trial_rows, _TRIAL_FIELDS),
        "failures.csv": (
            [row for row in trial_rows if row["status"] == "failed"],
            _TRIAL_FIELDS,
        ),
        "exclusions.csv": (
            [row for row in trial_rows if row["status"] == "excluded"],
            _TRIAL_FIELDS,
        ),
        "eligible_rankings.csv": (
            ranking_rows,
            (
                *(field.name for field in fields(RankedTrial)),
                "parameters",
                "strategy_parameters",
                "metrics",
                "qf5_run_id",
            ),
        ),
        "ineligible_trials.csv": (
            ineligible_rows,
            (
                *(field.name for field in fields(IneligibleTrial)),
                "parameters",
                "metrics",
            ),
        ),
        "stability.csv": (
            stability_rows,
            (*(field.name for field in fields(StabilitySummary)), "parameters"),
        ),
        "parameter_summary.csv": (
            validate_parameter_summaries(summary.get("parameter_summaries")),
            tuple(field.name for field in fields(ParameterSummary)),
        ),
    }
    indexed = {local_path(root, entry.path): entry for entry in entries}
    for name, (rows, headers) in tables.items():
        content = _csv_text(rows, headers).encode("utf-8")
        path = source / name
        entry = indexed.get(path)
        if entry is None or entry.sha256 != sha256(content).hexdigest():
            raise ManifestError(f"optimization CSV {name} differs from saved records")
        reads.expect(path, content)
