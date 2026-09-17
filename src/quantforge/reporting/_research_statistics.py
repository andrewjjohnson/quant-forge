"""Known producer statistic records used by count and data-quality warnings."""

from collections.abc import Iterator

from quantforge.configuration import Primitive, PrimitiveMapping
from quantforge.experiments import ArtifactType, ExperimentManifest
from quantforge.reporting._research_inputs import artifact_value, as_mapping
from quantforge.reporting.research_models import ReportArtifact

STATISTIC_CONTAINERS = frozenset(
    {
        "analysis",
        "summary",
        "metrics",
        "performance",
        "counts",
        "record_counts",
        "trial_counts",
        "combination_counts",
        "completeness",
        "coverage",
        "data_quality",
        "market_data",
        "dataset",
        "prediction_dataset",
        "manifest",
        "result",
        "artifact",
        "prediction_study",
        "prediction_window",
        "selection_evidence",
        "rankings",
        "eligible_rankings",
        "ineligible_trials",
        "top_objective_trials",
        "group_summaries",
        "bin_summaries",
    }
)
STATISTIC_ARTIFACTS = frozenset(
    {
        ArtifactType.PREDICTION_RESULT,
        ArtifactType.FEATURE_DATASET,
        ArtifactType.OUTCOME_LABELS,
        ArtifactType.TRIAL_RESULT,
        ArtifactType.PARAMETER_SUMMARY,
        ArtifactType.BACKTEST_RESULT,
        ArtifactType.WALK_FORWARD_WINDOW,
        ArtifactType.OOS_AGGREGATE,
        ArtifactType.HOLDOUT_RESULT,
        ArtifactType.FROZEN_SELECTION,
        ArtifactType.SOURCE_DATASET,
        ArtifactType.DATA_QUALITY,
    }
)


def statistic_records(
    value: Primitive, source: str
) -> Iterator[tuple[str, PrimitiveMapping]]:
    """Follow only statistic containers, never arbitrary configuration mappings."""
    if isinstance(value, dict):
        yield source, value
        for key in sorted(STATISTIC_CONTAINERS.intersection(value)):
            yield from statistic_records(value[key], source + "/" + key)
    elif isinstance(value, list):
        for index, record in enumerate(value):
            yield from statistic_records(record, source + f"/{index}")


def statistic_evidence(
    manifest: ExperimentManifest, artifacts: tuple[ReportArtifact, ...]
) -> Iterator[tuple[str, PrimitiveMapping]]:
    yield from statistic_records(
        manifest.provenance.observations.to_primitive(), "manifest/counts"
    )
    configuration = manifest.provenance.configuration.to_primitive()
    for key in ("market_data", "dataset", "prediction_dataset"):
        yield from statistic_records(
            configuration.get(key), "manifest/configuration/" + key
        )
    for item in artifacts:
        if item.content is None:
            continue
        value = artifact_value(item)
        source = item.entry.artifact_id
        if item.entry.artifact_type in STATISTIC_ARTIFACTS:
            yield from statistic_records(value, source)
        elif item.entry.artifact_type is ArtifactType.CONFIGURATION:
            # QF-9 indexes native producer manifests as CONFIGURATION. Only these
            # producer-owned members carry statistics; configuration is arbitrary.
            for key in (
                "record_counts",
                "performance",
                "market_data",
                "combination_counts",
            ):
                yield from statistic_records(
                    as_mapping(value).get(key), source + "/" + key
                )
