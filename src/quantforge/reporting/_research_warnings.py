"""Deterministic conditions over explicit producer metadata, never prose guesses."""

from collections.abc import Iterator
from decimal import Decimal, InvalidOperation

from quantforge.configuration import Primitive, PrimitiveMapping
from quantforge.experiments import ArtifactType, ExperimentManifest, StudyType
from quantforge.reporting._research_inputs import (
    artifact_value,
    as_mapping,
    lineage_artifacts,
    validation_observations,
)
from quantforge.reporting._research_statistics import statistic_evidence
from quantforge.reporting.research_models import (
    ReportArtifact,
    ResearchReportConfig,
    ResearchWarning,
)


def number(value: Primitive) -> Decimal | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        result = Decimal(str(value))
        return result if result.is_finite() else None
    except InvalidOperation:
        return None


def stability_records(
    artifact: ReportArtifact,
) -> Iterator[tuple[str, PrimitiveMapping]]:
    """Read explicit assessment fields, never nested strategy parameters."""
    value = artifact_value(artifact)
    source = artifact.entry.artifact_id
    category = artifact.entry.artifact_type
    candidates: list[tuple[str, Primitive]] = []
    if category is ArtifactType.CONFIGURATION_STABILITY:
        candidates.append((source, value))
    elif category is ArtifactType.OOS_AGGREGATE:
        candidates.append((source + "/stability", as_mapping(value).get("stability")))
    elif category is ArtifactType.PARAMETER_SUMMARY:
        for path, summary in (
            (source, value),
            (source + "/summary", as_mapping(value).get("summary")),
        ):
            candidates.append((path, summary))
            for key in ("stability", "summaries", "top_stability_trials"):
                candidates.append((path + "/" + key, as_mapping(summary).get(key)))
    for path, candidate in candidates:
        if isinstance(candidate, dict):
            yield path, candidate
        elif isinstance(candidate, list):
            for index, record in enumerate(candidate):
                if isinstance(record, dict):
                    yield path + f"/{index}", record


def build_warnings(
    manifest: ExperimentManifest,
    artifacts: tuple[ReportArtifact, ...],
    holdout: PrimitiveMapping,
    config: ResearchReportConfig,
) -> tuple[ResearchWarning, ...]:
    found: dict[tuple[str, str], ResearchWarning] = {}

    def add(code: str, message: str, source: str) -> None:
        found[(code, source)] = ResearchWarning(code, message, source)

    for item in artifacts:
        if item.status != "verified":
            add("ARTIFACT_INTEGRITY", item.status, item.entry.artifact_id)
    artifacts = lineage_artifacts(manifest, artifacts)
    for item in artifacts:
        if item.status != "verified" and item.entry.artifact_type in {
            ArtifactType.WALK_FORWARD_WINDOW,
            ArtifactType.FOLD_STATE,
            ArtifactType.OOS_AGGREGATE,
        }:
            add(
                "FAILED_OR_MISSING_OOS_WINDOWS",
                "Indexed OOS window or aggregate evidence is missing or invalid.",
                item.entry.artifact_id,
            )
    if holdout["state"] == "consumed":
        add(
            "HOLDOUT_ALREADY_CONSUMED",
            "Final holdout has been consumed; it cannot be treated as unseen.",
            "QF-40",
        )
    elif holdout["state"] == "unavailable":
        add(
            "HOLDOUT_STATE_UNAVAILABLE",
            "Current holdout state is unavailable.",
            "QF-40",
        )
    valid_oos = any(
        item.status == "verified"
        and item.entry.artifact_type
        in {
            ArtifactType.OOS_AGGREGATE,
            ArtifactType.WALK_FORWARD_WINDOW,
            ArtifactType.HOLDOUT_RESULT,
        }
        for item in artifacts
    )
    if not valid_oos:
        add(
            "IN_SAMPLE_ONLY",
            "No verified OOS or consumed-holdout result is linked. "
            "Search/standalone results do not establish out-of-sample performance.",
            "manifest",
        )

    configuration = manifest.provenance.configuration.to_primitive()
    minimum = config.minimum_sample_size
    if minimum is None:
        recorded = as_mapping(configuration.get("ranking")).get(
            "minimum_prediction_count"
        )
        if type(recorded) is int:
            minimum = recorded
    for location, record in statistic_evidence(manifest, artifacts):
        if minimum is not None:
            for key in (
                "prediction_count",
                "labeled_rows",
                "trade_count",
                "candidate_count",
            ):
                count = number(record.get(key))
                if count is not None and count < minimum:
                    add(
                        "LOW_SAMPLE_SIZE",
                        f"{key}={count} is below the declared minimum {minimum}.",
                        location + "/" + key,
                    )
        if config.high_trial_count is not None:
            for key in (
                "trials",
                "trial_count",
                "total_combinations",
                "total_cartesian_combinations",
            ):
                count = number(record.get(key))
                if count is not None and count >= config.high_trial_count:
                    add(
                        "HIGH_TRIAL_COUNT",
                        f"{key}={count} reaches the configured "
                        f"threshold {config.high_trial_count}.",
                        location + "/" + key,
                    )
        if (
            any(
                isinstance(record.get(key), list) and bool(record[key])
                for key in (
                    "missing_sessions",
                    "missing_intervals",
                    "incomplete_sessions",
                    "missing_expected_intervals",
                )
            )
            or record.get("coverage_complete") is False
        ):
            add(
                "INCOMPLETE_DATA_COVERAGE",
                "Source metadata records incomplete market-data coverage.",
                location,
            )
        if "expected_windows" in record and record.get("complete") is False:
            add(
                "FAILED_OR_MISSING_OOS_WINDOWS",
                "OOS aggregate covers only the "
                "reported completed windows; missing windows are not zero returns.",
                location,
            )
    for item in artifacts:
        for location, record in stability_records(item):
            if (
                record.get("classification") == "fragile"
                or record.get("is_isolated_peak") is True
            ):
                add(
                    "PARAMETER_INSTABILITY",
                    "Producer classified this configuration "
                    "as fragile or an isolated peak.",
                    location,
                )
            frequency = number(record.get("configuration_change_frequency"))
            maximum = config.maximum_configuration_change_frequency
            if maximum is not None and frequency is not None and frequency > maximum:
                add(
                    "PARAMETER_INSTABILITY",
                    f"Configuration change frequency {frequency} "
                    f"exceeds configured maximum {maximum}.",
                    location,
                )
    folds = validation_observations(manifest).get("folds")
    if isinstance(folds, list) and any(
        as_mapping(fold).get("status") != "completed" for fold in folds
    ):
        add(
            "FAILED_OR_MISSING_OOS_WINDOWS",
            "At least one planned fold is incomplete "
            "or failed; see persisted fold status.",
            "manifest/folds",
        )
    if manifest.provenance.study_type in {StudyType.BACKTEST, StudyType.OPTIMIZATION}:
        execution = as_mapping(configuration.get("backtest_configuration"))
        # QF-6 keeps the same QF-5 definition inside its identity inputs.
        if not execution:
            execution = as_mapping(configuration.get("backtest"))
        missing = [
            key
            for key in ("commission", "fees", "slippage")
            if execution.get(key) is None
        ]
        if missing:
            add(
                "MISSING_TRANSACTION_COST_ASSUMPTIONS",
                "Unavailable execution assumptions: " + ", ".join(missing) + ".",
                "manifest/configuration",
            )
    return tuple(found[key] for key in sorted(found))
