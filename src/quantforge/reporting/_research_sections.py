"""Explicit study-family views over already-persisted values."""

from pathlib import PurePosixPath

from quantforge.configuration import Primitive, PrimitiveMapping
from quantforge.experiments import (
    ArtifactFormat,
    ArtifactType,
    ExperimentManifest,
    RelationshipType,
    StudyType,
)
from quantforge.experiments._json import snapshot
from quantforge.reporting._research_inputs import (
    artifact_value,
    as_mapping,
    validation_observations,
)
from quantforge.reporting.research_models import (
    ReportArtifact,
    ReportPhase,
    ReportSection,
)

PREDICTION_FIELDS = (
    "prediction_count",
    "prediction_frequency",
    "direction_distribution",
    "accuracy",
    "accuracy_interval",
    "matched_baseline",
    "signed_outcome",
    "average_signed_outcome",
    "median_signed_outcome",
    "mfe",
    "mae",
    "event_outcomes",
    "period_comparisons",
    "weekday_comparisons",
    "matched_baseline_comparisons",
    "annual_summary",
    "regime_summary",
)


def primary_phase(manifest: ExperimentManifest) -> ReportPhase:
    entries = {item.artifact_id: item for item in manifest.artifacts.entries}
    for edge in manifest.artifacts.relationships:
        target = entries[edge.target_id]
        if (
            edge.relationship is RelationshipType.VALIDATES
            and target.artifact_type is ArtifactType.CONFIGURATION
            and target.producer_study_id == manifest.provenance.producer_study_id
        ):
            category = entries[edge.source_id].artifact_type
            if category is ArtifactType.HOLDOUT_RESULT:
                return ReportPhase.HOLDOUT
            if category is ArtifactType.WALK_FORWARD_WINDOW:
                return ReportPhase.OOS
    return ReportPhase.IN_SAMPLE


def build_sections(
    manifest: ExperimentManifest,
    artifacts: tuple[ReportArtifact, ...],
    holdout: PrimitiveMapping,
) -> tuple[ReportSection, ...]:
    sections: list[ReportSection] = []
    study_type = manifest.provenance.study_type
    configuration = manifest.provenance.configuration.to_primitive()
    observed = manifest.provenance.observations.to_primitive()
    phase = primary_phase(manifest)

    def add(
        title: str,
        values: Primitive,
        role: ReportPhase = phase,
        sources: tuple[str, ...] = (),
    ) -> None:
        sections.append(
            ReportSection(title, role, snapshot({"value": values}), sources)
        )

    def selected(*categories: ArtifactType) -> list[ReportArtifact]:
        return [
            item
            for item in artifacts
            if item.entry.artifact_type in categories and item.content is not None
        ]

    def show(
        title: str,
        categories: tuple[ArtifactType, ...],
        fields: tuple[str, ...] = (),
        role: ReportPhase = phase,
    ) -> None:
        found = False
        for item in selected(*categories):
            value = artifact_value(item)
            if fields:
                record = as_mapping(value)
                value = {key: record[key] for key in fields if key in record}
                if not value:
                    continue
            found = True
            add(title, value, role, (item.entry.artifact_id,))
        if not found:
            add(title, None, role)

    if study_type in {StudyType.PREDICTION, StudyType.PREDICTION_WINDOW}:
        add("Prediction counts", observed.get("record_counts"))
        add("Rule, outcome and indicator context", configuration)
        show(
            "Prediction metrics and comparisons",
            (ArtifactType.PREDICTION_RESULT,),
            ("analysis", "summary", "metrics", *PREDICTION_FIELDS),
        )
        add(
            "Prediction metric availability",
            {
                "policy": "Only published summaries are shown. Accuracy, confidence "
                "intervals, signed outcomes, MFE/MAE, event rates and segmented "
                "comparisons are unavailable when absent from indexed artifacts.",
                "execution_metrics": "Not applicable to prediction studies",
            },
        )
    elif study_type is StudyType.FEATURE_DATASET:
        add("Candidate dispositions and row count", observed.get("record_counts"))
        show("Feature and outcome schema", (ArtifactType.FEATURE_SCHEMA,))
        add(
            "Feature namespaces, outcome definitions and context provenance",
            configuration,
        )
        show(
            "Published feature / outcome distributions",
            (ArtifactType.FEATURE_DATASET, ArtifactType.OUTCOME_LABELS),
            (
                "summary",
                "distributions",
                "comparisons",
                "outcome_distributions",
                "group_summaries",
                "bin_summaries",
                "record_counts",
                "warning",
            ),
        )
        add("Related prediction studies", observed.get("prediction_study_ids"))
    elif study_type in {StudyType.PARAMETER_STUDY, StudyType.OPTIMIZATION}:
        add("Search space, constraints and minimum samples", configuration)
        add("Trial counts and execution status", observed)
        show(
            "Ranked configurations — published order",
            (ArtifactType.PARAMETER_SUMMARY,),
            (
                "rankings",
                "eligible_rankings",
                "ineligible_trials",
                "counts",
                "trial_counts",
                "best_objective",
                "best_stability",
                "recommended_robust",
                "recommendations",
                "best_objective_trial_id",
                "best_stability_trial_id",
                "recommended_robust_trial_id",
            ),
        )
        show(
            "Parameter neighborhoods and stability",
            (ArtifactType.PARAMETER_SUMMARY, ArtifactType.CONFIGURATION_STABILITY),
            ("stability", "parameter_summaries", "summaries", "recommendations"),
        )
        show(
            "Persisted trials and analysis",
            (ArtifactType.TRIAL_RESULT,),
            (
                "trial_id",
                "combination_id",
                "status",
                "parameters",
                "analysis",
                "failure",
                "error",
                "exclusion_reasons",
                "metrics",
                "performance",
                "backtest_configuration",
                "failure_type",
                "failure_category",
                "failure_message",
                "exclusion_code",
                "exclusion_reason",
            ),
        )
        add(
            "Search interpretation",
            "In-sample rankings are research comparisons; "
            "a top-ranked configuration is not a validated winner.",
        )

    if study_type is StudyType.BACKTEST:
        add("Strategy, capital, costs and execution assumptions", configuration)
        # QF-9's primary CONFIGURATION pointer is the exact native manifest.
        primary = [
            item
            for item in selected(
                ArtifactType.CONFIGURATION, ArtifactType.BACKTEST_RESULT
            )
            if item.entry.producer_study_id == manifest.provenance.producer_study_id
            and "performance" in as_mapping(artifact_value(item))
        ]
        # Directory exports may index the same manifest at only one category.
        seen: set[str] = set()
        for item in primary:
            if item.entry.path in seen:
                continue
            seen.add(item.entry.path)
            native = as_mapping(artifact_value(item))
            add(
                "Backtest performance and drawdown",
                native.get("performance"),
                sources=(item.entry.artifact_id,),
            )
            benchmark = as_mapping(native.get("benchmark"))
            add(
                "Benchmark comparison",
                {
                    key: benchmark.get(key)
                    for key in (
                        "benchmark_id",
                        "configuration",
                        "performance",
                        "dividend_accounting",
                    )
                },
                sources=(item.entry.artifact_id,),
            )
            add(
                "Trades and corporate-action accounting",
                {
                    key: native.get(key)
                    for key in (
                        "record_counts",
                        "corporate_action_accounting",
                        "warnings",
                        "limitations",
                    )
                },
                sources=(item.entry.artifact_id,),
            )
        if not primary:
            add("Backtest performance and drawdown", None)
            add("Benchmark comparison", None)
        for name, title in (
            ("equity.csv", "Equity and returns"),
            ("benchmark_equity.csv", "Benchmark equity"),
            ("trades.csv", "Trades"),
            ("fills.csv", "Fills"),
            ("positions.csv", "Positions"),
        ):
            matches = [
                item
                for item in selected(ArtifactType.BACKTEST_RESULT)
                if item.entry.producer_study_id == manifest.provenance.producer_study_id
                and PurePosixPath(item.entry.path).name == name
            ]
            if not matches:
                add(title, None)
            for item in matches:
                add(title, artifact_value(item), sources=(item.entry.artifact_id,))
        show(
            "Published monthly / period returns",
            (ArtifactType.BACKTEST_RESULT,),
            ("monthly_returns", "period_returns"),
        )

    validation = validation_observations(manifest)
    show(
        "Producer disclosures",
        (
            ArtifactType.CONFIGURATION,
            ArtifactType.PARAMETER_SUMMARY,
            ArtifactType.PREDICTION_RESULT,
            ArtifactType.FEATURE_DATASET,
        ),
        ("warnings", "limitations", "interpretation"),
    )
    show(
        "Validation plan and historical reservation definition",
        (ArtifactType.VALIDATION_PLAN,),
        role=ReportPhase.SELECTION,
    )
    show(
        "Frozen selections by fold",
        (ArtifactType.FROZEN_SELECTION,),
        (
            "fold_id",
            "selection_id",
            "selected_trial_id",
            "candidate",
            "membership",
            "selection_evidence",
            "selection_grid_study_id",
        ),
        ReportPhase.SELECTION,
    )
    add(
        "Fold coverage and validation lineage",
        {
            key: validation.get(key)
            for key in (
                "plan_id",
                "walk_forward_study_id",
                "lineage_id",
                "aggregate_id",
                "folds",
            )
        },
        ReportPhase.OOS,
    )
    show("Persisted fold status", (ArtifactType.FOLD_STATE,), role=ReportPhase.OOS)
    show(
        "Walk-forward OOS summary",
        (ArtifactType.OOS_AGGREGATE,),
        ("kind", "summary", "normalized_equity"),
        ReportPhase.OOS,
    )
    show(
        "Configuration turnover",
        (ArtifactType.CONFIGURATION_STABILITY,),
        role=ReportPhase.OOS,
    )
    # Native fold snapshots have different families and never enter search sections.
    for item in selected(ArtifactType.WALK_FORWARD_WINDOW):
        record = as_mapping(artifact_value(item))
        result = as_mapping(record.get("result"))
        native = as_mapping(result.get("manifest"))
        add(
            "Per-fold OOS result",
            {
                "result_id": record.get("result_id"),
                "selection_id": record.get("selection_id"),
                "kind": record.get("kind"),
                "record_counts": native.get("record_counts"),
                "performance": native.get("performance"),
                "summary": record.get("summary"),
            },
            ReportPhase.OOS,
            (item.entry.artifact_id,),
        )
    add("Final holdout state", holdout, ReportPhase.HOLDOUT)
    if holdout["state"] == "consumed":
        show(
            "Consumed final holdout result",
            (ArtifactType.HOLDOUT_RESULT,),
            ("kind", "request_id", "summary", "consumption_sha256"),
            ReportPhase.HOLDOUT,
        )

    add(
        "Code and execution provenance",
        manifest.execution.to_primitive(),
        ReportPhase.PROVENANCE,
    )
    add(
        "Research configuration and source lineage",
        configuration,
        ReportPhase.PROVENANCE,
    )
    # The current holdout section is authoritative; do not repeat stale state here.
    add(
        "Artifact relationships",
        manifest.artifacts.to_primitive()["relationships"],
        ReportPhase.PROVENANCE,
    )
    inspections = [
        item.entry.artifact_id
        for item in artifacts
        if item.entry.artifact_type in {ArtifactType.INSPECTION, ArtifactType.CHART}
    ]
    add(
        "Existing inspection reports and charts",
        None
        if not inspections
        else "Open the indexed artifacts below. Charts are not regenerated "
        "or embedded as raw HTML.",
        ReportPhase.PROVENANCE,
        tuple(inspections),
    )
    # Additional JSON summaries keep their own producer fields and schema; CSV/Parquet
    # and large row collections remain exact index links, not new statistics.
    for item in selected(ArtifactType.DATA_QUALITY):
        if item.entry.file_format is ArtifactFormat.JSON:
            add(
                "Source data quality",
                artifact_value(item),
                ReportPhase.PROVENANCE,
                (item.entry.artifact_id,),
            )
    return tuple(sections)
