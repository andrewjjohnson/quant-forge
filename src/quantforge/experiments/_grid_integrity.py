"""Compare existing grid records; never execute trials or calculate metrics."""

from collections import Counter

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.experiments._json import ManifestError, mapping, text
from quantforge.experiments.models import StudyType
from quantforge.optimization.models import TrialStatus


def validate_trial_status(study_type: StudyType, trial: PrimitiveMapping) -> None:
    """Require diagnostic/exclusion context and reject contradictory outcomes."""
    try:
        status = TrialStatus(text(trial.get("status")))
    except ValueError as error:
        raise ManifestError("invalid trial status context") from error
    failure_fields = ("failure_type", "failure_message")
    result_fields = ("artifact_location", "analysis", "artifact_fingerprint")
    if study_type is StudyType.OPTIMIZATION:
        failure_fields += ("failure_category",)
        result_fields = ("artifact_location", "metrics", "qf5_run_id")
    exclusion_fields = ("exclusion_code", "exclusion_reason")
    required = (
        failure_fields
        if status is TrialStatus.FAILED
        else exclusion_fields
        if status is TrialStatus.EXCLUDED
        else ()
    )
    for field in required:
        value = trial.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ManifestError("trial has incomplete diagnostic or exclusion context")
    prohibited = (
        (() if status is TrialStatus.FAILED else failure_fields)
        + (() if status is TrialStatus.EXCLUDED else exclusion_fields)
        + (() if status is TrialStatus.SUCCEEDED else result_fields)
    )
    if any(trial.get(field) is not None for field in prohibited):
        raise ManifestError("trial status contradicts its outcome context")


def validate_trial_counts(
    study_type: StudyType, summary: PrimitiveMapping, statuses: list[str]
) -> None:
    """Reconcile persisted trial coverage with the producer's summary."""
    counts = mapping(summary.get("counts"))
    observed = Counter(statuses)
    if set(observed) - {status.value for status in TrialStatus}:
        raise ManifestError("trial status is incompatible with grid summary")
    expected = {
        "failed": observed[TrialStatus.FAILED],
        "excluded": observed[TrialStatus.EXCLUDED],
    }
    if study_type is StudyType.PARAMETER_STUDY:
        expected.update(trials=len(statuses), succeeded=observed[TrialStatus.SUCCEEDED])
        if observed[TrialStatus.PENDING] or observed[TrialStatus.RUNNING]:
            raise ManifestError("unfinished trial is incompatible with grid summary")
    else:
        expected.update(
            recorded_trials=len(statuses),
            successful=observed[TrialStatus.SUCCEEDED],
            pending_or_running=observed[TrialStatus.PENDING]
            + observed[TrialStatus.RUNNING],
        )
    if any(
        type(counts.get(key)) is not int or counts[key] != count
        for key, count in expected.items()
    ):
        raise ManifestError("persisted trial counts do not match grid summary")


def validate_backtest_trial(
    trial: PrimitiveMapping,
    backtest: PrimitiveMapping,
    study_configuration: PrimitiveMapping,
) -> None:
    """Match QF-6 records to QF-5's recorded provenance and metrics.

    Use the grid's historical engine/schema versions, not today's defaults.
    The caller verifies the original QF-5 file integrity before this comparison.
    """
    strategy = mapping(backtest.get("strategy"))
    configuration = mapping(strategy.get("configuration"))
    run_id = text(backtest.get("run_id"))
    expected: PrimitiveMapping = {
        "qf5_run_id": run_id,
        "artifact_location": f"backtests/{text(trial.get('trial_id'))}/{run_id}",
        "dataset": mapping(backtest.get("market_data")),
        "backtest_configuration": mapping(backtest.get("backtest_configuration")),
        "metrics": mapping(backtest.get("performance")),
        "strategy_name": text(strategy.get("strategy_id")),
        "strategy_version": text(strategy.get("strategy_implementation_version")),
        "strategy_configuration_id": configuration_identity(configuration),
        "strategy_parameters": mapping(configuration.get("parameters")),
    }
    if (
        any(trial.get(key) != value for key, value in expected.items())
        or strategy.get("strategy_configuration_id")
        != expected["strategy_configuration_id"]
        or backtest.get("engine_version")
        != study_configuration.get("qf5_engine_version")
        or backtest.get("result_schema_version")
        != study_configuration.get("qf5_result_schema_version")
    ):
        raise ManifestError("trial provenance does not match its linked QF-5 artifact")
