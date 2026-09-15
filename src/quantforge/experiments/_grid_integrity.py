"""Compare existing grid records; never execute trials or calculate metrics."""

from collections import Counter

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.experiments._json import ManifestError, mapping, text
from quantforge.experiments._producer_integrity import validate_backtest_identity
from quantforge.experiments.models import StudyType
from quantforge.optimization.models import FailedTrialAttempt, TrialStatus
from quantforge.prediction.grid import (
    PredictionGridError,
    PredictionGridFailedAttempt,
    PredictionGridPersistenceError,
    PredictionTrialAnalysis,
)


def _parameters_at_grid_position(
    trial: PrimitiveMapping, study_configuration: PrimitiveMapping
) -> PrimitiveMapping:
    """Decode one saved Cartesian position without enumerating candidates."""
    index = trial.get("combination_index")
    axes = mapping(study_configuration.get("search_space")).get("parameters")
    if type(index) is not int or index < 0 or not isinstance(axes, list) or not axes:
        raise ManifestError("trial coordinates are invalid")
    remainder = index
    parameters: PrimitiveMapping = {}
    for raw_axis in reversed(axes):
        axis = mapping(raw_axis)
        name = text(axis.get("name"))
        values = axis.get("values")
        if name in parameters or not isinstance(values, list) or not values:
            raise ManifestError("trial coordinates have incompatible search axes")
        remainder, coordinate = divmod(remainder, len(values))
        parameters[name] = values[coordinate]
    if remainder or configuration_identity(parameters) != configuration_identity(
        mapping(trial.get("parameters"))
    ):
        raise ManifestError("trial parameters do not match grid coordinates")
    return parameters


def validate_prediction_trial_coordinates(
    trial: PrimitiveMapping, study_configuration: PrimitiveMapping
) -> None:
    """Bind QF-32 trial coordinates, identities and recorded component metadata."""
    parameters = _parameters_at_grid_position(trial, study_configuration)
    factory = mapping(study_configuration.get("study_factory"))
    schema_version = text(study_configuration.get("schema_version"))
    combination_id = configuration_identity(
        {
            "component": "quantforge_prediction_grid_combination",
            "schema_version": schema_version,
            "factory_name": text(factory.get("name")),
            "factory_version": text(factory.get("version")),
            "factory_configuration": mapping(factory.get("configuration")),
            "parameters": parameters,
        }
    )
    definition = (
        None
        if trial.get("status") == "excluded"
        else mapping(trial.get("trial_definition"))
    )
    identity: PrimitiveMapping = {
        "component": "quantforge_prediction_grid_trial",
        "schema_version": schema_version,
        "study_id": configuration_identity(study_configuration),
        "combination_id": combination_id,
        "dataset_family_fingerprint": study_configuration.get(
            "dataset_family_fingerprint"
        ),
        "indicator_backend": mapping(study_configuration.get("indicator_backend")),
        "trial_definition": definition,
    }
    expected: PrimitiveMapping = {
        **{key: value for key, value in identity.items() if key != "component"},
        "trial_id": configuration_identity(identity),
        "indicator_configuration_ids": []
        if definition is None
        else definition.get("indicator_configuration_ids"),
    }
    if any(trial.get(key) != value for key, value in expected.items()):
        raise ManifestError(
            "prediction trial identity or metadata differs from grid coordinates"
        )
    if definition is not None:
        from quantforge.experiments._prediction_trial_integrity import (
            frozen_prediction_components,
        )

        frozen_prediction_components(definition, study_configuration)


def validate_trial_coordinates(
    trial: PrimitiveMapping, study_configuration: PrimitiveMapping
) -> None:
    """Bind QF-6 coordinates and identities using saved metadata only.

    Never invoke a factory to resolve normalized strategy parameters.
    """
    parameters = _parameters_at_grid_position(trial, study_configuration)
    combination_id = configuration_identity(
        {
            "component": "quantforge_parameter_combination",
            "combination_schema_version": "1",
            "strategy_name": study_configuration.get("strategy_name"),
            "strategy_version": study_configuration.get("strategy_version"),
            "strategy_factory": mapping(study_configuration.get("strategy_factory")),
            "parameters": parameters,
        }
    )
    dataset = mapping(study_configuration.get("dataset"))
    trial_identity: PrimitiveMapping = {
        "component": "quantforge_optimization_trial",
        "study_id": configuration_identity(study_configuration),
        "combination_id": combination_id,
        "dataset_id": dataset.get("dataset_id"),
        "strategy_configuration_id": trial.get("strategy_configuration_id"),
        "strategy_parameters": mapping(trial.get("strategy_parameters")),
        **{
            key: study_configuration.get(key)
            for key in (
                "trial_schema_version",
                "strategy_name",
                "strategy_version",
                "backtest_configuration",
                "qf5_engine_version",
                "qf5_result_schema_version",
                "optimization_engine_version",
            )
        },
    }
    expected: PrimitiveMapping = {
        "combination_id": combination_id,
        "trial_id": configuration_identity(trial_identity),
        "schema_version": study_configuration.get("trial_schema_version"),
        **{
            key: trial_identity[key]
            for key in ("strategy_name", "strategy_version", "backtest_configuration")
        },
    }
    trial_dataset = mapping(trial.get("dataset"))
    if any(trial.get(key) != value for key, value in expected.items()) or any(
        trial_dataset.get(key) != value for key, value in dataset.items()
    ):
        raise ManifestError("trial identity is incompatible with grid coordinates")


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
    if study_type is StudyType.PARAMETER_STUDY and status is TrialStatus.SUCCEEDED:
        # A missing final summary does not relax the producer's completed-trial
        # contract. Deserialize stored analysis only; never invoke its analyzer.
        try:
            analysis = mapping(trial.get("analysis"))
            retained = PredictionTrialAnalysis.from_primitive(analysis).to_primitive()
        except (KeyError, TypeError, ValueError, PredictionGridError) as error:
            raise ManifestError(
                "successful prediction trial analysis is invalid"
            ) from error
        if configuration_identity(analysis) != configuration_identity(retained):
            raise ManifestError("successful prediction trial analysis is noncanonical")
    attempts = trial.get("failed_attempts", [])
    if not isinstance(attempts, list):
        raise ManifestError("archived trial attempts must be an array")
    for attempt in attempts:
        record = mapping(attempt)
        try:
            if study_type is StudyType.OPTIMIZATION:
                FailedTrialAttempt.from_primitive(record)
            else:
                PredictionGridFailedAttempt.from_primitive(record)
        except (
            KeyError,
            TypeError,
            ValueError,
            PredictionGridPersistenceError,
        ) as error:
            raise ManifestError("archived trial attempt is invalid") from error
        # QF-32's reader checks presence; require the declared text types too.
        for field in failure_fields:
            text(record.get(field))
        if study_type is StudyType.PARAMETER_STUDY:
            for field in ("started_at", "finished_at"):
                text(record.get(field))


def validate_trial_counts(
    study_type: StudyType,
    manifest: PrimitiveMapping,
    configuration: PrimitiveMapping,
    trials: list[PrimitiveMapping],
    summary: PrimitiveMapping | None,
) -> None:
    """Bind complete exports to the Cartesian size without enumerating trials."""
    axes = mapping(configuration.get("search_space")).get("parameters")
    if not isinstance(axes, list) or not axes:
        raise ManifestError("Cartesian search axes are invalid")
    total = 1
    names: set[str] = set()
    for raw_axis in axes:
        axis = mapping(raw_axis)
        name = text(axis.get("name"))
        values = axis.get("values")
        if name in names or not isinstance(values, list) or not values:
            raise ManifestError("Cartesian search axes are invalid")
        names.add(name)
        total *= len(values)
    if summary is not None and len(trials) != total:
        raise ManifestError("Cartesian trial coverage differs from completed summary")
    positions = [trial.get("combination_index") for trial in trials]
    if any(
        type(position) is not int or not 0 <= position < total for position in positions
    ):
        raise ManifestError("Cartesian trial coordinates are invalid")
    if len(set(positions)) != len(positions):
        raise ManifestError("Cartesian trial coordinates must be unique")
    declared: dict[str, int] = {}
    if study_type is StudyType.OPTIMIZATION:
        recorded = mapping(manifest.get("combination_counts"))
        for key in ("total_cartesian", "valid", "excluded"):
            count = recorded.get(key)
            if type(count) is not int or count < 0:
                raise ManifestError("Cartesian manifest counts are invalid")
            declared[key] = count
        if (
            declared["total_cartesian"] != total
            or declared["valid"] + declared["excluded"] != total
        ):
            raise ManifestError("Cartesian manifest counts differ from search space")
    if summary is None:
        return
    counts = mapping(summary.get("counts"))
    statuses = [text(trial.get("status")) for trial in trials]
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
        if (
            type(counts.get("total_cartesian_combinations")) is not int
            or counts["total_cartesian_combinations"] != total
            or declared["excluded"] != observed[TrialStatus.EXCLUDED]
            or declared["valid"] != total - observed[TrialStatus.EXCLUDED]
        ):
            raise ManifestError("Cartesian summary and manifest counts disagree")
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
    validate_backtest_identity(backtest)
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
