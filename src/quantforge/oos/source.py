"""Read QF-39 artifacts without factories, market prices, or execution callbacks."""

import csv
import json
from datetime import date, datetime
from pathlib import Path

from quantforge.backtesting import EvaluationInterval
from quantforge.backtesting.export import validate_backtest_result_artifact
from quantforge.configuration import (
    Primitive,
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.data.calendar import expected_sessions
from quantforge.data.exceptions import ValidationError
from quantforge.data.prediction_views import validate_prediction_view_lineage
from quantforge.oos._records import (
    OOSIntegrityError,
    mapping,
    records,
    text,
    texts,
    verify_identity,
)
from quantforge.oos.models import OOSSource, PredictionTrialWindow
from quantforge.optimization.models import TrialStatus
from quantforge.prediction import PredictionDecisionSchedule
from quantforge.prediction.errors import InvalidPredictionOutputError
from quantforge.prediction.grid import (
    PredictionGridPersistenceError,
    PredictionGridTrialRecord,
)
from quantforge.prediction.window_compact_validation import (
    validate_prediction_window_reader,
)
from quantforge.prediction.window_encoding import decode
from quantforge.prediction.window_reader import PredictionWindowReader
from quantforge.prediction.window_validation import validate_prediction_window_snapshot
from quantforge.timeframes import resolve_exchange_session
from quantforge.validation import (
    ExchangeSessionBoundary,
    PartitionRole,
    TimestampBoundary,
    ValidationPlan,
    ValidationWindow,
)
from quantforge.walk_forward.models import (
    BacktestOOSArtifact,
    CandidateConfiguration,
    FoldResult,
    FoldStatus,
    FrozenSelection,
    PredictionOOSArtifact,
    WalkForwardPersistenceError,
)
from quantforge.walk_forward.persistence import read_record
from quantforge.walk_forward.study import (
    _load_artifact,  # pyright: ignore[reportPrivateUsage]
)


def validation_lineage(definition: PrimitiveMapping) -> PrimitiveMappingSnapshot:
    """Exclude run labels/retry policy, retaining all scientific definitions."""
    config = mapping(definition["configuration"])
    plan = mapping(config["plan"])
    adapter = mapping(definition["adapter"])
    universe = mapping(adapter["universe"])
    grid = dict(mapping(universe["grid_definition"]))
    grid.pop("label", None)
    grid.pop("name", None)
    # Window names and reservation prose cannot restore a viewed holdout.
    windows: list[Primitive] = []
    for fold in records(plan["folds"]):
        windows.append(
            {
                role: None
                if fold.get(role) is None
                else {
                    k: v
                    for k, v in mapping(fold[role]).items()
                    if k not in {"name", "window_id"}
                }
                for role in ("development", "selection", "test")
            }
        )
    holdout = mapping(mapping(plan["final_holdout"])["window"])
    return PrimitiveMappingSnapshot.capture(
        {
            "component": "quantforge_validation_lineage",
            "version": "1",
            "environment": plan["environment"],
            "purge_policy": plan["purge_policy"],
            **(
                {"prediction_membership": plan["prediction_membership"]}
                if "prediction_membership" in plan
                else {}
            ),
            "training_window_mode": plan["training_window_mode"],
            "windows": windows,
            "holdout": {
                k: v for k, v in holdout.items() if k not in {"name", "window_id"}
            },
            "candidates": universe["candidates"],
            "grid": grid,
            "adapter": {k: v for k, v in adapter.items() if k != "universe"},
            "selection_policy": config["selection_policy"],
            "minimum_training_observations": config["minimum_training_observations"],
            "minimum_test_observations": config["minimum_test_observations"],
            "engine_version": definition["engine_version"],
        }
    )


def _role_window(plan: ValidationPlan, index: int, role: str) -> ValidationWindow:
    fold = plan.folds[index]
    window = (
        fold.development
        if role == "development"
        else fold.selection
        if role == "selection"
        else fold.test
        if role == "test"
        else None
    )
    if window is None:
        raise OOSIntegrityError("fold has no window for the requested role")
    return window


def _protected_window(plan: ValidationPlan, index: int, role: str) -> ValidationWindow:
    """The QF-8 window a role's outcomes must not reach (its purge boundary)."""
    fold = plan.folds[index]
    if role == "development":
        return fold.next_protected_window
    if role == "selection":
        return fold.test
    return (
        plan.folds[index + 1].test
        if index + 1 < len(plan.folds)
        else plan.final_holdout.window
    )


def _selection(
    record: PrimitiveMapping,
    definition: PrimitiveMapping,
    plan: ValidationPlan,
    index: int,
    study_id: str,
) -> FrozenSelection:
    verify_identity(record, "selection_id")
    universe = mapping(mapping(definition["adapter"])["universe"])
    if any(
        (
            record.get("study_id") != study_id,
            record.get("study_definition") != definition,
            record.get("plan_id") != plan.plan_id,
            record.get("fold_id") != plan.folds[index].fold_id,
            record.get("candidate_universe_id") != configuration_identity(universe),
            record.get("candidate") not in records(universe["candidates"]),
        )
    ):
        raise OOSIntegrityError("incompatible frozen selection/validation lineage")
    membership = mapping(record["membership"])
    for role, window in (
        ("development", plan.folds[index].development),
        ("selection", plan.folds[index].selection),
        ("test", plan.folds[index].test),
    ):
        if window is None:
            if membership[role] is not None:
                raise OOSIntegrityError("unexpected selection partition")
            continue
        part = mapping(membership[role])
        if part["window"] != window.to_primitive():
            raise OOSIntegrityError("artifact membership has the wrong window/role")
        member, purge = mapping(part["membership"]), mapping(part["purge"])
        protected = _protected_window(plan, index, role)
        verify_identity(member, "selection_id")
        verify_identity(purge, "result_id")
        if (
            member["window_id"] != window.window_id
            or member["warm_up_eligible_for_selection"] is not False
            or purge["source_window_id"] != window.window_id
            or purge["plan_id"] != plan.plan_id
            or purge["fold_id"] != plan.folds[index].fold_id
            or purge["protected_window_id"] != protected.window_id
            or member["source_dataset_id"]
            not in (
                (plan.prediction_membership.source_reference.dataset_id,)
                if plan.prediction_membership is not None
                else plan.environment.outcome_dataset.dataset_ids
            )
            or purge["source_dataset_id"] != member["source_dataset_id"]
        ):
            raise OOSIntegrityError("incompatible partition membership provenance")
        retained = records(purge["retained"])
        declared_sessions = texts(part["evaluation_sessions"])
        if len(retained) != len(declared_sessions):
            raise OOSIntegrityError(
                "evaluation sessions differ from retained membership"
            )
        timestamp_source = plan.prediction_membership
        if timestamp_source is not None:
            if (
                part.get("prediction_membership") != timestamp_source.to_primitive()
                or part.get("evaluation_timestamps")
                != [key.get("timestamp") for key in retained]
                or [text(key["timestamp"]) for key in retained]
                != sorted({text(key["timestamp"]) for key in retained})
                or member["source_timeframe_configuration_id"]
                != timestamp_source.schedule.primary_timeframe.configuration_id
                or member["source_family_manifest_id"]
                != timestamp_source.family_manifest_id
                or purge["source_timeframe_configuration_id"]
                != member["source_timeframe_configuration_id"]
            ):
                raise OOSIntegrityError("incompatible timestamp membership lineage")
        elif "prediction_membership" in part or "evaluation_timestamps" in part:
            raise OOSIntegrityError("unexpected timestamp membership on session plan")
        sessions: list[str] = []
        for key, declared_session in zip(retained, declared_sessions, strict=True):
            boundary = (
                ExchangeSessionBoundary(
                    date.fromisoformat(text(key["session"])),
                    window.interval.start.session_policy,
                )
                if isinstance(window.interval.start, ExchangeSessionBoundary)
                else TimestampBoundary(datetime.fromisoformat(text(key["timestamp"])))
            )
            if not window.interval.contains(boundary) or key not in records(
                member["study_observations"]
            ):
                raise OOSIntegrityError("retained observations escape their window")
            if isinstance(boundary, ExchangeSessionBoundary):
                sessions.append(boundary.session_date.isoformat())
            elif timestamp_source is not None:
                if (
                    timestamp_source.session_for(boundary.timestamp).isoformat()
                    != declared_session
                ):
                    raise OOSIntegrityError(
                        "timestamp differs from captured QF-42 schedule"
                    )
                sessions.append(declared_session)
            else:
                timeframe = plan.environment.outcome_dataset.standalone_timeframe
                assert timeframe is not None
                session = date.fromisoformat(declared_session)
                if (
                    resolve_exchange_session(
                        session, timeframe.session_policy
                    ).close_timestamp
                    != boundary.timestamp
                ):
                    raise OOSIntegrityError(
                        "timestamp does not identify the declared exchange session"
                    )
                sessions.append(declared_session)
        if (
            not sessions
            or sessions
            != sorted(sessions if timestamp_source is not None else set(sessions))
            or sessions != part["evaluation_sessions"]
        ):
            raise OOSIntegrityError(
                "evaluation sessions are duplicated or incompatible"
            )
    return FrozenSelection(
        PrimitiveMappingSnapshot.capture(
            {k: v for k, v in record.items() if k != "selection_id"}
        )
    )


def prediction_view_bounds(
    plan: ValidationPlan,
    window: ValidationWindow,
    part: PrimitiveMapping,
    first_decision: datetime,
) -> tuple[datetime, date | None]:
    """Derive view bounds from plan membership, never the candidate market record."""
    if plan.prediction_membership is not None:
        return first_decision, None
    source = plan.environment.outcome_dataset
    metadata, timeframe = source.market_data_metadata, source.standalone_timeframe
    if metadata is None or timeframe is None:
        raise OOSIntegrityError(
            "prediction ancestry requires canonical session metadata"
        )
    observed = tuple(
        session
        for session in expected_sessions(
            metadata.actual_first_session,
            metadata.actual_last_session,
            metadata.calendar,
        )
        if session not in metadata.missing_sessions
    )
    keys = [
        ExchangeSessionBoundary(session, timeframe.session_policy).to_primitive()
        if isinstance(window.interval.start, ExchangeSessionBoundary)
        else TimestampBoundary(
            resolve_exchange_session(session, timeframe.session_policy).close_timestamp
        ).to_primitive()
        for session in observed
    ]
    first = keys.index(window.interval.start.to_primitive())
    warm_up = window.warm_up_observations_for(timeframe)
    if (
        first < warm_up
        or records(mapping(part["membership"])["warm_up_context"])
        != (keys[first - warm_up : first])
    ):
        raise OOSIntegrityError("prediction ancestry warm-up differs from its window")
    retained = [date.fromisoformat(s) for s in texts(part["evaluation_sessions"])]
    start = observed[first - warm_up] if warm_up else retained[0]
    horizon = plan.purge_policy.label_horizon.exchange_sessions
    if horizon is None:
        raise OOSIntegrityError("session ancestry requires an exchange-session horizon")
    end = observed.index(retained[-1]) + horizon
    if end >= len(observed):
        raise OOSIntegrityError("prediction ancestry has insufficient outcome sessions")
    return (
        resolve_exchange_session(
            observed[end], timeframe.session_policy
        ).close_timestamp,
        start,
    )


def _validate_prediction(
    plan: ValidationPlan,
    selection: FrozenSelection,
    artifact: PredictionOOSArtifact,
    root: Path,
) -> PredictionWindowReader:
    frozen = selection.snapshot.to_primitive()
    payload = artifact.snapshot.to_primitive()
    reader = PredictionWindowReader.from_reference(payload, root=root / "test")
    _validate_prediction_partition(
        plan,
        selection,
        "test",
        mapping(mapping(frozen["candidate"])["definition"]),
        reader,
        payload,
        artifact.window_result_id,
    )
    return reader


def _validate_prediction_partition(
    plan: ValidationPlan,
    selection: FrozenSelection,
    role: str,
    candidate: PrimitiveMapping,
    reader: PredictionWindowReader,
    payload: PrimitiveMapping,
    window_result_id: Primitive,
) -> None:
    """Bind one QF-39 window to its frozen QF-8 role membership, offline.

    ``role`` is the persisted membership key (``development``, ``selection`` or
    ``test``) and ``candidate`` the executable definition the window must
    reproduce. The test path passes the frozen candidate; QF-32 trial windows
    pass their own universe candidate under the same frozen membership.
    """
    frozen = selection.snapshot.to_primitive()
    index = next(
        i for i, fold in enumerate(plan.folds) if fold.fold_id == frozen["fold_id"]
    )
    window = _role_window(plan, index, role)
    part = mapping(mapping(frozen["membership"])[role])
    adapter = mapping(mapping(frozen["study_definition"])["adapter"])
    if reader.schema_version != adapter.get("window_schema_version", "1"):
        raise OOSIntegrityError("window representation differs from frozen adapter")
    manifest = reader.manifest()
    configured = mapping(manifest["configuration"])
    for component in ("prediction_rule", "outcome_labeler", "evaluator"):
        for key, value in mapping(candidate[component]).items():
            if mapping(configured[component]).get(key) != value:
                raise OOSIntegrityError(
                    "prediction component differs from frozen selection"
                )
    if (
        configured["feature_configuration"] != candidate["feature_configuration"]
        or configured["prediction_context_requirements"]
        != candidate["prediction_context"]
        or manifest["indicator_backend_environment"]
        != candidate["indicator_backend_environment"]
    ):
        raise OOSIntegrityError("prediction feature/backend provenance differs")
    context = mapping(mapping(manifest["context_environment"])["configuration"])
    if context["plan_id"] != plan.plan_id or context["partition"] != part:
        raise OOSIntegrityError(f"prediction artifact is not from the {role} partition")
    grid = mapping(mapping(mapping(frozen["study_definition"])["adapter"])["universe"])
    primary_config = mapping(grid["grid_definition"])["primary_timeframe"]
    primary = next(
        t for t in plan.environment.timeframes if t.to_primitive() == primary_config
    )
    sessions = [date.fromisoformat(text(s)) for s in texts(part["evaluation_sessions"])]
    schedule = PredictionDecisionSchedule(
        primary,
        datetime.fromisoformat(texts(part["evaluation_timestamps"])[0])
        if plan.prediction_membership is not None
        else resolve_exchange_session(
            sessions[0], primary.session_policy
        ).open_timestamp,
        datetime.fromisoformat(texts(part["evaluation_timestamps"])[-1])
        if plan.prediction_membership is not None
        else resolve_exchange_session(
            sessions[-1], primary.session_policy
        ).close_timestamp,
    )
    market = mapping(manifest["market_data"])
    canonical_metadata = plan.environment.outcome_dataset.market_data_metadata
    if (
        canonical_metadata is not None
        and canonical_metadata.intraday_provenance is not None
    ):
        try:
            cutoff, start = prediction_view_bounds(
                plan, window, part, schedule.decision_timestamps[0]
            )
            validate_prediction_view_lineage(
                market,
                canonical_metadata,
                cutoff,
                start=start,
            )
        except (ValidationError, ValueError) as error:
            raise OOSIntegrityError(str(error)) from error
    if (
        market["dataset_id"] != part["bounded_dataset_id"]
        or market["bars_fingerprint"] != part["bounded_data_sha256"]
    ):
        raise OOSIntegrityError("prediction bounded dataset differs")
    outcome_sessions = tuple(
        s
        for s in expected_sessions(
            date.fromisoformat(text(market["actual_first_session"])),
            date.fromisoformat(text(market["actual_last_session"])),
            text(market["calendar"]),
        )
        if s.isoformat() not in texts(market["missing_sessions"])
    )
    identity = reader.evidence.identity_snapshot
    if reader.schema_version == "1":
        validate_prediction_window_snapshot(
            payload,
            expected_identity=identity,
            schedule=schedule,
            outcome_sessions=outcome_sessions,
            strategy_parameters=mapping(candidate["prediction_rule_parameters"]),
        )
    else:
        validate_prediction_window_reader(
            reader,
            expected_identity=identity,
            schedule=schedule,
            outcome_sessions=outcome_sessions,
            strategy_parameters=mapping(candidate["prediction_rule_parameters"]),
            canonical_metadata=canonical_metadata,
        )
    if window_result_id != manifest["window_result_id"]:
        raise OOSIntegrityError("prediction result ID differs")
    protected = _protected_window(plan, index, role)
    # A result must never include an outcome in the next protected window.
    for receipt in reader.iterate_decision_receipts():
        if receipt.decision is None:
            continue  # A sparse no-prediction receipt has no rows by construction.
        decision = receipt.decision.to_primitive()
        for row in records(mapping(decision["prediction_study"])["rows"]):
            if plan.prediction_membership is not None:
                assert isinstance(protected.interval.start, TimestampBoundary)
                reach = plan.purge_policy.label_horizon.elapsed
                embargo = plan.purge_policy.embargo.elapsed
                assert reach is not None
                assert embargo is not None
                if (
                    datetime.fromisoformat(text(decision["decision_timestamp"]))
                    + reach
                    + embargo
                    >= protected.interval.start.timestamp
                ):
                    raise OOSIntegrityError(
                        f"{role} outcome reaches a protected window"
                    )
            else:
                assert isinstance(protected.interval.start, ExchangeSessionBoundary)
                if (
                    date.fromisoformat(text(mapping(row["outcome"])["outcome_session"]))
                    >= protected.interval.start.session_date
                ):
                    raise OOSIntegrityError(
                        f"{role} outcome reaches a protected window"
                    )


def _validate_backtest(
    selection: FrozenSelection, artifact: BacktestOOSArtifact, root: Path
) -> None:
    frozen = selection.snapshot.to_primitive()
    part = mapping(mapping(frozen["membership"])["test"])
    payload = artifact.snapshot.to_primitive()
    manifest = mapping(payload["manifest"])
    universe = mapping(
        mapping(mapping(frozen["study_definition"])["adapter"])["universe"]
    )
    expected = dict(mapping(mapping(universe["grid_definition"])["backtest"]))
    sessions = part["evaluation_sessions"]
    if not isinstance(sessions, list):
        raise OOSIntegrityError("missing evaluation sessions")
    actual_config = mapping(manifest["backtest_configuration"])
    # QF-43 owns the interval schema; preserve all its policy fields.
    interval = EvaluationInterval(
        date.fromisoformat(text(sessions[0])), date.fromisoformat(text(sessions[-1]))
    ).to_primitive()
    expected["evaluation_interval"] = interval
    if (
        manifest["run_id"] != artifact.run_id
        or artifact.export_location != artifact.run_id
        or actual_config != expected
        or interval.get("start_session") != sessions[0]
        or interval.get("end_session") != sessions[-1]
        or mapping(manifest["strategy"])["configuration"]
        != mapping(frozen["candidate"])["definition"]
        or mapping(manifest["market_data"])["dataset_id"] != part["bounded_dataset_id"]
    ):
        raise OOSIntegrityError(
            "backtest artifact differs from frozen test configuration"
        )
    for key in ("daily_equity", "benchmark_daily_equity"):
        if [row["session"] for row in records(payload[key])] != sessions:
            raise OOSIntegrityError(
                "equity includes non-test sessions or missing observations"
            )
    destination = root / "test" / artifact.export_location
    validate_backtest_result_artifact(destination)
    if (
        configuration_identity(
            {"integrity": (destination / "integrity.json").read_text()}
        )
        != artifact.export_fingerprint
        or json.loads((destination / "manifest.json").read_text()) != manifest
    ):
        raise OOSIntegrityError("backtest snapshot/export fingerprint differs")
    tables = {
        "equity": "daily_equity",
        "benchmark_equity": "benchmark_daily_equity",
        "trades": "completed_trades",
        "signals": "signals",
        "orders": "orders",
        "fills": "fills",
        "positions": "positions",
        "dividend_cashflows": "dividend_cashflows",
        "split_adjustments": "split_adjustments",
        "benchmark_dividend_cashflows": "benchmark_dividend_cashflows",
        "benchmark_split_adjustments": "benchmark_split_adjustments",
    }
    for filename, key in tables.items():
        rows = records(payload[key])
        if filename == "trades":
            rows += records(payload["open_trades"])
        with (destination / f"{filename}.csv").open(newline="") as stream:
            reader = csv.DictReader(stream)
            actual = list(reader)
            fields = reader.fieldnames or []
        expected_rows: list[dict[str, str]] = []
        for row in rows:
            rendered: dict[str, str] = {}
            for field in fields:
                value = row.get(field)
                rendered[field] = (
                    json.dumps(
                        value,
                        ensure_ascii=True,
                        allow_nan=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    if isinstance(value, (dict, list))
                    else ""
                    if value is None
                    else str(value)
                )
            expected_rows.append(rendered)
        if actual != expected_rows:
            raise OOSIntegrityError("backtest tabular export differs from OOS snapshot")


def load_oos_source(plan: ValidationPlan, study_path: Path) -> OOSSource:
    """Load all planned folds; no selection/evaluation/resume execution occurs."""
    definition = read_record(study_path / "manifest.json")
    study_id = configuration_identity(definition)
    if (
        study_path.name != study_id
        or definition.get("component") != "quantforge_walk_forward"
        or definition.get("engine_version") != "1"
        or mapping(definition["configuration"])["plan"] != plan.to_manifest()
    ):
        raise OOSIntegrityError("incompatible QF-39 study/validation plan")
    universe = mapping(mapping(definition["adapter"])["universe"])
    if universe["study_type"] != plan.environment.study_type.value or definition[
        "candidate_universe_id"
    ] != configuration_identity(universe):
        raise OOSIntegrityError("incompatible candidate universe")
    expected_ids = {fold.fold_id for fold in plan.folds}
    if (study_path / "folds").exists() and any(
        path.name not in expected_ids for path in (study_path / "folds").iterdir()
    ):
        raise OOSIntegrityError("duplicate or unexpected fold artifact directory")
    folds: list[FoldResult] = []
    references: list[PrimitiveMappingSnapshot] = []
    artifact_ids: set[str] = set()
    prediction_windows: list[PredictionWindowReader | None] = []
    for index, fold in enumerate(plan.folds):
        prediction_reader = None
        root = study_path / "folds" / fold.fold_id
        state = (
            read_record(root / "state.json") if (root / "state.json").exists() else None
        )
        selection_record = (
            read_record(root / "selection.json")
            if (root / "selection.json").exists()
            else None
        )
        if state is None:
            if root.exists() and any(root.iterdir()):
                raise OOSIntegrityError("orphaned fold has no state")
            result = FoldResult(fold.fold_id, FoldStatus.PENDING, None, None, ())
        else:
            if state["study_id"] != study_id or state["fold_id"] != fold.fold_id:
                raise OOSIntegrityError("incompatible fold state")
            status = FoldStatus(text(state["status"]))
            selection = (
                None
                if selection_record is None
                else _selection(selection_record, definition, plan, index, study_id)
            )
            if state["selection_id"] is not None and (
                selection is None or selection.selection_id != state["selection_id"]
            ):
                raise OOSIntegrityError("fold lost its frozen selection")
            artifact = None
            if status is FoldStatus.COMPLETED:
                if selection is None:
                    raise OOSIntegrityError("completed fold has no frozen selection")
                content = read_record(root / "oos.json")
                digest = configuration_identity(content)
                if digest != state["artifact_id"] or digest in artifact_ids:
                    raise OOSIntegrityError("duplicate or incompatible OOS artifact")
                artifact_ids.add(digest)
                artifact = _load_artifact(content)
                if artifact.selection_id != selection.selection_id:
                    raise OOSIntegrityError(
                        "OOS artifact lost its frozen configuration"
                    )
                if (
                    isinstance(artifact, PredictionOOSArtifact)
                    and universe["study_type"] == "prediction"
                ):
                    prediction_reader = _validate_prediction(
                        plan, selection, artifact, root
                    )
                elif (
                    isinstance(artifact, BacktestOOSArtifact)
                    and universe["study_type"] == "trading_backtest"
                ):
                    _validate_backtest(selection, artifact, root)
                else:
                    raise OOSIntegrityError("OOS study family differs")
            result = FoldResult(
                fold.fold_id,
                status,
                selection,
                artifact,
                tuple(
                    PrimitiveMappingSnapshot.capture(item)
                    for item in records(state["failures"])
                ),
            )
        folds.append(result)
        prediction_windows.append(prediction_reader)
        references.append(
            PrimitiveMappingSnapshot.capture(
                {
                    "fold_id": fold.fold_id,
                    "window_id": fold.test.window_id,
                    "state": state,
                    "selection": selection_record,
                    "artifact_path": f"folds/{fold.fold_id}/oos.json"
                    if result.artifact
                    else None,
                    "artifact_sha256": None
                    if result.artifact is None
                    else configuration_identity(result.artifact.to_primitive()),
                    "result_id": None
                    if result.artifact is None
                    else result.artifact.to_primitive()["result_id"],
                    "status": "missing" if state is None else result.status.value,
                }
            )
        )
    return OOSSource(
        plan,
        study_id,
        PrimitiveMappingSnapshot.capture(definition),
        validation_lineage(definition),
        tuple(folds),
        tuple(references),
        tuple(prediction_windows),
    )


# Operational QF-32 manifest fields outside the grid's scientific identity.
_GRID_MANIFEST_RUNTIME_FIELDS = frozenset(
    {"study_id", "execution", "cache_policy", "interpretation"}
)
_TRIAL_WRAPPER_FIELDS = frozenset(
    {
        "schema_version",
        "grid_study_id",
        "trial_id",
        "prediction_window_id",
        "prediction_window",
        "analysis",
        "artifact_fingerprint",
    }
)


def _strict_json(path: Path) -> PrimitiveMapping:
    try:
        return decode(path.read_bytes())
    except (OSError, InvalidPredictionOutputError) as error:
        raise OOSIntegrityError(f"invalid QF-32 trial artifact: {path.name}") from error


def load_prediction_trial_window(
    source: OOSSource, study_path: Path, *, fold_id: str, combination_id: str
) -> PredictionTrialWindow:
    """Verify one persisted QF-32 trial window of a frozen QF-39 fold, offline.

    QF-39 runs every universe candidate on the fold's selection window, or on
    development when the plan declares no selection window. The trial is bound
    through the fold's immutable frozen selection (grid study and trial status),
    the QF-32 grid identity, trial record and wrapper fingerprints, and then the
    same plan membership, configuration, schedule, canonical lineage and
    protected-reach checks as a test window. No factory, market data or
    execution is needed. A trial window is in-sample evidence, never OOS.
    """
    plan = source.plan
    definition = source.definition.to_primitive()
    try:
        persisted = read_record(study_path / "manifest.json")
    except WalkForwardPersistenceError as error:
        raise OOSIntegrityError("trial window study manifest is invalid") from error
    if study_path.name != source.study_id or persisted != definition:
        raise OOSIntegrityError("trial window study differs from the verified source")
    index = next(
        (i for i, fold in enumerate(plan.folds) if fold.fold_id == fold_id), None
    )
    if index is None:
        raise OOSIntegrityError("trial window fold is not in the validation plan")
    selection = source.folds[index].selection
    if selection is None:
        raise OOSIntegrityError("trial windows require the fold's frozen selection")
    frozen = selection.snapshot.to_primitive()
    role = "selection" if plan.folds[index].selection is not None else "development"
    universe = mapping(mapping(definition["adapter"])["universe"])
    candidate = next(
        (
            item
            for item in records(universe["candidates"])
            if item.get("combination_id") == combination_id
        ),
        None,
    )
    if candidate is None:
        raise OOSIntegrityError("combination is outside the candidate universe")
    statuses = [
        item
        for item in records(mapping(frozen["selection_evidence"])["trial_statuses"])
        if item.get("combination_id") == combination_id
    ]
    if len(statuses) != 1 or statuses[0].get("status") != TrialStatus.SUCCEEDED.value:
        raise OOSIntegrityError(
            "frozen selection has no succeeded trial for the combination"
        )
    trial_id = text(statuses[0]["trial_id"])
    if candidate == frozen["candidate"] and frozen["selected_trial_id"] != trial_id:
        raise OOSIntegrityError("selected trial differs from the frozen selection")
    grid_id = text(frozen["selection_grid_study_id"])
    grid_root = study_path / "folds" / fold_id / "selection" / grid_id
    location = f"artifacts/{trial_id}/prediction-window.json"
    grid = _strict_json(grid_root / "manifest.json")
    wrapper = _strict_json(grid_root / location)
    try:
        record = PredictionGridTrialRecord.from_primitive(
            _strict_json(grid_root / "trials" / f"{trial_id}.json")
        )
    except PredictionGridPersistenceError as error:
        raise OOSIntegrityError("invalid QF-32 trial record") from error
    trial_definition = (
        None
        if record.trial_definition_snapshot is None
        else record.trial_definition_snapshot.to_primitive()
    )
    content = {k: v for k, v in wrapper.items() if k != "artifact_fingerprint"}
    if (
        grid.get("study_id") != grid_id
        or configuration_identity(
            {k: v for k, v in grid.items() if k not in _GRID_MANIFEST_RUNTIME_FIELDS}
        )
        != grid_id
        or record.study_id != grid_id
        or record.trial_id != trial_id
        or record.combination_id != combination_id
        or record.status is not TrialStatus.SUCCEEDED
        or record.parameters != candidate["parameters"]
        or record.artifact_location != location
        or record.analysis is None
        or trial_definition is None
        or {k: v for k, v in trial_definition.items() if k != "decision_schedule"}
        != candidate["definition"]
        or trial_definition.get("decision_schedule") != grid.get("decision_schedule")
        or frozenset(wrapper) != _TRIAL_WRAPPER_FIELDS
        or wrapper["grid_study_id"] != grid_id
        or wrapper["trial_id"] != trial_id
        or wrapper["analysis"] != record.analysis.to_primitive()
        or wrapper["artifact_fingerprint"] != configuration_identity(content)
        or wrapper["artifact_fingerprint"] != record.artifact_fingerprint
    ):
        raise OOSIntegrityError("QF-32 trial evidence differs from its frozen fold")
    payload = mapping(wrapper["prediction_window"])
    try:
        reader = PredictionWindowReader.from_reference(
            payload, root=(grid_root / location).parent
        )
    except InvalidPredictionOutputError as error:
        raise OOSIntegrityError("invalid QF-32 trial window reference") from error
    if reader.evidence.schedule.to_primitive() != grid.get("decision_schedule"):
        raise OOSIntegrityError("trial window schedule differs from its grid")
    _validate_prediction_partition(
        plan,
        selection,
        role,
        mapping(candidate["definition"]),
        reader,
        payload,
        wrapper["prediction_window_id"],
    )
    return PredictionTrialWindow(
        fold_id=fold_id,
        fold_index=index,
        role=PartitionRole.SELECTION
        if role == "selection"
        else PartitionRole.DEVELOPMENT,
        selection=selection,
        candidate=CandidateConfiguration(
            combination_id,
            PrimitiveMappingSnapshot.capture(mapping(candidate["parameters"])),
            PrimitiveMappingSnapshot.capture(mapping(candidate["definition"])),
        ),
        grid_study_id=grid_id,
        trial_id=trial_id,
        window_result_id=text(wrapper["prediction_window_id"]),
        reader=reader,
    )
