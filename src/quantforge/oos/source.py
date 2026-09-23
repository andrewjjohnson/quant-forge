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
from quantforge.oos.models import OOSSource
from quantforge.prediction import PredictionDecisionSchedule
from quantforge.prediction.window_validation import validate_prediction_window_snapshot
from quantforge.timeframes import resolve_exchange_session
from quantforge.validation import (
    ExchangeSessionBoundary,
    TimestampBoundary,
    ValidationPlan,
)
from quantforge.walk_forward.models import (
    BacktestOOSArtifact,
    FoldResult,
    FoldStatus,
    FrozenSelection,
    PredictionOOSArtifact,
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
        fold = plan.folds[index]
        protected = (
            fold.next_protected_window
            if role == "development"
            else fold.test
            if role == "selection"
            else plan.folds[index + 1].test
            if index + 1 < len(plan.folds)
            else plan.final_holdout.window
        )
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


def _validate_prediction(
    plan: ValidationPlan,
    selection: FrozenSelection,
    artifact: PredictionOOSArtifact,
) -> None:
    frozen = selection.snapshot.to_primitive()
    part = mapping(mapping(frozen["membership"])["test"])
    candidate = mapping(mapping(frozen["candidate"])["definition"])
    payload = artifact.snapshot.to_primitive()
    manifest = mapping(payload["manifest"])
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
        raise OOSIntegrityError("prediction artifact is not from the test partition")
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
    if canonical_metadata is not None:
        try:
            validate_prediction_view_lineage(
                market,
                canonical_metadata,
                schedule.decision_timestamps[0]
                if plan.prediction_membership is not None
                else resolve_exchange_session(
                    date.fromisoformat(text(market["actual_last_session"])),
                    primary.session_policy,
                ).close_timestamp,
                start=None
                if plan.prediction_membership is not None
                else date.fromisoformat(text(market["actual_first_session"])),
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
    identity = PrimitiveMappingSnapshot.capture(
        {
            k: v
            for k, v in manifest.items()
            if k
            not in {"window_id", "window_result_id", "schedule_id", "record_counts"}
        }
    )
    validate_prediction_window_snapshot(
        payload,
        expected_identity=identity,
        schedule=schedule,
        outcome_sessions=outcome_sessions,
        strategy_parameters=mapping(candidate["prediction_rule_parameters"]),
    )
    if artifact.window_result_id != manifest["window_result_id"]:
        raise OOSIntegrityError("prediction result ID differs")
    index = next(
        i for i, fold in enumerate(plan.folds) if fold.fold_id == frozen["fold_id"]
    )
    protected = (
        plan.folds[index + 1].test
        if index + 1 < len(plan.folds)
        else plan.final_holdout.window
    )
    # A test result must never include an outcome in the next protected window.
    for decision in records(payload["decisions"]):
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
                    raise OOSIntegrityError("test outcome reaches a protected window")
            else:
                assert isinstance(protected.interval.start, ExchangeSessionBoundary)
                if (
                    date.fromisoformat(text(mapping(row["outcome"])["outcome_session"]))
                    >= protected.interval.start.session_date
                ):
                    raise OOSIntegrityError("test outcome reaches a protected window")


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
    for index, fold in enumerate(plan.folds):
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
                    _validate_prediction(plan, selection, artifact)
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
    )
