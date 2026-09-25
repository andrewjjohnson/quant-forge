"""Durable one-way holdout exposure ledger for a local POSIX research store."""

import fcntl
import os
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from quantforge.configuration import (
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.data.calendar import expected_sessions
from quantforge.oos._records import OOSIntegrityError, mapping, text
from quantforge.oos.common import provenance
from quantforge.oos.holdout_evaluation import HoldoutEvaluation
from quantforge.oos.models import OOSSource
from quantforge.oos.prediction import (
    iter_prediction_observations,
    summarize_prediction_observations,
)
from quantforge.prediction.window_reader import PredictionWindowReader
from quantforge.timeframes import resolve_exchange_session
from quantforge.validation import ExchangeSessionBoundary, TimestampBoundary
from quantforge.walk_forward.models import PredictionOOSArtifact
from quantforge.walk_forward.persistence import read_record, write_record
from quantforge.walk_forward.study import (
    _load_artifact,  # pyright: ignore[reportPrivateUsage]
)


class HoldoutState(StrEnum):
    RESERVED = "reserved_unconsumed"
    CONSUMED = "consumed"


@dataclass(frozen=True, slots=True)
class HoldoutConsumptionRecord:
    state: HoldoutState
    reservation: PrimitiveMappingSnapshot
    consumption: PrimitiveMappingSnapshot | None
    result_reference: PrimitiveMappingSnapshot | None

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "schema_version": "1",
            "state": self.state.value,
            "reservation": self.reservation.to_primitive(),
            "consumption": None
            if self.consumption is None
            else self.consumption.to_primitive(),
            "result_reference": None
            if self.result_reference is None
            else self.result_reference.to_primitive(),
            "result_status": "available" if self.result_reference else "unavailable",
            "pristine": self.state is HoldoutState.RESERVED,
        }


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _durable_write(path: Path, payload: PrimitiveMapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Persist newly created directory entries as well as the file replacement.
    _sync_directory(path.parent.parent)
    write_record(path, payload, immutable=True)
    _sync_directory(path.parent)


def _exposure_scope(source: OOSSource) -> PrimitiveMapping:
    interval = source.plan.final_holdout.window.interval
    dataset = source.plan.environment.outcome_dataset
    metadata, timeframe = dataset.market_data_metadata, dataset.standalone_timeframe
    if metadata is None or timeframe is None:
        raise OOSIntegrityError("holdout exposure requires a daily dataset policy")
    if source.plan.prediction_membership is not None:
        membership = source.plan.prediction_membership
        sessions = tuple(
            membership.session_for(key.timestamp)
            for key in membership.observations
            if interval.contains(key)
        )
        if not sessions:
            raise OOSIntegrityError("holdout has no captured timestamp observations")
        # Exposure still conservatively guards whole exchange sessions, so a
        # different membership axis cannot restore a viewed portion of a session.
        start, end = sessions[0], sessions[-1]
    elif isinstance(interval.start, ExchangeSessionBoundary):
        assert isinstance(interval.end, ExchangeSessionBoundary)
        start, end = interval.start.session_date, interval.end.session_date
    else:
        # QF-39 timestamp keys are session closes. UTC/local calendar dates can
        # differ from their session labels, including midnight/overnight closes.
        sessions = tuple(
            session
            for session in expected_sessions(
                metadata.actual_first_session,
                metadata.actual_last_session,
                timeframe.session_policy.calendar_name,
            )
            if interval.contains(
                TimestampBoundary(
                    resolve_exchange_session(
                        session, timeframe.session_policy
                    ).close_timestamp
                )
            )
        )
        if not sessions:
            raise OOSIntegrityError("holdout exposure has no dataset session closes")
        start, end = sessions[0], sessions[-1]
    return {
        "date_basis": "exchange_session_labels_v1",
        "symbol": metadata.canonical_symbol,
        "start": start.isoformat(),
        "end": end.isoformat(),
    }


class HoldoutLedger:
    """One permanent store per research workspace; locks cover the entire attempt.

    Explicit initialization is distinct from opening a store. Lost/incompatible
    state is an error, never an inference that a holdout is unconsumed.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._verify_store()

    @classmethod
    def create(cls, root: Path) -> "HoldoutLedger":
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        if any(root.iterdir()):
            raise OOSIntegrityError("holdout store already exists; open it instead")
        _durable_write(
            root / "store.json",
            {"component": "quantforge_holdout_ledger", "schema_version": "1"},
        )
        (root / "lineages").mkdir()
        (root / "exposures").mkdir()
        _sync_directory(root)
        return cls(root)

    def _verify_store(self) -> None:
        if read_record(self.root / "store.json") != {
            "component": "quantforge_holdout_ledger",
            "schema_version": "1",
        }:
            raise OOSIntegrityError("missing or incompatible holdout store")

    @contextmanager
    def _locked(self) -> Generator[None]:
        self._verify_store()
        with (self.root / ".lock").open("a") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise OOSIntegrityError(
                    "conflicting holdout operation is already active"
                ) from error
            try:
                self._audit()
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def _audit(self) -> None:
        for path in sorted((self.root / "lineages").glob("*")):
            if not path.is_dir():
                raise OOSIntegrityError("invalid holdout lineage directory")
            reservation_path = path / "reservation.json"
            if not reservation_path.exists():
                raise OOSIntegrityError("holdout artifacts exist without a reservation")
            reservation = read_record(reservation_path)
            if reservation["lineage_id"] != path.name:
                raise OOSIntegrityError("holdout reservation lineage differs")
            exposure_path = self.root / "exposures" / f"{path.name}.json"
            if (
                any(p.name != "reservation.json" for p in path.iterdir())
                and not exposure_path.exists()
            ):
                raise OOSIntegrityError(
                    "holdout result/attempt exists without a compatible "
                    "consumption ledger"
                )
        for path in sorted((self.root / "exposures").glob("*.json")):
            marker = read_record(path)
            if (
                marker.get("state") != HoldoutState.CONSUMED.value
                or marker.get("lineage_id") != path.stem
                or configuration_identity(mapping(marker["request"]))
                != marker["request_id"]
            ):
                raise OOSIntegrityError("invalid permanent holdout consumption marker")
            if (
                mapping(marker["exposure_scope"]).get("date_basis")
                != "exchange_session_labels_v1"
            ):
                raise OOSIntegrityError(
                    "incompatible holdout exposure scope; preserve consumed evidence"
                )

    def _guard(self, source: OOSSource) -> None:
        provenance(source)
        scope = _exposure_scope(source)
        for path in sorted((self.root / "exposures").glob("*.json")):
            marker = read_record(path)
            previous = mapping(marker["exposure_scope"])
            if (
                previous["symbol"] == scope["symbol"]
                and text(previous["start"]) <= text(scope["end"])
                and text(scope["start"]) <= text(previous["end"])
                and marker["lineage_id"] != source.lineage_id
            ):
                raise OOSIntegrityError(
                    "overlapping holdout was already consumed in another lineage; "
                    "it cannot be pristine"
                )

    def _reservation(self, source: OOSSource) -> PrimitiveMapping:
        return {
            "schema_version": "1",
            "lineage_id": source.lineage_id,
            "lineage": source.lineage.to_primitive(),
            "exposure_scope": _exposure_scope(source),
        }

    def reserve(self, source: OOSSource) -> HoldoutConsumptionRecord:
        with self._locked():
            self._guard(source)
            _durable_write(
                self.root / "lineages" / source.lineage_id / "reservation.json",
                self._reservation(source),
            )
            return self._state(source)

    def state(self, source: OOSSource) -> HoldoutConsumptionRecord:
        with self._locked():
            self._guard(source)
            return self._state(source)

    def _state(self, source: OOSSource) -> HoldoutConsumptionRecord:
        root = self.root / "lineages" / source.lineage_id
        reservation = read_record(root / "reservation.json")
        if reservation != self._reservation(source):
            raise OOSIntegrityError("holdout reservation/lineage mismatch")
        path = self.root / "exposures" / f"{source.lineage_id}.json"
        marker = read_record(path) if path.exists() else None
        reference = None
        if (root / "result.json").exists():
            result = read_record(root / "result.json")
            if (
                marker is None
                or result["request_id"] != marker["request_id"]
                or result["consumption_sha256"] != configuration_identity(marker)
            ):
                raise OOSIntegrityError(
                    "holdout result has no compatible consumption marker"
                )
            artifact = mapping(result["artifact"])
            if result["artifact_sha256"] != configuration_identity(artifact):
                raise OOSIntegrityError("holdout result artifact hash mismatch")
            reference = PrimitiveMappingSnapshot.capture(
                {
                    "path": f"lineages/{source.lineage_id}/result.json",
                    "sha256": configuration_identity(result),
                    "summary": result["summary"],
                    "artifact_sha256": result["artifact_sha256"],
                    "result_id": artifact["result_id"],
                }
            )
        return HoldoutConsumptionRecord(
            HoldoutState.RESERVED if marker is None else HoldoutState.CONSUMED,
            PrimitiveMappingSnapshot.capture(reservation),
            None if marker is None else PrimitiveMappingSnapshot.capture(marker),
            reference,
        )

    def consume(
        self, evaluation: HoldoutEvaluation, *, run_id: str, reproduce: bool = False
    ) -> HoldoutConsumptionRecord:
        """Explicit first exposure or exact retry; no selection callback is invoked.

        A durable CONSUMED marker precedes every evaluator call. Failures propagate
        while retaining it. Reproduction uses the original run/timestamp and must
        match the exact request and any previous result bytes.
        """
        if not run_id.strip():
            raise OOSIntegrityError("consumption run ID is required")
        with self._locked():
            evaluation.validate()
            source = evaluation.source
            self._guard(source)
            state = self._state(source)
            request = evaluation.configuration()
            request_id = configuration_identity(request)
            marker_path = self.root / "exposures" / f"{source.lineage_id}.json"
            if state.consumption is not None:
                marker = state.consumption.to_primitive()
                if marker["request_id"] != request_id or marker["request"] != request:
                    raise OOSIntegrityError(
                        "consumed holdout prohibits configuration reselection"
                    )
                if not reproduce and state.result_reference is not None:
                    root = self.root / "lineages" / source.lineage_id
                    saved = read_record(root / "result.json")
                    evaluation.evaluator.validate_partition_artifact(
                        source.plan,
                        evaluation.permitted,
                        evaluation.selection,
                        _load_artifact(mapping(saved["artifact"])),
                        root / "evaluation",
                    )
                    return state
            else:
                marker: PrimitiveMapping = {
                    "schema_version": "1",
                    "state": HoldoutState.CONSUMED.value,
                    "lineage_id": source.lineage_id,
                    "exposure_scope": _exposure_scope(source),
                    "request_id": request_id,
                    "request": request,
                    "consumption_run_id": run_id,
                    "consumed_at": datetime.now(UTC).isoformat(),
                    "transition": (
                        "permanent_before_evaluation; interrupted_attempts_are_consumed"
                    ),
                }
            # A prior failed fsync can leave a visible but nondurable marker.
            # Revalidate and sync its original bytes before every evaluator call.
            _durable_write(marker_path, marker)
            root = self.root / "lineages" / source.lineage_id
            artifact = evaluation._evaluate(root / "evaluation")  # pyright: ignore[reportPrivateUsage]
            if isinstance(artifact, PredictionOOSArtifact):
                reader = PredictionWindowReader.from_reference(
                    artifact.snapshot.to_primitive(), root=root / "evaluation"
                )
                observations = iter_prediction_observations(
                    reader, artifact, "final_holdout"
                )
                summary = summarize_prediction_observations(
                    observations, reader.decision_count
                )
            else:
                # QF-5 fsyncs export files, but its directory rename is not durable.
                # Persist file entries and the renamed run before publishing references.
                _sync_directory(root / "evaluation" / artifact.export_location)
                _sync_directory(root / "evaluation")
                summary = mapping(
                    mapping(artifact.snapshot.to_primitive()["manifest"])["performance"]
                )
            artifact_record = artifact.to_primitive()
            if isinstance(artifact, PredictionOOSArtifact):
                # Export the summary already computed above as captured evidence.
                # Observational consumers must not rerun the QF-40 summarizer.
                artifact_record["holdout_summary"] = {
                    "schema_version": "1",
                    "window_result_id": artifact.window_result_id,
                    "summary": summary,
                }
            result: PrimitiveMapping = {
                "schema_version": "1",
                "kind": "final_holdout_result",
                "state": "consumed",
                "request_id": request_id,
                "consumption_sha256": configuration_identity(marker),
                "artifact": artifact_record,
                "artifact_sha256": configuration_identity(artifact_record),
                "summary": summary,
            }
            _durable_write(root / "result.json", result)
            # Returning any result reference occurs only after both writes and fsync.
            return self._state(source)

    def result(self, evaluation: HoldoutEvaluation) -> PrimitiveMappingSnapshot:
        """Validate the consumed request and underlying artifact before reading it."""
        with self._locked():
            evaluation.validate()
            source = evaluation.source
            self._guard(source)
            state = self._state(source)
            if state.consumption is None or state.result_reference is None:
                raise OOSIntegrityError("consumed holdout result is unavailable")
            if (
                state.consumption.to_primitive()["request"]
                != evaluation.configuration()
            ):
                raise OOSIntegrityError("holdout result request differs")
            root = self.root / "lineages" / source.lineage_id
            result = read_record(root / "result.json")
            evaluation.evaluator.validate_partition_artifact(
                source.plan,
                evaluation.permitted,
                evaluation.selection,
                _load_artifact(mapping(result["artifact"])),
                root / "evaluation",
            )
            return PrimitiveMappingSnapshot.capture(result)
