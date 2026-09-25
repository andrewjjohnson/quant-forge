"""Append-safe QF-55 decisions with an atomic committed-prefix checkpoint."""

import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import suppress
from copy import deepcopy
from pathlib import Path
from typing import BinaryIO

from quantforge.configuration import (
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.prediction.errors import InvalidPredictionOutputError
from quantforge.prediction.window_compact import (
    COMPACT_PREDICTION_WINDOW_SCHEMA_VERSION,
    CompactPredictionWindowDecision,
    PredictionWindowEvidence,
    WindowRecordCounts,
)
from quantforge.prediction.window_compact_validation import (
    PredictionWindowDecisionValidator,
)
from quantforge.prediction.window_encoding import (
    WindowResultIdentity,
    canonical,
    decode,
    mapping,
)
from quantforge.prediction.window_reader import PredictionWindowReader


class PredictionWindowPersistenceError(InvalidPredictionOutputError):
    """Incomplete, incompatible, or corrupt durable window state."""


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish(
    stream: BinaryIO, temporary: Path, destination: Path, *, replace: bool = False
) -> None:
    stream.flush()
    os.fsync(stream.fileno())
    if replace:
        os.replace(temporary, destination)
    else:
        # Atomically publish without overwriting an existing immutable artifact.
        os.link(temporary, destination)
    _sync_directory(destination.parent)


def _write_record(
    path: Path, record: PrimitiveMapping, *, replace: bool = False
) -> None:
    descriptor, filename = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    temporary = Path(filename)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical(record) + b"\n")
            _publish(stream, temporary, path, replace=replace)
    finally:
        temporary.unlink(missing_ok=True)


def _read(path: Path) -> PrimitiveMapping:
    return decode(path.read_bytes(), canonical_line=True)


class IncrementalPredictionWindowWriter:
    """One writer, three staging files, one completed record at a time.

    A successful append fsyncs the journal before atomically replacing its
    checkpoint. Recovery verifies the entire committed prefix, then truncates
    only bytes after that checkpoint's offset. Corrupt committed bytes fail
    closed. Reopen after any persistence failure. No completed payloads are kept.
    """

    def __init__(
        self, path: Path, validator: PredictionWindowDecisionValidator
    ) -> None:
        self.path = Path(path)
        self.staging_path = self.path.with_name(self.path.name + ".in-progress")
        self.validator = validator
        self.evidence = PredictionWindowEvidence(validator.expected_identity)
        self._schedule_id = self.evidence.schedule.schedule_id
        self.completed_count = 0
        self._byte_offset = 0
        self._last_decision_id: str | None = None
        self._digest = WindowResultIdentity(self.evidence.window_id)
        self._counts: WindowRecordCounts = WindowRecordCounts()
        self._finalized = False
        self._failed = False

    @classmethod
    def open(
        cls, path: Path, *, validator: PredictionWindowDecisionValidator
    ) -> "IncrementalPredictionWindowWriter":
        writer = cls(path, validator)
        try:
            if writer.path.exists():
                writer._validate_reader(PredictionWindowReader.open(writer.path))
                writer._finalized = True
                return writer
            writer.path.parent.mkdir(parents=True, exist_ok=True)
            if not writer.staging_path.exists():
                writer._initialize()
            if canonical(_read(writer.staging_path / "shared.json")) != canonical(
                writer.evidence.to_primitive()
            ):
                raise PredictionWindowPersistenceError(
                    "incompatible shared evidence/schema/schedule"
                )
            envelope = _read(writer.staging_path / "checkpoint.json")
            checkpoint = mapping(envelope.get("checkpoint"))
            if set(envelope) != {"checkpoint", "fingerprint"} or envelope[
                "fingerprint"
            ] != configuration_identity(checkpoint):
                raise PredictionWindowPersistenceError("corrupt checkpoint fingerprint")
            offset = checkpoint.get("byte_offset")
            if type(offset) is not int or offset < 0:
                raise PredictionWindowPersistenceError("invalid checkpoint byte offset")
            for decision, length in writer._records(offset):
                writer._accept(decision)
                writer._byte_offset += length
            if canonical(checkpoint) != canonical(writer.checkpoint()):
                raise PredictionWindowPersistenceError(
                    "corrupt completed-prefix checkpoint"
                )
            # Only after every compatibility, semantic, hash and count check.
            with writer.journal_path.open("r+b") as journal:
                journal.seek(0, os.SEEK_END)
                if journal.tell() > writer._byte_offset:
                    journal.truncate(writer._byte_offset)
                    journal.flush()
                    os.fsync(journal.fileno())
            return writer
        except (OSError, ValueError, TypeError, KeyError) as error:
            raise PredictionWindowPersistenceError(
                f"cannot open incremental window: {error}"
            ) from error

    @property
    def journal_path(self) -> Path:
        return self.staging_path / "decisions.jsonl"

    @property
    def finalized(self) -> bool:
        return self._finalized

    def _initialize(self) -> None:
        temporary = Path(
            tempfile.mkdtemp(prefix=".pending-window-", dir=self.path.parent)
        )
        try:
            _write_record(temporary / "shared.json", self.evidence.to_primitive())
            with (temporary / "decisions.jsonl").open("xb") as journal:
                journal.flush()
                os.fsync(journal.fileno())
            _write_record(temporary / "checkpoint.json", self._checkpoint_envelope())
            _sync_directory(temporary)
            os.rename(temporary, self.staging_path)
            _sync_directory(self.path.parent)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def checkpoint(self) -> PrimitiveMapping:
        """Return committed progress; append already persisted it atomically."""
        return {
            "checkpoint_version": "1",
            "schema_version": COMPACT_PREDICTION_WINDOW_SCHEMA_VERSION,
            "window_id": self.evidence.window_id,
            "shared_evidence_id": self.evidence.evidence_id,
            "schedule_id": self._schedule_id,
            "completed_count": self.completed_count,
            "byte_offset": self._byte_offset,
            "last_decision_id": self._last_decision_id,
            "prefix_result_id": self._digest.hexdigest(),
            "record_counts": deepcopy(self._counts.totals),
        }

    def _checkpoint_envelope(self) -> PrimitiveMapping:
        checkpoint = self.checkpoint()
        return {
            "checkpoint": checkpoint,
            "fingerprint": configuration_identity(checkpoint),
        }

    def _records(
        self, committed_offset: int
    ) -> Iterator[tuple[CompactPredictionWindowDecision, int]]:
        with self.journal_path.open("rb") as journal:
            remaining = committed_offset
            while remaining:
                line = journal.readline(remaining)
                if not line:
                    raise PredictionWindowPersistenceError(
                        "truncated committed journal"
                    )
                record = decode(line, canonical_line=True)
                remaining -= len(line)
                yield (
                    CompactPredictionWindowDecision(
                        PrimitiveMappingSnapshot.capture(record)
                    ),
                    len(line),
                )

    def _accept(self, decision: CompactPredictionWindowDecision) -> None:
        record = decision.to_primitive()
        index = self.completed_count
        timestamps = self.evidence.schedule.decision_timestamps
        if (
            index >= len(timestamps)
            or record["sequence"] != index
            or record["decision_timestamp"] != timestamps[index].isoformat()
            or record["shared_evidence_id"] != self.evidence.evidence_id
        ):
            raise PredictionWindowPersistenceError(
                "decision reference/order differs from expected prefix"
            )
        self.validator.validate(record, index)
        self._counts.update(record)
        self._digest.update(record)
        self.completed_count += 1
        self._last_decision_id = decision.decision_id

    def append(self, decision: CompactPredictionWindowDecision) -> None:
        """Fsync one canonical decision, then atomically commit its checkpoint."""
        if self._finalized or self._failed:
            raise PredictionWindowPersistenceError(
                "writer finalized or failed; reopen before use"
            )
        previous = (
            self.completed_count,
            self._byte_offset,
            self._last_decision_id,
            self._digest.copy(),
            deepcopy(self._counts),
        )
        try:
            self._accept(decision)
            encoded = decision.snapshot.canonical_json.encode() + b"\n"
            with self.journal_path.open("r+b") as journal:
                journal.seek(0, os.SEEK_END)
                if journal.tell() != self._byte_offset:
                    raise PredictionWindowPersistenceError(
                        "journal changed; reopen durable prefix"
                    )
                journal.write(encoded)
                journal.flush()
                os.fsync(journal.fileno())
            self._byte_offset += len(encoded)
            _write_record(
                self.staging_path / "checkpoint.json",
                self._checkpoint_envelope(),
                replace=True,
            )
        except BaseException as error:
            (
                self.completed_count,
                self._byte_offset,
                self._last_decision_id,
                self._digest,
                self._counts,
            ) = previous
            # The checkpoint may have been published before an I/O error/interrupt.
            # Always reopen to establish the authoritative committed prefix.
            if isinstance(error, OSError):
                self._failed = True
                raise PredictionWindowPersistenceError(
                    "decision persistence failed; reopen durable prefix"
                ) from error
            if not isinstance(error, InvalidPredictionOutputError):
                self._failed = True
            raise

    def _validate_reader(self, reader: PredictionWindowReader) -> None:
        if reader.schema_version != "2" or reader.evidence != self.evidence:
            raise PredictionWindowPersistenceError(
                "finalized window has incompatible evidence/version"
            )
        for decision in reader.iterate_decisions():
            self._accept(decision)
            self._byte_offset += len(decision.snapshot.canonical_json.encode()) + 1

    def finalize(self) -> PredictionWindowReader:
        """Validate and stream the same QF-55 JSONL bytes, then publish immutably."""
        if self._finalized:
            return PredictionWindowReader.open(self.path)
        if self._failed or self.completed_count != len(
            self.evidence.schedule.decision_timestamps
        ):
            raise PredictionWindowPersistenceError(
                "cannot finalize incomplete or failed window"
            )
        verified = type(self).open(self.path, validator=self.validator)
        if verified.checkpoint() != self.checkpoint():
            raise PredictionWindowPersistenceError(
                "durable prefix changed before finalization"
            )
        header: PrimitiveMapping = {
            "record_type": "header",
            "component": "quantforge_prediction_window",
            "schema_version": COMPACT_PREDICTION_WINDOW_SCHEMA_VERSION,
            "window_id": self.evidence.window_id,
            "window_result_id": self._digest.hexdigest(),
            "shared_evidence_id": self.evidence.evidence_id,
            "schedule_id": self._schedule_id,
            "decision_count": self.completed_count,
            "record_counts": self._counts.totals,
        }
        descriptor, filename = tempfile.mkstemp(
            prefix=".pending-", dir=self.path.parent
        )
        temporary = Path(filename)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(canonical(header) + b"\n")
                for source in (self.staging_path / "shared.json", self.journal_path):
                    with source.open("rb") as content:
                        shutil.copyfileobj(content, stream)
                stream.flush()
                check = type(self)(temporary, self.validator)
                check._validate_reader(PredictionWindowReader.open(temporary))
                _publish(stream, temporary, self.path)
            self._finalized = True
        finally:
            # Temporary-file cleanup must not mask publication errors or success.
            with suppress(OSError):
                temporary.unlink(missing_ok=True)
        # Invalid/stale work is preserved; only successfully finalized work is removed.
        # Cleanup errors leave redundant staging, never a failed research result.
        with suppress(OSError):
            shutil.rmtree(self.staging_path)
        return PredictionWindowReader.open(self.path)
