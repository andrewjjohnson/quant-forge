"""Atomic local-file study and trial persistence."""

import json
import os
import tempfile
from pathlib import Path
from typing import cast

from quantforge.configuration import Primitive, PrimitiveMapping
from quantforge.optimization.errors import (
    InvalidTrialTransitionError,
    StudyPersistenceError,
)
from quantforge.optimization.models import TrialRecord, TrialStatus


def _json_text(value: Primitive) -> str:
    try:
        return (
            json.dumps(
                value,
                ensure_ascii=True,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    except (TypeError, ValueError) as error:
        raise StudyPersistenceError(
            "study artifacts must contain finite JSON-compatible values"
        ) from error


def _atomic_write(path: Path, value: Primitive) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=str(path.parent), text=True
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(_json_text(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise StudyPersistenceError(
            f"failed to persist study artifact: {path}"
        ) from error


def _load_mapping(path: Path) -> PrimitiveMapping:
    try:
        loaded: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StudyPersistenceError(f"failed to load study artifact: {path}") from error
    if not isinstance(loaded, dict) or any(
        not isinstance(key, str) for key in cast(dict[object, object], loaded)
    ):
        raise StudyPersistenceError(f"study artifact must be a JSON object: {path}")
    return cast(PrimitiveMapping, loaded)


_ALLOWED_TRANSITIONS: dict[TrialStatus, frozenset[TrialStatus]] = {
    TrialStatus.PENDING: frozenset((TrialStatus.RUNNING,)),
    TrialStatus.RUNNING: frozenset(
        (TrialStatus.RUNNING, TrialStatus.SUCCEEDED, TrialStatus.FAILED)
    ),
    TrialStatus.FAILED: frozenset((TrialStatus.RUNNING,)),
    TrialStatus.SUCCEEDED: frozenset(),
    TrialStatus.EXCLUDED: frozenset(),
}


class FileStudyStore:
    """One parent-owned store below ``output_root/study_id``."""

    def __init__(self, output_root: Path, study_id: str) -> None:
        self.study_id = study_id
        self.study_path = output_root / study_id
        self.manifest_path = self.study_path / "manifest.json"
        self.trials_path = self.study_path / "trials"

    def initialize(self, manifest: PrimitiveMapping, *, resume: bool) -> None:
        """Create a new manifest or verify an exact compatible resume target."""
        if self.manifest_path.exists():
            existing = _load_mapping(self.manifest_path)
            if existing != manifest:
                raise StudyPersistenceError(
                    "persisted study manifest is incompatible with the requested study"
                )
            if not resume:
                raise StudyPersistenceError(
                    f"study already exists; use resume: {self.study_path}"
                )
            return
        if self.study_path.exists():
            try:
                is_orphaned = not self.study_path.is_dir() or (
                    next(self.study_path.iterdir(), None) is not None
                )
            except OSError as error:
                raise StudyPersistenceError(
                    f"failed to inspect study directory: {self.study_path}"
                ) from error
            if is_orphaned:
                raise StudyPersistenceError(
                    "study directory is non-empty but its manifest is missing; "
                    f"restore or remove the orphaned directory: {self.study_path}"
                )
        if resume:
            raise StudyPersistenceError(
                "cannot resume because the study manifest does not exist: "
                f"{self.study_path}"
            )
        _atomic_write(self.manifest_path, manifest)

    def load_manifest(self) -> PrimitiveMapping:
        return _load_mapping(self.manifest_path)

    def trial_path(self, trial_id: str) -> Path:
        return self.trials_path / f"{trial_id}.json"

    def load_trial(self, trial_id: str) -> TrialRecord | None:
        path = self.trial_path(trial_id)
        if not path.exists():
            return None
        record = TrialRecord.from_primitive(_load_mapping(path))
        if record.study_id != self.study_id or record.trial_id != trial_id:
            raise StudyPersistenceError(
                f"trial identity does not match its artifact path: {path}"
            )
        return record

    def write_trial(self, record: TrialRecord) -> None:
        """Atomically persist a valid state transition in the parent process."""
        if record.study_id != self.study_id:
            raise StudyPersistenceError("trial belongs to a different study")
        existing = self.load_trial(record.trial_id)
        if existing is not None:
            if existing.status in (TrialStatus.SUCCEEDED, TrialStatus.EXCLUDED):
                if existing.to_primitive() == record.to_primitive():
                    return
                raise InvalidTrialTransitionError(
                    f"immutable {existing.status.value} trial cannot be overwritten"
                )
            if record.status not in _ALLOWED_TRANSITIONS[existing.status]:
                raise InvalidTrialTransitionError(
                    f"invalid trial transition {existing.status.value} -> "
                    f"{record.status.value}"
                )
            expected_failed_attempts = existing.failed_attempts
            if (
                existing.status is TrialStatus.FAILED
                and record.status is TrialStatus.RUNNING
            ):
                try:
                    archived_failure = existing.failed_attempt_snapshot()
                except ValueError as error:
                    raise StudyPersistenceError(
                        "cannot retry a failed trial with incomplete failure context"
                    ) from error
                expected_failed_attempts = (*expected_failed_attempts, archived_failure)
            if record.failed_attempts != expected_failed_attempts:
                raise InvalidTrialTransitionError(
                    "trial transition must preserve complete failed-attempt history"
                )
        _atomic_write(self.trial_path(record.trial_id), record.to_primitive())

    def load_trials(self) -> tuple[TrialRecord, ...]:
        if not self.trials_path.exists():
            return ()
        records: list[TrialRecord] = []
        for path in sorted(self.trials_path.glob("*.json")):
            record = TrialRecord.from_primitive(_load_mapping(path))
            if record.study_id != self.study_id or path.stem != record.trial_id:
                raise StudyPersistenceError(
                    f"trial identity does not match its artifact path: {path}"
                )
            records.append(record)
        records.sort(key=lambda item: (item.combination_index, item.trial_id))
        return tuple(records)

    def write_derived(self, relative_path: str, value: Primitive) -> Path:
        destination = self.study_path / relative_path
        _atomic_write(destination, value)
        return destination
