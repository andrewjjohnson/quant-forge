"""Current-data scanning and research-only alerts for validated QF-31 rules."""

from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol, TextIO, cast

from quantforge.configuration import (
    Primitive,
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
    decimal_to_primitive,
)
from quantforge.data.intraday import IntradayBar
from quantforge.data.lineage import AdjustmentBasis
from quantforge.data.models import AdjustmentMode
from quantforge.data.multi_timeframe import (
    MultiTimeframeContext,
    MultiTimeframeContextError,
)
from quantforge.prediction.context import (
    PredictionContextError,
    PredictionContextFailurePolicy,
    PredictionContextRequirements,
    PredictionRuleContext,
    build_prediction_rule_context,
    skipped_prediction_context_manifest,
)
from quantforge.prediction.models import PredictionDirection
from quantforge.prediction.signal_feature_models import (
    SignalDisposition,
    SignalFeatureCandidate,
    SignalFeatureCandidateOutput,
)
from quantforge.prediction.technical_confluence import (
    TechnicalConfluenceEvaluation,
    TechnicalConfluenceOutcome,
)

PREDICTION_ALERT_SCHEMA_VERSION = "1"
PREDICTION_SCANNER_ENGINE_VERSION = "1"
HISTORICAL_PREDICTION_STUDY_REFERENCE_SCHEMA_VERSION = "1"
RESEARCH_ONLY_DISCLAIMER = (
    "Research only. This alert is not financial advice and does not submit or "
    "authorize any brokerage order or trade."
)


class PredictionScannerError(ValueError):
    """A current-data scan cannot be completed or reproduced safely."""


class HistoricalStudyMismatchError(PredictionScannerError):
    """A current rule cannot reproduce the referenced historical study."""


class AlertPersistenceError(PredictionScannerError):
    """An alert or deduplication marker cannot be persisted safely."""


class _PosixFileLocking(Protocol):
    LOCK_EX: int
    LOCK_UN: int

    def flock(self, descriptor: int, operation: int) -> None: ...


def _posix_file_locking() -> _PosixFileLocking:
    try:
        module = importlib.import_module("fcntl")
    except ModuleNotFoundError as error:
        raise AlertPersistenceError(
            "file-backed alert deduplication requires POSIX fcntl locking"
        ) from error
    return cast(_PosixFileLocking, module)


class AlertDeduplicationPolicy(StrEnum):
    """Select which causally distinct current evaluations may alert again."""

    EXACT_CONTEXT = "exact_context"
    DECISION_BAR = "decision_bar"


def _required_record_text(value: PrimitiveMapping, field_name: str) -> str:
    field_value = value[field_name]
    if not isinstance(field_value, str) or not field_value.strip():
        raise TypeError(f"{field_name} must be a nonempty string")
    return field_value


def _required_record_bool(value: PrimitiveMapping, field_name: str) -> bool:
    field_value = value[field_name]
    if not isinstance(field_value, bool):
        raise TypeError(f"{field_name} must be a boolean")
    return field_value


def _required_record_sha256(value: PrimitiveMapping, field_name: str) -> str:
    field_value = _required_record_text(value, field_name)
    if len(field_value) != 64 or any(
        character not in "0123456789abcdef" for character in field_value
    ):
        raise TypeError(f"{field_name} must be a lowercase SHA-256 fingerprint")
    return field_value


def _required_record_text_tuple(
    value: PrimitiveMapping, field_name: str
) -> tuple[str, ...]:
    field_value = value[field_name]
    if not isinstance(field_value, list) or any(
        not isinstance(item, str) or not item.strip() for item in field_value
    ):
        raise TypeError(f"{field_name} must be an array of nonempty strings")
    return tuple(cast(list[str], field_value))


def _optional_record_count(value: PrimitiveMapping, field_name: str) -> int | None:
    field_value = value.get(field_name)
    if field_value is None:
        return None
    if isinstance(field_value, bool) or not isinstance(field_value, int):
        raise TypeError(f"{field_name} must be an integer or null")
    return field_value


class PredictionScannerRule(Protocol):
    """QF-31 rule surface required by the scanner."""

    name: str
    implementation_version: str
    context_requirements: PredictionContextRequirements

    @property
    def configuration_id(self) -> str: ...

    def configuration(self) -> PrimitiveMapping: ...

    def evaluate(
        self, context: PredictionRuleContext
    ) -> TechnicalConfluenceEvaluation: ...

    def generate_with_context(
        self, context: PredictionRuleContext
    ) -> SignalFeatureCandidateOutput: ...


@dataclass(frozen=True, slots=True)
class HistoricalPredictionStudyReference:
    """Exact validated-study semantics that a current rule must reproduce."""

    study_id: str
    rule_id: str
    rule_implementation_version: str
    rule_configuration_id: str
    context_requirements_id: str
    validated_symbols: tuple[str, ...]
    historical_dataset_fingerprint: str
    adjustment_basis: AdjustmentBasis
    summary: PrimitiveMappingSnapshot | None = None
    sample_count: int | None = None

    def __post_init__(self) -> None:
        if any(
            not value
            for value in (
                self.study_id,
                self.rule_id,
                self.rule_implementation_version,
                self.rule_configuration_id,
                self.context_requirements_id,
                self.validated_symbols,
                self.historical_dataset_fingerprint,
            )
        ):
            raise PredictionScannerError(
                "historical study references require complete rule identity"
            )
        if any(
            not isinstance(cast(object, symbol), str) or not symbol.strip()
            for symbol in self.validated_symbols
        ) or self.validated_symbols != tuple(sorted(set(self.validated_symbols))):
            raise PredictionScannerError(
                "historical study validated symbols must be sorted and unique"
            )
        if len(self.historical_dataset_fingerprint) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.historical_dataset_fingerprint
        ):
            raise PredictionScannerError(
                "historical study dataset fingerprint must be a lowercase SHA-256"
            )
        if not isinstance(cast(object, self.adjustment_basis), AdjustmentBasis):
            raise PredictionScannerError("historical study adjustment basis is invalid")
        sample_count = cast(object, self.sample_count)
        if sample_count is not None and (
            isinstance(sample_count, bool)
            or not isinstance(sample_count, int)
            or sample_count < 0
        ):
            raise PredictionScannerError(
                "historical study sample count must be a nonnegative integer"
            )

    @classmethod
    def capture(
        cls,
        *,
        study_id: str,
        rule: PredictionScannerRule,
        validated_symbols: tuple[str, ...],
        historical_dataset_fingerprint: str,
        adjustment_basis: AdjustmentBasis,
        summary: PrimitiveMapping | None = None,
        sample_count: int | None = None,
    ) -> HistoricalPredictionStudyReference:
        """Capture the exact rule and backend semantics recorded by a study."""
        return cls(
            study_id=study_id,
            rule_id=rule.name,
            rule_implementation_version=rule.implementation_version,
            rule_configuration_id=rule.configuration_id,
            context_requirements_id=configuration_identity(
                rule.context_requirements.to_primitive()
            ),
            validated_symbols=tuple(sorted(set(validated_symbols))),
            historical_dataset_fingerprint=historical_dataset_fingerprint,
            adjustment_basis=adjustment_basis,
            summary=(
                None if summary is None else PrimitiveMappingSnapshot.capture(summary)
            ),
            sample_count=sample_count,
        )

    @classmethod
    def from_primitive(
        cls, value: PrimitiveMapping
    ) -> HistoricalPredictionStudyReference:
        """Load one immutable historical-study record and reject malformed input."""
        try:
            if (
                value["schema_version"]
                != HISTORICAL_PREDICTION_STUDY_REFERENCE_SCHEMA_VERSION
                or value["artifact_type"] != "historical_prediction_study_reference"
            ):
                raise ValueError("unsupported historical-study reference schema")
            adjustment_value = value["adjustment_basis"]
            if not isinstance(adjustment_value, dict):
                raise TypeError("adjustment_basis must be an object")
            adjustment = cast(PrimitiveMapping, adjustment_value)
            summary_value = value.get("summary")
            if summary_value is not None and not isinstance(summary_value, dict):
                raise TypeError("summary must be an object or null")
            return cls(
                study_id=_required_record_text(value, "study_id"),
                rule_id=_required_record_text(value, "rule_id"),
                rule_implementation_version=_required_record_text(
                    value, "rule_implementation_version"
                ),
                rule_configuration_id=_required_record_text(
                    value, "rule_configuration_id"
                ),
                context_requirements_id=_required_record_text(
                    value, "context_requirements_id"
                ),
                validated_symbols=_required_record_text_tuple(
                    value, "validated_symbols"
                ),
                historical_dataset_fingerprint=_required_record_sha256(
                    value, "historical_dataset_fingerprint"
                ),
                adjustment_basis=AdjustmentBasis(
                    AdjustmentMode(
                        _required_record_text(adjustment, "adjustment_mode")
                    ),
                    _required_record_text(adjustment, "ohlc_basis"),
                    _required_record_text(adjustment, "volume_basis"),
                    _required_record_text(adjustment, "corporate_action_policy"),
                    _required_record_bool(adjustment, "adjusted_fields_used"),
                ),
                summary=(
                    None
                    if summary_value is None
                    else PrimitiveMappingSnapshot.capture(
                        cast(PrimitiveMapping, summary_value)
                    )
                ),
                sample_count=_optional_record_count(value, "sample_count"),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise PredictionScannerError(
                "historical study reference record is invalid"
            ) from error

    def validate_rule(self, rule: PredictionScannerRule) -> None:
        """Fail closed if any logical, indicator, backend, or policy input differs."""
        current_configuration_id = configuration_identity(rule.configuration())
        current_requirements_id = configuration_identity(
            rule.context_requirements.to_primitive()
        )
        if (
            rule.name != self.rule_id
            or rule.implementation_version != self.rule_implementation_version
            or rule.configuration_id != self.rule_configuration_id
            or current_configuration_id != self.rule_configuration_id
            or current_requirements_id != self.context_requirements_id
        ):
            raise HistoricalStudyMismatchError(
                "current rule, indicator configuration/backend, or context policy "
                f"does not match historical study {self.study_id}"
            )

    def validate_adjustment_basis(self, adjustment_basis: AdjustmentBasis) -> None:
        """Reject current prices that use different historical adjustment semantics."""
        if adjustment_basis != self.adjustment_basis:
            raise HistoricalStudyMismatchError(
                "current data adjustment basis does not match historical study "
                f"{self.study_id}"
            )

    def validate_symbol(self, symbol: str) -> None:
        """Reject a current symbol outside the study's validated universe."""
        if symbol not in self.validated_symbols:
            raise HistoricalStudyMismatchError(
                f"current symbol {symbol} is outside historical study "
                f"{self.study_id} validated universe"
            )

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "schema_version": HISTORICAL_PREDICTION_STUDY_REFERENCE_SCHEMA_VERSION,
            "artifact_type": "historical_prediction_study_reference",
            "study_id": self.study_id,
            "rule_id": self.rule_id,
            "rule_implementation_version": self.rule_implementation_version,
            "rule_configuration_id": self.rule_configuration_id,
            "context_requirements_id": self.context_requirements_id,
            "validated_symbols": list(self.validated_symbols),
            "historical_dataset_fingerprint": self.historical_dataset_fingerprint,
            "adjustment_basis": self.adjustment_basis.to_primitive(),
            "summary": None if self.summary is None else self.summary.to_primitive(),
            "sample_count": self.sample_count,
        }


@dataclass(frozen=True, slots=True)
class PredictionScannerSnapshot:
    """Canonical current-data context returned by the ingestion/data boundary."""

    context: MultiTimeframeContext
    prediction_dataset_id: str
    symbol: str
    adjustment_basis: AdjustmentBasis

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.context), MultiTimeframeContext):
            raise PredictionScannerError("scanner source returned an invalid context")
        if not self.prediction_dataset_id or not self.symbol:
            raise PredictionScannerError(
                "scanner source snapshot requires dataset and symbol identity"
            )
        if not isinstance(cast(object, self.adjustment_basis), AdjustmentBasis):
            raise PredictionScannerError(
                "scanner source snapshot adjustment basis is invalid"
            )
        canonical_source_snapshot_ids = {
            reference.canonical_source_snapshot_id
            for timeframe_context in self.context.timeframes
            if (reference := timeframe_context.dataset_reference) is not None
        }
        if canonical_source_snapshot_ids != {self.prediction_dataset_id}:
            raise PredictionScannerError(
                "scanner prediction dataset ID does not match context lineage"
            )


class PredictionScannerDataSource(Protocol):
    """Refresh/load canonical data and rebuild the declared as-of context."""

    def prepare_context(
        self,
        requirements: PredictionContextRequirements,
        *,
        as_of: datetime,
        refresh: bool,
    ) -> PredictionScannerSnapshot: ...


@dataclass(frozen=True, slots=True)
class PredictionScannerRuleBinding:
    """One current rule paired with its authoritative historical-study record."""

    rule: PredictionScannerRule
    historical_study: HistoricalPredictionStudyReference

    def __post_init__(self) -> None:
        self.historical_study.validate_rule(self.rule)


@dataclass(frozen=True, slots=True)
class PredictionAlert:
    """Immutable research-only alert containing complete causal evidence."""

    alert_id: str
    symbol: str
    as_of: datetime
    decision_timestamp: datetime
    direction: PredictionDirection
    rule_id: str
    rule_implementation_version: str
    rule_configuration_id: str
    context_id: str
    completion_policy: str
    conditions: tuple[PrimitiveMappingSnapshot, ...]
    indicators: tuple[PrimitiveMappingSnapshot, ...]
    source_bars: tuple[PrimitiveMappingSnapshot, ...]
    provenance: PrimitiveMappingSnapshot
    historical_study: HistoricalPredictionStudyReference
    disclaimer: str = RESEARCH_ONLY_DISCLAIMER
    schema_version: str = PREDICTION_ALERT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != PREDICTION_ALERT_SCHEMA_VERSION:
            raise PredictionScannerError("prediction alert schema is invalid")
        if any(
            not value
            for value in (
                self.alert_id,
                self.symbol,
                self.rule_id,
                self.rule_implementation_version,
                self.rule_configuration_id,
                self.context_id,
                self.completion_policy,
                self.disclaimer,
            )
        ):
            raise PredictionScannerError("prediction alert identity is incomplete")
        if (
            self.as_of.utcoffset() is None
            or self.decision_timestamp.utcoffset() is None
        ):
            raise PredictionScannerError("prediction alert timestamps must be aware")
        if self.decision_timestamp > self.as_of:
            raise PredictionScannerError(
                "prediction alert decision timestamp cannot be after as-of"
            )
        if not self.conditions or not self.source_bars:
            raise PredictionScannerError(
                "prediction alerts require condition and source-bar evidence"
            )
        expected_id = configuration_identity(self.identity_primitive())
        if self.alert_id != expected_id:
            raise PredictionScannerError("prediction alert identity is inconsistent")

    def identity_primitive(self) -> PrimitiveMapping:
        """Return every field required to identify one exact alert decision."""
        return _alert_identity_primitive(
            schema_version=self.schema_version,
            symbol=self.symbol,
            rule_id=self.rule_id,
            rule_implementation_version=self.rule_implementation_version,
            rule_configuration_id=self.rule_configuration_id,
            historical_study=self.historical_study,
            indicators=self.indicators,
            decision_timestamp=self.decision_timestamp,
            context_id=self.context_id,
            completion_policy=self.completion_policy,
        )

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "schema_version": self.schema_version,
            "artifact_type": "prediction_alert",
            "alert_id": self.alert_id,
            "symbol": self.symbol,
            "as_of": self.as_of.astimezone(UTC).isoformat(),
            "decision_timestamp": self.decision_timestamp.astimezone(UTC).isoformat(),
            "direction": self.direction.value,
            "rule": {
                "rule_id": self.rule_id,
                "implementation_version": self.rule_implementation_version,
                "configuration_id": self.rule_configuration_id,
            },
            "context_id": self.context_id,
            "completion_policy": self.completion_policy,
            "conditions": [item.to_primitive() for item in self.conditions],
            "indicators": [item.to_primitive() for item in self.indicators],
            "source_bars": [item.to_primitive() for item in self.source_bars],
            "provenance": self.provenance.to_primitive(),
            "historical_study": self.historical_study.to_primitive(),
            "disclaimer": self.disclaimer,
        }

    def serialize(self) -> bytes:
        return (
            json.dumps(
                self.to_primitive(),
                ensure_ascii=True,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")


class PredictionAlertSink(Protocol):
    """Deliver one already-deduplicated structured prediction alert."""

    def emit(self, alert: PredictionAlert) -> None: ...


class ConsolePredictionAlertSink:
    """Write one complete JSON alert to a text stream."""

    def __init__(self, stream: TextIO | None = None) -> None:
        self._stream = stream

    def emit(self, alert: PredictionAlert) -> None:
        stream = sys.stdout if self._stream is None else self._stream
        stream.write(alert.serialize().decode("utf-8"))
        stream.flush()


class JsonFilePredictionAlertSink:
    """Atomically persist one content-addressed alert without overwrites."""

    def __init__(self, output_directory: Path) -> None:
        self.output_directory = output_directory

    def emit(self, alert: PredictionAlert) -> None:
        self.output_directory.mkdir(parents=True, exist_ok=True)
        path = self.output_directory / f"{alert.alert_id}.json"
        content = alert.serialize()
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{alert.alert_id}.",
            suffix=".tmp",
            dir=self.output_directory,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary_path, path)
            except FileExistsError:
                if path.read_bytes() != content:
                    raise AlertPersistenceError(
                        f"existing alert artifact differs: {path}"
                    ) from None
            _fsync_directory(self.output_directory)
        except AlertPersistenceError:
            raise
        except OSError as error:
            raise AlertPersistenceError(
                f"cannot persist alert artifact: {path}"
            ) from error
        finally:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError as error:
                raise AlertPersistenceError(
                    f"cannot remove temporary alert artifact: {temporary_path}"
                ) from error


def _fsync_directory(directory: Path) -> None:
    if sys.platform == "win32":
        return
    descriptor = os.open(
        directory,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class AlertDeduplicationClaim(Protocol):
    """One exclusive pending claim that can become published or be released."""

    def publish(self) -> None: ...

    def release(self) -> None: ...


@dataclass(frozen=True, slots=True)
class PublishedAlertDeduplication:
    """Identity of the published alert that caused a duplicate decision."""

    alert_id: str

    def __post_init__(self) -> None:
        if not self.alert_id:
            raise AlertPersistenceError("published duplicate alert ID is required")


class AlertDeduplicationStore(Protocol):
    """Atomically reserve an alert key through a recoverable claim lifecycle."""

    def claim(
        self, deduplication_key: str, alert_id: str
    ) -> AlertDeduplicationClaim | PublishedAlertDeduplication: ...


class InMemoryAlertDeduplicationStore:
    """Process-local deterministic deduplication for embedded scanners and tests."""

    def __init__(self) -> None:
        self._claims: dict[str, tuple[str, str]] = {}
        self._condition = threading.Condition()

    def claim(
        self, deduplication_key: str, alert_id: str
    ) -> AlertDeduplicationClaim | PublishedAlertDeduplication:
        with self._condition:
            while self._claims.get(deduplication_key, ("", ""))[1] == "pending":
                self._condition.wait()
            existing = self._claims.get(deduplication_key)
            if existing is not None:
                return PublishedAlertDeduplication(existing[0])
            self._claims[deduplication_key] = (alert_id, "pending")
        return _InMemoryAlertDeduplicationClaim(self, deduplication_key, alert_id)

    def publish_claim(self, deduplication_key: str, alert_id: str) -> None:
        with self._condition:
            if self._claims.get(deduplication_key) != (alert_id, "pending"):
                raise AlertPersistenceError(
                    "in-memory deduplication claim is not pending"
                )
            self._claims[deduplication_key] = (alert_id, "published")
            self._condition.notify_all()

    def release_claim(self, deduplication_key: str, alert_id: str) -> None:
        with self._condition:
            if self._claims.get(deduplication_key) == (alert_id, "pending"):
                del self._claims[deduplication_key]
                self._condition.notify_all()


@dataclass(slots=True)
class _InMemoryAlertDeduplicationClaim:
    store: InMemoryAlertDeduplicationStore
    deduplication_key: str
    alert_id: str
    _closed: bool = False

    def publish(self) -> None:
        if self._closed:
            raise AlertPersistenceError("deduplication claim is already closed")
        self.store.publish_claim(self.deduplication_key, self.alert_id)
        self._closed = True

    def release(self) -> None:
        if self._closed:
            return
        self.store.release_claim(self.deduplication_key, self.alert_id)
        self._closed = True


class JsonFileAlertDeduplicationStore:
    """Cross-run deduplication with stable locks and atomic state files."""

    def __init__(self, state_directory: Path) -> None:
        self.state_directory = state_directory

    def _path(self, deduplication_key: str) -> Path:
        return self.state_directory / f"{deduplication_key}.json"

    def _lock_path(self, deduplication_key: str) -> Path:
        return self.state_directory / f"{deduplication_key}.lock"

    def claim(
        self, deduplication_key: str, alert_id: str
    ) -> AlertDeduplicationClaim | PublishedAlertDeduplication:
        locking = _posix_file_locking()
        self.state_directory.mkdir(parents=True, exist_ok=True)
        path = self._path(deduplication_key)
        lock_descriptor = os.open(
            self._lock_path(deduplication_key),
            os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        try:
            locking.flock(lock_descriptor, locking.LOCK_EX)
        except BaseException:
            os.close(lock_descriptor)
            raise
        claim = _JsonFileAlertDeduplicationClaim(
            path, lock_descriptor, deduplication_key, alert_id, locking
        )
        try:
            existing_state = claim.read_state()
            if existing_state is not None:
                existing_lifecycle, existing_alert_id = existing_state
                if existing_lifecycle == "published":
                    claim.close()
                    return PublishedAlertDeduplication(existing_alert_id)
                if existing_alert_id != alert_id:
                    raise AlertPersistenceError(
                        "pending deduplication claim belongs to a different "
                        "alert identity"
                    )
            claim.write_state("pending")
        except BaseException:
            claim.close()
            raise
        return claim


@dataclass(slots=True)
class _JsonFileAlertDeduplicationClaim:
    path: Path
    lock_descriptor: int
    deduplication_key: str
    alert_id: str
    locking: _PosixFileLocking
    _closed: bool = False

    def read_state(self) -> tuple[str, str] | None:
        try:
            content = self.path.read_bytes()
        except FileNotFoundError:
            return None
        except OSError as error:
            raise AlertPersistenceError(
                f"cannot read deduplication state: {self.path}"
            ) from error
        try:
            if not content:
                return None
            decoded = cast(object, json.loads(content))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise AlertPersistenceError(
                f"cannot validate deduplication state: {self.path}"
            ) from error
        if not isinstance(decoded, dict):
            raise AlertPersistenceError(
                f"deduplication state is not a JSON object: {self.path}"
            )
        existing = cast(dict[str, object], decoded)
        if existing.get("deduplication_key") != self.deduplication_key:
            raise AlertPersistenceError(
                f"deduplication state identity is inconsistent: {self.path}"
            )
        existing_alert_id = existing.get("alert_id")
        if not isinstance(existing_alert_id, str) or not existing_alert_id:
            raise AlertPersistenceError(
                f"deduplication state alert identity is invalid: {self.path}"
            )
        state = existing.get("state")
        if state is None:
            # The original write-on-claim format did not distinguish delivery
            # from interruption. Recover it as pending so it cannot suppress an
            # undelivered alert forever.
            return "pending", existing_alert_id
        if state not in {"pending", "published"}:
            raise AlertPersistenceError(
                f"deduplication state lifecycle is invalid: {self.path}"
            )
        return cast(str, state), existing_alert_id

    def write_state(self, state: str) -> None:
        if state not in {"pending", "published"}:
            raise AlertPersistenceError("deduplication lifecycle state is invalid")
        content = json.dumps(
            {
                "alert_id": self.alert_id,
                "deduplication_key": self.deduplication_key,
                "state": state,
            },
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        try:
            _atomic_replace_private_file(self.path, content)
        except OSError as error:
            raise AlertPersistenceError(
                f"cannot persist deduplication state: {self.path}"
            ) from error

    def publish(self) -> None:
        if self._closed:
            raise AlertPersistenceError("deduplication claim is already closed")
        self.write_state("published")
        self.close()

    def release(self) -> None:
        if self._closed:
            return
        try:
            self.path.unlink(missing_ok=True)
            _fsync_directory(self.path.parent)
        except OSError as error:
            raise AlertPersistenceError(
                f"cannot release deduplication claim: {self.path}"
            ) from error
        finally:
            self.close()

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.locking.flock(self.lock_descriptor, self.locking.LOCK_UN)
        finally:
            os.close(self.lock_descriptor)
            self._closed = True


def _atomic_replace_private_file(path: Path, content: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    finally:
        temporary_path.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class PredictionRuleScanResult:
    """Auditable accepted, rejected, or deduplicated result for one rule."""

    rule_id: str
    rule_configuration_id: str
    context_id: str | None
    evaluation: PrimitiveMappingSnapshot | None
    alert: PredictionAlert | None
    duplicate_alert_id: str | None = None
    context_failure: PrimitiveMappingSnapshot | None = None

    def __post_init__(self) -> None:
        if self.context_failure is None:
            if self.context_id is None or self.evaluation is None:
                raise PredictionScannerError(
                    "evaluated scanner results require context and evaluation evidence"
                )
            return
        if any(
            value is not None
            for value in (
                self.context_id,
                self.evaluation,
                self.alert,
                self.duplicate_alert_id,
            )
        ):
            raise PredictionScannerError(
                "skipped scanner results cannot contain evaluation or alert evidence"
            )

    @property
    def skipped(self) -> bool:
        return self.context_failure is not None

    @property
    def accepted(self) -> bool:
        return self.alert is not None or self.duplicate_alert_id is not None


@dataclass(frozen=True, slots=True)
class PredictionScanResult:
    """One deterministic current-data scan across one or more validated rules."""

    as_of: datetime
    dry_run: bool
    deduplication_policy: AlertDeduplicationPolicy
    rule_results: tuple[PredictionRuleScanResult, ...]
    engine_version: str = PREDICTION_SCANNER_ENGINE_VERSION

    @property
    def alerts(self) -> tuple[PredictionAlert, ...]:
        return tuple(
            result.alert for result in self.rule_results if result.alert is not None
        )


@dataclass(slots=True)
class PredictionScanner:
    """Evaluate validated historical QF-31 rules against one current as-of view."""

    source: PredictionScannerDataSource
    bindings: tuple[PredictionScannerRuleBinding, ...]
    sinks: tuple[PredictionAlertSink, ...]
    deduplication_store: AlertDeduplicationStore = field(
        default_factory=InMemoryAlertDeduplicationStore
    )
    deduplication_policy: AlertDeduplicationPolicy = (
        AlertDeduplicationPolicy.DECISION_BAR
    )

    def __post_init__(self) -> None:
        if not self.bindings:
            raise PredictionScannerError(
                "prediction scanner requires at least one rule"
            )
        if not self.sinks:
            raise PredictionScannerError(
                "prediction scanner requires at least one sink"
            )
        identities = tuple(
            (item.rule.name, item.rule.configuration_id) for item in self.bindings
        )
        if len(identities) != len(set(identities)):
            raise PredictionScannerError(
                "prediction scanner rule bindings must be unique"
            )

    def scan(self, *, as_of: datetime, dry_run: bool = False) -> PredictionScanResult:
        """Prepare causal contexts, evaluate rules, deduplicate, and emit alerts."""
        if as_of.utcoffset() is None:
            raise PredictionScannerError(
                "scanner as-of timestamp must be timezone-aware"
            )
        decision_as_of = as_of.astimezone(UTC)
        results: list[PredictionRuleScanResult] = []
        for binding in self.bindings:
            binding.historical_study.validate_rule(binding.rule)
            snapshot: PredictionScannerSnapshot | None = None
            try:
                snapshot = self.source.prepare_context(
                    binding.rule.context_requirements,
                    as_of=decision_as_of,
                    refresh=not dry_run,
                )
                if snapshot.context.as_of != decision_as_of:
                    raise PredictionScannerError(
                        "scanner source context as-of does not match the requested "
                        "decision"
                    )
                binding.historical_study.validate_symbol(snapshot.symbol)
                binding.historical_study.validate_adjustment_basis(
                    snapshot.adjustment_basis
                )
                rule_context = build_prediction_rule_context(
                    binding.rule.context_requirements,
                    snapshot.context,
                    prediction_dataset_id=snapshot.prediction_dataset_id,
                    symbol=snapshot.symbol,
                    prediction_adjustment_basis=snapshot.adjustment_basis,
                )
            except (PredictionContextError, MultiTimeframeContextError) as error:
                binding.historical_study.validate_rule(binding.rule)
                if (
                    binding.rule.context_requirements.failure_policy
                    is PredictionContextFailurePolicy.FAIL
                ):
                    raise
                failure = skipped_prediction_context_manifest(
                    binding.rule.context_requirements,
                    str(error),
                    source_context=(None if snapshot is None else snapshot.context),
                )
                results.append(
                    PredictionRuleScanResult(
                        binding.rule.name,
                        binding.rule.configuration_id,
                        None,
                        None,
                        None,
                        context_failure=PrimitiveMappingSnapshot.capture(failure),
                    )
                )
                continue
            binding.historical_study.validate_rule(binding.rule)
            evaluation = binding.rule.evaluate(rule_context)
            output = binding.rule.generate_with_context(rule_context)
            binding.historical_study.validate_rule(binding.rule)
            candidate = _validated_rule_candidate(binding.rule, rule_context, output)
            _validate_candidate_evaluation(candidate, evaluation)
            alert = (
                None
                if candidate.disposition is not SignalDisposition.ACCEPTED
                else _build_alert(
                    binding,
                    snapshot,
                    rule_context,
                    evaluation,
                    cast(PredictionDirection, candidate.direction),
                )
            )
            if alert is None:
                results.append(
                    PredictionRuleScanResult(
                        binding.rule.name,
                        binding.rule.configuration_id,
                        rule_context.context_id,
                        PrimitiveMappingSnapshot.capture(evaluation.to_primitive()),
                        None,
                    )
                )
                continue
            deduplication_key = _deduplication_key(alert, self.deduplication_policy)
            deduplication_result = self.deduplication_store.claim(
                deduplication_key, alert.alert_id
            )
            if isinstance(deduplication_result, PublishedAlertDeduplication):
                results.append(
                    PredictionRuleScanResult(
                        binding.rule.name,
                        binding.rule.configuration_id,
                        rule_context.context_id,
                        PrimitiveMappingSnapshot.capture(evaluation.to_primitive()),
                        None,
                        deduplication_result.alert_id,
                    )
                )
                continue
            deduplication_claim = deduplication_result
            try:
                for sink in self.sinks:
                    sink.emit(alert)
                deduplication_claim.publish()
            except BaseException:
                deduplication_claim.release()
                raise
            results.append(
                PredictionRuleScanResult(
                    binding.rule.name,
                    binding.rule.configuration_id,
                    rule_context.context_id,
                    PrimitiveMappingSnapshot.capture(evaluation.to_primitive()),
                    alert,
                )
            )
        return PredictionScanResult(
            decision_as_of,
            dry_run,
            self.deduplication_policy,
            tuple(results),
        )


def _validated_rule_candidate(
    rule: PredictionScannerRule,
    context: PredictionRuleContext,
    output: SignalFeatureCandidateOutput,
) -> SignalFeatureCandidate:
    if (
        not isinstance(cast(object, output), SignalFeatureCandidateOutput)
        or output.contract_version != "1"
        or output.strategy_id != rule.name
        or output.strategy_configuration_id != rule.configuration_id
        or output.dataset_id != context.prediction_dataset_id
        or len(output.signals) != 1
    ):
        raise PredictionScannerError(
            "prediction rule returned an incompatible current-data output"
        )
    candidate = output.signals[0]
    if (
        candidate.symbol != context.symbol
        or candidate.signal_session != context.decision_session
        or candidate.strategy_id != rule.name
        or candidate.strategy_implementation_version != rule.implementation_version
        or candidate.strategy_configuration_id != rule.configuration_id
        or candidate.source_rule_id != rule.name
        or candidate.source_rule_implementation_version != rule.implementation_version
        or candidate.source_rule_configuration_id != rule.configuration_id
    ):
        raise PredictionScannerError(
            "prediction candidate identity does not match the current rule context"
        )
    return candidate


def _validate_candidate_evaluation(
    candidate: SignalFeatureCandidate,
    evaluation: TechnicalConfluenceEvaluation,
) -> None:
    expected_direction = (
        PredictionDirection.UP
        if evaluation.outcome is TechnicalConfluenceOutcome.UP
        else PredictionDirection.DOWN
        if evaluation.outcome is TechnicalConfluenceOutcome.DOWN
        else None
    )
    if candidate.direction is not expected_direction or (
        candidate.disposition is SignalDisposition.ACCEPTED
    ) is (expected_direction is None):
        raise PredictionScannerError(
            "prediction candidate and auditable rule evaluation disagree"
        )


def _build_alert(
    binding: PredictionScannerRuleBinding,
    snapshot: PredictionScannerSnapshot,
    context: PredictionRuleContext,
    evaluation: TechnicalConfluenceEvaluation,
    direction: PredictionDirection,
) -> PredictionAlert:
    indicators = _indicator_evidence(context)
    source_bars = _source_bar_evidence(context, snapshot.context)
    decision_timestamp = context.latest_bar_for(
        context.requirements.primary.timeframe
    ).end_timestamp
    conditions = tuple(
        PrimitiveMappingSnapshot.capture(item.to_primitive())
        for item in evaluation.condition_results
    )
    provenance = PrimitiveMappingSnapshot.capture(
        {
            "prediction_dataset_id": snapshot.prediction_dataset_id,
            "adjustment_basis": snapshot.adjustment_basis.to_primitive(),
            "context_source_consistency": (
                snapshot.context.source_consistency.to_primitive()
            ),
            "context_requirements": context.requirements.to_primitive(),
        }
    )
    alert_id = configuration_identity(
        _alert_identity_primitive(
            schema_version=PREDICTION_ALERT_SCHEMA_VERSION,
            symbol=context.symbol,
            rule_id=binding.rule.name,
            rule_implementation_version=binding.rule.implementation_version,
            rule_configuration_id=binding.rule.configuration_id,
            historical_study=binding.historical_study,
            indicators=indicators,
            decision_timestamp=decision_timestamp,
            context_id=context.context_id,
            completion_policy=snapshot.context.completion_policy.value,
        )
    )
    return PredictionAlert(
        alert_id=alert_id,
        symbol=context.symbol,
        as_of=context.as_of,
        decision_timestamp=decision_timestamp,
        direction=direction,
        rule_id=binding.rule.name,
        rule_implementation_version=binding.rule.implementation_version,
        rule_configuration_id=binding.rule.configuration_id,
        context_id=context.context_id,
        completion_policy=snapshot.context.completion_policy.value,
        conditions=conditions,
        indicators=indicators,
        source_bars=source_bars,
        provenance=provenance,
        historical_study=binding.historical_study,
    )


def _alert_identity_primitive(
    *,
    schema_version: str,
    symbol: str,
    rule_id: str,
    rule_implementation_version: str,
    rule_configuration_id: str,
    historical_study: HistoricalPredictionStudyReference,
    indicators: tuple[PrimitiveMappingSnapshot, ...],
    decision_timestamp: datetime,
    context_id: str,
    completion_policy: str,
) -> PrimitiveMapping:
    indicator_configurations: list[Primitive] = []
    for item in indicators:
        primitive = item.to_primitive()
        indicator_configurations.append(
            {
                "timeframe_configuration_id": primitive["timeframe_configuration_id"],
                "alias": primitive["alias"],
                "indicator_configuration_id": primitive["indicator_configuration_id"],
                "timeframe_indicator_configuration_id": primitive[
                    "timeframe_indicator_configuration_id"
                ],
                "backend": primitive["backend"],
            }
        )
    return {
        "schema_version": schema_version,
        "symbol": symbol,
        "rule_id": rule_id,
        "rule_implementation_version": rule_implementation_version,
        "rule_configuration_id": rule_configuration_id,
        "historical_study": historical_study.to_primitive(),
        "indicator_configurations": indicator_configurations,
        "decision_timestamp": decision_timestamp.astimezone(UTC).isoformat(),
        "context_id": context_id,
        "completion_policy": completion_policy,
    }


def _indicator_evidence(
    context: PredictionRuleContext,
) -> tuple[PrimitiveMappingSnapshot, ...]:
    values: list[PrimitiveMappingSnapshot] = []
    for timeframe_input in context.timeframes:
        requirements = {
            item.alias: item for item in timeframe_input.requirement.indicators
        }
        for named in timeframe_input.indicators:
            output = named.output
            requirement = requirements[named.alias]
            normalized_values: PrimitiveMapping = {}
            for field_output in output.fields:
                current = field_output.values[-1]
                normalized_values[field_output.name] = (
                    None if current is None else decimal_to_primitive(current)
                )
            values.append(
                PrimitiveMappingSnapshot.capture(
                    {
                        "timeframe_configuration_id": (
                            timeframe_input.requirement.timeframe.configuration_id
                        ),
                        "timeframe": (
                            timeframe_input.requirement.timeframe.to_primitive()
                        ),
                        "alias": named.alias,
                        "indicator_configuration_id": requirement.configuration_id,
                        "timeframe_indicator_configuration_id": (
                            output.configuration_id
                        ),
                        "backend": (
                            None
                            if output.backend_identity is None
                            else output.backend_identity.to_primitive()
                        ),
                        "normalized_values": normalized_values,
                        "source_bar_id": output.bar_ids[-1],
                        "source_bar_timestamp": (
                            output.bar_end_timestamps[-1].isoformat()
                        ),
                        "source_bar_completion": (output.completion_states[-1].value),
                    }
                )
            )
    return tuple(
        sorted(
            values,
            key=lambda item: (
                cast(str, item.to_primitive()["timeframe_configuration_id"]),
                cast(str, item.to_primitive()["alias"]),
            ),
        )
    )


def _source_bar_evidence(
    context: PredictionRuleContext,
    source_context: MultiTimeframeContext,
) -> tuple[PrimitiveMappingSnapshot, ...]:
    values: list[PrimitiveMappingSnapshot] = []
    for timeframe_input in context.timeframes:
        timeframe = timeframe_input.requirement.timeframe
        metadata = source_context.metadata_for(timeframe)
        latest = timeframe_input.bars[-1]
        if metadata.age is None:
            raise PredictionScannerError(
                "available scanner timeframe is missing age metadata"
            )
        session_dates = (
            (latest.session_date,)
            if isinstance(latest, IntradayBar)
            else latest.session_dates
        )
        values.append(
            PrimitiveMappingSnapshot.capture(
                {
                    "timeframe_configuration_id": timeframe.configuration_id,
                    "timeframe": timeframe.to_primitive(),
                    "session_policy": timeframe.session_policy.to_primitive(),
                    "feed_scope": (
                        timeframe_input.requirement.required_feed_scope.to_primitive()
                    ),
                    "dataset_reference": (
                        None
                        if metadata.dataset_reference is None
                        else metadata.dataset_reference.to_primitive(
                            include_feed_scope=True
                        )
                    ),
                    "availability": metadata.availability.value,
                    "age_microseconds": int(metadata.age.total_seconds() * 1_000_000),
                    "bar_id": latest.bar_id,
                    "start_timestamp": latest.start_timestamp.isoformat(),
                    "end_timestamp": latest.end_timestamp.isoformat(),
                    "completion": latest.completion.value,
                    "session_dates": [value.isoformat() for value in session_dates],
                }
            )
        )
    return tuple(
        sorted(
            values,
            key=lambda item: cast(
                str, item.to_primitive()["timeframe_configuration_id"]
            ),
        )
    )


def _deduplication_key(alert: PredictionAlert, policy: AlertDeduplicationPolicy) -> str:
    if policy is AlertDeduplicationPolicy.EXACT_CONTEXT:
        return alert.alert_id
    if policy is not AlertDeduplicationPolicy.DECISION_BAR:
        raise PredictionScannerError("prediction alert deduplication policy is invalid")
    indicator_configurations: list[Primitive] = []
    for item in alert.indicators:
        primitive = item.to_primitive()
        indicator_configurations.append(
            {
                "timeframe_configuration_id": primitive["timeframe_configuration_id"],
                "alias": primitive["alias"],
                "indicator_configuration_id": primitive["indicator_configuration_id"],
                "backend": primitive["backend"],
            }
        )
    return configuration_identity(
        {
            "deduplication_policy": policy.value,
            "symbol": alert.symbol,
            "rule_id": alert.rule_id,
            "rule_implementation_version": alert.rule_implementation_version,
            "rule_configuration_id": alert.rule_configuration_id,
            "historical_study": alert.historical_study.to_primitive(),
            "indicator_configurations": indicator_configurations,
            "decision_timestamp": alert.decision_timestamp.astimezone(UTC).isoformat(),
            "completion_policy": alert.completion_policy,
        }
    )


__all__ = [
    "HISTORICAL_PREDICTION_STUDY_REFERENCE_SCHEMA_VERSION",
    "PREDICTION_ALERT_SCHEMA_VERSION",
    "PREDICTION_SCANNER_ENGINE_VERSION",
    "RESEARCH_ONLY_DISCLAIMER",
    "AlertDeduplicationClaim",
    "AlertDeduplicationPolicy",
    "AlertDeduplicationStore",
    "AlertPersistenceError",
    "ConsolePredictionAlertSink",
    "HistoricalPredictionStudyReference",
    "HistoricalStudyMismatchError",
    "InMemoryAlertDeduplicationStore",
    "JsonFileAlertDeduplicationStore",
    "JsonFilePredictionAlertSink",
    "PredictionAlert",
    "PredictionAlertSink",
    "PredictionRuleScanResult",
    "PredictionScanResult",
    "PredictionScanner",
    "PredictionScannerDataSource",
    "PredictionScannerError",
    "PredictionScannerRule",
    "PredictionScannerRuleBinding",
    "PredictionScannerSnapshot",
    "PublishedAlertDeduplication",
]
