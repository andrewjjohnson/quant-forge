"""Current-data scanning and research-only alerts for validated QF-31 rules."""

from __future__ import annotations

import fcntl
import json
import os
import sys
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
from quantforge.data.multi_timeframe import MultiTimeframeContext
from quantforge.prediction.context import (
    PredictionContextRequirements,
    PredictionRuleContext,
    build_prediction_rule_context,
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


class AlertDeduplicationPolicy(StrEnum):
    """Select which causally distinct current evaluations may alert again."""

    EXACT_CONTEXT = "exact_context"
    DECISION_BAR = "decision_bar"


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
    rule_configuration: PrimitiveMappingSnapshot
    context_requirements: PrimitiveMappingSnapshot
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
            )
        ):
            raise PredictionScannerError(
                "historical study references require complete rule identity"
            )
        if self.sample_count is not None and self.sample_count < 0:
            raise PredictionScannerError(
                "historical study sample count cannot be negative"
            )

    @classmethod
    def capture(
        cls,
        *,
        study_id: str,
        rule: PredictionScannerRule,
        summary: PrimitiveMapping | None = None,
        sample_count: int | None = None,
    ) -> HistoricalPredictionStudyReference:
        """Capture the exact rule and backend semantics recorded by a study."""
        return cls(
            study_id=study_id,
            rule_id=rule.name,
            rule_implementation_version=rule.implementation_version,
            rule_configuration_id=rule.configuration_id,
            rule_configuration=PrimitiveMappingSnapshot.capture(rule.configuration()),
            context_requirements=PrimitiveMappingSnapshot.capture(
                rule.context_requirements.to_primitive()
            ),
            summary=(
                None if summary is None else PrimitiveMappingSnapshot.capture(summary)
            ),
            sample_count=sample_count,
        )

    def validate_rule(self, rule: PredictionScannerRule) -> None:
        """Fail closed if any logical, indicator, backend, or policy input differs."""
        current_configuration = PrimitiveMappingSnapshot.capture(rule.configuration())
        current_requirements = PrimitiveMappingSnapshot.capture(
            rule.context_requirements.to_primitive()
        )
        if (
            rule.name != self.rule_id
            or rule.implementation_version != self.rule_implementation_version
            or rule.configuration_id != self.rule_configuration_id
            or current_configuration != self.rule_configuration
            or current_requirements != self.context_requirements
            or configuration_identity(current_configuration.to_primitive())
            != self.rule_configuration_id
        ):
            raise HistoricalStudyMismatchError(
                "current rule, indicator configuration/backend, or context policy "
                f"does not match historical study {self.study_id}"
            )

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "study_id": self.study_id,
            "rule_id": self.rule_id,
            "rule_implementation_version": self.rule_implementation_version,
            "rule_configuration_id": self.rule_configuration_id,
            "rule_configuration": self.rule_configuration.to_primitive(),
            "context_requirements": self.context_requirements.to_primitive(),
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
    source_mode: str

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.context), MultiTimeframeContext):
            raise PredictionScannerError("scanner source returned an invalid context")
        if not self.prediction_dataset_id or not self.symbol or not self.source_mode:
            raise PredictionScannerError(
                "scanner source snapshot requires dataset, symbol, and source mode"
            )
        if not isinstance(cast(object, self.adjustment_basis), AdjustmentBasis):
            raise PredictionScannerError(
                "scanner source snapshot adjustment basis is invalid"
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
            historical_study_id=self.historical_study.study_id,
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
    """Persist one content-addressed JSON file per alert without overwrites."""

    def __init__(self, output_directory: Path) -> None:
        self.output_directory = output_directory

    def emit(self, alert: PredictionAlert) -> None:
        self.output_directory.mkdir(parents=True, exist_ok=True)
        path = self.output_directory / f"{alert.alert_id}.json"
        content = alert.serialize()
        try:
            with path.open("xb") as stream:
                stream.write(content)
        except FileExistsError:
            if path.read_bytes() != content:
                raise AlertPersistenceError(
                    f"existing alert artifact differs: {path}"
                ) from None


class AlertDeduplicationClaim(Protocol):
    """One exclusive pending claim that can become published or be released."""

    def publish(self) -> None: ...

    def release(self) -> None: ...


class AlertDeduplicationStore(Protocol):
    """Atomically reserve an alert key through a recoverable claim lifecycle."""

    def claim(
        self, deduplication_key: str, alert_id: str
    ) -> AlertDeduplicationClaim | None: ...


class InMemoryAlertDeduplicationStore:
    """Process-local deterministic deduplication for embedded scanners and tests."""

    def __init__(self) -> None:
        self._claims: dict[str, tuple[str, str]] = {}

    def claim(
        self, deduplication_key: str, alert_id: str
    ) -> AlertDeduplicationClaim | None:
        if deduplication_key in self._claims:
            return None
        self._claims[deduplication_key] = (alert_id, "pending")
        return _InMemoryAlertDeduplicationClaim(self, deduplication_key, alert_id)

    def publish_claim(self, deduplication_key: str, alert_id: str) -> None:
        if self._claims.get(deduplication_key) != (alert_id, "pending"):
            raise AlertPersistenceError("in-memory deduplication claim is not pending")
        self._claims[deduplication_key] = (alert_id, "published")

    def release_claim(self, deduplication_key: str, alert_id: str) -> None:
        if self._claims.get(deduplication_key) == (alert_id, "pending"):
            del self._claims[deduplication_key]


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
    """Cross-run deduplication with crash-recoverable locked state files."""

    def __init__(self, state_directory: Path) -> None:
        self.state_directory = state_directory

    def _path(self, deduplication_key: str) -> Path:
        return self.state_directory / f"{deduplication_key}.json"

    def claim(
        self, deduplication_key: str, alert_id: str
    ) -> AlertDeduplicationClaim | None:
        self.state_directory.mkdir(parents=True, exist_ok=True)
        path = self._path(deduplication_key)
        descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(descriptor)
            return None
        claim = _JsonFileAlertDeduplicationClaim(
            path, descriptor, deduplication_key, alert_id
        )
        try:
            existing_state = claim.read_state()
            if existing_state == "published":
                claim.close()
                return None
            claim.write_state("pending")
        except BaseException:
            claim.close()
            raise
        return claim


@dataclass(slots=True)
class _JsonFileAlertDeduplicationClaim:
    path: Path
    descriptor: int
    deduplication_key: str
    alert_id: str
    _closed: bool = False

    def read_state(self) -> str | None:
        os.lseek(self.descriptor, 0, os.SEEK_SET)
        try:
            content = os.read(self.descriptor, 64 * 1024)
            if not content:
                return None
            decoded = cast(object, json.loads(content))
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as error:
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
        state = existing.get("state")
        if state is None and isinstance(existing.get("alert_id"), str):
            # The original write-on-claim format did not distinguish delivery
            # from interruption. Recover it as pending so it cannot suppress an
            # undelivered alert forever.
            return "pending"
        if state not in {"pending", "published"}:
            raise AlertPersistenceError(
                f"deduplication state lifecycle is invalid: {self.path}"
            )
        return cast(str, state)

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
            os.lseek(self.descriptor, 0, os.SEEK_SET)
            os.ftruncate(self.descriptor, 0)
            with os.fdopen(self.descriptor, "wb", closefd=False) as stream:
                stream.write(content)
                stream.flush()
            os.fsync(self.descriptor)
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
            fcntl.flock(self.descriptor, fcntl.LOCK_UN)
        finally:
            os.close(self.descriptor)
            self._closed = True


@dataclass(frozen=True, slots=True)
class PredictionRuleScanResult:
    """Auditable accepted, rejected, or deduplicated result for one rule."""

    rule_id: str
    rule_configuration_id: str
    context_id: str
    evaluation: PrimitiveMappingSnapshot
    alert: PredictionAlert | None
    duplicate_alert_id: str | None = None

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
            snapshot = self.source.prepare_context(
                binding.rule.context_requirements,
                as_of=decision_as_of,
                refresh=not dry_run,
            )
            if snapshot.context.as_of != decision_as_of:
                raise PredictionScannerError(
                    "scanner source context as-of does not match the requested decision"
                )
            rule_context = build_prediction_rule_context(
                binding.rule.context_requirements,
                snapshot.context,
                prediction_dataset_id=snapshot.prediction_dataset_id,
                symbol=snapshot.symbol,
                prediction_adjustment_basis=snapshot.adjustment_basis,
            )
            evaluation = binding.rule.evaluate(rule_context)
            output = binding.rule.generate_with_context(rule_context)
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
            deduplication_claim = self.deduplication_store.claim(
                deduplication_key, alert.alert_id
            )
            if deduplication_claim is None:
                results.append(
                    PredictionRuleScanResult(
                        binding.rule.name,
                        binding.rule.configuration_id,
                        rule_context.context_id,
                        PrimitiveMappingSnapshot.capture(evaluation.to_primitive()),
                        None,
                        alert.alert_id,
                    )
                )
                continue
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
            "source_mode": snapshot.source_mode,
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
            historical_study_id=binding.historical_study.study_id,
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
    historical_study_id: str,
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
        "historical_study_id": historical_study_id,
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
            "historical_study_id": alert.historical_study.study_id,
            "indicator_configurations": indicator_configurations,
            "decision_timestamp": alert.decision_timestamp.astimezone(UTC).isoformat(),
            "completion_policy": alert.completion_policy,
        }
    )


__all__ = [
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
]
