"""Versions 2 and 3 of the QF-42 window: one scope, normalized decisions.

Construction consumes already completed results. Execution/checkpoint/resume
remain separate concerns. No method expands shared evidence into decisions.
Version 3 additionally stores visible-bar membership once in append-only QF-62
catalogues referenced by exact ranges; see ``window_membership``.
"""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, cast

from quantforge.configuration import (
    Primitive,
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.prediction.errors import InvalidPredictionOutputError
from quantforge.prediction.window import (
    PredictionDecisionSchedule,
    PredictionWindowResult,
    _window_record_counts,  # pyright: ignore[reportPrivateUsage]
)
from quantforge.prediction.window_encoding import (
    canonical,
    mapping,
    ordered_result_identity,
)
from quantforge.prediction.window_membership import (
    CATALOGUE_SEGMENTS_FIELD,
    MembershipCatalogues,
    MembershipView,
    prediction_context,
    with_prediction_context,
)
from quantforge.timeframes import Timeframe

COMPACT_PREDICTION_WINDOW_SCHEMA_VERSION = "2"
NORMALIZED_PREDICTION_WINDOW_SCHEMA_VERSION = "3"
# Physical JSONL representations sharing this reader/validator contract.
COMPACT_PREDICTION_WINDOW_SCHEMA_VERSIONS = (
    COMPACT_PREDICTION_WINDOW_SCHEMA_VERSION,
    NORMALIZED_PREDICTION_WINDOW_SCHEMA_VERSION,
)
_IDENTITY_FIELDS = {
    "component",
    "schema_version",
    "engine_version",
    "prediction_engine_version",
    "schedule",
    "market_data",
    "configuration",
    "dataset_family_fingerprint",
    "context_environment",
    "indicator_backend_environment",
}
_SHARED_STUDY_FIELDS = {"market_data", "configuration", "engine_version"}
_DECISION_FIELDS = {
    "decision_timestamp",
    "prediction_study_id",
    "context_id",
    "status",
    "prediction_study",
    "generated_signals",
}
_LOCAL_MANIFEST_FIELDS = {
    "component",
    "feature_outcome_boundary",
    "record_counts",
    "study_id",
    "prediction_context",
}


@dataclass(frozen=True, slots=True)
class PredictionWindowEvidence:
    """Immutable full scientific scope, including the unchanged QF-42 schedule.

    Current QF-42 windows require one market/configuration/engine scope. Different
    bounded cutoffs or sources therefore belong to different windows, never to
    an equality-by-coincidence evidence pool. ``schema_version`` is the physical
    representation; the nested scientific identity keeps its QF-42 schema "1".
    """

    identity_snapshot: PrimitiveMappingSnapshot
    schema_version: str = COMPACT_PREDICTION_WINDOW_SCHEMA_VERSION
    evidence_id: str = field(init=False)
    window_id: str = field(init=False)
    study_scope_id: str = field(init=False, repr=False)
    schedule: PredictionDecisionSchedule = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.schema_version not in COMPACT_PREDICTION_WINDOW_SCHEMA_VERSIONS:
            raise InvalidPredictionOutputError("unsupported compact window version")
        identity = self.identity_snapshot.to_primitive()
        if (
            set(identity) != _IDENTITY_FIELDS
            or identity.get("component") != "quantforge_prediction_window"
            or identity.get("schema_version") != "1"
        ):
            raise InvalidPredictionOutputError("unsupported shared window identity")
        mapping(identity["market_data"])
        mapping(identity["configuration"])
        schedule = mapping(identity["schedule"])
        try:
            typed = PredictionDecisionSchedule(
                Timeframe.from_primitive(mapping(schedule["primary_timeframe"])),
                datetime.fromisoformat(cast(str, schedule["start_timestamp"])),
                datetime.fromisoformat(cast(str, schedule["end_timestamp"])),
            )
        except (ValueError, TypeError, KeyError) as error:
            raise InvalidPredictionOutputError("invalid shared schedule") from error
        if typed.to_primitive() != schedule:
            raise InvalidPredictionOutputError("shared schedule membership is invalid")
        evidence_id = configuration_identity(self.to_primitive())
        object.__setattr__(self, "evidence_id", evidence_id)
        object.__setattr__(self, "schedule", typed)
        object.__setattr__(
            self,
            "study_scope_id",
            configuration_identity(
                {
                    "market_data": identity["market_data"],
                    "configuration": identity["configuration"],
                    "engine_version": identity["prediction_engine_version"],
                }
            ),
        )
        object.__setattr__(
            self,
            "window_id",
            configuration_identity(
                {
                    "component": "quantforge_prediction_window",
                    "schema_version": self.schema_version,
                    "shared_evidence_id": evidence_id,
                }
            ),
        )

    @property
    def normalized_membership(self) -> bool:
        return self.schema_version == NORMALIZED_PREDICTION_WINDOW_SCHEMA_VERSION

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "record_type": "shared_evidence",
            "schema_version": self.schema_version,
            "window_identity": self.identity_snapshot.to_primitive(),
        }

    @classmethod
    def from_primitive(cls, record: PrimitiveMapping) -> "PredictionWindowEvidence":
        version = record.get("schema_version")
        if (
            set(record) != {"record_type", "schema_version", "window_identity"}
            or record.get("record_type") != "shared_evidence"
            or version not in COMPACT_PREDICTION_WINDOW_SCHEMA_VERSIONS
        ):
            raise InvalidPredictionOutputError("invalid shared evidence record/version")
        return cls(
            PrimitiveMappingSnapshot.capture(mapping(record["window_identity"])),
            cast(str, version),
        )


@dataclass(frozen=True, slots=True)
class CompactPredictionWindowDecision:
    """One normalized result with a scope-bound reference and original study ID.

    Version 3 records carry catalogue segments and membership ranges. A reader
    attaches the accepted catalogue ``membership`` view so the exact expanded
    record can be reconstructed on demand; it is not part of record equality.
    """

    snapshot: PrimitiveMappingSnapshot
    membership: MembershipView | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        record = self.snapshot.to_primitive()
        physical = {"record_type", "sequence", "shared_evidence_id", "decision_id"}
        if CATALOGUE_SEGMENTS_FIELD in record:
            physical.add(CATALOGUE_SEGMENTS_FIELD)
        if (
            set(record) != _DECISION_FIELDS | physical
            or record.get("record_type") != "decision"
            or type(record.get("sequence")) is not int
            or cast(int, record["sequence"]) < 0
            or not isinstance(record.get("shared_evidence_id"), str)
            or not isinstance(record.get("decision_timestamp"), str)
            or record.get("status") not in ("evaluated", "skipped", "no_prediction")
        ):
            raise InvalidPredictionOutputError("invalid compact decision shape")
        study = mapping(record["prediction_study"])
        if set(study) != {"manifest", "rows"} or not isinstance(study["rows"], list):
            raise InvalidPredictionOutputError("invalid compact prediction study")
        manifest = mapping(study["manifest"])
        if set(manifest) != _LOCAL_MANIFEST_FIELDS:
            raise InvalidPredictionOutputError(
                "compact decision contains embedded evidence"
            )
        if not isinstance(record.get("generated_signals"), list):
            raise InvalidPredictionOutputError("compact generated signals are missing")
        for signal in cast(list[Primitive], record["generated_signals"]):
            mapping(mapping(mapping(signal).get("prediction")).get("values"))
            mapping(mapping(signal).get("features"))
        if record["decision_id"] != configuration_identity(
            {key: value for key, value in record.items() if key != "decision_id"}
        ):
            raise InvalidPredictionOutputError(
                "compact decision identity is inconsistent"
            )

    @classmethod
    def from_embedded(
        cls,
        decision: PrimitiveMapping,
        *,
        sequence: int,
        evidence: PredictionWindowEvidence,
        membership: MembershipCatalogues | None = None,
    ) -> "CompactPredictionWindowDecision":
        """Detach one v1 snapshot; never retain its embedded shared payload.

        Version 3 normalizes membership against ``membership``, the preceding
        accepted catalogue state of the same traversal, without changing it.
        """
        if set(decision) != _DECISION_FIELDS:
            raise InvalidPredictionOutputError("unsupported embedded decision fields")
        study = mapping(decision["prediction_study"])
        manifest = mapping(study["manifest"])
        if (
            configuration_identity(
                {key: manifest.get(key) for key in _SHARED_STUDY_FIELDS}
            )
            != evidence.study_scope_id
        ):
            raise InvalidPredictionOutputError("decision has foreign shared evidence")
        local: PrimitiveMapping = {
            **decision,
            "record_type": "decision",
            "sequence": sequence,
            "shared_evidence_id": evidence.evidence_id,
            "prediction_study": {
                **study,
                "manifest": {
                    key: value
                    for key, value in manifest.items()
                    if key not in _SHARED_STUDY_FIELDS
                },
            },
        }
        if evidence.normalized_membership:
            if membership is None:
                raise InvalidPredictionOutputError(
                    "normalized membership requires the window catalogue state"
                )
            context, segments = membership.normalize(prediction_context(local))
            local = with_prediction_context(local, context)
            local[CATALOGUE_SEGMENTS_FIELD] = segments
        elif membership is not None:
            raise InvalidPredictionOutputError(
                "schema 2 decisions do not use membership catalogues"
            )
        local["decision_id"] = configuration_identity(local)
        return cls(PrimitiveMappingSnapshot.capture(local))

    @property
    def schema_version(self) -> str:
        """Physical decision form; its evidence reference must agree."""
        return (
            NORMALIZED_PREDICTION_WINDOW_SCHEMA_VERSION
            if CATALOGUE_SEGMENTS_FIELD in self.to_primitive()
            else COMPACT_PREDICTION_WINDOW_SCHEMA_VERSION
        )

    def to_primitive(self) -> PrimitiveMapping:
        return self.snapshot.to_primitive()

    def expanded_record(self) -> PrimitiveMapping:
        """The record with exact expanded membership, for scientific validation.

        ``decision_id`` identifies the persisted normalized record, not this view.
        Schema 2 records are already expanded and are returned unchanged.
        """
        record = self.to_primitive()
        if CATALOGUE_SEGMENTS_FIELD not in record:
            return record
        if self.membership is None:
            raise InvalidPredictionOutputError(
                "normalized decision was not accepted with its window catalogues"
            )
        return self.membership.expand_record(record)

    @property
    def decision_id(self) -> str:
        return cast(str, self.to_primitive()["decision_id"])

    @property
    def prediction_study_id(self) -> str:
        return cast(str, self.to_primitive()["prediction_study_id"])


def checked_decisions(
    decisions: Iterable[CompactPredictionWindowDecision],
    evidence: PredictionWindowEvidence,
) -> Iterator[PrimitiveMapping]:
    timestamps = evidence.schedule.decision_timestamps
    count = 0
    for count, decision in enumerate(decisions, 1):
        record = decision.to_primitive()
        if (
            count > len(timestamps)
            or record["sequence"] != count - 1
            or (CATALOGUE_SEGMENTS_FIELD in record) != evidence.normalized_membership
            or record["shared_evidence_id"] != evidence.evidence_id
            or record["decision_timestamp"] != timestamps[count - 1].isoformat()
        ):
            raise InvalidPredictionOutputError(
                "compact decision reference/order differs from window"
            )
        yield record
    if count != len(timestamps):
        raise InvalidPredictionOutputError(
            "compact window decision membership is incomplete"
        )


class WindowRecordCounts:
    """Only aggregate integers and disposition totals; retain no decisions."""

    def __init__(self) -> None:
        self.totals = _window_record_counts([])

    def update(self, decision: PrimitiveMapping) -> None:
        dispositions = mapping(self.totals["signal_dispositions"])
        for key, value in _window_record_counts([decision]).items():
            if key == "signal_dispositions":
                for disposition, count in mapping(value).items():
                    dispositions[disposition] = cast(
                        int, dispositions.get(disposition, 0)
                    ) + cast(int, count)
            else:
                self.totals[key] = cast(int, self.totals[key]) + cast(int, value)


def decision_counts(decisions: Iterable[PrimitiveMapping]) -> PrimitiveMapping:
    counts = WindowRecordCounts()
    for decision in decisions:
        counts.update(decision)
    return counts.totals


def window_header(
    evidence: PredictionWindowEvidence,
    *,
    window_result_id: str,
    decision_count: int,
    record_counts: PrimitiveMapping,
    membership: MembershipCatalogues | None = None,
) -> PrimitiveMapping:
    """Finalized header; version 3 also names each complete catalogue's content."""
    header: PrimitiveMapping = {
        "record_type": "header",
        "component": "quantforge_prediction_window",
        "schema_version": evidence.schema_version,
        "window_id": evidence.window_id,
        "window_result_id": window_result_id,
        "shared_evidence_id": evidence.evidence_id,
        "schedule_id": evidence.schedule.schedule_id,
        "decision_count": decision_count,
        "record_counts": record_counts,
    }
    if evidence.normalized_membership:
        header["membership_catalogues"] = (
            MembershipCatalogues() if membership is None else membership
        ).summaries()
    return header


@dataclass(frozen=True, slots=True)
class CompactPredictionWindowResult:
    """A completed, immutable representation of the existing historical window."""

    evidence: PredictionWindowEvidence
    decisions: tuple[CompactPredictionWindowDecision, ...]
    header_snapshot: PrimitiveMappingSnapshot = field(init=False, repr=False)

    def __post_init__(self) -> None:
        membership = (
            MembershipCatalogues() if self.evidence.normalized_membership else None
        )

        def accepted() -> Iterator[PrimitiveMapping]:
            for record in checked_decisions(self.decisions, self.evidence):
                if membership is not None:
                    membership.accept(record)
                yield record

        result_id = ordered_result_identity(self.evidence.window_id, accepted())
        object.__setattr__(
            self,
            "header_snapshot",
            PrimitiveMappingSnapshot.capture(
                window_header(
                    self.evidence,
                    window_result_id=result_id,
                    decision_count=len(self.decisions),
                    record_counts=decision_counts(
                        item.to_primitive() for item in self.decisions
                    ),
                    membership=membership,
                )
            ),
        )

    @classmethod
    def from_window(
        cls,
        window: PredictionWindowResult[Any, Any, Any],
        *,
        schema_version: str = COMPACT_PREDICTION_WINDOW_SCHEMA_VERSION,
    ) -> "CompactPredictionWindowResult":
        evidence = PredictionWindowEvidence(window.identity_snapshot, schema_version)
        membership = MembershipCatalogues() if evidence.normalized_membership else None
        decisions: list[CompactPredictionWindowDecision] = []
        for index, decision in enumerate(window.decisions):
            compact = CompactPredictionWindowDecision.from_embedded(
                decision.to_primitive(),
                sequence=index,
                evidence=evidence,
                membership=membership,
            )
            if membership is not None:
                compact = replace(
                    compact, membership=membership.accept(compact.to_primitive())
                )
            decisions.append(compact)
        return cls(evidence, tuple(decisions))

    @property
    def window_id(self) -> str:
        return self.evidence.window_id

    @property
    def window_result_id(self) -> str:
        return cast(str, self.header_snapshot.to_primitive()["window_result_id"])

    @property
    def decision_count(self) -> int:
        return len(self.decisions)

    def iter_serialized_records(self) -> Iterator[bytes]:
        """Finalized JSONL, shared evidence once; no incremental execution writer."""
        yield canonical(self.header_snapshot.to_primitive()) + b"\n"
        yield canonical(self.evidence.to_primitive()) + b"\n"
        for decision in self.decisions:
            yield decision.snapshot.canonical_json.encode("utf-8") + b"\n"

    def serialize(self) -> bytes:
        return b"".join(self.iter_serialized_records())

    def to_primitive(self) -> PrimitiveMapping:
        """Normalized inspection form; canonical disk encoding is JSONL."""
        return {
            "header": self.header_snapshot.to_primitive(),
            "shared_evidence": self.evidence.to_primitive(),
            "decisions": cast(
                list[Primitive], [d.to_primitive() for d in self.decisions]
            ),
        }
