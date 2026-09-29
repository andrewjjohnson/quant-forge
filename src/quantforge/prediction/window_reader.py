"""Version-aware read-only access to embedded and compact historical windows."""

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from quantforge.configuration import (
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.prediction.errors import InvalidPredictionOutputError
from quantforge.prediction.window_compact import (
    COMPACT_PREDICTION_WINDOW_SCHEMA_VERSIONS,
    CompactDecisionReceipt,
    CompactPredictionWindowDecision,
    PredictionWindowEvidence,
    WindowRecordCounts,
    checked_decisions,
    checked_receipts,
)
from quantforge.prediction.window_coverage import (
    PredictionDecisionReceipt,
    PredictionWindowObservation,
    decision_receipt,
    sparse_decision_receipt,
)
from quantforge.prediction.window_encoding import (
    WindowResultIdentity,
    canonical,
    decode,
    mapping,
    ordered_result_identity,
)
from quantforge.prediction.window_membership import MembershipCatalogues


@dataclass(frozen=True, slots=True)
class PredictionWindowReader:
    """Header/shared inspection without decision loading; lazy v2 verification.

    Iteration verifies each reference before yielding and verifies totals/result
    identity at exhaustion. Call verify_integrity() before accepting an artifact
    if only a prefix will be read. Scientific verification additionally uses
    validate_prediction_window_reader with independently trusted inputs.

    V1 remains a materialized JSON document because its existing format provides
    no record boundaries. V2 retains shared evidence, schedule, and one decision.
    V3 additionally retains its append-only membership catalogues, which grow
    with the source bars observed, never with the number of decisions. Yielded
    v3 decisions keep ranges; ``expanded_record()`` reconstructs lists on demand.
    V4 (QF-64) holds one coverage receipt per decision and nests rich evidence
    only for evaluated and skipped decisions. ``iterate_decision_receipts()`` and
    ``iterate_observations()`` serve every version; ``iterate_decisions()``
    rejects v4 because it has no rich record for every scheduled timestamp.
    A fresh file handle per traversal avoids shared cursor state.
    """

    path: Path
    schema_version: str
    header_snapshot: PrimitiveMappingSnapshot
    evidence: PredictionWindowEvidence
    _legacy: PrimitiveMappingSnapshot | None = field(default=None, repr=False)

    @classmethod
    def open(cls, path: Path) -> "PredictionWindowReader":
        with path.open("rb") as stream:
            first = stream.readline()
            # A v2 header is always exactly one canonical line. Pretty-printed
            # legacy objects may span lines, so only those use a complete read.
            try:
                record = decode(first)
            except InvalidPredictionOutputError:
                stream.seek(0)
                record = decode(stream.read())
            if "record_type" in record:
                header = decode(first, canonical_line=True)
                version = header.get("schema_version")
                if version not in COMPACT_PREDICTION_WINDOW_SCHEMA_VERSIONS:
                    raise InvalidPredictionOutputError(
                        "unsupported compact window schema version"
                    )
                evidence = PredictionWindowEvidence.from_primitive(
                    decode(stream.readline(), canonical_line=True)
                )
                expected = {
                    "record_type": "header",
                    "component": "quantforge_prediction_window",
                    "schema_version": evidence.schema_version,
                    "window_id": evidence.window_id,
                    "shared_evidence_id": evidence.evidence_id,
                    "schedule_id": evidence.schedule.schedule_id,
                    "decision_count": len(evidence.schedule.decision_timestamps),
                }
                verified_later = {"window_result_id", "record_counts"}
                if evidence.normalized_membership:
                    verified_later.add("membership_catalogues")
                if (
                    set(header) != set(expected) | verified_later
                    or any(header.get(key) != value for key, value in expected.items())
                    or type(header.get("decision_count")) is not int
                    or not isinstance(header.get("window_result_id"), str)
                    or not isinstance(header.get("record_counts"), dict)
                    or not isinstance(header.get("membership_catalogues", []), list)
                ):
                    raise InvalidPredictionOutputError(
                        "unsupported or inconsistent compact header/version"
                    )
                return cls(
                    path,
                    evidence.schema_version,
                    PrimitiveMappingSnapshot.capture(header),
                    evidence,
                )
            stream.seek(0)
            snapshot = decode(stream.read())
        return cls.from_snapshot(snapshot, path=path)

    @classmethod
    def from_snapshot(
        cls, snapshot: PrimitiveMapping, *, path: Path = Path(".")
    ) -> "PredictionWindowReader":
        """Read an existing embedded v1 snapshot without writing or migrating it."""
        if set(snapshot) != {"manifest", "decisions"}:
            raise InvalidPredictionOutputError("unsupported embedded window shape")
        manifest = mapping(snapshot["manifest"])
        if manifest.get("schema_version") != "1":
            raise InvalidPredictionOutputError(
                "unsupported embedded window schema version"
            )
        if not isinstance(snapshot["decisions"], list):
            raise InvalidPredictionOutputError("embedded window decisions are missing")
        identity = {
            key: value
            for key, value in manifest.items()
            if key
            not in {"window_id", "window_result_id", "schedule_id", "record_counts"}
        }
        evidence = PredictionWindowEvidence(PrimitiveMappingSnapshot.capture(identity))
        if (
            manifest.get("window_id") != configuration_identity(identity)
            or manifest.get("schedule_id") != evidence.schedule.schedule_id
        ):
            raise InvalidPredictionOutputError(
                "embedded window identity is inconsistent"
            )
        return cls(
            path,
            "1",
            PrimitiveMappingSnapshot.capture(manifest),
            evidence,
            PrimitiveMappingSnapshot.capture(snapshot),
        )

    @classmethod
    def from_reference(
        cls, snapshot: PrimitiveMapping, *, root: Path
    ) -> "PredictionWindowReader":
        """Resolve a QF-56 reference, or normalize an embedded v1 snapshot.

        The producer's fixed relative filename is part of its contract. It must
        resolve inside the supplied artifact directory, including through symlinks.
        """
        if "manifest" in snapshot:
            return cls.from_snapshot(snapshot)
        path = root / "prediction-window.jsonl"
        if (
            snapshot.get("schema_version")
            not in COMPACT_PREDICTION_WINDOW_SCHEMA_VERSIONS
            or snapshot.get("path") != path.name
            or set(snapshot) != {"schema_version", "path", "header"}
            or not path.resolve().is_relative_to(root.resolve())
        ):
            raise InvalidPredictionOutputError("invalid compact window reference")
        reader = cls.open(path)
        if (
            reader.schema_version != snapshot["schema_version"]
            or reader.header() != snapshot["header"]
        ):
            raise InvalidPredictionOutputError(
                "compact reference differs from artifact"
            )
        return reader

    def manifest(self) -> PrimitiveMapping:
        """Scientific metadata and truthful physical identity, without decisions.

        The shared identity retains QF-42's versioned scientific scope. The
        returned schema/window/result identities describe the actual artifact.
        This is an inspection view, never an embedded-window serialization.
        """
        return {**self.evidence.identity_snapshot.to_primitive(), **self.header()}

    @property
    def decision_count(self) -> int:
        return len(self.evidence.schedule.decision_timestamps)

    def header(self) -> PrimitiveMapping:
        return self.header_snapshot.to_primitive()

    def shared_evidence(self) -> PredictionWindowEvidence:
        return self.evidence

    def _lines(self) -> Iterator[bytes]:
        with self.path.open("rb") as stream:
            if (
                stream.readline() != canonical(self.header()) + b"\n"
                or stream.readline() != canonical(self.evidence.to_primitive()) + b"\n"
            ):
                raise InvalidPredictionOutputError(
                    "window header/evidence changed during reading"
                )
            yield from stream

    def _records(self) -> Iterator[CompactPredictionWindowDecision]:
        if self._legacy is not None:
            snapshot = self._legacy.to_primitive()
            decisions = cast(list[PrimitiveMapping], snapshot["decisions"])
            if (
                ordered_result_identity(
                    cast(str, self.header()["window_id"]), decisions
                )
                != self.header()["window_result_id"]
            ):
                raise InvalidPredictionOutputError(
                    "embedded window result identity is inconsistent"
                )
            for index, decision in enumerate(decisions):
                yield CompactPredictionWindowDecision.from_embedded(
                    decision, sequence=index, evidence=self.evidence
                )
            return
        for line in self._lines():
            yield CompactPredictionWindowDecision(
                PrimitiveMappingSnapshot.capture(decode(line, canonical_line=True))
            )

    def _receipts(self) -> Iterator[CompactDecisionReceipt]:
        for line in self._lines():
            yield CompactDecisionReceipt(
                PrimitiveMappingSnapshot.capture(decode(line, canonical_line=True))
            )

    def _coverage(self) -> Iterator[PredictionDecisionReceipt]:
        """Checked coverage of every scheduled decision, in exact order.

        Each record is verified before it is yielded; totals, ordered result
        identity and catalogue contents are verified at exhaustion. Retained
        state is bounded: hash state, counters and (v3/v4) catalogues.
        """
        digest = WindowResultIdentity(self.evidence.window_id)
        counts = WindowRecordCounts()
        membership = (
            MembershipCatalogues() if self.evidence.normalized_membership else None
        )
        timestamps = self.evidence.schedule.decision_timestamps
        observations = 0

        def count(update: Callable[[], None]) -> None:
            try:
                update()
            except (KeyError, TypeError, ValueError) as error:
                raise InvalidPredictionOutputError(
                    "invalid decision counts payload"
                ) from error

        if self.evidence.sparse_decisions:
            assert membership is not None
            for receipt in checked_receipts(self._receipts(), self.evidence):
                decision = receipt.decision
                if decision is not None:
                    view = membership.accept(decision.to_primitive())
                    decision = CompactPredictionWindowDecision(decision.snapshot, view)
                digest.update(receipt.to_primitive())
                count(lambda: counts.update_receipt(receipt))
                yield sparse_decision_receipt(
                    receipt,
                    decision,
                    decision_timestamp=timestamps[receipt.sequence],
                    observation_start=observations,
                )
                observations += receipt.observation_count
        else:
            for record in checked_decisions(self._records(), self.evidence):
                view = None if membership is None else membership.accept(record)
                digest.update(record)
                count(lambda: counts.update(record))
                item = decision_receipt(
                    record,
                    CompactPredictionWindowDecision(
                        PrimitiveMappingSnapshot.capture(record), view
                    ),
                    decision_timestamp=timestamps[cast(int, record["sequence"])],
                    observation_start=observations,
                )
                yield item
                observations += item.observation_count
        header = self.header()
        if configuration_identity(counts.totals) != configuration_identity(
            mapping(header.get("record_counts"))
        ):
            raise InvalidPredictionOutputError("window record counts are inconsistent")
        if (
            self.schema_version != "1"
            and digest.hexdigest() != header["window_result_id"]
        ):
            raise InvalidPredictionOutputError(
                "compact window result identity is inconsistent"
            )
        if membership is not None and canonical(
            {"catalogues": membership.summaries()}
        ) != canonical({"catalogues": header["membership_catalogues"]}):
            raise InvalidPredictionOutputError(
                "window membership catalogues are inconsistent"
            )

    def iterate_decision_receipts(self) -> Iterator[PredictionDecisionReceipt]:
        """Coverage of every scheduled decision, for every physical version."""
        return self._coverage()

    def iterate_observations(self) -> Iterator[PredictionWindowObservation]:
        """Generated signals only, in schedule then signal order.

        The whole window is still verified; a v4 no-prediction receipt is a
        small line that is checked but never expanded into a rich object.
        """
        for receipt in self._coverage():
            if receipt.decision is not None:
                yield from receipt.observations()

    def iterate_decisions(self) -> Iterator[CompactPredictionWindowDecision]:
        """Versions 1-3 expose one normalized record per scheduled decision.

        V3 catalogue segments and ranges are authenticated against the preceding
        catalogue state before a decision is yielded with its catalogue view.
        Version 4 fails explicitly: it retains rich decisions only where they
        carry observations or skipped-context evidence.
        """
        if self.evidence.sparse_decisions:
            raise InvalidPredictionOutputError(
                "sparse windows have no rich record per decision; use "
                "iterate_decision_receipts() or iterate_observations()"
            )
        for receipt in self._coverage():
            assert receipt.decision is not None
            yield receipt.decision

    def verify_integrity(self) -> None:
        """Exhaustively check identities, references, membership, order and counts."""
        for _ in self._coverage():
            pass
