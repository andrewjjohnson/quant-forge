"""Offline scientific verification over the common normalized window reader."""

from datetime import date
from typing import cast

from quantforge.configuration import (
    Primitive,
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.data.exceptions import ValidationError
from quantforge.data.models import (
    BoundedPredictionProvenance,
    DatasetMetadata,
    PredictionInputProvenance,
)
from quantforge.data.prediction_inputs import (
    validate_prediction_provenance,
    validate_prediction_source_reference,
)
from quantforge.data.prediction_views import validate_prediction_view_lineage
from quantforge.data.prepared_prediction_views import PreparedProjectionRegistry
from quantforge.prediction.errors import InvalidPredictionOutputError
from quantforge.prediction.outcome_temporal import (
    ElapsedDurationHorizon,
    outcome_temporal_configuration,
)
from quantforge.prediction.window import PredictionDecisionSchedule
from quantforge.prediction.window_encoding import StudyIdentity, mapping
from quantforge.prediction.window_reader import PredictionWindowReader
from quantforge.prediction.window_validation import (
    _validate_decision_record,  # pyright: ignore[reportPrivateUsage]
)
from quantforge.timeframes import Timeframe


def _reference(
    provenance: PredictionInputProvenance,
    reference: PrimitiveMapping,
    timeframe: Timeframe,
) -> None:
    if (
        reference.get("timeframe_configuration_id") != timeframe.configuration_id
        or configuration_identity(timeframe.session_policy.to_primitive())
        != provenance.session_policy_id
    ):
        raise InvalidPredictionOutputError(
            "prediction source timeframe/session differs"
        )
    validate_prediction_source_reference(provenance, reference)


def _outcome_source(
    identity: PrimitiveMapping,
    provenance: PredictionInputProvenance | None,
) -> None:
    labeler = mapping(mapping(identity["configuration"])["outcome_labeler"])
    temporal = outcome_temporal_configuration(
        mapping(labeler["configuration"]),
        required_future_sessions=cast(
            int | None, labeler.get("required_future_sessions")
        ),
    )
    source = labeler.get("outcome_source")
    if source is None:
        if isinstance(temporal.horizon, ElapsedDurationHorizon):
            raise InvalidPredictionOutputError(
                "elapsed outcomes require canonical source evidence"
            )
        return
    source = mapping(source)
    timeframe = temporal.observation_timeframe
    reference = mapping(source["source_reference"])
    if (
        timeframe is None
        or reference.get("timeframe_configuration_id") != timeframe.configuration_id
    ):
        raise InvalidPredictionOutputError(
            "outcome source observation timeframe differs"
        )
    if provenance is not None:
        if (
            source.get("family_manifest_id")
            != provenance.family_manifest.to_primitive()["manifest_id"]
        ):
            raise InvalidPredictionOutputError("outcome source family manifest differs")
        _reference(provenance, reference, timeframe)


def _context_sources(
    context: PrimitiveMapping,
    provenance: PredictionInputProvenance | None,
) -> None:
    # A rejected context intentionally records its own incompatible provenance;
    # QF-42 still validates that rejected snapshot's internal consistency below.
    if provenance is None or context.get("status") == "skipped":
        return
    source = mapping(context["source_context"])
    if (
        source.get("dataset_family_manifest_id")
        != provenance.family_manifest.to_primitive()["manifest_id"]
    ):
        raise InvalidPredictionOutputError(
            "context family manifest differs from shared input"
        )
    requirements = mapping(context["requirements"])
    declarations = [
        mapping(requirements["primary"]),
        *(mapping(item) for item in cast(list[Primitive], requirements["contextual"])),
        *(
            mapping(mapping(item)["requirement"])
            for item in cast(list[Primitive], context["timeframes"])
        ),
    ]
    for declaration in declarations:
        if (
            configuration_identity(mapping(declaration["feed_scope"]))
            != provenance.feed_scope_id
        ):
            raise InvalidPredictionOutputError("context feed differs from shared input")
    feed = declarations[0]["feed_scope"]
    for item in cast(list[Primitive], source["timeframes"]):
        aligned = mapping(item)
        reference = dict(mapping(aligned["dataset_reference"]))
        reference.setdefault("feed_scope", feed)
        timeframe = Timeframe.from_primitive(
            mapping(
                mapping(mapping(aligned["requirement"])["timeframe"])["configuration"]
            )
        )
        _reference(provenance, reference, timeframe)


class PredictionWindowDecisionValidator:
    """QF-55 scientific checks reusable for append, resume and finalized reading."""

    def __init__(
        self,
        *,
        expected_identity: PrimitiveMappingSnapshot,
        schedule: PredictionDecisionSchedule,
        outcome_sessions: tuple[date, ...],
        strategy_parameters: PrimitiveMapping,
        canonical_metadata: DatasetMetadata | None = None,
        projection_registry: PreparedProjectionRegistry | None = None,
        projection_scope: PrimitiveMappingSnapshot | None = None,
    ) -> None:
        self.expected_identity = expected_identity
        self.schedule = schedule
        self.identity = expected_identity.to_primitive()
        if self.identity.get("schedule") != schedule.to_primitive():
            raise InvalidPredictionOutputError(
                "window scope differs from expected schedule"
            )
        market = mapping(self.identity["market_data"])
        self.provenance = validate_prediction_provenance(market)
        if isinstance(self.provenance, BoundedPredictionProvenance):
            if canonical_metadata is None:
                raise InvalidPredictionOutputError(
                    "bounded window requires independent canonical ancestry"
                )
            if (
                schedule.decision_timestamps
                and self.provenance.causal_cutoff > schedule.decision_timestamps[0]
            ):
                raise InvalidPredictionOutputError(
                    "bounded cutoff is after the first decision"
                )
            start = date.fromisoformat(cast(str, market["actual_first_session"]))
            if projection_registry is None:
                validate_prediction_view_lineage(
                    market,
                    canonical_metadata,
                    self.provenance.causal_cutoff,
                    start=start,
                )
            else:
                if projection_scope is None:
                    raise InvalidPredictionOutputError(
                        "prepared lineage requires a scope"
                    )
                projection_registry.verify_lineage(
                    market,
                    canonical_metadata,
                    self.provenance.causal_cutoff,
                    start=start,
                    scope=projection_scope,
                )
        _outcome_source(self.identity, self.provenance)
        self.study_identity = StudyIdentity(self.identity)
        self.session_indexes = {
            session.isoformat(): index for index, session in enumerate(outcome_sessions)
        }
        self.primary: PrimitiveMapping = {
            "configuration_id": schedule.primary_timeframe.configuration_id,
            "configuration": schedule.primary_timeframe.to_primitive(),
        }
        self.strategy_parameters = PrimitiveMappingSnapshot.capture(strategy_parameters)

    def validate(self, decision: PrimitiveMapping, sequence: int) -> None:
        """Check a single normalized record without expanding shared evidence."""
        try:
            manifest = mapping(mapping(decision["prediction_study"])["manifest"])
            context = mapping(manifest["prediction_context"])
            _context_sources(context, self.provenance)
            _validate_decision_record(
                decision,
                self.identity,
                self.schedule.decision_timestamps[sequence].isoformat(),
                study_id=self.study_identity.for_context(context),
                primary_timeframe=self.primary,
                decision_session=self.schedule.decision_sessions[sequence].isoformat(),
                session_indexes=self.session_indexes,
                strategy_parameters=self.strategy_parameters.to_primitive(),
            )
        except (ValueError, TypeError, KeyError, IndexError, ValidationError) as error:
            raise InvalidPredictionOutputError(
                f"invalid historical window evidence: {error}"
            ) from error


def validate_prediction_window_reader(
    reader: PredictionWindowReader,
    *,
    expected_identity: PrimitiveMappingSnapshot,
    schedule: PredictionDecisionSchedule,
    outcome_sessions: tuple[date, ...],
    strategy_parameters: PrimitiveMapping,
    canonical_metadata: DatasetMetadata | None = None,
    projection_registry: PreparedProjectionRegistry | None = None,
    projection_scope: PrimitiveMappingSnapshot | None = None,
) -> None:
    """Verify v1/v2 with trusted inputs and independent bounded-view ancestry."""
    try:
        if (
            reader.evidence.identity_snapshot != expected_identity
            or reader.evidence.schedule != schedule
        ):
            raise InvalidPredictionOutputError(
                "window scope differs from expected identity/schedule"
            )
        validator = PredictionWindowDecisionValidator(
            expected_identity=expected_identity,
            schedule=schedule,
            outcome_sessions=outcome_sessions,
            strategy_parameters=strategy_parameters,
            canonical_metadata=canonical_metadata,
            projection_registry=projection_registry,
            projection_scope=projection_scope,
        )
        for index, compact in enumerate(reader.iterate_decisions()):
            validator.validate(compact.to_primitive(), index)
    except (ValueError, TypeError, KeyError, IndexError, ValidationError) as error:
        raise InvalidPredictionOutputError(
            f"invalid historical window evidence: {error}"
        ) from error
