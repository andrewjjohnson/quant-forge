"""Verified authoritative event sources and holdout isolation (QF-67).

Sources come only from existing verifiers; callers cannot assert a role:

- walk-forward test windows: QF-40 ``load_oos_source`` (plan, frozen selection,
  QF-8 test membership, schedule, lineage, scientific window validation and
  protected-window reach);
- selection/development windows: QF-40 ``load_prediction_trial_window``, the
  same checks under the fold's frozen QF-8 membership for that role plus the
  QF-32 trial evidence;
- development windows of folds whose trials ran on selection: QF-40
  ``load_prediction_development_window``, the frozen candidate's recorded
  QF-69 development evidence under the frozen development membership (only
  when the study recorded it; otherwise the role stays an exclusion);
- final holdout: only ``HoldoutLedger.result`` of an **already consumed**
  holdout in the workspace's permanent ledger. Building never consumes.

The workspace's permanent ledger (``reports/holdout-ledger``) is required and
never created; there is no override and no ledger object can be substituted.
Its shared lock is held while sources are read and rows are checked, so no
reservation or consumption can change the protected scopes meanwhile.

QF-72 rapid results are exploratory and are refused at this boundary: they have
no verifier, and an object or file declaring ``authoritative: false`` (or using
the ``.rapid.json`` suffix) is rejected explicitly. Promising rapid
configurations must first be reproduced through QF-32/QF-39/QF-40.
"""

from collections.abc import Generator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import cast

from quantforge.configuration import (
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.experiments.artifacts import NON_AUTHORITATIVE_SUFFIX
from quantforge.ml.errors import (
    EventDatasetIntegrityError,
    EventHoldoutError,
    EventPartitionError,
    EventPopulationError,
    NonAuthoritativeSourceError,
)
from quantforge.oos import (
    HoldoutEvaluation,
    HoldoutLedger,
    OOSIntegrityError,
    OOSSource,
    has_development_evidence,
    load_oos_source,
    load_prediction_development_window,
    load_prediction_trial_window,
)
from quantforge.prediction.errors import InvalidPredictionOutputError
from quantforge.prediction.window_reader import PredictionWindowReader
from quantforge.validation import (
    PartitionRole,
    ResearchStudyType,
    TimestampBoundary,
    ValidationPlan,
    ValidationWindow,
)
from quantforge.walk_forward.models import (
    CandidateConfiguration,
    FoldStatus,
    FrozenSelection,
    PredictionOOSArtifact,
    WalkForwardError,
)
from quantforge.walk_forward.study import (
    _load_artifact,  # pyright: ignore[reportPrivateUsage]
)

# The research workspace's permanent holdout authority (QF-40/QF-45/QF-72).
WORKSPACE_HOLDOUT_LEDGER = Path("reports", "holdout-ledger")
FOLD_ROLES = (
    PartitionRole.DEVELOPMENT,
    PartitionRole.SELECTION,
    PartitionRole.WALK_FORWARD_TEST,
)
ROLE_ORDER = {role: index for index, role in enumerate((*FOLD_ROLES,))} | {
    PartitionRole.FINAL_HOLDOUT: len(FOLD_ROLES)
}
_VERIFIED = object()


class DispositionPolicy(StrEnum):
    """Which generated signals of the population become rows."""

    ACCEPTED_ONLY = "accepted_only"
    ALL_GENERATED_SIGNALS = "all_generated_signals"


def reject_non_authoritative(value: object, *, label: str = "source") -> None:
    """Refuse QF-72 rapid (exploratory) results at an authoritative boundary."""
    if (
        getattr(value, "authoritative", None) is False
        or getattr(value, "mode", None) == "exploratory"
        or type(value).__module__.startswith("quantforge.rapid")
    ):
        raise NonAuthoritativeSourceError(
            f"{label} is a non-authoritative exploratory result; reproduce the "
            "configuration through QF-32/QF-39/QF-40 before building an ML dataset"
        )
    if isinstance(value, Path) and value.name.lower().endswith(
        NON_AUTHORITATIVE_SUFFIX
    ):
        raise NonAuthoritativeSourceError(
            f"{label} is a {NON_AUTHORITATIVE_SUFFIX} exploratory export"
        )


@dataclass(frozen=True, slots=True)
class EventPopulation:
    """The frozen strategy population every row is conditional on.

    Rows exist only where this exact rule configuration emitted a generated
    signal; no-trigger decisions are coverage evidence, never negatives. The
    population says nothing about timestamps where the rule did not trigger.
    """

    plan_id: str
    candidate: CandidateConfiguration
    symbol: str
    source: PrimitiveMappingSnapshot
    disposition_policy: DispositionPolicy

    @classmethod
    def capture(
        cls,
        plan: ValidationPlan,
        candidate: CandidateConfiguration,
        disposition_policy: DispositionPolicy,
    ) -> "EventPopulation":
        if type(cast(object, disposition_policy)) is not DispositionPolicy:
            raise EventPopulationError("disposition policy is unsupported")
        membership = plan.prediction_membership
        metadata = plan.environment.outcome_dataset.market_data_metadata
        if (
            plan.environment.study_type is not ResearchStudyType.PREDICTION
            or membership is None
            or metadata is None
        ):
            raise EventPopulationError(
                "event datasets require a timestamp prediction plan with "
                "canonical market metadata"
            )
        return cls(
            plan.plan_id,
            candidate,
            metadata.canonical_symbol,
            PrimitiveMappingSnapshot.capture(
                {
                    "schedule_id": membership.schedule.schedule_id,
                    "primary_timeframe_configuration_id": (
                        membership.schedule.primary_timeframe.configuration_id
                    ),
                    "source_reference": membership.source_reference.to_primitive(
                        include_feed_scope=True
                    ),
                    "family_manifest_id": membership.family_manifest_id,
                }
            ),
            disposition_policy,
        )

    @property
    def definition(self) -> PrimitiveMapping:
        return self.candidate.definition.to_primitive()

    @property
    def rule(self) -> PrimitiveMapping:
        return cast(PrimitiveMapping, self.definition["prediction_rule"])

    def to_primitive(self) -> PrimitiveMapping:
        definition = self.definition
        rule = self.rule
        return {
            "contract_version": "1",
            "plan_id": self.plan_id,
            "candidate": self.candidate.to_primitive(),
            "strategy": {
                "strategy_id": rule["name"],
                "implementation_version": rule["implementation_version"],
                "configuration_id": rule["configuration_id"],
                "parameters": definition["prediction_rule_parameters"],
            },
            "symbol": self.symbol,
            "source": self.source.to_primitive(),
            "event_selection": {
                "decisions": "qf42_schedule_equal_to_verified_qf8_retained_membership",
                "observation_unit": "generated_signal",
                "no_trigger_decisions": "coverage_receipts_counted_never_rows",
            },
            "disposition_policy": self.disposition_policy.value,
            "interpretation": (
                "conditional on this exact strategy configuration triggering; "
                "not a sample of all market timestamps"
            ),
        }

    @property
    def population_id(self) -> str:
        return configuration_identity(self.to_primitive())

    def require_window(self, reader: PredictionWindowReader) -> None:
        """A window must execute exactly this candidate's executable definition."""
        manifest = reader.manifest()
        configured = cast(PrimitiveMapping, manifest["configuration"])
        definition = self.definition
        for component in ("prediction_rule", "outcome_labeler", "evaluator"):
            expected = cast(PrimitiveMapping, definition[component])
            actual = cast(PrimitiveMapping, configured.get(component))
            if any(actual.get(key) != value for key, value in expected.items()):
                raise EventPopulationError(
                    "source window executed a different strategy configuration"
                )
        if (
            configured.get("feature_configuration")
            != definition["feature_configuration"]
            or configured.get("prediction_context_requirements")
            != definition["prediction_context"]
            or manifest.get("indicator_backend_environment")
            != definition["indicator_backend_environment"]
        ):
            raise EventPopulationError("source window differs from the population")

    def signal_identity(self) -> PrimitiveMapping:
        """The exact prediction identity every generated signal must carry."""
        rule = self.rule
        return {
            "symbol": self.symbol,
            "strategy_id": rule["name"],
            "strategy_implementation_version": rule["implementation_version"],
            "strategy_configuration_id": rule["configuration_id"],
            "strategy_parameters": self.definition["prediction_rule_parameters"],
        }

    @staticmethod
    def require_signal(
        signal: PrimitiveMapping, identity: PrimitiveMapping
    ) -> PrimitiveMapping:
        """Prediction values of a signal carrying exactly ``identity``."""
        try:
            prediction = cast(PrimitiveMapping, signal["prediction"])
            values = cast(PrimitiveMapping, prediction["values"])
        except (KeyError, TypeError) as error:
            raise EventDatasetIntegrityError(
                "generated signal is incomplete"
            ) from error
        if any(prediction.get(key) != value for key, value in identity.items()):
            raise EventPopulationError(
                "observation was generated outside the declared population"
            )
        return values

    def includes(self, disposition: object) -> bool:
        if disposition not in (None, "accepted", "rejected", "blocked", "overlapping"):
            raise EventDatasetIntegrityError("unknown signal disposition")
        return (
            self.disposition_policy is DispositionPolicy.ALL_GENERATED_SIGNALS
            or disposition in (None, "accepted")
        )


@dataclass(frozen=True, slots=True)
class EventSourceWindow:
    """One verified window and its plan role; only loaders construct it.

    ``membership`` holds the scientific QF-8 membership identities of the role;
    ``physical`` holds operational references outside scientific identity.
    The QF-39 ``selection_id`` is physical too: it hashes the study definition,
    including the physical window schema, while the frozen candidate and the
    QF-8 membership it binds are recorded scientifically.
    """

    role: PartitionRole
    fold_id: str | None
    fold_index: int | None
    window: ValidationWindow
    membership: PrimitiveMappingSnapshot
    candidate: CandidateConfiguration
    selection_id: str
    physical: PrimitiveMappingSnapshot
    reader: PredictionWindowReader = field(compare=False, repr=False)
    _token: object = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self._token is not _VERIFIED:
            raise EventDatasetIntegrityError(
                "event sources are created only by the verified QF-39/QF-40 loaders"
            )

    @property
    def scientific_window_id(self) -> str:
        """QF-42 window identity; independent of the physical window schema."""
        return configuration_identity(
            self.reader.evidence.identity_snapshot.to_primitive()
        )

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "role": self.role.value,
            "fold_id": self.fold_id,
            "fold_index": self.fold_index,
            "qf8_window_id": self.window.window_id,
            "membership": self.membership.to_primitive(),
            "candidate_combination_id": self.candidate.combination_id,
            "scientific_window_id": self.scientific_window_id,
            "schedule_id": self.reader.evidence.schedule.schedule_id,
            "scheduled_decisions": self.reader.decision_count,
        }


def _membership(part: PrimitiveMapping) -> PrimitiveMappingSnapshot:
    member = cast(PrimitiveMapping, part["membership"])
    purge = part.get("purge")
    timestamps = part.get("evaluation_timestamps")
    return PrimitiveMappingSnapshot.capture(
        {
            "membership_selection_id": member["selection_id"],
            "purge_result_id": None
            if purge is None
            else cast(PrimitiveMapping, purge)["result_id"],
            "retained_decision_count": len(cast(list[object], timestamps))
            if isinstance(timestamps, list)
            else None,
        }
    )


def _part(selection: FrozenSelection, role: str) -> PrimitiveMapping:
    membership = cast(PrimitiveMapping, selection.snapshot.to_primitive()["membership"])
    return cast(PrimitiveMapping, membership[role])


def _physical(
    reader: PredictionWindowReader, **values: object
) -> PrimitiveMappingSnapshot:
    return PrimitiveMappingSnapshot.capture(
        {
            **cast(PrimitiveMapping, values),
            "window_schema_version": reader.schema_version,
            "window_result_id": reader.header()["window_result_id"],
            "shared_evidence_id": reader.evidence.evidence_id,
        }
    )


@dataclass(frozen=True, slots=True)
class StudyEventSources:
    """Verified study sources plus explicit, reasoned exclusions."""

    source: OOSSource = field(repr=False)
    windows: tuple[EventSourceWindow, ...]
    exclusions: tuple[PrimitiveMappingSnapshot, ...]


def universe_candidate(
    source: OOSSource, combination_id: str
) -> CandidateConfiguration:
    """The exact candidate of the verified study universe; never caller-built."""
    adapter = cast(PrimitiveMapping, source.definition.to_primitive()["adapter"])
    universe = cast(PrimitiveMapping, adapter["universe"])
    for item in cast(list[PrimitiveMapping], universe["candidates"]):
        if item.get("combination_id") == combination_id:
            return CandidateConfiguration(
                combination_id,
                PrimitiveMappingSnapshot.capture(
                    cast(PrimitiveMapping, item["parameters"])
                ),
                PrimitiveMappingSnapshot.capture(
                    cast(PrimitiveMapping, item["definition"])
                ),
            )
    raise EventPopulationError("combination is outside the study's candidate universe")


def load_study_event_sources(
    plan: ValidationPlan,
    study_path: Path,
    *,
    combination_id: str,
    roles: frozenset[PartitionRole],
) -> StudyEventSources:
    """Verify every available window of one candidate in a QF-39 study.

    Folds that froze another candidate contribute no test window, and folds
    without a frozen selection contribute nothing; both are recorded as
    exclusions rather than silently skipped.
    """
    reject_non_authoritative(study_path, label="study path")
    if not roles:
        raise EventPartitionError("at least one fold role must be requested")
    if not roles <= frozenset(FOLD_ROLES):
        raise EventHoldoutError(
            "requested roles must be fold roles; final-holdout rows require an "
            "explicitly consumed holdout evaluation"
        )
    try:
        source = load_oos_source(plan, study_path)
    except (OOSIntegrityError, WalkForwardError, InvalidPredictionOutputError) as error:
        raise EventDatasetIntegrityError(
            f"QF-39 study evidence failed verification: {error}"
        ) from error
    candidate = universe_candidate(source, combination_id)
    windows: list[EventSourceWindow] = []
    exclusions: list[PrimitiveMappingSnapshot] = []

    def exclude(fold_id: str, role: PartitionRole, reason: str) -> None:
        exclusions.append(
            PrimitiveMappingSnapshot.capture(
                {"fold_id": fold_id, "role": role.value, "reason": reason}
            )
        )

    for index, fold in enumerate(source.folds):
        plan_fold = plan.folds[index]
        trial_role = (
            PartitionRole.SELECTION
            if plan_fold.selection is not None
            else PartitionRole.DEVELOPMENT
        )
        if trial_role in roles:
            if fold.selection is None:
                exclude(fold.fold_id, trial_role, f"fold_{fold.status.value}")
            else:
                try:
                    trial = load_prediction_trial_window(
                        source,
                        study_path,
                        fold_id=fold.fold_id,
                        combination_id=combination_id,
                    )
                except (
                    OOSIntegrityError,
                    WalkForwardError,
                    InvalidPredictionOutputError,
                ) as error:
                    raise EventDatasetIntegrityError(
                        f"QF-32 trial evidence failed verification: {error}"
                    ) from error
                role_key, window = (
                    ("selection", plan_fold.selection)
                    if trial.role is PartitionRole.SELECTION
                    else ("development", plan_fold.development)
                )
                assert window is not None
                windows.append(
                    EventSourceWindow(
                        trial.role,
                        fold.fold_id,
                        index,
                        window,
                        _membership(_part(trial.selection, role_key)),
                        trial.candidate,
                        trial.selection.selection_id,
                        _physical(
                            trial.reader,
                            study_id=source.study_id,
                            selection_id=trial.selection.selection_id,
                            grid_study_id=trial.grid_study_id,
                            trial_id=trial.trial_id,
                            path=(
                                f"folds/{fold.fold_id}/selection/{trial.grid_study_id}"
                                f"/artifacts/{trial.trial_id}/prediction-window"
                            ),
                        ),
                        trial.reader,
                        _VERIFIED,
                    )
                )
        unrequested = {PartitionRole.DEVELOPMENT, PartitionRole.SELECTION} - {
            trial_role
        }
        if (
            PartitionRole.DEVELOPMENT in unrequested & roles
            and fold.selection is not None
            and has_development_evidence(study_path, fold.fold_id)
        ):
            # QF-69: the frozen candidate's verified development evidence.
            unrequested.discard(PartitionRole.DEVELOPMENT)
            frozen = fold.selection.snapshot.to_primitive()
            if cast(PrimitiveMapping, frozen["candidate"])["combination_id"] != (
                combination_id
            ):
                exclude(
                    fold.fold_id,
                    PartitionRole.DEVELOPMENT,
                    "frozen_selection_is_another_candidate",
                )
            else:
                try:
                    development = load_prediction_development_window(
                        source, study_path, fold_id=fold.fold_id
                    )
                except (
                    OOSIntegrityError,
                    WalkForwardError,
                    InvalidPredictionOutputError,
                ) as error:
                    raise EventDatasetIntegrityError(
                        f"QF-39 development evidence failed verification: {error}"
                    ) from error
                windows.append(
                    EventSourceWindow(
                        PartitionRole.DEVELOPMENT,
                        fold.fold_id,
                        index,
                        plan_fold.development,
                        _membership(_part(fold.selection, "development")),
                        development.candidate,
                        fold.selection.selection_id,
                        _physical(
                            development.reader,
                            study_id=source.study_id,
                            selection_id=fold.selection.selection_id,
                            path=f"folds/{fold.fold_id}/development/prediction-window",
                        ),
                        development.reader,
                        _VERIFIED,
                    )
                )
        for role in sorted(unrequested & roles, key=ROLE_ORDER.__getitem__):
            exclude(fold.fold_id, role, "role_not_executed_by_qf39_selection")
        if PartitionRole.WALK_FORWARD_TEST not in roles:
            continue
        artifact, reader = fold.artifact, source.prediction_windows[index]
        if fold.status is not FoldStatus.COMPLETED or fold.selection is None:
            exclude(
                fold.fold_id,
                PartitionRole.WALK_FORWARD_TEST,
                f"fold_{fold.status.value}",
            )
            continue
        frozen = fold.selection.snapshot.to_primitive()
        if cast(PrimitiveMapping, frozen["candidate"])["combination_id"] != (
            combination_id
        ):
            exclude(
                fold.fold_id,
                PartitionRole.WALK_FORWARD_TEST,
                "frozen_selection_is_another_candidate",
            )
            continue
        if not isinstance(artifact, PredictionOOSArtifact) or reader is None:
            raise EventDatasetIntegrityError("completed fold has no prediction window")
        windows.append(
            EventSourceWindow(
                PartitionRole.WALK_FORWARD_TEST,
                fold.fold_id,
                index,
                plan_fold.test,
                _membership(_part(fold.selection, "test")),
                candidate,
                fold.selection.selection_id,
                _physical(
                    reader,
                    study_id=source.study_id,
                    selection_id=fold.selection.selection_id,
                    path=f"folds/{fold.fold_id}/test/prediction-window",
                ),
                reader,
                _VERIFIED,
            )
        )
    return StudyEventSources(source, tuple(windows), tuple(exclusions))


def workspace_ledger(workspace: Path) -> HoldoutLedger:
    """Open the workspace's permanent ledger; it is never created here."""
    reject_non_authoritative(workspace, label="workspace")
    if not isinstance(cast(object, workspace), Path):
        raise EventHoldoutError("event datasets require a research workspace path")
    try:
        return HoldoutLedger(workspace.resolve() / WORKSPACE_HOLDOUT_LEDGER)
    except (OOSIntegrityError, WalkForwardError) as error:
        raise EventHoldoutError(
            "event datasets require the research workspace's permanent holdout "
            "ledger at reports/holdout-ledger; it is never created by this builder"
        ) from error


def consumed_holdout_source(
    ledger: HoldoutLedger,
    evaluation: HoldoutEvaluation,
    *,
    plan: ValidationPlan,
    study_id: str,
    candidate: CandidateConfiguration,
) -> EventSourceWindow:
    """Final-holdout rows from an explicitly consumed ledger result only.

    ``HoldoutLedger.result`` validates the exact consumed request and its
    artifact; a reserved or unconsumed holdout fails closed. This never calls
    ``consume`` and never reads source rows that physically exist elsewhere.
    """
    if type(cast(object, evaluation)) is not HoldoutEvaluation:
        reject_non_authoritative(evaluation, label="holdout evaluation")
        raise EventHoldoutError("final-holdout rows require a HoldoutEvaluation")
    if (
        evaluation.source.plan.plan_id != plan.plan_id
        or evaluation.source.study_id != study_id
    ):
        raise EventHoldoutError("holdout evaluation belongs to another plan or study")
    try:
        result = ledger.result(evaluation).to_primitive()
    except (OOSIntegrityError, WalkForwardError) as error:
        raise EventHoldoutError(
            "final-holdout rows require a holdout already consumed through the "
            f"permanent ledger: {error}"
        ) from error
    frozen = evaluation.selection.snapshot.to_primitive()
    if cast(PrimitiveMapping, frozen["candidate"]) != candidate.to_primitive():
        raise EventPopulationError(
            "consumed holdout evaluated another candidate configuration"
        )
    artifact = _load_artifact(cast(PrimitiveMapping, result["artifact"]))
    if not isinstance(artifact, PredictionOOSArtifact):
        raise EventHoldoutError("consumed holdout has no prediction window")
    lineage = evaluation.source.lineage_id
    try:
        reader = PredictionWindowReader.from_reference(
            artifact.snapshot.to_primitive(),
            root=ledger.root / "lineages" / lineage / "evaluation",
        )
    except InvalidPredictionOutputError as error:
        raise EventDatasetIntegrityError(
            "consumed holdout window is invalid"
        ) from error
    if reader.header()["window_result_id"] != artifact.window_result_id:
        raise EventDatasetIntegrityError("consumed holdout window changed")
    membership = evaluation.permitted.membership
    return EventSourceWindow(
        PartitionRole.FINAL_HOLDOUT,
        None,
        None,
        plan.final_holdout.window,
        PrimitiveMappingSnapshot.capture(
            {
                "membership_selection_id": membership.selection_id,
                "purge_result_id": None,
                "retained_decision_count": len(
                    evaluation.permitted.decision_timestamps
                ),
                "holdout_id": plan.final_holdout.holdout_id,
                "consumption_request_id": result["request_id"],
                "tail_policy": "outcome_horizon_remains_inside_holdout",
            }
        ),
        candidate,
        evaluation.selection.selection_id,
        _physical(
            reader,
            study_id=study_id,
            selection_id=evaluation.selection.selection_id,
            lineage_id=lineage,
            path=f"reports/holdout-ledger/lineages/{lineage}/evaluation/prediction-window",
        ),
        reader,
        _VERIFIED,
    )


@dataclass(frozen=True, slots=True)
class HoldoutIsolation:
    """Plan holdout boundary plus permanent-ledger scopes, captured under lock."""

    symbol: str
    holdout_start: datetime
    holdout_end: datetime
    horizon: timedelta
    embargo: timedelta
    ledger_scopes: tuple[tuple[date, date], ...]

    @classmethod
    def capture(
        cls, plan: ValidationPlan, scopes: tuple[PrimitiveMapping, ...], symbol: str
    ) -> "HoldoutIsolation":
        interval = plan.final_holdout.window.interval
        horizon = plan.purge_policy.label_horizon.elapsed
        embargo = plan.purge_policy.embargo.elapsed
        if (
            not isinstance(interval.start, TimestampBoundary)
            or not isinstance(interval.end, TimestampBoundary)
            or horizon is None
            or embargo is None
        ):
            raise EventHoldoutError("event datasets require a timestamp holdout plan")
        return cls(
            symbol,
            interval.start.timestamp,
            interval.end.timestamp,
            horizon,
            embargo,
            tuple(
                (
                    date.fromisoformat(cast(str, scope["start"])),
                    date.fromisoformat(cast(str, scope["end"])),
                )
                for scope in scopes
                if scope.get("symbol") == symbol
            ),
        )

    def require(self, role: PartitionRole, decision: datetime, session: date) -> None:
        """Fold rows stay outside every protected scope; holdout rows inside it.

        Fold rows use the QF-39 purge rule (label horizon plus embargo must end
        before the holdout); holdout rows use the QF-40 tail rule (the horizon
        ends inside the reserved interval).
        """
        if role is PartitionRole.FINAL_HOLDOUT:
            if not (
                self.holdout_start <= decision
                and decision + self.horizon <= self.holdout_end
            ):
                raise EventHoldoutError("holdout row escapes the reserved interval")
            return
        if decision + self.horizon + self.embargo >= self.holdout_start:
            raise EventHoldoutError(
                "row outcome reaches the plan's reserved final holdout"
            )
        if any(start <= session <= end for start, end in self.ledger_scopes):
            raise EventHoldoutError(
                "row session overlaps a reserved or consumed holdout in the "
                "permanent ledger"
            )


@contextmanager
def held_isolation(
    ledger: HoldoutLedger, plan: ValidationPlan, symbol: str
) -> Generator[HoldoutIsolation]:
    """Hold the ledger's shared lock while rows are read and checked."""
    with ExitStack() as stack:
        try:
            scopes = stack.enter_context(ledger.held_exposure_scopes())
        except (OOSIntegrityError, WalkForwardError) as error:
            raise EventHoldoutError(
                f"permanent holdout ledger refused access: {error}"
            ) from error
        yield HoldoutIsolation.capture(plan, scopes, symbol)


__all__ = [
    "FOLD_ROLES",
    "WORKSPACE_HOLDOUT_LEDGER",
    "DispositionPolicy",
    "EventPopulation",
    "EventSourceWindow",
    "HoldoutIsolation",
    "StudyEventSources",
    "consumed_holdout_source",
    "held_isolation",
    "load_study_event_sources",
    "reject_non_authoritative",
    "universe_candidate",
    "workspace_ledger",
]
