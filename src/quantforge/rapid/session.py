"""Execution-local rapid exploratory research sessions (QF-72).

NON-AUTHORITATIVE / EXPLORATORY. A session prepares one permitted QF-8 plan
window once, through the authoritative components, then scans rules on
prepared causal values:

- QF-8/QF-39 ``partition``: exact retained (purged) decision membership and the
  QF-52 bounded input view used by authoritative decisions;
- QF-59 ``PreparedPredictionContext``: exact causal ``[start, stop)`` positions;
- QF-63 ``PreparedContextScope``: validated runs and content-keyed indicator
  series with first-use proofs, shared by every scanned configuration;
- QF-11 dataset session with its QF-61 outcome registry.

A scan evaluates only eligible decisions and labels outcomes only for triggers.
It never builds QF-11 study identities, QF-42 decisions, schema-4 receipts,
journals, checkpoints, manifests or QF-9 artifacts, and nothing is persisted.
All state is released when the session closes.
"""

from collections.abc import Generator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from itertools import pairwise
from pathlib import Path
from time import perf_counter
from typing import Any, cast
from zoneinfo import ZoneInfo

from quantforge.configuration import PrimitiveMapping, PrimitiveMappingSnapshot
from quantforge.data import MarketDataset, TimeframeBarSeries
from quantforge.data.exceptions import ValidationError as MarketDataValidationError
from quantforge.data.lineage import AdjustmentBasis
from quantforge.data.prepared_canonical import canonical_preparation
from quantforge.data.prepared_prediction_views import PreparedProjectionRegistry
from quantforge.prediction.context import (
    PredictionContextError,
    build_prediction_rule_context,
)
from quantforge.prediction.contracts import PredictionStudy, evaluate_outcome_request
from quantforge.prediction.models import PredictionDirection
from quantforge.prediction.outcome_resolution import OutcomeEvaluationRequest
from quantforge.prediction.outcome_temporal import OutcomeAnchor
from quantforge.prediction.prepared_features import (
    PreparedContextScope,
    validate_prepared_context_sources,
)
from quantforge.prediction.prepared_outcomes import PreparedOutcomeSource
from quantforge.prediction.signal_feature_models import (
    OUTCOME_SCHEMA_VERSION,
    SignalFeatureCandidate,
)
from quantforge.prediction.study import (
    PredictionStudyConfiguration,
    PredictionStudyDatasetSession,
    _capture_study_configuration,  # pyright: ignore[reportPrivateUsage]
    _prediction_outcome,  # pyright: ignore[reportPrivateUsage]
    prepare_prediction_study_dataset,
)
from quantforge.prediction.timestamp_execution import bounded_outcome_source
from quantforge.prediction.window import PredictionDecisionSchedule
from quantforge.rapid.admission import (
    AdmittedRapidOutcome,
    AdmittedRapidRule,
    admit_outcomes,
    admit_rule,
)
from quantforge.rapid.models import (
    RapidAdmissionError,
    RapidDecisionWindow,
    RapidEvent,
    RapidOutcomeConfiguration,
    RapidOutcomeSummary,
    RapidOutcomeValues,
    RapidPeriodSummary,
    RapidResearchWindow,
    RapidScanError,
    RapidScanProfile,
    RapidScanResult,
    RapidSourceReference,
    RapidValue,
    summarize_outcomes,
)
from quantforge.rapid.scope import (
    HoldoutIsolation,
    bar_sessions,
    exploratory_window,
    workspace_holdout_ledger,
)
from quantforge.timeframes import Timeframe
from quantforge.validation import (
    PartitionRole,
    TimestampBoundary,
    ValidationPlan,
)
from quantforge.validation.errors import ValidationPlanError
from quantforge.validation.prepared_context import PreparedPredictionContext
from quantforge.walk_forward.partitions import partition


@dataclass(frozen=True, slots=True)
class _Eligibility:
    """Frozen eligible decisions for one clock timezone and decision window."""

    timestamps: tuple[datetime, ...]
    clocks: tuple[time, ...]


@dataclass(frozen=True, slots=True)
class _Positions:
    """Exact QF-59 positions of each eligible decision for one timeframe."""

    starts: tuple[int, ...]
    stops: tuple[int, ...]
    sessions: tuple[date, ...] | None


@dataclass(frozen=True, slots=True)
class _OutcomeRuntime:
    admitted: AdmittedRapidOutcome
    prepared: PreparedOutcomeSource
    configuration: PredictionStudyConfiguration


class _Clock:
    """Named cumulative timers for one operation."""

    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}
        self._started: dict[str, float] = {}

    def start(self, name: str) -> None:
        self._started[name] = perf_counter()

    def stop(self, name: str) -> None:
        elapsed = perf_counter() - self._started.pop(name)
        self.seconds[name] = self.seconds.get(name, 0.0) + elapsed


class RapidResearchSession:
    """Prepared exploratory state for one permitted plan window.

    Obtain it only through ``rapid_research_session``. It owns no global cache;
    closing the session releases every prepared structure.
    """

    __slots__ = (
        "_adjustment_basis",
        "_canonical_dataset_id",
        "_closed",
        "_component_dataset",
        "_eligibility",
        "_family_fingerprint",
        "_footprint_start",
        "_isolation",
        "_plan",
        "_positions",
        "_preparation",
        "_prepared",
        "_primary",
        "_research_window",
        "_schedule",
        "_scope",
        "_sources",
        "_statistics",
        "_study_dataset",
        "_symbol",
        "_warm_up",
    )
    _adjustment_basis: AdjustmentBasis
    _canonical_dataset_id: str
    _closed: bool
    _component_dataset: MarketDataset
    _eligibility: dict[tuple[str, RapidDecisionWindow | None], _Eligibility]
    _family_fingerprint: str
    _footprint_start: datetime
    _isolation: HoldoutIsolation
    _plan: ValidationPlan
    _positions: dict[tuple[str, RapidDecisionWindow | None, str], _Positions]
    _prepared: PreparedPredictionContext
    _preparation: dict[str, float]
    _primary: TimeframeBarSeries
    _research_window: RapidResearchWindow
    _schedule: PredictionDecisionSchedule
    _scope: PreparedContextScope
    _sources: dict[str, TimeframeBarSeries]
    _statistics: dict[str, int]
    _study_dataset: PredictionStudyDatasetSession
    _symbol: str
    _warm_up: dict[str, int]

    def __init__(self) -> None:
        raise TypeError("open a rapid session with rapid_research_session()")

    @classmethod
    def _open(
        cls,
        *,
        plan: ValidationPlan,
        fold_index: int,
        role: PartitionRole,
        dataset: MarketDataset,
        series: tuple[TimeframeBarSeries, ...],
        workspace: Path,
        projection_registry: PreparedProjectionRegistry,
    ) -> "RapidResearchSession":
        clock = _Clock()
        clock.start("total")
        # Scope and holdout guards precede any market-value preparation.
        clock.start("scope_and_holdout_guard")
        window = exploratory_window(plan, fold_index, role)
        if not isinstance(cast(object, dataset), MarketDataset):
            raise RapidAdmissionError("rapid sessions require a canonical QF-3 input")
        symbol = dataset.metadata.canonical_symbol
        isolation = HoldoutIsolation.capture(
            plan, workspace_holdout_ledger(workspace), symbol=symbol
        )
        membership = plan.prediction_membership
        assert membership is not None
        timezone = ZoneInfo(
            membership.schedule.primary_timeframe.session_policy.timezone_name
        )
        start = cast(TimestampBoundary, window.interval.start).timestamp
        end = cast(TimestampBoundary, window.interval.end).timestamp
        isolation.require_isolated(
            first_timestamp=start,
            last_timestamp=end,
            first_session=start.astimezone(timezone).date(),
            last_session=end.astimezone(timezone).date(),
        )
        clock.stop("scope_and_holdout_guard")

        clock.start("qf8_partition_and_bounded_view")
        permitted = partition(
            dataset,
            plan,
            fold_index,
            role,
            minimum_observations=1,
            projection_registry=projection_registry,
        )
        clock.stop("qf8_partition_and_bounded_view")

        clock.start("sources_and_schedule")
        ordered = tuple(
            sorted(series, key=lambda item: item.timeframe.configuration_id)
        )
        identifiers = tuple(item.timeframe.configuration_id for item in ordered)
        if len(set(identifiers)) != len(identifiers):
            raise RapidAdmissionError("rapid session sources need unique timeframes")
        if {item.dataset_reference for item in ordered} != set(
            plan.environment.dataset.family_references
        ) or any(
            item.dataset_family_manifest_id
            != plan.environment.dataset.family_manifest_id
            for item in ordered
        ):
            raise RapidAdmissionError("rapid session sources differ from the plan")
        primary_timeframe = membership.schedule.primary_timeframe
        primary = next(
            (item for item in ordered if item.timeframe == primary_timeframe), None
        )
        if primary is None:
            raise RapidAdmissionError("rapid session needs the plan's primary source")
        membership.validate_source(primary)
        timestamps = permitted.decision_timestamps
        schedule = PredictionDecisionSchedule(
            primary_timeframe, timestamps[0], timestamps[-1]
        )
        if schedule.decision_timestamps != timestamps:
            raise RapidScanError("QF-42 schedule differs from QF-8 retained membership")
        clock.stop("sources_and_schedule")

        clock.start("qf59_qf63_preparation")
        try:
            prepared = PreparedPredictionContext.capture(
                plan,
                window,
                series=ordered,
                input_identity=permitted.dataset.metadata.dataset_id,
            )
        except ValidationPlanError as error:
            raise RapidAdmissionError(
                f"rapid session sources cannot be prepared: {error}"
            ) from error
        if prepared is None:
            raise RapidAdmissionError(
                "rapid session sources are not admissible to immutable preparation; "
                "use the authoritative pipeline"
            )
        scope = PreparedContextScope.capture(
            tuple(
                (index.source, *index.extent(prepared.end_timestamp))
                for index in prepared.indexes
            )
        )
        if scope is None:
            raise RapidAdmissionError(
                "rapid session sources are not admissible to prepared runs; use the "
                "authoritative pipeline"
            )
        runs = tuple(scope.runs.values())
        first_bar = min(
            (run.bars[0] for run in runs), key=lambda bar: bar.start_timestamp
        )
        footprint_first = min(bar_sessions(run.bars[0])[0] for run in runs)
        isolation.require_isolated(
            first_timestamp=first_bar.start_timestamp,
            last_timestamp=schedule.decision_timestamps[-1],
            first_session=footprint_first,
            last_session=schedule.decision_sessions[-1],
        )
        clock.stop("qf59_qf63_preparation")

        clock.start("qf11_dataset_session")
        study_dataset = prepare_prediction_study_dataset(permitted.dataset)
        clock.stop("qf11_dataset_session")

        self = object.__new__(cls)
        metadata = study_dataset.component_dataset.metadata
        fold = plan.folds[fold_index]
        warm_up = tuple(
            (item.timeframe.configuration_id, item.observations)
            for item in window.warm_up_by_timeframe
        )
        holdout = plan.final_holdout.window.interval
        research_window = RapidResearchWindow(
            plan.plan_id,
            fold_index,
            fold.name,
            role,
            window.name,
            start,
            end,
            warm_up,
            len(timestamps),
            len(permitted.purge.purged),
            timestamps[0],
            timestamps[-1],
            footprint_first,
            schedule.decision_sessions[-1],
            cast(TimestampBoundary, holdout.start).timestamp,
            cast(TimestampBoundary, holdout.end).timestamp,
            len(isolation.ledger_scopes),
        )
        self._closed = False
        self._plan = plan
        self._schedule = schedule
        self._prepared = prepared
        self._scope = scope
        self._sources = {item.timeframe.configuration_id: item for item in ordered}
        self._primary = primary
        self._warm_up = dict(warm_up)
        self._study_dataset = study_dataset
        self._component_dataset = study_dataset.component_dataset
        self._symbol = symbol
        self._adjustment_basis = AdjustmentBasis(
            adjustment_mode=metadata.adjustment_mode,
            ohlc_basis=metadata.ohlc_basis,
            volume_basis=metadata.volume_basis,
            corporate_action_policy=metadata.corporate_action_policy,
            adjusted_fields_used=metadata.adjusted_fields_used,
        )
        self._family_fingerprint = ordered[0].dataset_reference.family_id
        self._isolation = isolation
        self._footprint_start = first_bar.start_timestamp
        self._eligibility = {}
        self._positions = {}
        self._research_window = research_window
        self._canonical_dataset_id = dataset.metadata.dataset_id
        self._statistics = dict.fromkeys(
            (
                "scans",
                "eligibility_builds",
                "eligibility_reuses",
                "position_builds",
                "position_reuses",
                "anchor_contexts",
                "outcome_requests",
            ),
            0,
        )
        clock.stop("total")
        self._preparation = dict(clock.seconds)
        return self

    # -- public ---------------------------------------------------------------

    @property
    def research_window(self) -> RapidResearchWindow:
        return self._research_window

    @property
    def preparation_seconds(self) -> dict[str, float]:
        return dict(self._preparation)

    def statistics(self) -> dict[str, int]:
        """Session reuse counters plus QF-63 series/lookup statistics."""
        self._require_open()
        return {
            **self._statistics,
            **{f"qf63_{key}": value for key, value in self._scope.statistics().items()},
        }

    def eligible_timestamps(self, rule: object) -> tuple[datetime, ...]:
        """The frozen eligible decisions a scan of ``rule`` evaluates, in order."""
        self._require_open()
        admitted = admit_rule(rule, sources=self._sources, warm_up=self._warm_up)
        return self._eligible(admitted).timestamps

    def scan(
        self,
        rule: object,
        *,
        outcomes: tuple[object, ...] = (),
        record_values: bool = True,
    ) -> RapidScanResult:
        """Scan one configured rule over the session window. NON-AUTHORITATIVE."""
        self._require_open()
        clock = _Clock()
        clock.start("total")
        clock.start("admission")
        admitted = admit_rule(rule, sources=self._sources, warm_up=self._warm_up)
        horizon = self._plan.purge_policy.label_horizon.elapsed
        assert horizon is not None
        configured = admit_outcomes(
            outcomes, primary=self._primary, maximum_reach=horizon
        )
        reach = max((item.future_reach for item in configured), default=None)
        # The ledger was audited and read at session open (its records can be
        # large); every scan re-checks its own maximum outcome reach.
        self._isolation.require_isolated(
            first_timestamp=self._footprint_start,
            last_timestamp=self._schedule.decision_timestamps[-1]
            + (reach if reach is not None else timedelta(0)),
            first_session=self._research_window.footprint_first_session,
            last_session=self._research_window.footprint_last_session,
        )
        clock.stop("admission")

        clock.start("eligibility")
        eligibility = self._eligible(admitted)
        clock.stop("eligibility")
        clock.start("positions")
        positions = tuple(
            self._positions_for(admitted, timeframe, eligibility)
            for timeframe in admitted.timeframes
        )
        primary_sessions = positions[0].sessions
        assert primary_sessions is not None
        clock.stop("positions")

        before = self._scope.statistics()
        triggers: list[tuple[int, PredictionDirection, tuple[RapidValue, ...]]] = []
        count = len(eligibility.timestamps)
        segment_start = 0
        anchors = 0
        decide = admitted.specification.decide
        while segment_start < count:
            starts = tuple(item.starts[segment_start] for item in positions)
            segment_end = segment_start + 1
            while segment_end < count and all(
                item.starts[segment_end] == first
                for item, first in zip(positions, starts, strict=True)
            ):
                segment_end += 1
            clock.start("anchor_contexts")
            arrays = self._anchor_values(
                admitted, eligibility, positions, segment_end - 1
            )
            anchors += 1
            clock.stop("anchor_contexts")
            clock.start("kernel")
            lookups = tuple(
                (
                    arrays[index],
                    positions[item.timeframe_index].stops,
                    starts[item.timeframe_index] + 1 + item.lag,
                )
                for index, item in enumerate(admitted.inputs)
            )
            for position in range(segment_start, segment_end):
                values = tuple(
                    array[stops[position] - offset]
                    if stops[position] >= offset
                    else None
                    for array, stops, offset in lookups
                )
                direction = decide(eligibility.clocks[position], values)
                if direction is not None:
                    if not isinstance(cast(object, direction), PredictionDirection):
                        raise RapidScanError(
                            "rapid kernel returned an invalid decision"
                        )
                    triggers.append((position, direction, values))
            clock.stop("kernel")
            segment_start = segment_end
        after = self._scope.statistics()

        clock.start("outcome_preparation")
        runtimes = self._outcome_runtimes(admitted, configured) if triggers else ()
        clock.stop("outcome_preparation")
        names = tuple(item.name for item in admitted.inputs)
        strategy = admitted.configuration
        parameters = strategy.parameters.to_primitive()
        primary_positions = positions[0]
        events: list[RapidEvent] = []
        for position, direction, values in triggers:
            timestamp = eligibility.timestamps[position]
            session = primary_sessions[position]
            clock.start("event_construction")
            signal = admitted.specification.candidate(
                self._symbol, session, timestamp, values
            )
            # The QF-11 checks applied to every emitted authoritative signal.
            if (
                type(signal) is not SignalFeatureCandidate
                or signal.direction is not direction
                or signal.decision_timestamp != timestamp
                or signal.signal_session != session
                or signal.symbol != self._symbol
                or signal.strategy_id != strategy.name
                or signal.strategy_implementation_version
                != strategy.implementation_version
                or signal.strategy_configuration_id != strategy.configuration_id
                or signal.strategy_parameters.to_primitive() != parameters
            ):
                raise RapidScanError(
                    "rapid candidate builder disagrees with the shared kernel"
                )
            if (
                primary_positions.stops[position] - primary_positions.starts[position]
                < admitted.warm_up_observations
            ):
                raise RapidScanError(
                    "rapid trigger precedes the strategy's declared warm-up"
                )
            clock.stop("event_construction")
            clock.start("outcome_resolution")
            outcome_values = tuple(
                self._label(runtime, signal, timestamp, session) for runtime in runtimes
            )
            clock.stop("outcome_resolution")
            events.append(
                RapidEvent(
                    timestamp,
                    session,
                    direction,
                    tuple(zip(names, values, strict=True)) if record_values else (),
                    outcome_values,
                )
            )

        clock.start("summaries")
        outcome_configurations = tuple(
            RapidOutcomeConfiguration(
                item.outcome.namespace,
                item.outcome.labeler_configuration_id,
                item.outcome.evaluator_configuration_id,
                item.future_reach,
                self._primary.dataset_reference.dataset_id,
                item.value_fields,
            )
            for item in configured
        )
        summaries = self._summaries(configured, events)
        periods = self._periods(configured, primary_sessions, events)
        clock.stop("summaries")
        clock.stop("total")
        self._statistics["scans"] += 1
        self._statistics["anchor_contexts"] += anchors
        counts = (
            ("permitted_decisions", len(self._schedule.decision_timestamps)),
            ("eligible_decisions", count),
            ("evaluated_decisions", count),
            ("triggers", len(events)),
            ("anchor_contexts", anchors),
            ("outcome_requests", len(events) * len(runtimes)),
            *(
                (f"qf63_{key}", after[key] - before[key])
                for key in ("series_built", "series_reused", "series_verified")
            ),
        )
        return RapidScanResult(
            admitted.configuration,
            RapidSourceReference(
                self._symbol,
                self._family_fingerprint,
                self._primary.dataset_family_manifest_id,
                self._canonical_dataset_id,
                tuple(
                    (identifier, source.dataset_reference.dataset_id)
                    for identifier, source in sorted(self._sources.items())
                ),
            ),
            self._research_window,
            admitted.specification.decision_window,
            admitted.specification.clock_timezone,
            count,
            count,
            tuple(events),
            outcome_configurations,
            summaries,
            periods,
            record_values,
            RapidScanProfile(
                tuple(sorted(clock.seconds.items())),
                counts,
            ),
        )

    # -- internals ------------------------------------------------------------

    def _require_open(self) -> None:
        if self._closed:
            raise RapidScanError("rapid research session is closed")

    def _release(self) -> None:
        """Drop every prepared structure; later scans fail closed."""
        self._closed = True
        self._eligibility.clear()
        self._positions.clear()
        self._sources.clear()
        del self._prepared, self._scope, self._study_dataset
        del self._component_dataset, self._primary

    def _eligible(self, admitted: AdmittedRapidRule) -> _Eligibility:
        specification = admitted.specification
        key = (specification.clock_timezone, specification.decision_window)
        cached = self._eligibility.get(key)
        if cached is not None:
            self._statistics["eligibility_reuses"] += 1
            return cached
        timezone = ZoneInfo(specification.clock_timezone)
        window: RapidDecisionWindow | None = specification.decision_window
        timestamps: list[datetime] = []
        clocks: list[time] = []
        for timestamp in self._schedule.decision_timestamps:
            clock = timestamp.astimezone(timezone).time()
            if window is None or window.contains(clock):
                timestamps.append(timestamp)
                clocks.append(clock)
        eligibility = _Eligibility(tuple(timestamps), tuple(clocks))
        self._eligibility[key] = eligibility
        self._statistics["eligibility_builds"] += 1
        return eligibility

    def _positions_for(
        self,
        admitted: AdmittedRapidRule,
        timeframe: Timeframe,
        eligibility: _Eligibility,
    ) -> _Positions:
        specification = admitted.specification
        key = (
            specification.clock_timezone,
            specification.decision_window,
            timeframe.configuration_id,
        )
        cached = self._positions.get(key)
        if cached is not None:
            self._statistics["position_reuses"] += 1
            return cached
        starts: list[int] = []
        stops: list[int] = []
        primary = timeframe == admitted.requirements.primary.timeframe
        run = self._scope.runs[timeframe.configuration_id]
        sessions: list[date] = []
        for timestamp in eligibility.timestamps:
            _, start, stop = self._prepared.bounds_for(
                timeframe, as_of=TimestampBoundary(timestamp)
            )
            if primary:
                # QF-42 requires the exact scheduled primary bar; plan membership
                # makes a gap impossible, so any mismatch is an integrity failure.
                bar = run.bars[stop - 1 - run.offset] if stop > run.offset else None
                if bar is None or bar.end_timestamp != timestamp:
                    raise RapidScanError(
                        "rapid context is missing the scheduled primary bar"
                    )
                sessions.append(bar_sessions(bar)[1])
            starts.append(start)
            stops.append(stop)
        if any(later < earlier for earlier, later in pairwise(stops)):
            raise RapidScanError("rapid causal positions are not chronological")
        positions = _Positions(
            tuple(starts), tuple(stops), tuple(sessions) if primary else None
        )
        self._positions[key] = positions
        self._statistics["position_builds"] += 1
        return positions

    def _anchor_values(
        self,
        admitted: AdmittedRapidRule,
        eligibility: _Eligibility,
        positions: tuple[_Positions, ...],
        anchor: int,
    ) -> tuple[tuple[RapidValue, ...], ...]:
        """Values visible at the anchor, through the unchanged QF-63 rule context.

        Every decision in the anchor's segment shares its per-timeframe input
        starts, so its visible values are exact prefixes of these arrays.
        """
        requirements = admitted.requirements
        timestamp = eligibility.timestamps[anchor]
        context = self._scope.context_at(
            requirements.primary.timeframe,
            requirements.context_timeframe_requirements(),
            as_of=timestamp,
            positions=tuple(
                (
                    timeframe.configuration_id,
                    item.starts[anchor],
                    item.stops[anchor],
                )
                for timeframe, item in zip(admitted.timeframes, positions, strict=True)
            ),
        )
        if context.source_consistency.family_id != self._family_fingerprint:
            raise RapidScanError("rapid context has the wrong dataset family")
        before = self._scope.statistics()
        try:
            if not validate_prepared_context_sources(self._component_dataset, context):
                raise RapidScanError("rapid context was not prepared")
            rule_context = build_prediction_rule_context(
                requirements,
                context,
                prediction_dataset_id=self._component_dataset.metadata.dataset_id,
                symbol=self._symbol,
                prediction_adjustment_basis=self._adjustment_basis,
            )
        except (MarketDataValidationError, PredictionContextError) as error:
            raise RapidScanError(
                f"rapid context failed its authoritative checks: {error}"
            ) from error
        after = self._scope.statistics()
        if not rule_context.values_guarded or any(
            after[key] != before[key]
            for key in (
                "reference_indicator_fallbacks",
                "series_verification_fallbacks",
            )
        ):
            raise RapidScanError(
                "an indicator could not be served from a proven prepared series"
            )
        arrays: list[tuple[RapidValue, ...]] = []
        for item in admitted.inputs:
            timeframe = admitted.timeframes[item.timeframe_index]
            expected = (
                positions[item.timeframe_index].stops[anchor]
                - positions[item.timeframe_index].starts[anchor]
            )
            if item.alias is not None:
                assert item.output is not None
                values = rule_context.indicator_for(timeframe, item.alias).values_for(
                    item.output
                )
            else:
                assert item.bar_field is not None
                values = tuple(
                    cast(RapidValue, getattr(bar, item.bar_field))
                    for bar in rule_context.bars_for(timeframe)
                )
            if len(values) != expected:
                raise RapidScanError("rapid values differ from their causal positions")
            arrays.append(values)
        rule_context.validate_values_unchanged()
        return tuple(arrays)

    def _outcome_runtimes(
        self,
        admitted: AdmittedRapidRule,
        configured: tuple[AdmittedRapidOutcome, ...],
    ) -> tuple[_OutcomeRuntime, ...]:
        study_dataset: PredictionStudyDatasetSession = self._study_dataset
        runtimes: list[_OutcomeRuntime] = []
        for item in configured:
            outcome = item.outcome
            assert outcome.outcome_source is not None
            try:
                prepared = study_dataset.outcome_sources.prepare(
                    self._component_dataset, outcome.outcome_source
                )
            except MarketDataValidationError as error:
                raise RapidScanError(
                    f"rapid outcome source failed validation: {error}"
                ) from error
            if prepared is None:
                raise RapidAdmissionError(
                    "rapid outcome source is not admissible to QF-61 preparation"
                )
            key = (id(outcome.labeler), outcome.labeler_configuration_id)
            if key not in study_dataset.validated_labelers:
                outcome.labeler.validate_dataset(self._component_dataset)
                study_dataset.validated_labelers.add(key)
            configuration = _capture_study_configuration(
                PredictionStudy[Any, Any, Any].create(
                    cast(Any, admitted.rule),
                    outcome.labeler,
                    outcome.evaluator,
                    result_schema_version=OUTCOME_SCHEMA_VERSION,
                    outcome_source=outcome.outcome_source,
                )
            )
            runtimes.append(_OutcomeRuntime(item, prepared, configuration))
        return tuple(runtimes)

    def _label(
        self,
        runtime: _OutcomeRuntime,
        signal: SignalFeatureCandidate,
        timestamp: datetime,
        session: date,
    ) -> RapidOutcomeValues:
        """The QF-11 request/resolution/label/evaluation flow for one trigger."""
        admitted = runtime.admitted
        outcome = admitted.outcome
        market_data = self._study_dataset.market_data
        temporal = admitted.temporal
        request = OutcomeEvaluationRequest(
            OutcomeAnchor(temporal.anchor_kind, session, timestamp),
            temporal,
            outcome.labeler_configuration_id,
            market_data.dataset_id,
            market_data.bars_fingerprint,
            runtime.prepared.source.dataset_reference,
        )
        bounded, resolution = bounded_outcome_source(
            runtime.prepared.source, request, prepared=runtime.prepared
        )
        if any(bar_sessions(bar) != (session, session) for bar in bounded.bars):
            raise RapidScanError("rapid outcome path left its decision session")
        label = evaluate_outcome_request(
            outcome.labeler,
            self._component_dataset,
            request,
            source=deepcopy(bounded),
            resolution=deepcopy(resolution),
        )
        if (
            label is None
            or label.signal_session != session
            or label.outcome_session != session
        ):
            raise RapidScanError("elapsed outcomes require a same-session label")
        prediction_outcome = _prediction_outcome(
            market_data,
            runtime.configuration,
            session,
            session,
            label.values,
            PrimitiveMappingSnapshot.capture(label.values.to_primitive()),
            PrimitiveMappingSnapshot.capture(resolution.to_primitive()),
        )
        values = cast(
            PrimitiveMapping,
            outcome.evaluator.evaluate(signal, prediction_outcome).to_primitive(),
        )
        if "outcome_session" in admitted.field_names:
            values["outcome_session"] = session.isoformat()
        if frozenset(values) != admitted.field_names:
            raise RapidScanError(
                f"outcome {outcome.namespace} values do not match their schema"
            )
        self._statistics["outcome_requests"] += 1
        return RapidOutcomeValues(
            outcome.namespace,
            PrimitiveMappingSnapshot.capture(
                {name: values[name] for name in admitted.value_fields}
            ),
        )

    @staticmethod
    def _summaries(
        configured: tuple[AdmittedRapidOutcome, ...], events: list[RapidEvent]
    ) -> tuple[RapidOutcomeSummary, ...]:
        return tuple(
            summarize_outcomes(
                item.outcome.namespace,
                tuple(event.outcome(item.outcome.namespace) for event in events),
            )
            for item in configured
        )

    @staticmethod
    def _periods(
        configured: tuple[AdmittedRapidOutcome, ...],
        sessions: tuple[date, ...],
        events: list[RapidEvent],
    ) -> tuple[RapidPeriodSummary, ...]:
        eligible: dict[str, int] = {}
        for session in sessions:
            period = session.strftime("%Y-%m")
            eligible[period] = eligible.get(period, 0) + 1
        by_period: dict[str, list[RapidEvent]] = {}
        for event in events:
            by_period.setdefault(event.signal_session.strftime("%Y-%m"), []).append(
                event
            )
        return tuple(
            RapidPeriodSummary(
                period,
                eligible[period],
                len(by_period.get(period, [])),
                RapidResearchSession._summaries(configured, by_period.get(period, [])),
            )
            for period in sorted(eligible)
        )


@contextmanager
def rapid_research_session(
    *,
    plan: ValidationPlan,
    fold_index: int,
    role: PartitionRole,
    dataset: MarketDataset,
    series: tuple[TimeframeBarSeries, ...],
    workspace: Path,
) -> Generator[RapidResearchSession]:
    """Open one bounded exploratory session; state is released on exit.

    ``workspace`` is the research workspace root; its permanent holdout ledger
    (``reports/holdout-ledger``) must exist and is read once, read-only. The
    session joins the enclosing QF-65 canonical preparation, or owns one, and
    owns a QF-60 projection registry for its bounded input view.
    """
    registry = PreparedProjectionRegistry()
    try:
        with canonical_preparation():
            session = RapidResearchSession._open(  # pyright: ignore[reportPrivateUsage]
                plan=plan,
                fold_index=fold_index,
                role=role,
                dataset=dataset,
                series=series,
                workspace=workspace,
                projection_registry=registry,
            )
            try:
                yield session
            finally:
                session._release()  # pyright: ignore[reportPrivateUsage]
    finally:
        registry.clear()


__all__ = ["RapidResearchSession", "rapid_research_session"]
