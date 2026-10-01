"""Rapid exploratory scan contracts and results (QF-72).

NON-AUTHORITATIVE / EXPLORATORY. These records screen hypotheses. They are not
QF-9 artifacts, QF-42 windows, QF-39 selections, OOS or holdout results, and
nothing here subclasses or aliases an authoritative result. A promising rule is
promoted by configuration only and must be reproduced through the authoritative
QF-32/QF-39/QF-40 pipeline before any research conclusion.
"""

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from decimal import Decimal, localcontext
from typing import ClassVar, Literal, Protocol, cast
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from quantforge.configuration import (
    Primitive,
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
    decimal_to_primitive,
)
from quantforge.prediction.models import PredictionDirection
from quantforge.prediction.signal_feature_models import SignalFeatureCandidate
from quantforge.timeframes import Timeframe
from quantforge.validation import PartitionRole

RAPID_SCAN_SCHEMA_VERSION = "1"
RAPID_SPECIFICATION_VERSION = "1"
RAPID_SCAN_MODE: Literal["exploratory"] = "exploratory"
NON_AUTHORITATIVE_NOTICE = (
    "NON-AUTHORITATIVE EXPLORATORY RESULT: must be reproduced through the "
    "authoritative QuantForge pipeline before any research conclusion"
)
# Reviewed OHLCV fields a rapid rule may read from visible completed bars.
BAR_VALUE_FIELDS = frozenset({"open", "high", "low", "close", "volume"})

type RapidValue = Decimal | None


class RapidScanError(ValueError):
    """A rapid scan cannot run exactly; nothing was approximated."""


class RapidAdmissionError(RapidScanError):
    """A rule, feature, source or outcome is not reviewed for rapid scans."""


class RapidScopeError(RapidScanError):
    """The requested research scope is not permitted for exploration."""


class RapidHoldoutError(RapidScopeError):
    """The request would read a reserved final holdout."""


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise RapidAdmissionError(f"rapid {name} must be nonempty text")
    return value


def _lag(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RapidAdmissionError("rapid input lag must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class RapidIndicatorInput:
    """One declared indicator output, ``lag`` bars before the latest visible bar."""

    name: str
    timeframe: Timeframe
    alias: str
    output: str
    lag: int = 0

    def __post_init__(self) -> None:
        _text(self.name, "input name")
        _text(self.alias, "indicator alias")
        _text(self.output, "indicator output")
        _lag(self.lag)
        if not isinstance(cast(object, self.timeframe), Timeframe):
            raise RapidAdmissionError("rapid input timeframe is invalid")

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "kind": "indicator",
            "name": self.name,
            "timeframe_configuration_id": self.timeframe.configuration_id,
            "alias": self.alias,
            "output": self.output,
            "lag": self.lag,
        }


@dataclass(frozen=True, slots=True)
class RapidBarInput:
    """One OHLCV field of a completed bar, ``lag`` bars before the latest one."""

    name: str
    timeframe: Timeframe
    field: str
    lag: int = 0

    def __post_init__(self) -> None:
        _text(self.name, "input name")
        _lag(self.lag)
        if self.field not in BAR_VALUE_FIELDS:
            raise RapidAdmissionError(f"rapid bar field is not reviewed: {self.field}")
        if not isinstance(cast(object, self.timeframe), Timeframe):
            raise RapidAdmissionError("rapid input timeframe is invalid")

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "kind": "bar",
            "name": self.name,
            "timeframe_configuration_id": self.timeframe.configuration_id,
            "field": self.field,
            "lag": self.lag,
        }


type RapidInput = RapidIndicatorInput | RapidBarInput


@dataclass(frozen=True, slots=True)
class RapidDecisionWindow:
    """Inclusive local decision-clock bounds, frozen before any scan.

    This is a causal schedule filter only: it may skip evaluation of decisions
    the rule could never accept. The kernel still applies the complete rule.
    """

    start: time
    end: time

    def __post_init__(self) -> None:
        if (
            type(self.start) is not time
            or type(self.end) is not time
            or self.start.tzinfo is not None
            or self.end.tzinfo is not None
            or self.start > self.end
        ):
            raise RapidAdmissionError(
                "rapid decision window needs naive ordered local clock bounds"
            )

    def contains(self, clock: time) -> bool:
        return self.start <= clock <= self.end

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "boundaries": "both_inclusive",
        }


class RapidDecisionKernel(Protocol):
    """The rule's pure decision on causal scalars, in declared input order."""

    def __call__(
        self, decision_clock: time, values: tuple[RapidValue, ...], /
    ) -> PredictionDirection | None: ...


class RapidCandidateBuilder(Protocol):
    """The rule's own candidate construction, shared with authoritative output."""

    def __call__(
        self,
        symbol: str,
        signal_session: date,
        decision_timestamp: datetime,
        values: tuple[RapidValue, ...],
        /,
    ) -> SignalFeatureCandidate: ...


def _callable_name(value: object) -> str:
    owner = getattr(value, "__module__", None) or type(value).__module__
    name = getattr(value, "__qualname__", None) or type(value).__qualname__
    return f"{owner}.{name}"


@dataclass(frozen=True, slots=True)
class RapidRuleSpecification:
    """A rule's explicit, reviewed capability to run on prepared causal values.

    ``inputs`` name every value the kernel reads. ``decide`` must be the same
    predicate the authoritative rule applies, and ``candidate`` the same
    candidate construction, so both paths share one strategy definition.
    ``decision_clock`` values are local to ``clock_timezone``.
    """

    inputs: tuple[RapidInput, ...]
    decide: RapidDecisionKernel
    candidate: RapidCandidateBuilder
    clock_timezone: str
    decision_window: RapidDecisionWindow | None = None
    version: str = RAPID_SPECIFICATION_VERSION

    def __post_init__(self) -> None:
        inputs = cast(object, self.inputs)
        if (
            not isinstance(inputs, tuple)
            or not inputs
            or any(
                not isinstance(item, (RapidIndicatorInput, RapidBarInput))
                for item in cast(tuple[object, ...], inputs)
            )
        ):
            raise RapidAdmissionError("rapid inputs must be a nonempty typed tuple")
        names = [item.name for item in self.inputs]
        if len(names) != len(set(names)):
            raise RapidAdmissionError("rapid input names must be unique")
        if not callable(self.decide) or not callable(self.candidate):
            raise RapidAdmissionError("rapid kernel and candidate builder required")
        try:
            ZoneInfo(_text(self.clock_timezone, "clock timezone"))
        except (ZoneInfoNotFoundError, ValueError) as error:
            raise RapidAdmissionError("rapid clock timezone is invalid") from error
        if self.decision_window is not None and not isinstance(
            cast(object, self.decision_window), RapidDecisionWindow
        ):
            raise RapidAdmissionError("rapid decision window is invalid")
        if self.version != RAPID_SPECIFICATION_VERSION:
            raise RapidAdmissionError("unsupported rapid specification version")

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "version": self.version,
            "inputs": [item.to_primitive() for item in self.inputs],
            "clock_timezone": self.clock_timezone,
            "decision_window": None
            if self.decision_window is None
            else self.decision_window.to_primitive(),
            "kernel": _callable_name(self.decide),
        }


@dataclass(frozen=True, slots=True)
class RapidStrategyConfiguration:
    """Exact rule configuration, in the format authoritative execution captures."""

    name: str
    implementation_version: str
    configuration_id: str
    configuration: PrimitiveMappingSnapshot
    parameters: PrimitiveMappingSnapshot
    specification: PrimitiveMappingSnapshot

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "name": self.name,
            "implementation_version": self.implementation_version,
            "configuration_id": self.configuration_id,
            "configuration": self.configuration.to_primitive(),
            "parameters": self.parameters.to_primitive(),
            "rapid_specification": self.specification.to_primitive(),
        }


@dataclass(frozen=True, slots=True)
class RapidSourceReference:
    """References to the authenticated canonical inputs that were scanned."""

    symbol: str
    family_id: str
    family_manifest_id: str | None
    canonical_prediction_dataset_id: str
    timeframes: tuple[tuple[str, str], ...]

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "symbol": self.symbol,
            "family_id": self.family_id,
            "family_manifest_id": self.family_manifest_id,
            "canonical_prediction_dataset_id": self.canonical_prediction_dataset_id,
            "timeframes": [
                {"timeframe_configuration_id": timeframe, "dataset_id": dataset}
                for timeframe, dataset in self.timeframes
            ],
        }


@dataclass(frozen=True, slots=True)
class RapidResearchWindow:
    """The permitted non-holdout plan window and its checked data footprint."""

    research_plan_id: str
    fold_index: int
    fold_name: str
    role: PartitionRole
    window_name: str
    interval_start: datetime
    interval_end: datetime
    warm_up: tuple[tuple[str, int], ...]
    permitted_decisions: int
    purged_decisions: int
    first_decision: datetime
    last_decision: datetime
    footprint_first_session: date
    footprint_last_session: date
    reserved_holdout_start: datetime
    reserved_holdout_end: datetime
    ledger_exposure_scopes_checked: int

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "research_plan_id": self.research_plan_id,
            "fold_index": self.fold_index,
            "fold_name": self.fold_name,
            "role": self.role.value,
            "window_name": self.window_name,
            "interval_start": self.interval_start.isoformat(),
            "interval_end": self.interval_end.isoformat(),
            "warm_up_bars": [
                {"timeframe_configuration_id": timeframe, "bars": bars}
                for timeframe, bars in self.warm_up
            ],
            "permitted_decisions": self.permitted_decisions,
            "purged_decisions": self.purged_decisions,
            "first_decision": self.first_decision.isoformat(),
            "last_decision": self.last_decision.isoformat(),
            "footprint_first_session": self.footprint_first_session.isoformat(),
            "footprint_last_session": self.footprint_last_session.isoformat(),
            "reserved_holdout": {
                "start": self.reserved_holdout_start.isoformat(),
                "end": self.reserved_holdout_end.isoformat(),
                "read": False,
            },
            "ledger_exposure_scopes_checked": self.ledger_exposure_scopes_checked,
        }


@dataclass(frozen=True, slots=True)
class RapidOutcomeConfiguration:
    """Envelope-level outcome definition; per-event values omit these constants."""

    namespace: str
    labeler_configuration_id: str
    evaluator_configuration_id: str
    future_reach: timedelta
    source_dataset_id: str
    value_fields: tuple[str, ...]

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "namespace": self.namespace,
            "labeler_configuration_id": self.labeler_configuration_id,
            "evaluator_configuration_id": self.evaluator_configuration_id,
            "future_reach_microseconds": self.future_reach // timedelta(microseconds=1),
            "source_dataset_id": self.source_dataset_id,
            "value_fields": list(self.value_fields),
        }


@dataclass(frozen=True, slots=True)
class RapidOutcomeValues:
    """One configured outcome's evaluated value fields for one trigger."""

    namespace: str
    values: PrimitiveMappingSnapshot

    def to_primitive(self) -> PrimitiveMapping:
        return {"namespace": self.namespace, "values": self.values.to_primitive()}


@dataclass(frozen=True, slots=True)
class RapidEvent:
    """One trigger: timestamp, direction, optional rule values and outcomes."""

    decision_timestamp: datetime
    signal_session: date
    direction: PredictionDirection
    values: tuple[tuple[str, RapidValue], ...]
    outcomes: tuple[RapidOutcomeValues, ...]

    def value(self, name: str) -> RapidValue:
        for key, value in self.values:
            if key == name:
                return value
        raise KeyError(name)

    def outcome(self, namespace: str) -> PrimitiveMapping:
        for item in self.outcomes:
            if item.namespace == namespace:
                return item.values.to_primitive()
        raise KeyError(namespace)

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "decision_timestamp": self.decision_timestamp.isoformat(),
            "signal_session": self.signal_session.isoformat(),
            "direction": self.direction.value,
            "values": {
                name: None if value is None else decimal_to_primitive(value)
                for name, value in self.values
            },
            "outcomes": {
                item.namespace: item.values.to_primitive() for item in self.outcomes
            },
        }


@dataclass(frozen=True, slots=True)
class RapidOutcomeSummary:
    """Descriptive exploratory statistics; never an OOS or validated metric."""

    namespace: str
    requested: int
    available: int
    statistics: PrimitiveMappingSnapshot

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "namespace": self.namespace,
            "requested": self.requested,
            "available": self.available,
            "statistics": self.statistics.to_primitive(),
        }


@dataclass(frozen=True, slots=True)
class RapidPeriodSummary:
    """Chronological exploratory counts by exchange-session month."""

    period: str
    eligible_decisions: int
    triggers: int
    outcomes: tuple[RapidOutcomeSummary, ...]

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "period": self.period,
            "eligible_decisions": self.eligible_decisions,
            "triggers": self.triggers,
            "outcomes": [item.to_primitive() for item in self.outcomes],
        }


@dataclass(frozen=True, slots=True)
class RapidScanProfile:
    """Operational timings and counts; excluded from scientific comparisons."""

    seconds: tuple[tuple[str, float], ...]
    counts: tuple[tuple[str, int], ...]

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "seconds": {name: value for name, value in self.seconds},
            "counts": {name: value for name, value in self.counts},
        }


_NUMERIC_STATISTICS = ("raw_return", "mfe_percentage", "mae_percentage")
_CATEGORICAL_STATISTICS = ("label", "status")


def summarize_outcomes(
    namespace: str, rows: tuple[PrimitiveMapping, ...]
) -> RapidOutcomeSummary:
    """Counts, means and strictly-positive fractions of available values."""
    statistics: dict[str, Primitive] = {}
    for name in _NUMERIC_STATISTICS:
        values = [
            Decimal(cast(str, row[name]))
            for row in rows
            if isinstance(row.get(name), str)
        ]
        if not values:
            continue
        with localcontext() as arithmetic:
            arithmetic.prec = 34
            mean = sum(values, Decimal(0)) / len(values)
            positive = Decimal(sum(value > 0 for value in values)) / len(values)
        statistics[name] = {
            "count": len(values),
            "mean": decimal_to_primitive(mean),
            "positive_fraction": decimal_to_primitive(positive),
        }
    for name in _CATEGORICAL_STATISTICS:
        labels = [row[name] for row in rows if isinstance(row.get(name), str)]
        if labels:
            statistics[f"{name}_counts"] = {
                label: labels.count(label) for label in sorted(set(map(str, labels)))
            }
    return RapidOutcomeSummary(
        namespace,
        len(rows),
        sum(row.get("available") is True for row in rows),
        PrimitiveMappingSnapshot.capture(statistics),
    )


@dataclass(frozen=True, slots=True)
class RapidScanResult:
    """One exploratory scan. ``authoritative`` is always ``False``.

    No result, study, context, receipt, checkpoint or artifact identity exists
    for rapid output. ``scientific_primitive`` excludes operational timing so
    repeated scans can be compared exactly.
    """

    strategy: RapidStrategyConfiguration
    source: RapidSourceReference
    window: RapidResearchWindow
    decision_window: RapidDecisionWindow | None
    clock_timezone: str
    eligible_decisions: int
    evaluated_decisions: int
    events: tuple[RapidEvent, ...]
    outcome_configurations: tuple[RapidOutcomeConfiguration, ...]
    outcome_summaries: tuple[RapidOutcomeSummary, ...]
    periods: tuple[RapidPeriodSummary, ...]
    values_recorded: bool
    profile: RapidScanProfile = field(compare=False)

    schema_version: ClassVar[str] = RAPID_SCAN_SCHEMA_VERSION
    notice: ClassVar[str] = NON_AUTHORITATIVE_NOTICE

    @property
    def authoritative(self) -> Literal[False]:
        return False

    @property
    def mode(self) -> Literal["exploratory"]:
        return RAPID_SCAN_MODE

    @property
    def trigger_count(self) -> int:
        return len(self.events)

    def scientific_primitive(self) -> PrimitiveMapping:
        return {
            "notice": NON_AUTHORITATIVE_NOTICE,
            "component": "quantforge_rapid_scan",
            "schema_version": RAPID_SCAN_SCHEMA_VERSION,
            "mode": RAPID_SCAN_MODE,
            "authoritative": False,
            "authoritative_reproduction_required": True,
            "strategy": self.strategy.to_primitive(),
            "source": self.source.to_primitive(),
            "window": self.window.to_primitive(),
            "eligibility": {
                "clock_timezone": self.clock_timezone,
                "decision_window": None
                if self.decision_window is None
                else self.decision_window.to_primitive(),
                "frozen_before_scan": True,
            },
            "counts": {
                "permitted_decisions": self.window.permitted_decisions,
                "eligible_decisions": self.eligible_decisions,
                "evaluated_decisions": self.evaluated_decisions,
                "triggers": self.trigger_count,
            },
            "values_recorded": self.values_recorded,
            "outcome_configurations": [
                item.to_primitive() for item in self.outcome_configurations
            ],
            "events": [item.to_primitive() for item in self.events],
            "exploratory_outcome_summaries": [
                item.to_primitive() for item in self.outcome_summaries
            ],
            "exploratory_chronological_summaries": [
                item.to_primitive() for item in self.periods
            ],
        }

    def to_primitive(self) -> PrimitiveMapping:
        return {**self.scientific_primitive(), "profile": self.profile.to_primitive()}


@dataclass(frozen=True, slots=True)
class PromotedStrategyConfiguration:
    """A frozen rule configuration to reproduce authoritatively; no results.

    Promotion never marks the rapid result authoritative. The authoritative run
    must build its rule from ``parameters`` and pass ``require_same_rule``.
    """

    strategy: RapidStrategyConfiguration
    research_plan_id: str
    fold_index: int
    role: PartitionRole

    @property
    def parameters(self) -> PrimitiveMapping:
        return self.strategy.parameters.to_primitive()

    def require_same_rule(self, rule: object) -> None:
        """Prove an authoritative rule has the identical scientific configuration."""
        configuration = getattr(rule, "configuration", None)
        if not callable(configuration):
            raise RapidScanError("promoted configuration requires a configured rule")
        primitive = cast(PrimitiveMapping, configuration())
        if (
            getattr(rule, "name", None) != self.strategy.name
            or getattr(rule, "implementation_version", None)
            != self.strategy.implementation_version
            or getattr(rule, "configuration_id", None) != self.strategy.configuration_id
            or configuration_identity(primitive) != self.strategy.configuration_id
            or PrimitiveMappingSnapshot.capture(primitive)
            != self.strategy.configuration
        ):
            raise RapidScanError(
                "authoritative rule differs from the promoted configuration"
            )

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "component": "quantforge_rapid_promotion",
            "status": "requires_authoritative_reproduction",
            "authoritative": False,
            "strategy": self.strategy.to_primitive(),
            "source_research_window": {
                "research_plan_id": self.research_plan_id,
                "fold_index": self.fold_index,
                "role": self.role.value,
            },
        }


def promote_strategy_configuration(
    result: RapidScanResult,
) -> PromotedStrategyConfiguration:
    """Freeze the exact scanned configuration; rapid outcomes are not carried."""
    if type(result) is not RapidScanResult:
        raise RapidScanError("only rapid scan results can be promoted")
    return PromotedStrategyConfiguration(
        result.strategy,
        result.window.research_plan_id,
        result.window.fold_index,
        result.window.role,
    )
