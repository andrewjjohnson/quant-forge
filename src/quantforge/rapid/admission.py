"""Explicit rapid-scan admission of rules, prepared features and outcomes (QF-72).

A component is rapid-compatible only when it is reviewed for exact prepared
evaluation: a declared rapid specification over completed-bar inputs, QF-63
prefix-stable indicators, warm-up covered by the plan window, and reviewed
QF-47/QF-49 outcomes on the canonical primary source within the plan's purge
horizon. Everything else is refused; nothing is approximated.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, cast

from quantforge.configuration import (
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.data import TimeframeBarSeries
from quantforge.data.multi_timeframe import ContextCompletionPolicy
from quantforge.prediction.context import (
    PredictionContextRequirements,
    PredictionIndicatorRequirement,
)
from quantforge.prediction.feature_dataset import PredictionStudyOutcome
from quantforge.prediction.intraday_forward_return import (
    IntradayForwardReturnEvaluator,
    IntradayForwardReturnOutcomeLabeler,
)
from quantforge.prediction.intraday_path import IntradayPathOutcomeLabeler
from quantforge.prediction.intraday_path_evaluation import (
    IntradayExcursionEvaluator,
    IntradayTargetStopEvaluator,
)
from quantforge.prediction.outcome_temporal import (
    OutcomeTemporalConfiguration,
    outcome_temporal_configuration,
)
from quantforge.prediction.prepared_features import prefix_stable_indicator
from quantforge.rapid.models import (
    RapidAdmissionError,
    RapidIndicatorInput,
    RapidRuleSpecification,
    RapidStrategyConfiguration,
)
from quantforge.timeframes import Timeframe

# Reviewed exact labeler/evaluator pairs whose evaluation depends only on the
# QF-46 resolution, the bounded same-session path and the signal direction.
_REVIEWED_OUTCOMES: frozenset[tuple[type, type]] = frozenset(
    {
        (IntradayForwardReturnOutcomeLabeler, IntradayForwardReturnEvaluator),
        (IntradayPathOutcomeLabeler, IntradayExcursionEvaluator),
        (IntradayPathOutcomeLabeler, IntradayTargetStopEvaluator),
    }
)
# Constant per outcome configuration (kept once in the result envelope) or
# repeated from the event itself; identities are never repeated per event.
_CONFIGURATION_FIELDS = frozenset(
    {
        "anchor_kind",
        "decision_timestamp",
        "direction",
        "elapsed_duration_microseconds",
        "horizon_kind",
        "outcome_session",
        "same_bar_conflict_policy",
        "signal_session",
        "source_reference",
        "stop_percentage",
        "target_percentage",
    }
)


@dataclass(frozen=True, slots=True)
class ResolvedRapidInput:
    """One admitted input bound to its timeframe position and value accessor."""

    name: str
    timeframe_index: int
    lag: int
    alias: str | None
    output: str | None
    bar_field: str | None


@dataclass(frozen=True, slots=True)
class AdmittedRapidRule:
    rule: object
    requirements: PredictionContextRequirements
    specification: RapidRuleSpecification
    timeframes: tuple[Timeframe, ...]
    inputs: tuple[ResolvedRapidInput, ...]
    configuration: RapidStrategyConfiguration
    warm_up_observations: int


@dataclass(frozen=True, slots=True)
class AdmittedRapidOutcome:
    outcome: PredictionStudyOutcome[Any, Any]
    temporal: OutcomeTemporalConfiguration
    future_reach: timedelta
    field_names: frozenset[str]
    value_fields: tuple[str, ...]


def _indicator(
    requirements: PredictionContextRequirements, timeframe: Timeframe, alias: str
) -> PredictionIndicatorRequirement:
    for item in requirements.all_timeframes:
        if item.timeframe.configuration_id == timeframe.configuration_id:
            for indicator in item.indicators:
                if indicator.alias == alias:
                    return indicator
    raise RapidAdmissionError(f"rapid input names an undeclared indicator: {alias}")


def admit_rule(
    rule: object,
    *,
    sources: Mapping[str, TimeframeBarSeries],
    warm_up: Mapping[str, int],
) -> AdmittedRapidRule:
    """Admit one configured rule for exact prepared rapid evaluation."""
    specify = getattr(rule, "rapid_specification", None)
    if not callable(specify):
        raise RapidAdmissionError(
            f"{type(rule).__qualname__} declares no reviewed rapid specification; "
            "use the authoritative pipeline"
        )
    requirements = getattr(rule, "context_requirements", None)
    if not isinstance(requirements, PredictionContextRequirements):
        raise RapidAdmissionError("rapid rules require QF-28 context requirements")
    if requirements.context_completion_policy is not (
        ContextCompletionPolicy.COMPLETED_BARS_ONLY
    ) or any(
        item.completion_policy is not ContextCompletionPolicy.COMPLETED_BARS_ONLY
        or item.maximum_age is not None
        for item in requirements.all_timeframes
    ):
        raise RapidAdmissionError(
            "rapid scans support completed-bar timeframes without staleness limits"
        )
    timeframes = tuple(item.timeframe for item in requirements.all_timeframes)
    for item in requirements.all_timeframes:
        source = sources.get(item.timeframe.configuration_id)
        if source is None:
            raise RapidAdmissionError(
                "rapid rule timeframe has no session source: "
                f"{item.timeframe.configuration_id}"
            )
        if source.dataset_reference.feed_scope != item.required_feed_scope:
            raise RapidAdmissionError("rapid rule feed scope differs from its source")
        for indicator in item.indicators:
            if not prefix_stable_indicator(indicator.indicator):
                raise RapidAdmissionError(
                    f"indicator {indicator.alias} is not a reviewed prefix-stable "
                    "type and backend"
                )
    specification: object = specify()
    if not isinstance(specification, RapidRuleSpecification):
        raise RapidAdmissionError("rapid specification has the wrong type")
    identifiers = tuple(timeframe.configuration_id for timeframe in timeframes)
    resolved: list[ResolvedRapidInput] = []
    for item in specification.inputs:
        identifier = item.timeframe.configuration_id
        if identifier not in identifiers:
            raise RapidAdmissionError(
                f"rapid input {item.name} names an undeclared timeframe"
            )
        declared = warm_up.get(identifier)
        if declared is None:
            raise RapidAdmissionError("plan window declares no warm-up for a timeframe")
        if isinstance(item, RapidIndicatorInput):
            indicator = _indicator(requirements, item.timeframe, item.alias)
            if item.output not in indicator.indicator.output_fields:
                raise RapidAdmissionError(
                    f"rapid input {item.name} names an undeclared indicator output"
                )
            required = indicator.indicator.warm_up_observations - 1 + item.lag
            resolved.append(
                ResolvedRapidInput(
                    item.name,
                    identifiers.index(identifier),
                    item.lag,
                    item.alias,
                    item.output,
                    None,
                )
            )
        else:
            required = item.lag
            resolved.append(
                ResolvedRapidInput(
                    item.name,
                    identifiers.index(identifier),
                    item.lag,
                    None,
                    None,
                    item.field,
                )
            )
        if declared < required:
            raise RapidAdmissionError(
                f"plan window warm-up ({declared} bars) cannot supply rapid input "
                f"{item.name} ({required} bars required)"
            )
    configuration = _strategy_configuration(rule, specification)
    warm_up_observations = getattr(rule, "warm_up_observations", None)
    if (
        isinstance(warm_up_observations, bool)
        or not isinstance(warm_up_observations, int)
        or warm_up_observations < 1
    ):
        raise RapidAdmissionError("rapid rules require a positive declared warm-up")
    return AdmittedRapidRule(
        rule,
        requirements,
        specification,
        timeframes,
        tuple(resolved),
        configuration,
        warm_up_observations,
    )


def _strategy_configuration(
    rule: object, specification: RapidRuleSpecification
) -> RapidStrategyConfiguration:
    configure = getattr(rule, "configuration", None)
    parameters = getattr(rule, "parameters", None)
    to_primitive = getattr(parameters, "to_primitive", None)
    name = getattr(rule, "name", None)
    version = getattr(rule, "implementation_version", None)
    configuration_id = getattr(rule, "configuration_id", None)
    if not callable(configure) or not callable(to_primitive):
        raise RapidAdmissionError("rapid rules require configuration and parameters")
    configuration = cast(PrimitiveMapping, configure())
    if (
        not isinstance(name, str)
        or not isinstance(version, str)
        or not isinstance(configuration_id, str)
        or configuration_identity(configuration) != configuration_id
    ):
        raise RapidAdmissionError("rapid rule configuration identity is invalid")
    return RapidStrategyConfiguration(
        name,
        version,
        configuration_id,
        PrimitiveMappingSnapshot.capture(configuration),
        PrimitiveMappingSnapshot.capture(cast(PrimitiveMapping, to_primitive())),
        PrimitiveMappingSnapshot.capture(specification.to_primitive()),
    )


def admit_outcomes(
    outcomes: tuple[object, ...],
    *,
    primary: TimeframeBarSeries,
    maximum_reach: timedelta,
) -> tuple[AdmittedRapidOutcome, ...]:
    """Admit reviewed configured outcomes on the canonical primary source."""
    if not isinstance(cast(object, outcomes), tuple):
        raise RapidAdmissionError("rapid outcomes must be a tuple")
    admitted: list[AdmittedRapidOutcome] = []
    namespaces: set[str] = set()
    for item in outcomes:
        if type(item) is not PredictionStudyOutcome:
            raise RapidAdmissionError("rapid outcomes must be QF-7 study outcomes")
        outcome = cast(PredictionStudyOutcome[Any, Any], item)
        pair = (type(outcome.labeler), type(outcome.evaluator))
        if pair not in _REVIEWED_OUTCOMES:
            raise RapidAdmissionError(
                f"outcome {outcome.namespace} is not a reviewed rapid outcome"
            )
        if outcome.outcome_source is None or outcome.outcome_source != primary:
            raise RapidAdmissionError(
                f"outcome {outcome.namespace} must label the canonical primary source"
            )
        if outcome.namespace in namespaces:
            raise RapidAdmissionError("rapid outcome namespaces must be unique")
        namespaces.add(outcome.namespace)
        temporal = outcome_temporal_configuration(outcome.labeler.configuration())
        reach = temporal.future_temporal_reach.elapsed
        if reach is None or reach > maximum_reach:
            raise RapidAdmissionError(
                f"outcome {outcome.namespace} reaches beyond the plan purge horizon"
            )
        names = frozenset(field.name for field in outcome.fields)
        admitted.append(
            AdmittedRapidOutcome(
                outcome,
                temporal,
                reach,
                names,
                tuple(
                    sorted(
                        name
                        for name in names
                        if name not in _CONFIGURATION_FIELDS
                        and not name.endswith("_id")
                        and not name.endswith("_convention")
                    )
                ),
            )
        )
    return tuple(admitted)
