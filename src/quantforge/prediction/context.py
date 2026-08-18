"""Declarative, restricted multi-timeframe inputs for prediction rules."""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import StrEnum
from typing import Protocol, cast

from quantforge.configuration import (
    Primitive,
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.data.intraday import IntradayBar
from quantforge.data.lineage import AdjustmentBasis, FeedScope
from quantforge.data.multi_timeframe import (
    ContextAvailability,
    ContextBar,
    ContextCompletionPolicy,
    ContextTimeframeRequirement,
    MultiTimeframeContext,
)
from quantforge.indicators import (
    DevelopingBarSupport,
    IndicatorBackendIdentity,
    TimeframeIndicatorOutput,
    TimeframeNeutralIndicator,
    evaluate_indicator,
)
from quantforge.timeframes import BarCompletion, IntradayInterval, Timeframe

PREDICTION_CONTEXT_CONTRACT_VERSION = "1"


class PredictionContextError(ValueError):
    """Declared prediction context cannot be resolved safely."""


class PredictionContextAccessError(PredictionContextError):
    """A rule attempted to access an undeclared context input."""


class PredictionContextFailurePolicy(StrEnum):
    """How the runner handles unavailable or incompatible declared context."""

    FAIL = "fail"
    SKIP = "skip"


def _duration_microseconds(value: timedelta | None) -> int | None:
    if value is None:
        return None
    return (value.days * 86_400 + value.seconds) * 1_000_000 + value.microseconds


def _timeframe_primitive(timeframe: Timeframe) -> PrimitiveMapping:
    return {
        "configuration_id": timeframe.configuration_id,
        "configuration": timeframe.to_primitive(),
    }


@dataclass(frozen=True, slots=True)
class PredictionIndicatorRequirement:
    """One named backend-neutral indicator required by a prediction rule."""

    alias: str
    indicator: TimeframeNeutralIndicator = field(repr=False, compare=False)
    _configuration_snapshot: PrimitiveMappingSnapshot = field(init=False, repr=False)
    _configuration_id: str = field(init=False, repr=False)
    _indicator_name: str = field(init=False, repr=False)
    _backend_identity: IndicatorBackendIdentity | None = field(init=False, repr=False)
    _developing_bar_support: DevelopingBarSupport = field(init=False, repr=False)
    _output_fields: tuple[str, ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.alias), str) or not self.alias:
            raise PredictionContextError("indicator requirement alias is required")
        indicator = cast(object, self.indicator)
        try:
            configuration = self.indicator.configuration()
            configuration_snapshot = PrimitiveMappingSnapshot.capture(configuration)
            configuration_id = self.indicator.configuration_id
            indicator_name = self.indicator.name
            backend_identity = cast(
                object, getattr(self.indicator, "backend_identity", None)
            )
            standard_definition = getattr(self.indicator, "standard_definition", None)
            output_fields = self.indicator.output_fields
            developing_support = self.indicator.developing_bar_support
            required_fields = self.indicator.required_fields
            warm_up = self.indicator.warm_up_observations
        except (AttributeError, TypeError, ValueError) as error:
            raise PredictionContextError(
                "indicator requirement must implement the timeframe-neutral contract"
            ) from error
        if (
            indicator is None
            or not isinstance(cast(object, configuration_id), str)
            or not configuration_id
            or configuration_identity(configuration_snapshot.to_primitive())
            != configuration_id
            or not isinstance(cast(object, indicator_name), str)
            or not indicator_name
            or (
                backend_identity is not None
                and not isinstance(backend_identity, IndicatorBackendIdentity)
            )
            or (standard_definition is not None and backend_identity is None)
            or not isinstance(cast(object, output_fields), tuple)
            or not output_fields
            or any(
                not isinstance(cast(object, name), str) or not name
                for name in output_fields
            )
            or not isinstance(cast(object, required_fields), frozenset)
            or isinstance(warm_up, bool)
            or not isinstance(cast(object, warm_up), int)
            or warm_up < 1
            or not isinstance(cast(object, developing_support), DevelopingBarSupport)
        ):
            raise PredictionContextError(
                "indicator requirement metadata or backend identity is invalid"
            )
        object.__setattr__(self, "_configuration_snapshot", configuration_snapshot)
        object.__setattr__(self, "_configuration_id", configuration_id)
        object.__setattr__(self, "_indicator_name", indicator_name)
        object.__setattr__(self, "_backend_identity", backend_identity)
        object.__setattr__(self, "_developing_bar_support", developing_support)
        object.__setattr__(self, "_output_fields", output_fields)

    @property
    def configuration_id(self) -> str:
        return self._configuration_id

    @property
    def backend_identity(self) -> IndicatorBackendIdentity | None:
        return self._backend_identity

    @property
    def developing_bar_support(self) -> DevelopingBarSupport:
        return self._developing_bar_support

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "alias": self.alias,
            "indicator": {
                "name": self._indicator_name,
                "configuration_id": self._configuration_id,
                "configuration": self._configuration_snapshot.to_primitive(),
                "backend": (
                    None
                    if self._backend_identity is None
                    else self._backend_identity.to_primitive()
                ),
                "developing_bar_support": self._developing_bar_support.value,
                "output_fields": list(self._output_fields),
            },
        }

    def evaluate(
        self,
        context: MultiTimeframeContext,
        timeframe: Timeframe,
        completion_policy: ContextCompletionPolicy,
    ) -> TimeframeIndicatorOutput:
        """Evaluate after proving the declaration did not change."""
        self.validate_unchanged()
        return evaluate_indicator(
            self.indicator,
            context,
            timeframe,
            completion_policy=completion_policy,
        )

    def validate_unchanged(self) -> None:
        """Prove the live indicator still matches its captured declaration."""
        if (
            PrimitiveMappingSnapshot.capture(self.indicator.configuration())
            != self._configuration_snapshot
            or self.indicator.configuration_id != self._configuration_id
            or self.indicator.name != self._indicator_name
            or getattr(self.indicator, "backend_identity", None)
            != self._backend_identity
            or self.indicator.developing_bar_support is not self._developing_bar_support
            or self.indicator.output_fields != self._output_fields
        ):
            raise PredictionContextError(
                f"declared indicator changed before evaluation: {self.alias}"
            )


@dataclass(frozen=True, slots=True)
class PredictionTimeframeRequirement:
    """One declared timeframe, its indicators, and causal source policies."""

    timeframe: Timeframe
    required_feed_scope: FeedScope
    indicators: tuple[PredictionIndicatorRequirement, ...] = ()
    completion_policy: ContextCompletionPolicy = (
        ContextCompletionPolicy.COMPLETED_BARS_ONLY
    )
    maximum_age: timedelta | None = None

    def __post_init__(self) -> None:
        if not isinstance(cast(object, self.timeframe), Timeframe):
            raise PredictionContextError("prediction timeframe is invalid")
        if not isinstance(cast(object, self.required_feed_scope), FeedScope):
            raise PredictionContextError("prediction feed scope is invalid")
        if not isinstance(
            cast(object, self.completion_policy), ContextCompletionPolicy
        ):
            raise PredictionContextError("prediction completion policy is invalid")
        if self.maximum_age is not None and self.maximum_age <= timedelta(0):
            raise PredictionContextError(
                "prediction context maximum age must be positive"
            )
        if not isinstance(cast(object, self.indicators), tuple) or any(
            not isinstance(item, PredictionIndicatorRequirement)
            for item in cast(tuple[object, ...], self.indicators)
        ):
            raise PredictionContextError("prediction indicators must be a tuple")
        aliases = tuple(item.alias for item in self.indicators)
        if len(aliases) != len(set(aliases)):
            raise PredictionContextError(
                "prediction indicator aliases must be unique per timeframe"
            )
        if (
            self.completion_policy is ContextCompletionPolicy.DEVELOPING_BAR_AS_OF
            and any(
                item.developing_bar_support is not DevelopingBarSupport.DEVELOPING_AS_OF
                for item in self.indicators
            )
        ):
            raise PredictionContextError(
                "developing prediction timeframe requires indicators that support "
                "developing bars"
            )
        object.__setattr__(
            self,
            "indicators",
            tuple(sorted(self.indicators, key=lambda item: item.alias)),
        )

    def to_primitive(self) -> PrimitiveMapping:
        session_policy = self.timeframe.session_policy.to_primitive()
        return {
            "timeframe": _timeframe_primitive(self.timeframe),
            "session_policy": {
                "configuration_id": configuration_identity(session_policy),
                "configuration": session_policy,
            },
            "feed_scope": self.required_feed_scope.to_primitive(),
            "completion_policy": self.completion_policy.value,
            "maximum_age_microseconds": _duration_microseconds(self.maximum_age),
            "indicators": [item.to_primitive() for item in self.indicators],
        }


@dataclass(frozen=True, slots=True)
class PredictionContextRequirements:
    """Complete declarative multi-timeframe contract owned by one rule."""

    primary: PredictionTimeframeRequirement
    contextual: tuple[PredictionTimeframeRequirement, ...]
    failure_policy: PredictionContextFailurePolicy = PredictionContextFailurePolicy.FAIL
    contract_version: str = PREDICTION_CONTEXT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.contract_version != PREDICTION_CONTEXT_CONTRACT_VERSION:
            raise PredictionContextError(
                "prediction context contract version is invalid"
            )
        if not isinstance(cast(object, self.primary), PredictionTimeframeRequirement):
            raise PredictionContextError("primary prediction timeframe is invalid")
        if not isinstance(self.primary.timeframe.interval, IntradayInterval):
            raise PredictionContextError(
                "primary prediction timeframe must be intraday"
            )
        if not isinstance(cast(object, self.contextual), tuple) or not self.contextual:
            raise PredictionContextError(
                "multi-timeframe prediction requires contextual timeframes"
            )
        if any(
            not isinstance(item, PredictionTimeframeRequirement)
            for item in cast(tuple[object, ...], self.contextual)
        ):
            raise PredictionContextError("contextual prediction timeframes are invalid")
        if not isinstance(
            cast(object, self.failure_policy), PredictionContextFailurePolicy
        ):
            raise PredictionContextError("prediction context failure policy is invalid")
        ordered = tuple(
            sorted(self.contextual, key=lambda item: item.timeframe.configuration_id)
        )
        identifiers = (
            self.primary.timeframe.configuration_id,
            *(item.timeframe.configuration_id for item in ordered),
        )
        if len(identifiers) != len(set(identifiers)):
            raise PredictionContextError(
                "primary and contextual prediction timeframes must be unique"
            )
        declared = (self.primary, *ordered)
        if any(
            item.timeframe.session_policy != self.primary.timeframe.session_policy
            for item in declared[1:]
        ):
            raise PredictionContextError(
                "prediction timeframes require one compatible session policy"
            )
        if any(
            item.required_feed_scope != self.primary.required_feed_scope
            for item in declared[1:]
        ):
            raise PredictionContextError(
                "prediction timeframes require one compatible feed scope"
            )
        if (
            self.primary.completion_policy
            is ContextCompletionPolicy.DEVELOPING_BAR_AS_OF
        ):
            raise PredictionContextError(
                "the primary decision timeframe must use completed bars"
            )
        object.__setattr__(self, "contextual", ordered)

    @property
    def all_timeframes(self) -> tuple[PredictionTimeframeRequirement, ...]:
        return (self.primary, *self.contextual)

    @property
    def context_completion_policy(self) -> ContextCompletionPolicy:
        if any(
            item.completion_policy is ContextCompletionPolicy.DEVELOPING_BAR_AS_OF
            for item in self.contextual
        ):
            return ContextCompletionPolicy.DEVELOPING_BAR_AS_OF
        return ContextCompletionPolicy.COMPLETED_BARS_ONLY

    def context_timeframe_requirements(
        self,
    ) -> tuple[ContextTimeframeRequirement, ...]:
        """Return the QF-20 requirements a provider passes to the data builder."""
        return tuple(
            ContextTimeframeRequirement(item.timeframe, item.maximum_age)
            for item in self.contextual
        )

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "contract_version": self.contract_version,
            "primary": self.primary.to_primitive(),
            "contextual": [item.to_primitive() for item in self.contextual],
            "context_completion_policy": self.context_completion_policy.value,
            "failure_policy": self.failure_policy.value,
        }


class PredictionContextProvider(Protocol):
    """Obtain one QF-20/QF-21 context for declared prediction requirements."""

    def get_context(
        self, requirements: PredictionContextRequirements
    ) -> MultiTimeframeContext: ...


class PredictionIndicatorOutputCache(Protocol):
    """Resolve normalized indicator output under an identity-aware cache policy."""

    def resolve(
        self,
        requirement: PredictionIndicatorRequirement,
        context: MultiTimeframeContext,
        timeframe: Timeframe,
        completion_policy: ContextCompletionPolicy,
    ) -> TimeframeIndicatorOutput: ...


@dataclass(frozen=True, slots=True)
class NamedPredictionIndicatorOutput:
    """A rule-facing alias paired only with a normalized QuantForge output."""

    alias: str
    output: TimeframeIndicatorOutput

    def manifest_primitive(self) -> PrimitiveMapping:
        return {
            "alias": self.alias,
            "indicator_name": self.output.indicator_name,
            "configuration_id": self.output.configuration_id,
            "backend": (
                None
                if self.output.backend_identity is None
                else self.output.backend_identity.to_primitive()
            ),
            "source_timeframe": _timeframe_primitive(self.output.source_timeframe),
            "source_fields": [item.value for item in self.output.source_fields],
            "completion_policy": self.output.completion_policy.value,
            "dataset_reference": self.output.dataset_reference.to_primitive(
                include_feed_scope=True
            ),
            "warm_up_bars": self.output.warm_up_bars,
            "visible_bar_ids": list(self.output.bar_ids),
            "output_fields": [item.name for item in self.output.fields],
        }


@dataclass(frozen=True, slots=True)
class PredictionTimeframeInput:
    """Restricted bars and normalized indicators for one declared timeframe."""

    requirement: PredictionTimeframeRequirement
    bars: tuple[ContextBar, ...]
    indicators: tuple[NamedPredictionIndicatorOutput, ...]

    def indicator(self, alias: str) -> TimeframeIndicatorOutput:
        for item in self.indicators:
            if item.alias == alias:
                return item.output
        raise PredictionContextAccessError(
            f"indicator was not declared for this timeframe: {alias}"
        )

    def manifest_primitive(self) -> PrimitiveMapping:
        return {
            "requirement": self.requirement.to_primitive(),
            "visible_bar_ids": [bar.bar_id for bar in self.bars],
            "indicators": [item.manifest_primitive() for item in self.indicators],
        }


@dataclass(frozen=True, slots=True)
class PredictionRuleContext:
    """Rule-facing view containing exactly the declared causal inputs."""

    prediction_dataset_id: str
    symbol: str
    adjustment_basis: AdjustmentBasis
    requirements: PredictionContextRequirements
    source_context_snapshot: PrimitiveMappingSnapshot
    timeframes: tuple[PredictionTimeframeInput, ...]

    @property
    def as_of(self) -> datetime:
        value = self.source_context_snapshot.to_primitive()["as_of"]
        return datetime.fromisoformat(cast(str, value))

    @property
    def context_id(self) -> str:
        value = self.source_context_snapshot.to_primitive()["context_id"]
        return cast(str, value)

    @property
    def decision_session(self) -> date:
        """Return the sole session for which this as-of snapshot may emit a signal."""
        latest_primary_bar = self.latest_bar_for(self.requirements.primary.timeframe)
        if isinstance(latest_primary_bar, IntradayBar):
            return latest_primary_bar.session_date
        return latest_primary_bar.session_dates[-1]

    def _for_timeframe(self, timeframe: Timeframe) -> PredictionTimeframeInput:
        if not isinstance(cast(object, timeframe), Timeframe):
            raise PredictionContextAccessError(
                "requested prediction timeframe is invalid"
            )
        for item in self.timeframes:
            if (
                item.requirement.timeframe.configuration_id
                == timeframe.configuration_id
            ):
                return item
        raise PredictionContextAccessError(
            "timeframe was not declared by the prediction rule: "
            f"{timeframe.configuration_id}"
        )

    def bars_for(self, timeframe: Timeframe) -> tuple[ContextBar, ...]:
        return self._for_timeframe(timeframe).bars

    def latest_bar_for(self, timeframe: Timeframe) -> ContextBar:
        return self.bars_for(timeframe)[-1]

    def indicator_for(
        self, timeframe: Timeframe, alias: str
    ) -> TimeframeIndicatorOutput:
        return self._for_timeframe(timeframe).indicator(alias)

    def manifest_primitive(self) -> PrimitiveMapping:
        return {
            "prediction_dataset_id": self.prediction_dataset_id,
            "symbol": self.symbol,
            "adjustment_basis": self.adjustment_basis.to_primitive(),
            "decision_session": self.decision_session.isoformat(),
            "requirements": self.requirements.to_primitive(),
            "source_context": self.source_context_snapshot.to_primitive(),
            "timeframes": [item.manifest_primitive() for item in self.timeframes],
        }

    def values_primitive(self) -> PrimitiveMapping:
        """Return the complete rule input for mutation checks, including values."""
        timeframe_values: list[Primitive] = []
        for item in self.timeframes:
            indicator_values: list[Primitive] = [
                {
                    "alias": indicator.alias,
                    "rows": cast(list[Primitive], indicator.output.to_rows()),
                }
                for indicator in item.indicators
            ]
            timeframe_values.append(
                {
                    "timeframe_configuration_id": (
                        item.requirement.timeframe.configuration_id
                    ),
                    "bars": cast(
                        list[Primitive], [bar.to_primitive() for bar in item.bars]
                    ),
                    "indicators": indicator_values,
                }
            )
        return {
            **self.manifest_primitive(),
            "timeframe_values": timeframe_values,
        }


def build_prediction_rule_context(
    requirements: PredictionContextRequirements,
    context: MultiTimeframeContext,
    *,
    prediction_dataset_id: str,
    symbol: str,
    prediction_adjustment_basis: AdjustmentBasis,
    indicator_output_cache: PredictionIndicatorOutputCache | None = None,
) -> PredictionRuleContext:
    """Validate, restrict, and evaluate one declared prediction context."""
    if not isinstance(cast(object, requirements), PredictionContextRequirements):
        raise PredictionContextError("prediction context requirements are invalid")
    if not isinstance(cast(object, context), MultiTimeframeContext):
        raise PredictionContextError(
            "prediction context provider returned an invalid context"
        )
    if not prediction_dataset_id or not symbol:
        raise PredictionContextError(
            "prediction dataset identity and symbol are required"
        )
    if not isinstance(cast(object, prediction_adjustment_basis), AdjustmentBasis):
        raise PredictionContextError("prediction dataset adjustment basis is invalid")
    if context.primary_timeframe != requirements.primary.timeframe:
        raise PredictionContextError(
            "prediction context primary timeframe is incompatible"
        )
    if context.completion_policy is not requirements.context_completion_policy:
        raise PredictionContextError(
            "prediction context completion policy is incompatible"
        )
    expected_contextual = requirements.context_timeframe_requirements()
    if context.required_timeframes != expected_contextual:
        raise PredictionContextError(
            "prediction context timeframes or staleness policies are incompatible"
        )

    try:
        primary_context_bars = context.bars_for(requirements.primary.timeframe)
    except ValueError as error:
        raise PredictionContextError(str(error)) from error
    primary_bars = tuple(
        bar
        for bar in primary_context_bars
        if bar.completion is not BarCompletion.DEVELOPING
    )
    if not primary_bars:
        raise PredictionContextError(
            "prediction context has no completed primary decision bar"
        )
    primary_decision_boundary = primary_bars[-1].end_timestamp

    visible_context_bars = tuple(
        bar for timeframe in context.timeframes for bar in timeframe.bars
    )
    if any(bar.symbol != symbol for bar in visible_context_bars):
        raise PredictionContextError(
            "prediction context symbol is incompatible with the prediction dataset"
        )
    context_adjustment_bases = {
        bar.provenance.adjustment_basis
        for bar in visible_context_bars
        if isinstance(bar, IntradayBar)
    }
    if context_adjustment_bases != {prediction_adjustment_basis}:
        raise PredictionContextError(
            "prediction context adjustment basis is incompatible with the "
            "prediction dataset"
        )

    resolved: list[PredictionTimeframeInput] = []
    for requirement in requirements.all_timeframes:
        try:
            metadata = context.metadata_for(requirement.timeframe)
        except ValueError as error:
            raise PredictionContextError(str(error)) from error
        if metadata.availability is ContextAvailability.MISSING:
            raise PredictionContextError(
                "prediction context is missing declared timeframe: "
                f"{requirement.timeframe.configuration_id}"
            )
        if metadata.dataset_reference is None:
            raise PredictionContextError(
                "prediction context timeframe has no dataset provenance"
            )
        if metadata.dataset_reference.feed_scope != requirement.required_feed_scope:
            raise PredictionContextError(
                "prediction context feed scope is incompatible for timeframe: "
                f"{requirement.timeframe.configuration_id}"
            )
        try:
            context_bars = context.bars_for(requirement.timeframe)
        except ValueError as error:
            raise PredictionContextError(str(error)) from error
        bars = (
            tuple(
                bar
                for bar in context_bars
                if bar.completion is not BarCompletion.DEVELOPING
            )
            if requirement.completion_policy
            is ContextCompletionPolicy.COMPLETED_BARS_ONLY
            else context_bars
        )
        if not bars:
            raise PredictionContextError(
                "prediction context has no bar under the declared completion policy"
            )
        if any(bar.end_timestamp > primary_decision_boundary for bar in bars):
            raise PredictionContextError(
                "prediction context exposes a bar after the primary decision boundary"
            )
        age = context.as_of - bars[-1].end_timestamp
        if metadata.availability is ContextAvailability.STALE or (
            requirement.maximum_age is not None and age > requirement.maximum_age
        ):
            raise PredictionContextError(
                "prediction context is stale for timeframe: "
                f"{requirement.timeframe.configuration_id}"
            )
        outputs = tuple(
            NamedPredictionIndicatorOutput(
                item.alias,
                (
                    item.evaluate(
                        context,
                        requirement.timeframe,
                        requirement.completion_policy,
                    )
                    if indicator_output_cache is None
                    else indicator_output_cache.resolve(
                        item,
                        context,
                        requirement.timeframe,
                        requirement.completion_policy,
                    )
                ),
            )
            for item in requirement.indicators
        )
        resolved.append(PredictionTimeframeInput(requirement, bars, outputs))

    source_snapshot = PrimitiveMappingSnapshot.capture(context.to_primitive())
    rule_context = PredictionRuleContext(
        prediction_dataset_id,
        symbol,
        prediction_adjustment_basis,
        requirements,
        source_snapshot,
        tuple(resolved),
    )
    # Force deterministic JSON-compatible manifest validation at the boundary.
    PrimitiveMappingSnapshot.capture(rule_context.manifest_primitive())
    return rule_context


def skipped_prediction_context_manifest(
    requirements: PredictionContextRequirements,
    reason: str,
    source_context: MultiTimeframeContext | None = None,
) -> PrimitiveMapping:
    """Return stable manifest evidence when explicit policy skips rule execution."""
    return {
        "status": "skipped",
        "requirements": requirements.to_primitive(),
        "reason": reason,
        "source_context": (
            source_context.to_primitive()
            if isinstance(source_context, MultiTimeframeContext)
            else None
        ),
    }


def available_prediction_context_manifest(
    context: PredictionRuleContext,
) -> PrimitiveMapping:
    return {"status": "available", **context.manifest_primitive()}


__all__ = [
    "PREDICTION_CONTEXT_CONTRACT_VERSION",
    "NamedPredictionIndicatorOutput",
    "PredictionContextAccessError",
    "PredictionContextError",
    "PredictionContextFailurePolicy",
    "PredictionContextProvider",
    "PredictionContextRequirements",
    "PredictionIndicatorRequirement",
    "PredictionRuleContext",
    "PredictionTimeframeInput",
    "PredictionTimeframeRequirement",
    "available_prediction_context_manifest",
    "build_prediction_rule_context",
    "skipped_prediction_context_manifest",
]
