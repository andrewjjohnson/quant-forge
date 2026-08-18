"""Configurable, auditable multi-timeframe technical-confluence predictions."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from typing import cast

from quantforge.configuration import (
    Primitive,
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
    decimal_to_primitive,
)
from quantforge.data.lineage import FeedScope
from quantforge.data.multi_timeframe import ContextBar, ContextCompletionPolicy
from quantforge.indicators import (
    AVERAGE_DIRECTIONAL_INDEX_OUTPUT,
    BOLLINGER_BANDWIDTH_OUTPUT,
    BOLLINGER_MIDDLE_BAND_OUTPUT,
    EXPONENTIAL_MOVING_AVERAGE_OUTPUT,
    MACD_HISTOGRAM_OUTPUT,
    RELATIVE_VOLUME_OUTPUT,
    SIMPLE_MOVING_AVERAGE_OUTPUT,
    STOCHASTIC_D_OUTPUT,
    STOCHASTIC_K_OUTPUT,
    TALIB_INDICATOR_BACKEND,
    VOLUME_MOVING_AVERAGE_OUTPUT,
    WILDER_AVERAGE_TRUE_RANGE_OUTPUT,
    WILDER_RSI_OUTPUT,
    BollingerBands,
    BollingerBandsParameters,
    ExponentialMovingAverage,
    ExponentialMovingAverageParameters,
    Indicator,
    MarketField,
    MovingAverageConvergenceDivergence,
    RelativeVolume,
    RelativeVolumeParameters,
    SimpleMovingAverage,
    SimpleMovingAverageParameters,
    StochasticOscillator,
    TimeframeNeutralIndicator,
    VolumeMovingAverage,
    VolumeMovingAverageParameters,
    WilderAverageTrueRange,
    WilderAverageTrueRangeParameters,
    WilderDirectionalMovement,
    WilderDirectionalMovementParameters,
    WilderRelativeStrengthIndex,
    WilderRelativeStrengthIndexParameters,
)
from quantforge.prediction.context import (
    PredictionContextError,
    PredictionContextFailurePolicy,
    PredictionContextRequirements,
    PredictionIndicatorRequirement,
    PredictionRuleContext,
    PredictionTimeframeRequirement,
)
from quantforge.prediction.errors import InvalidPredictionConfigurationError
from quantforge.prediction.models import PredictionDirection
from quantforge.prediction.multi_timeframe_features import (
    MultiTimeframeFeatureRequest,
)
from quantforge.prediction.signal_feature_models import (
    SchemaField,
    SchemaFieldCategory,
    SignalDisposition,
    SignalFeatureCandidate,
    SignalFeatureCandidateOutput,
    SignalFeatureValue,
)
from quantforge.timeframes import (
    IntradayInterval,
    SessionInterval,
    Timeframe,
    TradingWeekInterval,
)

TECHNICAL_CONFLUENCE_RULE_CONTRACT_VERSION = "1"
TECHNICAL_CONFLUENCE_RULE_IMPLEMENTATION_VERSION = "1"


def _valid_name(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and value[0].isalpha()
        and value == value.lower()
        and all(character.isalnum() or character == "_" for character in value)
    )


def _timeframe_primitive(timeframe: Timeframe) -> PrimitiveMapping:
    return {
        "configuration_id": timeframe.configuration_id,
        "configuration": timeframe.to_primitive(),
    }


class TechnicalConditionOperator(StrEnum):
    """Supported exact comparisons over normalized values or canonical bar fields."""

    GREATER_THAN = "greater_than"
    GREATER_THAN_OR_EQUAL = "greater_than_or_equal"
    LESS_THAN = "less_than"
    LESS_THAN_OR_EQUAL = "less_than_or_equal"
    EQUAL = "equal"
    CROSSES_ABOVE = "crosses_above"
    CROSSES_BELOW = "crosses_below"


class TechnicalConditionStatus(StrEnum):
    """Auditable result of one configured condition."""

    PASSED = "passed"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"
    DISABLED = "disabled"


class TechnicalConfluenceOutcome(StrEnum):
    """Directional or explicitly abstaining result of the complete rule."""

    UP = "up"
    DOWN = "down"
    NO_PREDICTION = "no_prediction"


@dataclass(frozen=True, slots=True)
class TechnicalConditionOperand:
    """One normalized indicator output or canonical bar field on a timeframe."""

    unit: str
    indicator_alias: str | None = None
    normalized_output_name: str | None = None
    bar_field: MarketField | None = None

    def __post_init__(self) -> None:
        indicator_source = (
            self.indicator_alias is not None or self.normalized_output_name is not None
        )
        if (
            not isinstance(cast(object, self.unit), str)
            or not self.unit
            or indicator_source == (self.bar_field is not None)
            or (
                indicator_source
                and (
                    not _valid_name(self.indicator_alias)
                    or not _valid_name(self.normalized_output_name)
                )
            )
            or (
                self.bar_field is not None
                and not isinstance(cast(object, self.bar_field), MarketField)
            )
        ):
            raise InvalidPredictionConfigurationError(
                "condition operands require one normalized indicator output or "
                "canonical bar field and a unit"
            )

    @classmethod
    def indicator(
        cls,
        alias: str,
        normalized_output_name: str,
        unit: str,
    ) -> "TechnicalConditionOperand":
        return cls(unit, alias, normalized_output_name)

    @classmethod
    def bar(cls, field: MarketField, unit: str) -> "TechnicalConditionOperand":
        return cls(unit, bar_field=field)

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "source": "indicator" if self.indicator_alias is not None else "bar",
            "indicator_alias": self.indicator_alias,
            "normalized_output_name": self.normalized_output_name,
            "bar_field": None if self.bar_field is None else self.bar_field.value,
            "unit": self.unit,
        }


type TechnicalConditionRight = TechnicalConditionOperand | Decimal


@dataclass(frozen=True, slots=True)
class TechnicalCondition:
    """One named, explicitly timeframed comparison used by a directional side."""

    name: str
    timeframe_name: str
    timeframe: Timeframe
    left: TechnicalConditionOperand
    operator: TechnicalConditionOperator
    right: TechnicalConditionRight
    enabled: bool = True

    def __post_init__(self) -> None:
        if (
            not _valid_name(self.name)
            or not _valid_name(self.timeframe_name)
            or not isinstance(cast(object, self.timeframe), Timeframe)
            or not isinstance(cast(object, self.left), TechnicalConditionOperand)
            or not isinstance(cast(object, self.operator), TechnicalConditionOperator)
            or not isinstance(cast(object, self.enabled), bool)
            or not isinstance(
                cast(object, self.right), (TechnicalConditionOperand, Decimal)
            )
        ):
            raise InvalidPredictionConfigurationError(
                "technical conditions require stable names, timeframe, operands, "
                "operator, and enabled state"
            )
        if isinstance(self.right, Decimal):
            if not self.right.is_finite():
                raise InvalidPredictionConfigurationError(
                    "technical condition thresholds must be finite"
                )
        elif self.right.unit != self.left.unit:
            raise InvalidPredictionConfigurationError(
                "technical condition operands must use the same unit"
            )

    @property
    def requires_previous(self) -> bool:
        return self.operator in {
            TechnicalConditionOperator.CROSSES_ABOVE,
            TechnicalConditionOperator.CROSSES_BELOW,
        }

    def to_primitive(self) -> PrimitiveMapping:
        right: PrimitiveMapping = (
            {
                "kind": "threshold",
                "value": decimal_to_primitive(self.right),
                "unit": self.left.unit,
            }
            if isinstance(self.right, Decimal)
            else {"kind": "operand", "operand": self.right.to_primitive()}
        )
        return {
            "enabled": self.enabled,
            "left": self.left.to_primitive(),
            "name": self.name,
            "operator": self.operator.value,
            "right": right,
            "timeframe": _timeframe_primitive(self.timeframe),
            "timeframe_name": self.timeframe_name,
        }


@dataclass(frozen=True, slots=True)
class TechnicalConfluenceParameters:
    """Complete directional condition sets; every enabled set uses all-of semantics."""

    up_conditions: tuple[TechnicalCondition, ...]
    down_conditions: tuple[TechnicalCondition, ...]
    reference_name: str = "custom"

    def __post_init__(self) -> None:
        if (
            not _valid_name(self.reference_name)
            or not isinstance(cast(object, self.up_conditions), tuple)
            or not isinstance(cast(object, self.down_conditions), tuple)
            or not self.up_conditions
            or not self.down_conditions
            or any(
                not isinstance(item, TechnicalCondition)
                for item in cast(
                    tuple[object, ...],
                    (*self.up_conditions, *self.down_conditions),
                )
            )
        ):
            raise InvalidPredictionConfigurationError(
                "technical confluence requires named nonempty UP and DOWN conditions"
            )
        conditions = (*self.up_conditions, *self.down_conditions)
        names = tuple(item.name for item in conditions)
        if len(names) != len(set(names)):
            raise InvalidPredictionConfigurationError(
                "technical condition names must be globally unique"
            )
        if not any(item.enabled for item in self.up_conditions) or not any(
            item.enabled for item in self.down_conditions
        ):
            raise InvalidPredictionConfigurationError(
                "UP and DOWN each require at least one enabled condition"
            )
        aliases: dict[str, str] = {}
        for item in conditions:
            timeframe_id = item.timeframe.configuration_id
            existing = aliases.setdefault(item.timeframe_name, timeframe_id)
            if existing != timeframe_id:
                raise InvalidPredictionConfigurationError(
                    "one timeframe name cannot identify multiple timeframe "
                    "configurations"
                )
        object.__setattr__(
            self,
            "up_conditions",
            tuple(sorted(self.up_conditions, key=lambda item: item.name)),
        )
        object.__setattr__(
            self,
            "down_conditions",
            tuple(sorted(self.down_conditions, key=lambda item: item.name)),
        )

    @property
    def conditions(self) -> tuple[TechnicalCondition, ...]:
        return (*self.up_conditions, *self.down_conditions)

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "condition_combination": "all_enabled_conditions_per_direction",
            "direction_resolution": (
                "UP when only UP passes; DOWN when only DOWN passes; otherwise "
                "NO_PREDICTION"
            ),
            "down_conditions": [item.to_primitive() for item in self.down_conditions],
            "reference_name": self.reference_name,
            "up_conditions": [item.to_primitive() for item in self.up_conditions],
        }


@dataclass(frozen=True, slots=True)
class TechnicalConditionResult:
    """Values and timestamps used to decide one configured condition."""

    condition: TechnicalCondition
    status: TechnicalConditionStatus
    left_value: Decimal | None
    right_value: Decimal | None
    previous_left_value: Decimal | None
    previous_right_value: Decimal | None
    source_timestamp: datetime | None

    @property
    def passed(self) -> bool | None:
        if self.status is TechnicalConditionStatus.PASSED:
            return True
        if self.status is TechnicalConditionStatus.FAILED:
            return False
        return None

    def to_primitive(self) -> PrimitiveMapping:
        def value(item: Decimal | None) -> str | None:
            return None if item is None else decimal_to_primitive(item)

        return {
            "condition": self.condition.to_primitive(),
            "left_value": value(self.left_value),
            "passed": self.passed,
            "previous_left_value": value(self.previous_left_value),
            "previous_right_value": value(self.previous_right_value),
            "right_value": value(self.right_value),
            "source_timestamp": (
                None
                if self.source_timestamp is None
                else self.source_timestamp.isoformat()
            ),
            "status": self.status.value,
        }


@dataclass(frozen=True, slots=True)
class TechnicalConfluenceEvaluation:
    """Complete auditable result before it is adapted to a QF-7 candidate."""

    outcome: TechnicalConfluenceOutcome
    up_conditions_passed: bool
    down_conditions_passed: bool
    condition_results: tuple[TechnicalConditionResult, ...]
    latest_source_timestamps: tuple[tuple[str, datetime], ...]

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "condition_results": [
                item.to_primitive() for item in self.condition_results
            ],
            "down_conditions_passed": self.down_conditions_passed,
            "latest_source_timestamps": {
                name: timestamp.isoformat()
                for name, timestamp in self.latest_source_timestamps
            },
            "outcome": self.outcome.value,
            "up_conditions_passed": self.up_conditions_passed,
        }


@dataclass(frozen=True, slots=True)
class _OperandValues:
    current: Decimal | None
    previous: Decimal | None
    current_timestamp: datetime


class TechnicalConfluencePredictionRule:
    """Evaluate typed all-of directional conditions over a restricted QF-28 context."""

    name = "multi_timeframe_technical_confluence"
    implementation_version = TECHNICAL_CONFLUENCE_RULE_IMPLEMENTATION_VERSION
    warm_up_observations = 1

    def __init__(
        self,
        parameters: TechnicalConfluenceParameters,
        context_requirements: PredictionContextRequirements,
    ) -> None:
        self._parameters = parameters
        self.context_requirements = context_requirements
        self._validate_configuration()

    @property
    def parameters(self) -> TechnicalConfluenceParameters:
        return self._parameters

    @property
    def required_indicators(self) -> tuple[Indicator, ...]:
        indicators: dict[str, Indicator] = {}
        for timeframe in self.context_requirements.all_timeframes:
            for requirement in timeframe.indicators:
                indicator = cast(Indicator, requirement.indicator)
                indicators.setdefault(requirement.configuration_id, indicator)
        return tuple(indicators[name] for name in sorted(indicators))

    def configuration(self) -> PrimitiveMapping:
        return {
            "component_name": self.name,
            "component_type": "prediction_strategy",
            "context_requirements": self.context_requirements.to_primitive(),
            "contract_version": TECHNICAL_CONFLUENCE_RULE_CONTRACT_VERSION,
            "implementation_version": self.implementation_version,
            "parameters": self.parameters.to_primitive(),
            "required_indicators": [
                item.to_primitive()
                for timeframe in self.context_requirements.all_timeframes
                for item in timeframe.indicators
            ],
            "warm_up_observations": self.warm_up_observations,
        }

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    @property
    def multi_timeframe_feature_requests(
        self,
    ) -> tuple[MultiTimeframeFeatureRequest, ...]:
        """Return every normalized condition input for QF-29 provenance capture."""
        requests: dict[str, MultiTimeframeFeatureRequest] = {}
        for condition in self.parameters.conditions:
            for operand in (condition.left, condition.right):
                if not isinstance(operand, TechnicalConditionOperand):
                    continue
                if operand.indicator_alias is None:
                    continue
                request = MultiTimeframeFeatureRequest(
                    condition.timeframe_name,
                    condition.timeframe,
                    operand.indicator_alias,
                    cast(str, operand.normalized_output_name),
                    operand.unit,
                )
                existing = requests.setdefault(request.name, request)
                if existing != request:
                    raise InvalidPredictionConfigurationError(
                        f"condition feature request name is ambiguous: {request.name}"
                    )
        return tuple(requests[name] for name in sorted(requests))

    @property
    def strategy_feature_definitions(self) -> tuple[SchemaField, ...]:
        """Describe every auditable condition and source timestamp for QF-7."""
        timing = "available at the causal multi-timeframe decision timestamp"
        definitions: list[SchemaField] = [
            SchemaField(
                "down_conditions_passed",
                SchemaFieldCategory.CONTEMPORANEOUS_FEATURE,
                "boolean",
                "flag",
                False,
                "all enabled DOWN technical conditions passed",
                timing,
            ),
            SchemaField(
                "prediction_outcome",
                SchemaFieldCategory.CONTEMPORANEOUS_FEATURE,
                "string",
                "enum",
                False,
                "UP, DOWN, or NO_PREDICTION confluence outcome",
                timing,
            ),
            SchemaField(
                "up_conditions_passed",
                SchemaFieldCategory.CONTEMPORANEOUS_FEATURE,
                "boolean",
                "flag",
                False,
                "all enabled UP technical conditions passed",
                timing,
            ),
        ]
        for condition in self.parameters.conditions:
            prefix = f"condition_{condition.name}"
            for suffix, data_type, unit, nullable, source in (
                ("enabled", "boolean", "flag", False, "configured enabled state"),
                ("passed", "boolean", "flag", True, "comparison pass/fail result"),
                ("status", "string", "enum", False, "condition evaluation status"),
                ("left_value", "decimal", condition.left.unit, True, "left operand"),
                ("right_value", "decimal", condition.left.unit, True, "right operand"),
                (
                    "previous_left_value",
                    "decimal",
                    condition.left.unit,
                    True,
                    "prior left operand for crossover comparisons",
                ),
                (
                    "previous_right_value",
                    "decimal",
                    condition.left.unit,
                    True,
                    "prior right operand for crossover comparisons",
                ),
                (
                    "source_timestamp",
                    "string",
                    "iso8601_timestamp",
                    True,
                    "latest source bar used by the condition",
                ),
            ):
                definitions.append(
                    SchemaField(
                        f"{prefix}_{suffix}",
                        SchemaFieldCategory.CONTEMPORANEOUS_FEATURE,
                        data_type,
                        unit,
                        nullable,
                        source,
                        timing,
                    )
                )
        for timeframe_name in self._timeframe_aliases():
            definitions.append(
                SchemaField(
                    f"timeframe_{timeframe_name}_latest_source_timestamp",
                    SchemaFieldCategory.CONTEMPORANEOUS_FEATURE,
                    "string",
                    "iso8601_timestamp",
                    False,
                    "latest rule-visible source bar for the configured timeframe",
                    timing,
                )
            )
        return tuple(sorted(definitions, key=lambda field: field.name))

    def evaluate(self, context: PredictionRuleContext) -> TechnicalConfluenceEvaluation:
        """Evaluate only the latest values visible through the restricted context."""
        if context.requirements != self.context_requirements:
            raise PredictionContextError(
                "technical confluence context requirements do not match the rule"
            )
        results = tuple(
            self._evaluate_condition(context, condition)
            for condition in self.parameters.conditions
        )
        by_name = {item.condition.name: item for item in results}
        up_passed = all(
            by_name[item.name].status is TechnicalConditionStatus.PASSED
            for item in self.parameters.up_conditions
            if item.enabled
        )
        down_passed = all(
            by_name[item.name].status is TechnicalConditionStatus.PASSED
            for item in self.parameters.down_conditions
            if item.enabled
        )
        outcome = (
            TechnicalConfluenceOutcome.UP
            if up_passed and not down_passed
            else TechnicalConfluenceOutcome.DOWN
            if down_passed and not up_passed
            else TechnicalConfluenceOutcome.NO_PREDICTION
        )
        latest_timestamps = tuple(
            sorted(
                (
                    name,
                    context.latest_bar_for(timeframe).end_timestamp,
                )
                for name, timeframe in self._timeframe_aliases().items()
            )
        )
        return TechnicalConfluenceEvaluation(
            outcome,
            up_passed,
            down_passed,
            results,
            latest_timestamps,
        )

    def generate_with_context(
        self, context: PredictionRuleContext
    ) -> SignalFeatureCandidateOutput:
        evaluation = self.evaluate(context)
        direction = (
            PredictionDirection.UP
            if evaluation.outcome is TechnicalConfluenceOutcome.UP
            else PredictionDirection.DOWN
            if evaluation.outcome is TechnicalConfluenceOutcome.DOWN
            else None
        )
        accepted = direction is not None
        selected_reason = (
            None
            if direction is None
            else f"technical_confluence_{evaluation.outcome.value}"
        )
        features = self._feature_values(evaluation)
        candidate = SignalFeatureCandidate(
            symbol=context.symbol,
            signal_session=context.decision_session,
            strategy_id=self.name,
            strategy_implementation_version=self.implementation_version,
            strategy_configuration_id=self.configuration_id,
            source_rule_id=self.name,
            source_rule_implementation_version=self.implementation_version,
            source_rule_configuration_id=self.configuration_id,
            strategy_parameters=PrimitiveMappingSnapshot.capture(
                self.parameters.to_primitive()
            ),
            disposition=(
                SignalDisposition.ACCEPTED if accepted else SignalDisposition.REJECTED
            ),
            reason_codes=(f"technical_confluence_{evaluation.outcome.value}",),
            explanation=(
                f"{evaluation.outcome.value}: "
                f"UP all-of={evaluation.up_conditions_passed}; "
                f"DOWN all-of={evaluation.down_conditions_passed}"
            ),
            direction=direction,
            selected_rule_reason=selected_reason,
            matched_rule_reasons=(
                () if selected_reason is None else (selected_reason,)
            ),
            strategy_features=features,
        )
        return SignalFeatureCandidateOutput(
            self.name,
            self.configuration_id,
            context.prediction_dataset_id,
            (candidate,),
        )

    def _validate_configuration(self) -> None:
        if not isinstance(
            cast(object, self.context_requirements), PredictionContextRequirements
        ):
            raise InvalidPredictionConfigurationError(
                "technical confluence requires QF-28 context requirements"
            )
        declared = {
            item.timeframe.configuration_id: item
            for item in self.context_requirements.all_timeframes
        }
        contextual_intervals = tuple(
            item.timeframe.interval for item in self.context_requirements.contextual
        )
        required_context_ids = {
            item.timeframe.configuration_id
            for item in self.context_requirements.contextual
            if (
                (
                    isinstance(item.timeframe.interval, IntradayInterval)
                    and item.timeframe.interval.nominal_duration == timedelta(hours=4)
                )
                or (
                    isinstance(item.timeframe.interval, SessionInterval)
                    and item.timeframe.interval.session_count == 1
                )
                or (
                    isinstance(item.timeframe.interval, TradingWeekInterval)
                    and item.timeframe.interval.week_count == 1
                )
            )
        }
        if (
            not any(
                isinstance(item, IntradayInterval)
                and item.nominal_duration == timedelta(hours=4)
                for item in contextual_intervals
            )
            or not any(
                isinstance(item, SessionInterval) and item.session_count == 1
                for item in contextual_intervals
            )
            or not any(
                isinstance(item, TradingWeekInterval) and item.week_count == 1
                for item in contextual_intervals
            )
        ):
            raise InvalidPredictionConfigurationError(
                "technical confluence requires explicit 4h, daily, and weekly context"
            )
        condition_context_ids = {
            item.timeframe.configuration_id
            for item in self.parameters.conditions
            if item.enabled
        }
        if not required_context_ids.issubset(condition_context_ids):
            raise InvalidPredictionConfigurationError(
                "technical confluence requires conditions on 4h, daily, and weekly"
            )
        for condition in self.parameters.conditions:
            requirement = declared.get(condition.timeframe.configuration_id)
            if requirement is None:
                raise InvalidPredictionConfigurationError(
                    f"condition timeframe is undeclared: {condition.name}"
                )
            indicators = {item.alias: item for item in requirement.indicators}
            for operand in (condition.left, condition.right):
                if not isinstance(operand, TechnicalConditionOperand):
                    continue
                if operand.indicator_alias is None:
                    continue
                indicator = indicators.get(operand.indicator_alias)
                if indicator is None or operand.normalized_output_name not in (
                    indicator.indicator.output_fields
                ):
                    raise InvalidPredictionConfigurationError(
                        "condition references an undeclared normalized indicator "
                        f"output: {condition.name}"
                    )

    def _timeframe_aliases(self) -> dict[str, Timeframe]:
        aliases: dict[str, Timeframe] = {}
        for condition in self.parameters.conditions:
            aliases.setdefault(condition.timeframe_name, condition.timeframe)
        return dict(sorted(aliases.items()))

    def _operand_values(
        self,
        context: PredictionRuleContext,
        timeframe: Timeframe,
        operand: TechnicalConditionOperand,
    ) -> _OperandValues:
        if operand.bar_field is not None:
            bars: tuple[ContextBar, ...] = context.bars_for(timeframe)
            if not bars:
                raise PredictionContextError(
                    "technical condition timeframe has no visible bars"
                )
            latest_bar = bars[-1]
            current = cast(Decimal, getattr(latest_bar, operand.bar_field.value))
            previous = (
                None
                if len(bars) < 2
                else cast(Decimal, getattr(bars[-2], operand.bar_field.value))
            )
            return _OperandValues(current, previous, latest_bar.end_timestamp)
        output = context.indicator_for(timeframe, cast(str, operand.indicator_alias))
        values = output.values_for(cast(str, operand.normalized_output_name))
        return _OperandValues(
            values[-1],
            None if len(values) < 2 else values[-2],
            output.bar_end_timestamps[-1],
        )

    def _evaluate_condition(
        self,
        context: PredictionRuleContext,
        condition: TechnicalCondition,
    ) -> TechnicalConditionResult:
        if not condition.enabled:
            return TechnicalConditionResult(
                condition,
                TechnicalConditionStatus.DISABLED,
                None,
                None,
                None,
                None,
                None,
            )
        left = self._operand_values(context, condition.timeframe, condition.left)
        if isinstance(condition.right, Decimal):
            right = _OperandValues(
                condition.right,
                condition.right,
                left.current_timestamp,
            )
        else:
            right = self._operand_values(context, condition.timeframe, condition.right)
        source_timestamp = max(left.current_timestamp, right.current_timestamp)
        previous_left = left.previous if condition.requires_previous else None
        previous_right = right.previous if condition.requires_previous else None
        required_values = (
            (left.current, right.current, previous_left, previous_right)
            if condition.requires_previous
            else (left.current, right.current)
        )
        if any(value is None for value in required_values):
            return TechnicalConditionResult(
                condition,
                TechnicalConditionStatus.UNAVAILABLE,
                left.current,
                right.current,
                previous_left,
                previous_right,
                source_timestamp,
            )
        current_left = cast(Decimal, left.current)
        current_right = cast(Decimal, right.current)
        passed = _compare(
            condition.operator,
            current_left,
            current_right,
            previous_left,
            previous_right,
        )
        return TechnicalConditionResult(
            condition,
            (
                TechnicalConditionStatus.PASSED
                if passed
                else TechnicalConditionStatus.FAILED
            ),
            current_left,
            current_right,
            previous_left,
            previous_right,
            source_timestamp,
        )

    def _feature_values(
        self, evaluation: TechnicalConfluenceEvaluation
    ) -> tuple[SignalFeatureValue, ...]:
        values: dict[str, Primitive | Decimal] = {
            "down_conditions_passed": evaluation.down_conditions_passed,
            "prediction_outcome": evaluation.outcome.value,
            "up_conditions_passed": evaluation.up_conditions_passed,
        }
        for result in evaluation.condition_results:
            prefix = f"condition_{result.condition.name}"
            values.update(
                {
                    f"{prefix}_enabled": result.condition.enabled,
                    f"{prefix}_left_value": result.left_value,
                    f"{prefix}_passed": result.passed,
                    f"{prefix}_previous_left_value": result.previous_left_value,
                    f"{prefix}_previous_right_value": result.previous_right_value,
                    f"{prefix}_right_value": result.right_value,
                    f"{prefix}_source_timestamp": (
                        None
                        if result.source_timestamp is None
                        else result.source_timestamp.isoformat()
                    ),
                    f"{prefix}_status": result.status.value,
                }
            )
        for timeframe_name, timestamp in evaluation.latest_source_timestamps:
            values[f"timeframe_{timeframe_name}_latest_source_timestamp"] = (
                timestamp.isoformat()
            )
        return tuple(
            SignalFeatureValue(name, value) for name, value in sorted(values.items())
        )


def _compare(
    operator: TechnicalConditionOperator,
    left: Decimal,
    right: Decimal,
    previous_left: Decimal | None,
    previous_right: Decimal | None,
) -> bool:
    if operator is TechnicalConditionOperator.GREATER_THAN:
        return left > right
    if operator is TechnicalConditionOperator.GREATER_THAN_OR_EQUAL:
        return left >= right
    if operator is TechnicalConditionOperator.LESS_THAN:
        return left < right
    if operator is TechnicalConditionOperator.LESS_THAN_OR_EQUAL:
        return left <= right
    if operator is TechnicalConditionOperator.EQUAL:
        return left == right
    if previous_left is None or previous_right is None:
        return False
    if operator is TechnicalConditionOperator.CROSSES_ABOVE:
        return previous_left <= previous_right and left > right
    return previous_left >= previous_right and left < right


def create_reference_technical_confluence_rule(
    *,
    primary_timeframe: Timeframe,
    four_hour_timeframe: Timeframe,
    daily_timeframe: Timeframe,
    weekly_timeframe: Timeframe,
    feed_scope: FeedScope,
    completion_policy: ContextCompletionPolicy = (
        ContextCompletionPolicy.COMPLETED_BARS_ONLY
    ),
    failure_policy: PredictionContextFailurePolicy = (
        PredictionContextFailurePolicy.FAIL
    ),
) -> TechnicalConfluencePredictionRule:
    """Build the documented fixed SPY reference rule using explicit TA-Lib backends."""

    def indicator(
        alias: str, value: TimeframeNeutralIndicator
    ) -> PredictionIndicatorRequirement:
        return PredictionIndicatorRequirement(alias, value)

    weekly = PredictionTimeframeRequirement(
        weekly_timeframe,
        feed_scope,
        (
            indicator(
                "ema_10",
                ExponentialMovingAverage(
                    ExponentialMovingAverageParameters(10),
                    backend_id=TALIB_INDICATOR_BACKEND,
                ),
            ),
            indicator("macd", MovingAverageConvergenceDivergence()),
        ),
        completion_policy,
    )
    daily = PredictionTimeframeRequirement(
        daily_timeframe,
        feed_scope,
        (
            indicator(
                "atr_14",
                WilderAverageTrueRange(
                    WilderAverageTrueRangeParameters(14),
                    backend_id=TALIB_INDICATOR_BACKEND,
                ),
            ),
            indicator(
                "bollinger_20_2",
                BollingerBands(
                    BollingerBandsParameters(20),
                    backend_id=TALIB_INDICATOR_BACKEND,
                ),
            ),
            indicator(
                "directional_14",
                WilderDirectionalMovement(
                    WilderDirectionalMovementParameters(14),
                    backend_id=TALIB_INDICATOR_BACKEND,
                ),
            ),
            indicator(
                "ema_50",
                ExponentialMovingAverage(
                    ExponentialMovingAverageParameters(50),
                    backend_id=TALIB_INDICATOR_BACKEND,
                ),
            ),
            indicator(
                "relative_volume_20",
                RelativeVolume(RelativeVolumeParameters(20, feed_scope)),
            ),
            indicator(
                "rsi_14",
                WilderRelativeStrengthIndex(
                    WilderRelativeStrengthIndexParameters(14),
                    backend_id=TALIB_INDICATOR_BACKEND,
                ),
            ),
            indicator(
                "sma_20",
                SimpleMovingAverage(
                    SimpleMovingAverageParameters(20),
                    backend_id=TALIB_INDICATOR_BACKEND,
                ),
            ),
            indicator(
                "volume_average_20",
                VolumeMovingAverage(VolumeMovingAverageParameters(20, feed_scope)),
            ),
        ),
        completion_policy,
    )
    four_hour = PredictionTimeframeRequirement(
        four_hour_timeframe,
        feed_scope,
        (
            indicator(
                "ema_9",
                ExponentialMovingAverage(
                    ExponentialMovingAverageParameters(9),
                    backend_id=TALIB_INDICATOR_BACKEND,
                ),
            ),
            indicator(
                "ema_21",
                ExponentialMovingAverage(
                    ExponentialMovingAverageParameters(21),
                    backend_id=TALIB_INDICATOR_BACKEND,
                ),
            ),
            indicator("macd", MovingAverageConvergenceDivergence()),
            indicator("stochastic", StochasticOscillator()),
        ),
        completion_policy,
    )
    requirements = PredictionContextRequirements(
        PredictionTimeframeRequirement(primary_timeframe, feed_scope),
        (weekly, daily, four_hour),
        failure_policy,
    )

    def ind(alias: str, output: str, unit: str) -> TechnicalConditionOperand:
        return TechnicalConditionOperand.indicator(alias, output, unit)

    close = TechnicalConditionOperand.bar(MarketField.CLOSE, "price_per_share")
    volume = TechnicalConditionOperand.bar(MarketField.VOLUME, "shares")
    price = "price_per_share"
    points = "index_points"
    ratio = "ratio"
    up = (
        TechnicalCondition(
            "up_weekly_close_above_ema",
            "weekly",
            weekly_timeframe,
            close,
            TechnicalConditionOperator.GREATER_THAN,
            ind("ema_10", EXPONENTIAL_MOVING_AVERAGE_OUTPUT, price),
        ),
        TechnicalCondition(
            "up_weekly_macd_positive",
            "weekly",
            weekly_timeframe,
            ind("macd", MACD_HISTOGRAM_OUTPUT, price),
            TechnicalConditionOperator.GREATER_THAN,
            Decimal("0"),
        ),
        TechnicalCondition(
            "up_daily_fast_above_slow",
            "daily",
            daily_timeframe,
            ind("sma_20", SIMPLE_MOVING_AVERAGE_OUTPUT, price),
            TechnicalConditionOperator.GREATER_THAN,
            ind("ema_50", EXPONENTIAL_MOVING_AVERAGE_OUTPUT, price),
        ),
        TechnicalCondition(
            "up_daily_close_above_middle_band",
            "daily",
            daily_timeframe,
            close,
            TechnicalConditionOperator.GREATER_THAN_OR_EQUAL,
            ind("bollinger_20_2", BOLLINGER_MIDDLE_BAND_OUTPUT, price),
        ),
        TechnicalCondition(
            "up_daily_bandwidth",
            "daily",
            daily_timeframe,
            ind("bollinger_20_2", BOLLINGER_BANDWIDTH_OUTPUT, ratio),
            TechnicalConditionOperator.GREATER_THAN_OR_EQUAL,
            Decimal("0.02"),
        ),
        TechnicalCondition(
            "up_daily_relative_volume",
            "daily",
            daily_timeframe,
            ind("relative_volume_20", RELATIVE_VOLUME_OUTPUT, ratio),
            TechnicalConditionOperator.GREATER_THAN_OR_EQUAL,
            Decimal("1"),
        ),
        TechnicalCondition(
            "up_daily_volume_above_average",
            "daily",
            daily_timeframe,
            volume,
            TechnicalConditionOperator.GREATER_THAN_OR_EQUAL,
            ind("volume_average_20", VOLUME_MOVING_AVERAGE_OUTPUT, "shares"),
        ),
        TechnicalCondition(
            "up_daily_rsi_ceiling",
            "daily",
            daily_timeframe,
            ind("rsi_14", WILDER_RSI_OUTPUT, points),
            TechnicalConditionOperator.LESS_THAN_OR_EQUAL,
            Decimal("70"),
        ),
        TechnicalCondition(
            "up_daily_adx_floor",
            "daily",
            daily_timeframe,
            ind("directional_14", AVERAGE_DIRECTIONAL_INDEX_OUTPUT, points),
            TechnicalConditionOperator.GREATER_THAN_OR_EQUAL,
            Decimal("20"),
        ),
        TechnicalCondition(
            "up_daily_atr_ceiling",
            "daily",
            daily_timeframe,
            ind("atr_14", WILDER_AVERAGE_TRUE_RANGE_OUTPUT, price),
            TechnicalConditionOperator.LESS_THAN_OR_EQUAL,
            Decimal("15"),
        ),
        TechnicalCondition(
            "up_4h_fast_above_slow",
            "four_hour",
            four_hour_timeframe,
            ind("ema_9", EXPONENTIAL_MOVING_AVERAGE_OUTPUT, price),
            TechnicalConditionOperator.GREATER_THAN,
            ind("ema_21", EXPONENTIAL_MOVING_AVERAGE_OUTPUT, price),
        ),
        TechnicalCondition(
            "up_4h_macd_cross",
            "four_hour",
            four_hour_timeframe,
            ind("macd", MACD_HISTOGRAM_OUTPUT, price),
            TechnicalConditionOperator.CROSSES_ABOVE,
            Decimal("0"),
        ),
        TechnicalCondition(
            "up_4h_stochastic_cross",
            "four_hour",
            four_hour_timeframe,
            ind("stochastic", STOCHASTIC_K_OUTPUT, points),
            TechnicalConditionOperator.CROSSES_ABOVE,
            ind("stochastic", STOCHASTIC_D_OUTPUT, points),
        ),
        TechnicalCondition(
            "up_4h_stochastic_ceiling",
            "four_hour",
            four_hour_timeframe,
            ind("stochastic", STOCHASTIC_K_OUTPUT, points),
            TechnicalConditionOperator.LESS_THAN_OR_EQUAL,
            Decimal("30"),
        ),
    )
    down = (
        TechnicalCondition(
            "down_weekly_close_below_ema",
            "weekly",
            weekly_timeframe,
            close,
            TechnicalConditionOperator.LESS_THAN,
            ind("ema_10", EXPONENTIAL_MOVING_AVERAGE_OUTPUT, price),
        ),
        TechnicalCondition(
            "down_weekly_macd_negative",
            "weekly",
            weekly_timeframe,
            ind("macd", MACD_HISTOGRAM_OUTPUT, price),
            TechnicalConditionOperator.LESS_THAN,
            Decimal("0"),
        ),
        TechnicalCondition(
            "down_daily_fast_below_slow",
            "daily",
            daily_timeframe,
            ind("sma_20", SIMPLE_MOVING_AVERAGE_OUTPUT, price),
            TechnicalConditionOperator.LESS_THAN,
            ind("ema_50", EXPONENTIAL_MOVING_AVERAGE_OUTPUT, price),
        ),
        TechnicalCondition(
            "down_daily_close_below_middle_band",
            "daily",
            daily_timeframe,
            close,
            TechnicalConditionOperator.LESS_THAN_OR_EQUAL,
            ind("bollinger_20_2", BOLLINGER_MIDDLE_BAND_OUTPUT, price),
        ),
        TechnicalCondition(
            "down_daily_bandwidth",
            "daily",
            daily_timeframe,
            ind("bollinger_20_2", BOLLINGER_BANDWIDTH_OUTPUT, ratio),
            TechnicalConditionOperator.GREATER_THAN_OR_EQUAL,
            Decimal("0.02"),
        ),
        TechnicalCondition(
            "down_daily_relative_volume",
            "daily",
            daily_timeframe,
            ind("relative_volume_20", RELATIVE_VOLUME_OUTPUT, ratio),
            TechnicalConditionOperator.GREATER_THAN_OR_EQUAL,
            Decimal("1"),
        ),
        TechnicalCondition(
            "down_daily_volume_above_average",
            "daily",
            daily_timeframe,
            volume,
            TechnicalConditionOperator.GREATER_THAN_OR_EQUAL,
            ind("volume_average_20", VOLUME_MOVING_AVERAGE_OUTPUT, "shares"),
        ),
        TechnicalCondition(
            "down_daily_rsi_floor",
            "daily",
            daily_timeframe,
            ind("rsi_14", WILDER_RSI_OUTPUT, points),
            TechnicalConditionOperator.GREATER_THAN_OR_EQUAL,
            Decimal("30"),
        ),
        TechnicalCondition(
            "down_daily_adx_floor",
            "daily",
            daily_timeframe,
            ind("directional_14", AVERAGE_DIRECTIONAL_INDEX_OUTPUT, points),
            TechnicalConditionOperator.GREATER_THAN_OR_EQUAL,
            Decimal("20"),
        ),
        TechnicalCondition(
            "down_daily_atr_ceiling",
            "daily",
            daily_timeframe,
            ind("atr_14", WILDER_AVERAGE_TRUE_RANGE_OUTPUT, price),
            TechnicalConditionOperator.LESS_THAN_OR_EQUAL,
            Decimal("15"),
        ),
        TechnicalCondition(
            "down_4h_fast_below_slow",
            "four_hour",
            four_hour_timeframe,
            ind("ema_9", EXPONENTIAL_MOVING_AVERAGE_OUTPUT, price),
            TechnicalConditionOperator.LESS_THAN,
            ind("ema_21", EXPONENTIAL_MOVING_AVERAGE_OUTPUT, price),
        ),
        TechnicalCondition(
            "down_4h_macd_cross",
            "four_hour",
            four_hour_timeframe,
            ind("macd", MACD_HISTOGRAM_OUTPUT, price),
            TechnicalConditionOperator.CROSSES_BELOW,
            Decimal("0"),
        ),
        TechnicalCondition(
            "down_4h_stochastic_cross",
            "four_hour",
            four_hour_timeframe,
            ind("stochastic", STOCHASTIC_K_OUTPUT, points),
            TechnicalConditionOperator.CROSSES_BELOW,
            ind("stochastic", STOCHASTIC_D_OUTPUT, points),
        ),
        TechnicalCondition(
            "down_4h_stochastic_floor",
            "four_hour",
            four_hour_timeframe,
            ind("stochastic", STOCHASTIC_K_OUTPUT, points),
            TechnicalConditionOperator.GREATER_THAN_OR_EQUAL,
            Decimal("70"),
        ),
    )
    return TechnicalConfluencePredictionRule(
        TechnicalConfluenceParameters(up, down, "spy_reference_v1"),
        requirements,
    )


__all__ = [
    "TECHNICAL_CONFLUENCE_RULE_CONTRACT_VERSION",
    "TECHNICAL_CONFLUENCE_RULE_IMPLEMENTATION_VERSION",
    "TechnicalCondition",
    "TechnicalConditionOperand",
    "TechnicalConditionOperator",
    "TechnicalConditionResult",
    "TechnicalConditionStatus",
    "TechnicalConfluenceEvaluation",
    "TechnicalConfluenceOutcome",
    "TechnicalConfluenceParameters",
    "TechnicalConfluencePredictionRule",
    "create_reference_technical_confluence_rule",
]
