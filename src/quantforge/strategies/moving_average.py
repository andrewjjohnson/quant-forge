"""Reference long-only moving-average crossover strategy."""

from decimal import Decimal

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.data.models import MarketDataset
from quantforge.indicators.base import Indicator
from quantforge.indicators.models import MarketField
from quantforge.indicators.moving_average import (
    SIMPLE_MOVING_AVERAGE_OUTPUT,
    SimpleMovingAverage,
    SimpleMovingAverageParameters,
)
from quantforge.strategies.base import next_exchange_session
from quantforge.strategies.exceptions import InvalidStrategyOutputError
from quantforge.strategies.models import (
    ExecutionSessionStatus,
    ExecutionTiming,
    IndicatorObservation,
    MarketDataReference,
    ParameterValue,
    PositionIntent,
    StrategyDecision,
    StrategyOutput,
)
from quantforge.strategies.parameters import (
    MovingAverageCrossoverParameters,
    StrategyParameters,
)
from quantforge.strategies.sizing import (
    PositionSizingPolicy,
    TargetWeightSizingPolicy,
)


class MovingAverageCrossoverStrategy:
    """Emit state changes when a fast average crosses a slow average.

    Equality emits no decision. A move from equality to a strict relation is a
    crossover because the prior relation is non-strict. The first available
    average pair cannot signal because there is no valid prior pair.
    """

    name = "moving_average_crossover"
    timing = ExecutionTiming.NEXT_SESSION_AFTER_CLOSE
    asset_assumptions = ("single stock or ETF symbol", "long-only")

    def __init__(self, parameters: MovingAverageCrossoverParameters) -> None:
        self._parameters = parameters
        self._sizing_policy = TargetWeightSizingPolicy(parameters.target_long_weight)
        self._required_indicators: tuple[Indicator, ...] = (
            SimpleMovingAverage(
                SimpleMovingAverageParameters(
                    parameters.fast_window, parameters.source_field
                )
            ),
            SimpleMovingAverage(
                SimpleMovingAverageParameters(
                    parameters.slow_window, parameters.source_field
                )
            ),
        )

    @property
    def parameters(self) -> StrategyParameters:
        return self._parameters

    @property
    def required_fields(self) -> frozenset[MarketField]:
        return frozenset((self._parameters.source_field,))

    @property
    def required_indicators(self) -> tuple[Indicator, ...]:
        return self._required_indicators

    @property
    def warm_up_observations(self) -> int:
        return max(
            indicator.warm_up_observations for indicator in self._required_indicators
        )

    @property
    def sizing_policy(self) -> PositionSizingPolicy:
        return self._sizing_policy

    def configuration(self) -> PrimitiveMapping:
        return {
            "component_type": "strategy",
            "component_name": self.name,
            "contract_version": "1",
            "parameters": self._parameters.to_primitive(),
            "required_fields": [field.value for field in sorted(self.required_fields)],
            "required_indicators": [
                indicator.configuration() for indicator in self._required_indicators
            ],
            "warm_up_observations": self.warm_up_observations,
            "timing_convention": self.timing.value,
            "sizing": self._sizing_policy.configuration(),
            "asset_assumptions": list(self.asset_assumptions),
        }

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    def generate(self, dataset: MarketDataset) -> StrategyOutput:
        """Calculate owned indicators and emit only target-state transitions."""
        fast_output = self._required_indicators[0].calculate(dataset)
        slow_output = self._required_indicators[1].calculate(dataset)
        fast_values = fast_output.values_for(SIMPLE_MOVING_AVERAGE_OUTPUT)
        slow_values = slow_output.values_for(SIMPLE_MOVING_AVERAGE_OUTPUT)
        parameters = _parameter_snapshot(self._parameters.to_primitive())
        decisions: list[StrategyDecision] = []
        current_target = PositionIntent.FLAT

        for index in range(1, len(dataset.bars)):
            previous_fast = fast_values[index - 1]
            previous_slow = slow_values[index - 1]
            current_fast = fast_values[index]
            current_slow = slow_values[index]
            if None in (
                previous_fast,
                previous_slow,
                current_fast,
                current_slow,
            ):
                continue
            assert isinstance(previous_fast, Decimal)
            assert isinstance(previous_slow, Decimal)
            assert isinstance(current_fast, Decimal)
            assert isinstance(current_slow, Decimal)

            target: PositionIntent | None = None
            reason: str | None = None
            if (
                previous_fast <= previous_slow
                and current_fast > current_slow
                and current_target is PositionIntent.FLAT
            ):
                target = PositionIntent.LONG
                reason = "fast moving average crossed above slow moving average"
            elif (
                previous_fast >= previous_slow
                and current_fast < current_slow
                and current_target is PositionIntent.LONG
            ):
                target = PositionIntent.FLAT
                reason = "fast moving average crossed below slow moving average"
            if target is None:
                continue

            current_target = target
            sizing_intent = self._sizing_policy.size(target)
            signal_session = dataset.bars[index].session_date
            decisions.append(
                StrategyDecision(
                    canonical_symbol=dataset.metadata.canonical_symbol,
                    signal_session=signal_session,
                    earliest_executable_session=next_exchange_session(
                        signal_session, dataset.metadata.calendar
                    ),
                    execution_timing=self.timing,
                    execution_session_status=ExecutionSessionStatus.PENDING,
                    target_position=target,
                    target_weight=sizing_intent.target_weight,
                    strategy_id=self.name,
                    strategy_configuration_id=self.configuration_id,
                    strategy_parameters=parameters,
                    reason=reason,
                    indicator_values=tuple(
                        sorted(
                            (
                                IndicatorObservation("fast_sma", current_fast),
                                IndicatorObservation(
                                    "previous_fast_sma", previous_fast
                                ),
                                IndicatorObservation(
                                    "previous_slow_sma", previous_slow
                                ),
                                IndicatorObservation("slow_sma", current_slow),
                            ),
                            key=lambda observation: observation.name,
                        )
                    ),
                )
            )

        return StrategyOutput(
            self.name,
            self.configuration_id,
            MarketDataReference.from_dataset(dataset),
            tuple(decisions),
        )


def _parameter_snapshot(
    values: PrimitiveMapping,
) -> tuple[ParameterValue, ...]:
    snapshot: list[ParameterValue] = []
    for name, value in sorted(values.items()):
        if isinstance(value, (str, int, float, bool)) or value is None:
            snapshot.append(ParameterValue(name, value))
        else:
            raise InvalidStrategyOutputError(
                "strategy parameters must serialize to primitive scalar values"
            )
    return tuple(snapshot)
