"""Provider-neutral prediction-strategy contract."""

from typing import Protocol

from quantforge.configuration import PrimitiveMapping
from quantforge.data.models import MarketDataset
from quantforge.indicators.base import Indicator
from quantforge.prediction.models import PredictionStrategyOutput


class PredictionStrategyParameters(Protocol):
    """Typed prediction parameters with stable primitive serialization."""

    def to_primitive(self) -> PrimitiveMapping: ...


class PredictionStrategy(Protocol):
    """Generate causal direction guesses without calculating forward labels."""

    @property
    def name(self) -> str: ...

    @property
    def implementation_version(self) -> str: ...

    @property
    def parameters(self) -> PredictionStrategyParameters: ...

    @property
    def required_indicators(self) -> tuple[Indicator, ...]: ...

    @property
    def warm_up_observations(self) -> int: ...

    @property
    def configuration_id(self) -> str: ...

    def configuration(self) -> PrimitiveMapping: ...

    def generate(self, dataset: MarketDataset) -> PredictionStrategyOutput: ...
