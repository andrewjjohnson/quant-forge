"""QF-45 fixed bullish EMA hypothesis; no acquisition or outcome mathematics."""

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

from quantforge.configuration import (
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.data import FeedScope, TimeframeBarSeries
from quantforge.indicators import (
    EXPONENTIAL_MOVING_AVERAGE_OUTPUT,
    TALIB_INDICATOR_BACKEND,
    ExponentialMovingAverage,
    ExponentialMovingAverageParameters,
    Indicator,
)
from quantforge.optimization import (
    CategoricalValues,
    ParameterSearchSpace,
)
from quantforge.prediction import (
    MultiTimeframeFeatureRequest,
    PredictionContextRequirements,
    PredictionDirection,
    PredictionGridConfig,
    PredictionIndicatorBackendEnvironment,
    PredictionIndicatorRequirement,
    PredictionRankingConfig,
    PredictionRuleContext,
    PredictionStudy,
    PredictionTimeframeRequirement,
    SchemaField,
    SchemaFieldCategory,
    SignalDisposition,
    SignalFeatureCandidate,
    SignalFeatureCandidateOutput,
    SignalFeatureValue,
    intraday_excursion_outcome,
    intraday_forward_return_outcome,
    intraday_target_stop_outcome,
)
from quantforge.prediction.grid import PredictionTrialAnalysis
from quantforge.prediction.window import PredictionWindowResult
from quantforge.prediction.window_encoding import mapping
from quantforge.prediction.window_reader import PredictionWindowReader
from quantforge.rapid.models import (
    RapidBarInput,
    RapidDecisionWindow,
    RapidIndicatorInput,
    RapidRuleSpecification,
    RapidValue,
)
from quantforge.timeframes import IntradayInterval, SessionInterval, Timeframe

ONE_MINUTE = Timeframe.us_equity(IntradayInterval(timedelta(minutes=1)))
TWO_MINUTES = Timeframe.us_equity(IntradayInterval(timedelta(minutes=2)))
DAILY = Timeframe.us_equity(SessionInterval(1))
FORWARD_MINUTES = (10, 30, 60, 120)
EMA_PAIRS = (("8/40", (8, 40)), ("8/48", (8, 48)), ("12/60", (12, 60)))
DECISION_TIMEZONE = "America/New_York"
DECISION_START = time(11)
DECISION_END = time(14)


def midday_bullish_cross(
    decision_clock: time,
    previous_fast: Decimal | None,
    previous_slow: Decimal | None,
    current_fast: Decimal | None,
    current_slow: Decimal | None,
    daily_close: Decimal | None,
    daily_ema50: Decimal | None,
) -> bool:
    """The complete QF-45 rule on causal scalars; both execution paths call it.

    ``decision_clock`` is the decision bar end in New York time. EMA values are
    the previous and current completed 2m values and the latest completed daily
    close/EMA50; ``None`` means unavailable and never triggers.
    """
    return (
        DECISION_START <= decision_clock <= DECISION_END
        and previous_fast is not None
        and previous_slow is not None
        and current_fast is not None
        and current_slow is not None
        and daily_close is not None
        and daily_ema50 is not None
        and previous_fast <= previous_slow
        and current_fast > current_slow
        and daily_close > daily_ema50
    )


def _rapid_decision(
    decision_clock: time, values: tuple[RapidValue, ...]
) -> PredictionDirection | None:
    """QF-72 rapid kernel: the unchanged predicate on declared input order."""
    previous_fast, previous_slow, current_fast, current_slow, close, ema50 = values
    if midday_bullish_cross(
        decision_clock,
        previous_fast,
        previous_slow,
        current_fast,
        current_slow,
        close,
        ema50,
    ):
        return PredictionDirection.UP
    return None


@dataclass(frozen=True)
class EmaParameters:
    """The only searched parameters; all other hypothesis choices stay fixed."""

    fast: int = 8
    slow: int = 48

    def __post_init__(self) -> None:
        if type(self.fast) is not int or type(self.slow) is not int:
            raise ValueError("EMA periods must be integers")
        if not 1 < self.fast < self.slow:
            raise ValueError("EMA periods require 1 < fast < slow")

    def to_primitive(self) -> PrimitiveMapping:
        return {"fast": self.fast, "slow": self.slow, "daily": 50}

    @classmethod
    def from_primitive(cls, value: PrimitiveMapping) -> "EmaParameters":
        """Rebuild promoted parameters exactly; the daily period is fixed at 50."""
        if set(value) != {"fast", "slow", "daily"} or value["daily"] != 50:
            raise ValueError("EMA parameters require fast, slow and daily=50")
        return cls(cast(int, value["fast"]), cast(int, value["slow"]))


class EmaSmokeRule:
    """Emit only an UP transition with completed daily context in [11:00, 14:00]."""

    name = "spy_midday_ema_smoke"
    implementation_version = "2"

    def __init__(self, parameters: EmaParameters = EmaParameters()) -> None:
        self.parameters = parameters
        self.context_requirements = PredictionContextRequirements(
            PredictionTimeframeRequirement(
                TWO_MINUTES,
                FeedScope.consolidated(),
                tuple(
                    self._ema(alias, period)
                    for alias, period in (
                        ("fast", parameters.fast),
                        ("slow", parameters.slow),
                    )
                ),
            ),
            (
                PredictionTimeframeRequirement(
                    DAILY, FeedScope.consolidated(), (self._ema("trend", 50),)
                ),
            ),
        )

    @staticmethod
    def _ema(alias: str, period: int) -> PredictionIndicatorRequirement:
        return PredictionIndicatorRequirement(
            alias,
            ExponentialMovingAverage(
                ExponentialMovingAverageParameters(period),
                backend_id=TALIB_INDICATOR_BACKEND,
            ),
        )

    @property
    def warm_up_observations(self) -> int:
        return self.parameters.slow + 1

    @property
    def required_indicators(self) -> tuple[Indicator, ...]:
        return tuple(
            cast(Indicator, item.indicator)
            for requirement in self.context_requirements.all_timeframes
            for item in requirement.indicators
        )

    @property
    def strategy_feature_definitions(self) -> tuple[SchemaField, ...]:
        return tuple(
            SchemaField(
                name,
                SchemaFieldCategory.CONTEMPORANEOUS_FEATURE,
                "decimal",
                "price",
                False,
                source,
                "completed bars at decision timestamp",
            )
            for name, source in sorted(
                (
                    ("previous_fast", "previous normalized talib_v1 fast EMA"),
                    ("previous_slow", "previous normalized talib_v1 slow EMA"),
                    ("current_fast", "current normalized talib_v1 fast EMA"),
                    ("current_slow", "current normalized talib_v1 slow EMA"),
                    ("daily_close", "latest permitted completed daily close"),
                    ("daily_ema50", "latest normalized talib_v1 daily EMA50"),
                )
            )
        )

    def configuration(self) -> PrimitiveMapping:
        return {
            "component_name": self.name,
            "component_type": "prediction_strategy",
            "implementation_version": self.implementation_version,
            "parameters": self.parameters.to_primitive(),
            "context_requirements": self.context_requirements.to_primitive(),
            "warm_up_observations": self.warm_up_observations,
            "rule": (
                "previous_fast<=previous_slow AND fast>slow AND daily_close>daily_ema50"
            ),
            "decision_interval": {
                "start": "11:00:00",
                "end": "14:00:00",
                "timezone": "America/New_York",
                "boundaries": "both_inclusive",
            },
        }

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    @property
    def multi_timeframe_feature_requests(
        self,
    ) -> tuple[MultiTimeframeFeatureRequest, ...]:
        return tuple(
            MultiTimeframeFeatureRequest(
                name, timeframe, alias, EXPONENTIAL_MOVING_AVERAGE_OUTPUT, "price"
            )
            for name, timeframe, alias in (
                ("two_minute", TWO_MINUTES, "fast"),
                ("two_minute", TWO_MINUTES, "slow"),
                ("daily", DAILY, "trend"),
            )
        )

    def generate_with_context(
        self, context: PredictionRuleContext
    ) -> SignalFeatureCandidateOutput:
        signals: tuple[SignalFeatureCandidate, ...] = ()
        clock = context.as_of.astimezone(ZoneInfo(DECISION_TIMEZONE)).time()
        fast = context.indicator_for(TWO_MINUTES, "fast").values_for(
            EXPONENTIAL_MOVING_AVERAGE_OUTPUT
        )
        slow = context.indicator_for(TWO_MINUTES, "slow").values_for(
            EXPONENTIAL_MOVING_AVERAGE_OUTPUT
        )
        trend = context.indicator_for(DAILY, "trend").values_for(
            EXPONENTIAL_MOVING_AVERAGE_OUTPUT
        )
        daily = context.latest_bar_for(DAILY)
        if (
            len(fast) >= 2
            and len(slow) >= 2
            and trend
            and midday_bullish_cross(
                clock, fast[-2], slow[-2], fast[-1], slow[-1], daily.close, trend[-1]
            )
        ):
            signals = (
                self.candidate(
                    context.symbol,
                    context.decision_session,
                    context.as_of,
                    (fast[-2], slow[-2], fast[-1], slow[-1], daily.close, trend[-1]),
                ),
            )
        return SignalFeatureCandidateOutput(
            self.name, self.configuration_id, context.prediction_dataset_id, signals
        )

    def candidate(
        self,
        symbol: str,
        signal_session: date,
        decision_timestamp: datetime,
        values: tuple[RapidValue, ...],
    ) -> SignalFeatureCandidate:
        """The accepted UP candidate for causal values in rapid input order."""
        previous_fast, previous_slow, current_fast, current_slow, close, ema50 = values
        return SignalFeatureCandidate(
            symbol=symbol,
            signal_session=signal_session,
            strategy_id=self.name,
            strategy_implementation_version=self.implementation_version,
            strategy_configuration_id=self.configuration_id,
            source_rule_id=self.name,
            source_rule_implementation_version=self.implementation_version,
            source_rule_configuration_id=self.configuration_id,
            strategy_parameters=PrimitiveMappingSnapshot.capture(
                self.parameters.to_primitive()
            ),
            disposition=SignalDisposition.ACCEPTED,
            reason_codes=("midday_bullish_cross",),
            explanation=(
                "Completed two-minute bullish cross above completed daily EMA50 trend"
            ),
            direction=PredictionDirection.UP,
            selected_rule_reason="midday_bullish_cross",
            matched_rule_reasons=("midday_bullish_cross",),
            strategy_features=tuple(
                SignalFeatureValue(name, value)
                for name, value in sorted(
                    (
                        ("previous_fast", previous_fast),
                        ("previous_slow", previous_slow),
                        ("current_fast", current_fast),
                        ("current_slow", current_slow),
                        ("daily_close", close),
                        ("daily_ema50", ema50),
                    )
                )
            ),
            decision_timestamp=decision_timestamp,
        )

    def rapid_specification(self) -> RapidRuleSpecification:
        """QF-72 capability: the same kernel and candidate on prepared values."""
        ema = EXPONENTIAL_MOVING_AVERAGE_OUTPUT
        return RapidRuleSpecification(
            inputs=(
                RapidIndicatorInput("previous_fast", TWO_MINUTES, "fast", ema, lag=1),
                RapidIndicatorInput("previous_slow", TWO_MINUTES, "slow", ema, lag=1),
                RapidIndicatorInput("current_fast", TWO_MINUTES, "fast", ema),
                RapidIndicatorInput("current_slow", TWO_MINUTES, "slow", ema),
                RapidBarInput("daily_close", DAILY, "close"),
                RapidIndicatorInput("daily_ema50", DAILY, "trend", ema),
            ),
            decide=_rapid_decision,
            candidate=self.candidate,
            clock_timezone=DECISION_TIMEZONE,
            decision_window=RapidDecisionWindow(DECISION_START, DECISION_END),
        )


def configured_outcomes(source: TimeframeBarSeries) -> tuple[Any, ...]:
    """Use QF-49 endpoints and QF-47 complete paths on the canonical 2m source."""
    return (
        *(
            intraday_forward_return_outcome(timedelta(minutes=m), source)
            for m in FORWARD_MINUTES
        ),
        intraday_excursion_outcome(timedelta(minutes=60), source),
        intraday_target_stop_outcome(
            timedelta(minutes=60), source, Decimal("0.003"), Decimal("0.002")
        ),
    )


@dataclass(frozen=True)
class EmaStudyFactory:
    source: TimeframeBarSeries
    name = "spy_ema_smoke_factory"
    version = "2"
    parameter_order = ("ema_pair",)
    required_parameter_names = frozenset(parameter_order)

    def configuration(self) -> PrimitiveMapping:
        return {
            "name": self.name,
            "version": self.version,
            "ema_pairs": {name: list(pair) for name, pair in EMA_PAIRS},
            "daily_period": 50,
            "outcome_minutes": 30,
            "source": self.source.dataset_reference.to_primitive(),
        }

    def build(self, parameters: PrimitiveMapping) -> PredictionStudy[Any, Any, Any]:
        rule = EmaSmokeRule(
            EmaParameters(*dict(EMA_PAIRS)[cast(str, parameters["ema_pair"])])
        )
        outcome = intraday_forward_return_outcome(timedelta(minutes=30), self.source)
        return PredictionStudy[Any, Any, Any].create(
            rule, outcome.labeler, outcome.evaluator, outcome_source=self.source
        )


def backend_environment() -> PredictionIndicatorBackendEnvironment:
    identity = ExponentialMovingAverage(
        ExponentialMovingAverageParameters(50), backend_id=TALIB_INDICATOR_BACKEND
    ).backend_identity
    return PredictionIndicatorBackendEnvironment.create(
        backend_id=identity.backend_id,
        library_name=identity.library_name,
        library_version=identity.library_version,
        contract_version=identity.contract_version,
        runtime_library_name=identity.runtime_library_name,
        runtime_library_version=identity.runtime_library_version,
        configuration={"selection_policy": "fixed_for_ordinary_grid"},
    )


def grid_configuration(
    output_root: Path, *, fixed: bool = False
) -> PredictionGridConfig:
    return PredictionGridConfig(
        "QF-45 fixed EMA" if fixed else "QF-45 three EMA configurations",
        ParameterSearchSpace(
            {
                "ema_pair": CategoricalValues(
                    ("8/48",) if fixed else tuple(name for name, _ in EMA_PAIRS)
                ),
            }
        ),
        # QF-45-only exception approved in AGENTS.md: this platform smoke test
        # exercises comparison, deterministic ranking, frozen selection, OOS,
        # holdout handling, manifests, and reporting, not a deployable policy.
        # Exactly three EMA pairs, study windows, and mean 30m raw return were
        # fixed before inspecting outcomes. One available observation suffices;
        # a single event may select the configuration. The matched always-UP
        # baseline is descriptive only, not an independent quality/risk gate.
        # Selection does not establish robustness, profitability, or production
        # suitability. Do not add sample-size/risk/stable-region/other eligibility
        # gates or switch to FIRST_STABLE; preserve the frozen science and schemas.
        # Ordinary QuantForge research still prohibits return-only optimization.
        PredictionRankingConfig(
            "mean_30m_return", "always_up_matched", minimum_prediction_count=1
        ),
        output_root,
        maximum_combinations=3,
        window_schema_version="2",
    )


class EmaWindowAnalyzer:
    """Describe existing 30m labels; QF-32 owns ranking and selection."""

    name = "ema_smoke_30m_return"
    version = "1"

    def configuration(self) -> PrimitiveMapping:
        return {
            "name": self.name,
            "version": self.version,
            "metric": "mean_30m_return",
            "directional_interpretation": "raw_return>0",
            "baseline": "always_up_matched",
            "baseline_scope": "same UP candidates; identical by construction",
            "decimal_precision": 34,
        }

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration())

    def analyze_window(
        self, result: PredictionWindowResult[Any, Any, Any]
    ) -> PredictionTrialAnalysis:
        observations = [
            (decision.decision_timestamp, row.evaluation.values.raw_return)
            for decision in result.decisions
            for row in decision.result.rows
            if row.evaluation.values.raw_return is not None
        ]
        return self._analyze(observations)

    def analyze_compact_window(
        self, reader: PredictionWindowReader
    ) -> PredictionTrialAnalysis:
        """Exhaust compact records; retain only available timestamp/return scalars."""
        observations: list[tuple[datetime, Decimal]] = []
        for receipt in reader.iterate_decision_receipts():
            if receipt.decision is None:
                continue  # Sparse no-prediction receipts carry no rows.
            record = receipt.decision.to_primitive()
            timestamp = datetime.fromisoformat(cast(str, record["decision_timestamp"]))
            rows = cast(
                list[PrimitiveMapping], mapping(record["prediction_study"])["rows"]
            )
            for row in rows:
                value = mapping(mapping(row["evaluation"])["values"])["raw_return"]
                if value is not None:
                    observations.append((timestamp, Decimal(cast(str, value))))
        return self._analyze(observations)

    @staticmethod
    def _analyze(
        observations: list[tuple[datetime, Decimal]],
    ) -> PredictionTrialAnalysis:
        from decimal import localcontext

        def summary(values: list[Decimal]) -> PrimitiveMapping:
            with localcontext() as arithmetic:
                arithmetic.prec = 34
                return {
                    "count": len(values),
                    "mean_30m_return": str(sum(values, Decimal(0)) / len(values))
                    if values
                    else None,
                    "positive_30m_count": sum(v > 0 for v in values),
                    "positive_30m_fraction": str(
                        Decimal(sum(v > 0 for v in values)) / len(values)
                    )
                    if values
                    else None,
                }

        # QF-32 represents undefined objectives by absence, not a null numeric
        # metric or an invented zero. Empty populations stay unrankable.
        metrics: PrimitiveMapping = {
            key: value
            for key, value in summary([v for _, v in observations]).items()
            if value is not None
        }
        return PredictionTrialAnalysis.create(
            prediction_count=len(observations),
            metrics=metrics,
            period_comparisons=tuple(
                {
                    "period": period,
                    **summary(
                        [v for t, v in observations if t.strftime("%Y-%m") == period]
                    ),
                }
                for period in sorted({t.strftime("%Y-%m") for t, _ in observations})
            ),
            weekday_comparisons=tuple(
                {
                    "weekday": weekday,
                    **summary([v for t, v in observations if t.weekday() == weekday]),
                }
                for weekday in sorted({t.weekday() for t, _ in observations})
            ),
            matched_baseline_comparisons=(
                {"baseline_name": "always_up_matched", **metrics},
            )
            if observations
            else (),
        )
