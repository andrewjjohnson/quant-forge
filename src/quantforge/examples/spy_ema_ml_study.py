"""QF-69 frozen design: the SPY EMA 8/48 event population as an ML smoke study.

This module only *declares* the study. The rule kernel, decision semantics,
factory, analyzer, backend, feature schema and target are the unchanged QF-45
and QF-67 components; QF-45's own configuration, dates, artifacts and its
AGENTS.md selection exception are untouched. One candidate (8/48 with the fixed
completed daily EMA50) is executed; no EMA pair is compared and nothing is
selected by return.

Windows are inclusive XNYS sessions of the complete, authenticated 2025 cache,
chosen from coverage and existing exposure restrictions before any outcome or
model was inspected:

- warm-up/context: 2025-01-02 .. 2025-03-18 (61 2m and 50 daily bars);
- development: 2025-03-19 .. 2025-06-27, every post-warm-up session before
  QF-45's reserved holdout scope;
- QF-45's reserved holdout scope 2025-06-30 .. 2025-07-31 holds no QF-69
  decision or label (it remains reserved and unread);
- selection: 2025-08-01 .. 2025-08-29;
- walk-forward test (OOS): 2025-09-02 .. 2025-10-31;
- reserved final holdout: 2025-11-03 .. 2025-12-31.

The purge horizon is the only declared outcome's reach (30 minutes plus one 2m
alignment interval) with zero embargo; partitions are separated by at least a
weekend. See ``docs/conditional-ml-smoke-study.md``.
"""

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import ROUND_HALF_EVEN, Decimal, localcontext
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

from quantforge.configuration import (
    Primitive,
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
)
from quantforge.data import TimeframeBarSeries
from quantforge.examples.spy_ema import (
    DAILY,
    DECISION_END,
    DECISION_START,
    DECISION_TIMEZONE,
    TWO_MINUTES,
    EmaParameters,
    EmaSmokeRule,
    EmaStudyFactory,
    EmaWindowAnalyzer,
    backend_environment,
)
from quantforge.examples.spy_ema_inputs import SmokeInputs
from quantforge.examples.spy_ema_ml_dataset import (
    ema_event_feature_schema,
    ema_forward_return_target,
)
from quantforge.examples.spy_ema_plan import validation_window
from quantforge.indicators import (
    TALIB_INDICATOR_BACKEND,
    ExponentialMovingAverage,
    ExponentialMovingAverageParameters,
)
from quantforge.ml import DispositionPolicy, EventDataset
from quantforge.ml.modeling import ModelConfiguration
from quantforge.ml.study import ConditionalStudySpecification
from quantforge.optimization import CategoricalValues, ParameterSearchSpace
from quantforge.prediction import (
    IntradayForwardReturnOutcomeLabeler,
    PredictionDecisionSchedule,
    PredictionGridConfig,
    PredictionRankingConfig,
    intraday_forward_return_outcome,
)
from quantforge.validation import (
    ConfigurationReference,
    DatasetProvenance,
    FinalHoldout,
    IndicatorComponent,
    IndicatorProvenance,
    OutcomeProvenance,
    PartitionRole,
    PredictionMembershipSource,
    PurgePolicy,
    ResearchEnvironment,
    ResearchRuleProvenance,
    ResearchStudyType,
    TemporalOffset,
    TimestampBoundary,
    TrainingWindowMode,
    ValidationFold,
    ValidationPlan,
    ValidationWindow,
    select_prediction_context_observations,
)
from quantforge.walk_forward import (
    PredictionEvaluator,
    SelectionPolicy,
    WalkForwardConfig,
)

STUDY_NAME = "qf69_spy_ema_8_48_conditional_ml_smoke"
STUDY_VERSION = "1"
EMA_PAIR = "8/48"
TARGET_MINUTES = 30
# Inclusive XNYS sessions; fixed before any outcome or model was inspected.
WARMUP = ("2025-01-02", "2025-03-18")
DEVELOPMENT = ("2025-03-19", "2025-06-27")
SELECTION = ("2025-08-01", "2025-08-29")
TEST = ("2025-09-02", "2025-10-31")
HOLDOUT = ("2025-11-03", "2025-12-31")
# Existing same-symbol protected scope (QF-45 reserved holdout): no QF-69
# decision session or label may fall inside it.
QF45_RESERVED_SCOPE = ("2025-06-30", "2025-07-31")
# Plumbing floors frozen before fitting (see docs/conditional-ml-smoke-study.md).
MINIMUM_TRAINING_OBSERVATIONS = 30
MINIMUM_TRAINING_CLASS_OBSERVATIONS = 10
MINIMUM_EVALUATION_OBSERVATIONS = 10
SCORE_BUCKETS = 5


def model_configuration() -> ModelConfiguration:
    """QF-68's fixed L2 logistic regression; probability-only (no threshold)."""
    return ModelConfiguration(
        "qf69_conditional_logistic",
        "1",
        minimum_training_observations=MINIMUM_TRAINING_OBSERVATIONS,
        minimum_evaluation_observations=MINIMUM_EVALUATION_OBSERVATIONS,
        class_threshold=None,
        calibration_bins=SCORE_BUCKETS,
    )


def grid_configuration(output_root: Path) -> PredictionGridConfig:
    """One frozen candidate; QF-32 ranks it by event count, never by return.

    With a single candidate there is no comparison. QF-39 still freezes it
    through its ordinary selection contract, which requires at least one
    labeled selection event to rank; the count objective keeps returns out of
    that mechanical step.
    """
    return PredictionGridConfig(
        "QF-69 frozen 8/48 EMA event population",
        ParameterSearchSpace({"ema_pair": CategoricalValues((EMA_PAIR,))}),
        PredictionRankingConfig(
            "count", "always_up_matched", minimum_prediction_count=1
        ),
        output_root,
        maximum_combinations=1,
        window_schema_version="4",
    )


def prepare_ml_walk_forward(
    inputs: SmokeInputs, output_root: Path
) -> tuple[WalkForwardConfig, PredictionEvaluator]:
    """The QF-8 plan and QF-39 adapter of the frozen QF-69 strategy population."""
    factory = EmaStudyFactory(inputs.primary)
    adapter = PredictionEvaluator(
        dataset=inputs.dataset,
        series=(inputs.primary, inputs.daily),
        primary_timeframe=TWO_MINUTES,
        study_factory=factory,
        analyzer=EmaWindowAnalyzer(),
        indicator_backend=backend_environment(),
        grid_config=grid_configuration(output_root),
    )
    (candidate,) = adapter.universe.candidates
    study = factory.build(candidate.parameters.to_primitive())
    indicators = tuple(
        IndicatorProvenance.capture(
            cast(IndicatorComponent, indicator.indicator), requirement.timeframe
        )
        for requirement in getattr(
            study.strategy, "context_requirements"
        ).all_timeframes
        for indicator in requirement.indicators
    )
    outcome = intraday_forward_return_outcome(
        timedelta(minutes=TARGET_MINUTES), inputs.primary
    )
    labeler = cast(IntradayForwardReturnOutcomeLabeler, outcome.labeler)
    environment = ResearchEnvironment(
        ResearchStudyType.PREDICTION,
        DatasetProvenance.from_dataset_family(
            inputs.family,
            (
                inputs.primary.dataset_reference.dataset_id,
                inputs.daily.dataset_reference.dataset_id,
            ),
        ),
        (TWO_MINUTES, DAILY),
        ResearchRuleProvenance.capture_prediction(study.strategy),
        aggregation_policies=(
            ConfigurationReference.capture_aggregation_policy(
                inputs.family.aggregation_policy
            ),
        ),
        indicators=indicators,
        outcomes=(OutcomeProvenance.capture_timestamp(labeler),),
        prediction_dataset=DatasetProvenance.from_market_dataset(inputs.dataset),
    )
    fold = ValidationFold(
        "qf69-fold-0",
        validation_window("qf69-development-0", PartitionRole.DEVELOPMENT, DEVELOPMENT),
        validation_window("qf69-test-0", PartitionRole.WALK_FORWARD_TEST, TEST),
        validation_window("qf69-selection-0", PartitionRole.SELECTION, SELECTION),
    )
    schedule = PredictionDecisionSchedule(
        TWO_MINUTES,
        inputs.primary.bars[0].end_timestamp,
        inputs.primary.bars[-1].end_timestamp,
    )
    plan = ValidationPlan(
        "QF-69 frozen SPY EMA 8/48 conditional ML smoke study",
        environment,
        (fold,),
        FinalHoldout(
            validation_window(
                "qf69-final-holdout", PartitionRole.FINAL_HOLDOUT, HOLDOUT
            ),
            "Reserved; the QF-69 pre-holdout gate never consumes it",
        ),
        PurgePolicy(
            labeler.temporal_configuration.future_temporal_reach,
            TemporalOffset.duration(timedelta(0)),
        ),
        TrainingWindowMode.EXPANDING,
        prediction_membership=PredictionMembershipSource.capture(
            schedule, inputs.primary
        ),
    )
    return WalkForwardConfig(
        "QF-69 frozen 8/48 conditional ML smoke study",
        plan,
        selection_policy=SelectionPolicy.BEST_ELIGIBLE,
        continue_on_failure=False,
    ), adapter


@dataclass(frozen=True)
class StrategyDesign:
    """Human-readable frozen strategy, source, window and policy declarations."""

    inputs: SmokeInputs

    def to_primitive(self) -> PrimitiveMapping:
        source = self.inputs.source
        request = source.request
        return {
            "strategy": {
                "rule": "spy_midday_ema_smoke",
                "rule_kernel": "quantforge.examples.spy_ema.midday_bullish_cross",
                "parameters": {"ema_pair": EMA_PAIR, "daily_ema": 50},
                "definition": (
                    "UP only when previous fast EMA <= previous slow EMA, current "
                    "fast > current slow (completed 2m bars) and the latest "
                    "completed daily close > its daily EMA50"
                ),
                "decision_window": {
                    "start": DECISION_START.isoformat(),
                    "end": DECISION_END.isoformat(),
                    "timezone": DECISION_TIMEZONE,
                    "boundaries": "both_inclusive_on_decision_bar_end",
                },
                "indicator_backend": "talib_v1 (normalized, fixed)",
                "candidates": [EMA_PAIR],
                "selection": (
                    "single frozen candidate; QF-39 BEST_ELIGIBLE over one "
                    "candidate ranked by event count (no return-based selection)"
                ),
            },
            "source": {
                "symbol": request.symbol,
                "provider": source.metadata.provider_name,
                "canonical_1m_dataset_id": source.metadata.dataset_id,
                "raw_snapshot_ids": list(source.metadata.raw_snapshot_ids),
                "two_minute_dataset_id": (
                    self.inputs.primary.dataset_reference.dataset_id
                ),
                "daily_dataset_id": self.inputs.daily.dataset_reference.dataset_id,
                "timeframes": ["1m canonical RTH", "2m session-open", "daily"],
                "calendar": "XNYS",
                "timezone": "America/New_York (decisions); UTC (timestamps)",
                "timestamps": "2m bar end (decision) after the bar completed",
                "adjustment_policy": request.adjustment_basis.to_primitive(),
            },
            "population": {
                "disposition_policy": DispositionPolicy.ACCEPTED_ONLY.value,
                "observation_unit": "generated signal (trigger); no-trigger "
                "decisions are coverage receipts, never rows or negatives",
            },
            "windows": {
                "calendar": "inclusive XNYS sessions",
                "warm_up_context": list(WARMUP),
                "development": list(DEVELOPMENT),
                "selection": list(SELECTION),
                "walk_forward_test": list(TEST),
                "final_holdout": list(HOLDOUT),
                "excluded_protected_scope": {
                    "sessions": list(QF45_RESERVED_SCOPE),
                    "reason": (
                        "QF-45 reserved final holdout (same symbol); no QF-69 "
                        "decision or label is placed inside it"
                    ),
                },
                "warm_up": {"2m_bars": 61, "daily_bars": 50},
                "outcome_reach": "30 minutes plus one 2m alignment interval",
                "purge": "QF-8 label horizon equal to the outcome reach",
                "embargo": "0",
                "window_rationale": (
                    "development spans every session after warm-up before the "
                    "QF-45 protected scope; selection, test and holdout take the "
                    "next one, two and two calendar months of the complete 2025 "
                    "cache. Chosen from coverage only; no outcome was inspected."
                ),
            },
            "prior_exposure": {
                "development": (
                    "overlaps QF-45 development (2025-03-19..04-30, never "
                    "executed), selection (May; 8/48 fixed and comparison trials "
                    "inspected) and OOS (June; 12/60 OOS inspected). Reused as "
                    "training data only; not a fresh untouched period."
                ),
                "selection_test_holdout": (
                    "2025-08-01..12-31 has no recorded QuantForge strategy "
                    "evaluation before QF-69; 2025 market history is public, so "
                    "it is not blind in an absolute sense."
                ),
                "warm_up_context": (
                    "indicator warm-up for windows after the QF-45 scope reads "
                    "completed July 2025 bars as causal context only (no "
                    "decision, label or outcome there), as the QF-67/QF-72 "
                    "session-granular protected-scope contracts allow."
                ),
            },
        }


def design_primitive(inputs: SmokeInputs) -> PrimitiveMapping:
    """Strategy/source/window declarations plus fixed evaluation conventions."""
    return {
        **StrategyDesign(inputs).to_primitive(),
        "feature_schema_name": ema_event_feature_schema().name,
        "target": ema_forward_return_target().to_primitive(),
        "minimums": {
            "minimum_training_observations": MINIMUM_TRAINING_OBSERVATIONS,
            "minimum_training_class_observations": (
                MINIMUM_TRAINING_CLASS_OBSERVATIONS
            ),
            "minimum_evaluation_observations": MINIMUM_EVALUATION_OBSERVATIONS,
            "rationale": (
                "plumbing floors, not statistical adequacy: at least five fitting "
                "rows per standardized feature (6 features) and ten rows of each "
                "class so the fit is not determined by a handful of events; "
                "metrics on fewer than ten labeled rows are reported undefined. "
                "QF-45's one-observation exception does not apply."
            ),
        },
    }


# Exploratory QF-72 planning evidence, recorded before the specification froze:
# a rapid scan of the development window only, with no outcome requested.
PLANNING_EVIDENCE: PrimitiveMapping = {
    "method": (
        "QF-72 rapid scan (non-authoritative) of the development window only, "
        "outcomes=() and record_values=False; used only to confirm that the "
        "frozen floors are reachable before freezing the windows"
    ),
    "development_causal_trigger_count": 55,
    "development_triggers_by_month": {"2025-05": 28, "2025-06": 27},
    "retained_decisions": {
        "development_training": 13_650,
        "validation_selection": 4_095,
        "walk_forward_test": 8_580,
    },
    "purged_decisions": 0,
    "selection_or_test_outcomes_inspected": False,
}
_FEATURES = {
    "ema_fast_previous": ("fast", 2),
    "ema_slow_previous": ("slow", 2),
    "ema_fast": ("fast", 1),
    "ema_slow": ("slow", 1),
}


def event_row_audit(
    inputs: SmokeInputs, plan: ValidationPlan
) -> Callable[[EventDataset], PrimitiveMapping]:
    """Recompute every row's features and label from canonical bars.

    For each row, QF-8 selects the exact completed context bars of its role
    window at the decision timestamp. The audit asserts that every context bar
    ended at or before the decision (the latest 2m bar ends exactly at it; the
    latest daily bar is from an earlier session), recomputes both 2m EMAs and
    the daily EMA50 with the rule's own indicator definitions, and recomputes
    the 30-minute raw return from canonical 2m closes under QF-49's arithmetic
    policy. It bypasses QF-59/QF-63 prepared series, QF-42/QF-11 execution and
    QF-64/QF-67 persistence and extraction, so a shifted or future-derived
    value fails it.
    """
    rule = EmaSmokeRule(EmaParameters(8, 48))
    periods = {"fast": rule.parameters.fast, "slow": rule.parameters.slow}
    by_end = {bar.end_timestamp: bar for bar in inputs.primary.bars}
    new_york = ZoneInfo(DECISION_TIMEZONE)

    def ema(period: int, bars: tuple[Any, ...]) -> tuple[Decimal | None, ...]:
        indicator = ExponentialMovingAverage(
            ExponentialMovingAverageParameters(period),
            backend_id=TALIB_INDICATOR_BACKEND,
        )
        (field,) = indicator.calculate_bar_fields(bars)
        return field.values

    def context(
        series: TimeframeBarSeries, window: ValidationWindow, decision: datetime
    ) -> tuple[Any, ...]:
        selection = select_prediction_context_observations(
            plan, window, source=series, as_of=TimestampBoundary(decision)
        ).source_selection
        keys = {*selection.warm_up_context, *selection.study_observations}
        return tuple(
            bar for bar in series.bars if TimestampBoundary(bar.end_timestamp) in keys
        )

    def audit(dataset: EventDataset) -> PrimitiveMapping:
        failures: list[Primitive] = []
        labels: Counter[str] = Counter()
        fold = plan.folds[0]
        windows = {
            PartitionRole.DEVELOPMENT: fold.development,
            PartitionRole.SELECTION: fold.selection,
            PartitionRole.WALK_FORWARD_TEST: fold.test,
        }
        for index, row in enumerate(dataset.rows):
            decision = row.decision_timestamp
            window = windows.get(row.partition_role)
            values = dict(zip(dataset.feature_columns, row.features, strict=True))
            problems: list[str] = []
            if window is None:
                problems.append("row outside the audited fold roles")
            else:
                clock = decision.astimezone(new_york).time()
                if not DECISION_START <= clock <= DECISION_END:
                    problems.append("decision outside 11:00-14:00 New York")
                two_minute = context(inputs.primary, window, decision)
                if not two_minute or two_minute[-1].end_timestamp != decision:
                    problems.append("latest 2m context bar does not end at decision")
                if any(bar.end_timestamp > decision for bar in two_minute):
                    problems.append("2m context includes an incomplete bar")
                for alias, period in periods.items():
                    series = ema(period, two_minute)
                    for name, (feature_alias, back) in _FEATURES.items():
                        if feature_alias == alias and (
                            len(series) < back
                            or series[-back] != Decimal(cast(str, values[name]))
                        ):
                            problems.append(f"{name} differs from recomputation")
                daily = context(inputs.daily, window, decision)
                latest = max(
                    bar.end_timestamp
                    for bar in inputs.daily.bars
                    if bar.end_timestamp <= decision
                )
                if (
                    not daily
                    or any(bar.end_timestamp > decision for bar in daily)
                    or daily[-1].end_timestamp != latest
                ):
                    problems.append("daily context is not the latest completed bar")
                elif daily[-1].close != Decimal(cast(str, values["daily_close"])) or (
                    ema(50, daily)[-1] != Decimal(cast(str, values["daily_ema50"]))
                ):
                    problems.append("daily inputs differ from recomputation")
                reference = by_end.get(decision)
                future = by_end.get(decision + timedelta(minutes=TARGET_MINUTES))
                if row.label.value is None:
                    labels[f"unavailable:{row.label.status}"] += 1
                elif (
                    reference is None
                    or future is None
                    or future.end_timestamp.astimezone(new_york).date()
                    != reference.end_timestamp.astimezone(new_york).date()
                ):
                    problems.append("available label without a same-session endpoint")
                else:
                    with localcontext() as arithmetic:
                        arithmetic.prec = 34
                        arithmetic.rounding = ROUND_HALF_EVEN
                        expected = future.close / reference.close - Decimal(1)
                    if Decimal(cast(str, row.label.source_value)) != expected:
                        problems.append("30m raw return differs from canonical bars")
                    labels["available"] += 1
            if problems:
                failures.append(
                    {
                        "row_index": index,
                        "decision_timestamp": decision.isoformat(),
                        "problems": cast(list[Primitive], problems),
                    }
                )
        return {
            "status": "failed" if failures else "passed",
            "rows_checked": dataset.row_count,
            "labels_checked": dict(sorted(labels.items())),
            "failures": failures[:20],
            "failure_count": len(failures),
            "method": (
                "QF-8 completed context bars at each decision; every bar ends at "
                "or before it; EMAs and daily EMA50 recomputed with the rule's "
                "indicator definitions; 30m return recomputed from canonical "
                "2m closes (34-digit half-even)"
            ),
        }

    return audit


def specification(
    inputs: SmokeInputs,
    config: WalkForwardConfig,
    adapter: PredictionEvaluator,
    study_id: str,
) -> ConditionalStudySpecification:
    """The complete frozen QF-69 specification, persisted before research."""
    (candidate,) = adapter.universe.candidates
    return ConditionalStudySpecification(
        name=STUDY_NAME,
        version=STUDY_VERSION,
        plan=config.plan,
        fold_id=config.plan.folds[0].fold_id,
        qf39_study_id=study_id,
        candidate=candidate,
        feature_schema=ema_event_feature_schema(),
        target=ema_forward_return_target(),
        disposition_policy=DispositionPolicy.ACCEPTED_ONLY,
        model=model_configuration(),
        minimum_training_class_observations=MINIMUM_TRAINING_CLASS_OBSERVATIONS,
        score_buckets=SCORE_BUCKETS,
        design=PrimitiveMappingSnapshot.capture(
            {**design_primitive(inputs), "planning_evidence": PLANNING_EVIDENCE}
        ),
    )


__all__ = [
    "DEVELOPMENT",
    "EMA_PAIR",
    "HOLDOUT",
    "MINIMUM_EVALUATION_OBSERVATIONS",
    "MINIMUM_TRAINING_CLASS_OBSERVATIONS",
    "MINIMUM_TRAINING_OBSERVATIONS",
    "PLANNING_EVIDENCE",
    "QF45_RESERVED_SCOPE",
    "SCORE_BUCKETS",
    "SELECTION",
    "STUDY_NAME",
    "STUDY_VERSION",
    "TEST",
    "WARMUP",
    "StrategyDesign",
    "design_primitive",
    "event_row_audit",
    "grid_configuration",
    "model_configuration",
    "prepare_ml_walk_forward",
    "specification",
]
