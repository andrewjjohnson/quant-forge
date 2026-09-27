"""Performance-independent 2025 session splits and generic QF-8/QF-39 wiring."""

from datetime import date, timedelta
from pathlib import Path
from typing import cast

from quantforge.configuration import PrimitiveMapping
from quantforge.examples.spy_ema import (
    DAILY,
    TWO_MINUTES,
    EmaSmokeRule,
    EmaStudyFactory,
    EmaWindowAnalyzer,
    backend_environment,
    configured_outcomes,
    grid_configuration,
)
from quantforge.examples.spy_ema_inputs import SmokeInputs
from quantforge.optimization import TrialStatus
from quantforge.prediction import PredictionDecisionSchedule
from quantforge.timeframes import resolve_exchange_session
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
    TimeframeWarmUpRequirement,
    TimestampBoundary,
    TrainingWindowMode,
    ValidationFold,
    ValidationInterval,
    ValidationPlan,
    ValidationWindow,
)
from quantforge.walk_forward import (
    PredictionEvaluator,
    SelectionEvidence,
    WalkForwardConfig,
    WalkForwardError,
)

# Inclusive actual XNYS session dates, chosen before evaluating any returns.
WARMUP = ("2025-01-02", "2025-03-18")
SPLITS = (
    (
        ("2025-03-19", "2025-04-30"),
        ("2025-05-01", "2025-05-30"),
        ("2025-06-02", "2025-06-27"),
    ),
)
HOLDOUT = ("2025-06-30", "2025-07-31")


class SmokePredictionEvaluator(PredictionEvaluator):
    """Keep QF-45 comparison completion separate from scientific eligibility."""

    def configuration(self) -> PrimitiveMapping:
        # Earlier unchecked folds must not satisfy this adapter's identity.
        return {
            **super().configuration(),
            "qf45_trial_completion_policy": "all_candidates_succeeded_v1",
        }

    def select(
        self, config: WalkForwardConfig, fold_index: int, output_root: Path
    ) -> SelectionEvidence:
        evidence = super().select(config, fold_index, output_root)
        # QF-32 verifies/preserves trial records and permits failed neighbors.
        # QF-45 needs the whole comparison to finish successfully before QF-39
        # freezes selection or evaluates OOS. Successful unrankable trials count.
        statuses = cast(
            list[PrimitiveMapping], evidence.evidence.to_primitive()["trial_statuses"]
        )
        expected = {
            candidate.combination_id: TrialStatus.SUCCEEDED.value
            for candidate in self.universe.candidates
        }
        actual = {
            cast(str, trial["combination_id"]): trial["status"] for trial in statuses
        }
        if len(statuses) != len(expected) or actual != expected:
            raise WalkForwardError(
                "QF-45 requires every comparison trial to succeed before selection "
                "is frozen; preserve failure evidence"
            )
        return evidence


def validation_window(
    name: str, role: PartitionRole, dates: tuple[str, str]
) -> ValidationWindow:
    policy = TWO_MINUTES.session_policy
    first = resolve_exchange_session(date.fromisoformat(dates[0]), policy)
    last = resolve_exchange_session(date.fromisoformat(dates[1]), policy)
    return ValidationWindow(
        name,
        role,
        ValidationInterval(
            TimestampBoundary(first.open_timestamp + timedelta(minutes=2)),
            TimestampBoundary(last.close_timestamp),
        ),
        warm_up_by_timeframe=(
            TimeframeWarmUpRequirement(TWO_MINUTES, 61),
            TimeframeWarmUpRequirement(DAILY, 50),
        ),
    )


def prepare_walk_forward(
    inputs: SmokeInputs, output_root: Path, *, fixed: bool = False
) -> tuple[WalkForwardConfig, PredictionEvaluator]:
    """Declare the complete label reach before any selection or evaluation."""
    factory = EmaStudyFactory(inputs.primary)
    adapter = SmokePredictionEvaluator(
        dataset=inputs.dataset,
        series=(inputs.primary, inputs.daily),
        primary_timeframe=TWO_MINUTES,
        study_factory=factory,
        analyzer=EmaWindowAnalyzer(),
        indicator_backend=backend_environment(),
        grid_config=grid_configuration(output_root, fixed=fixed),
    )
    studies = [
        factory.build(candidate.parameters.to_primitive())
        for candidate in adapter.universe.candidates
    ]
    indicators = {
        (
            requirement.timeframe.configuration_id,
            indicator.configuration_id,
        ): IndicatorProvenance.capture(
            cast(IndicatorComponent, indicator.indicator), requirement.timeframe
        )
        for study in studies
        for requirement in getattr(
            study.strategy, "context_requirements"
        ).all_timeframes
        for indicator in requirement.indicators
    }
    outcomes = configured_outcomes(inputs.primary)
    # Target/stop and excursions intentionally share one generic path labeler.
    provenance = {
        item.labeler.configuration_id: OutcomeProvenance.capture_timestamp(item.labeler)
        for item in outcomes
    }
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
        ResearchRuleProvenance.capture_prediction(studies[0].strategy),
        aggregation_policies=(
            ConfigurationReference.capture_aggregation_policy(
                inputs.family.aggregation_policy
            ),
        ),
        indicators=tuple(indicators.values()),
        outcomes=tuple(provenance.values()),
        prediction_dataset=DatasetProvenance.from_market_dataset(inputs.dataset),
    )
    folds = tuple(
        ValidationFold(
            f"fold-{index}",
            validation_window(
                f"development-{index}", PartitionRole.DEVELOPMENT, development
            ),
            validation_window(f"test-{index}", PartitionRole.WALK_FORWARD_TEST, test),
            validation_window(f"selection-{index}", PartitionRole.SELECTION, selection),
        )
        for index, (development, selection, test) in enumerate(SPLITS, 1)
    )
    schedule = PredictionDecisionSchedule(
        TWO_MINUTES,
        inputs.primary.bars[0].end_timestamp,
        inputs.primary.bars[-1].end_timestamp,
    )
    horizon = outcomes[3].labeler.temporal_configuration.future_temporal_reach
    plan = ValidationPlan(
        "QF-45 shortened 2025 single-fold EMA smoke study",
        environment,
        folds,
        FinalHoldout(
            validation_window("final-holdout", PartitionRole.FINAL_HOLDOUT, HOLDOUT),
            "Reserved for one explicit post-inspection consumption; no retuning",
        ),
        PurgePolicy(horizon, TemporalOffset.duration(timedelta(0))),
        TrainingWindowMode.EXPANDING,
        prediction_membership=PredictionMembershipSource.capture(
            schedule, inputs.primary
        ),
    )
    return WalkForwardConfig(
        "QF-45 fixed" if fixed else "QF-45 three-config comparison",
        plan,
        continue_on_failure=False,
    ), adapter


def verify_configuration(inputs: SmokeInputs, output_root: Path) -> dict[str, object]:
    """Check calendar, warm-up and maximum reach without evaluating any rule."""
    from quantforge.indicators import EXPONENTIAL_MOVING_AVERAGE_OUTPUT
    from quantforge.prediction.context import build_prediction_rule_context
    from quantforge.walk_forward.partitions import partition
    from quantforge.walk_forward.prediction import (
        _PermittedContextProvider,  # pyright: ignore[reportPrivateUsage]
    )

    config, adapter = prepare_walk_forward(inputs, output_root)
    adapter.validate(config.plan)
    counts: dict[str, object] = {}
    for role in (
        PartitionRole.DEVELOPMENT,
        PartitionRole.SELECTION,
        PartitionRole.WALK_FORWARD_TEST,
    ):
        permitted = partition(
            inputs.dataset, config.plan, 0, role, minimum_observations=1
        )
        schedule = PredictionDecisionSchedule(
            TWO_MINUTES,
            permitted.decision_timestamps[0],
            permitted.decision_timestamps[-1],
        )
        if schedule.decision_timestamps != permitted.decision_timestamps:
            raise ValueError("QF-42 schedule differs from QF-8 retained membership")
        provider = _PermittedContextProvider(
            config.plan, permitted, (inputs.primary, inputs.daily), schedule
        )
        requirements = cast(
            EmaSmokeRule, adapter.factory.build({"ema_pair": "12/60"}).strategy
        ).context_requirements
        context = provider.get_context_at(
            requirements, as_of=schedule.decision_timestamps[0]
        )
        normalized = build_prediction_rule_context(
            requirements,
            context,
            prediction_dataset_id=permitted.dataset.metadata.dataset_id,
            symbol="SPY",
            prediction_adjustment_basis=inputs.source.request.adjustment_basis,
        )
        for timeframe, alias in ((TWO_MINUTES, "slow"), (DAILY, "trend")):
            if (
                normalized.indicator_for(timeframe, alias).values_for(
                    EXPONENTIAL_MOVING_AVERAGE_OUTPUT
                )[-1]
                is None
            ):
                raise ValueError("insufficient declared EMA warm-up")
        counts[role.value] = {
            "decisions": len(schedule.decision_timestamps),
            "purged": len(permitted.purge.purged),
            "first": schedule.decision_timestamps[0].isoformat(),
            "last": schedule.decision_timestamps[-1].isoformat(),
            "latest_daily_at_first_decision": normalized.latest_bar_for(
                DAILY
            ).end_timestamp.isoformat(),
        }
    return {
        "plan_id": config.plan.plan_id,
        "universe_id": adapter.universe.universe_id,
        "configurations": [
            c.parameters.to_primitive() for c in adapter.universe.candidates
        ],
        "windows": counts,
        "maximum_reach_minutes": 122,
        "embargo_minutes": 0,
        "calendar_correction": None,
        "holdout_projected_or_evaluated": False,
    }
