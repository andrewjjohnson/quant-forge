from dataclasses import replace
from datetime import date
from decimal import Decimal

from quantforge.configuration import PrimitiveMappingSnapshot
from quantforge.prediction import (
    DirectionalExcursionEvaluator,
    ExcursionOutcomeLabeler,
    ForwardReturnOutcomeLabeler,
    PredictionDirection,
    PredictionOutcome,
    SameSessionConflictPolicy,
    SignalDisposition,
    SignalFeatureCandidate,
    TargetStopEvaluator,
    TargetStopLabel,
    TargetStopOutcomeLabeler,
)

from ..helpers import make_dataset


def _candidate(
    signal_session: date, direction: PredictionDirection | None
) -> SignalFeatureCandidate:
    disposition = (
        SignalDisposition.ACCEPTED
        if direction is not None
        else SignalDisposition.REJECTED
    )
    return SignalFeatureCandidate(
        symbol="SPY",
        signal_session=signal_session,
        strategy_id="candidate_rule",
        strategy_implementation_version="1",
        strategy_configuration_id="candidate-config",
        source_rule_id="source_rule",
        source_rule_implementation_version="1",
        source_rule_configuration_id="source-config",
        strategy_parameters=PrimitiveMappingSnapshot.capture({"period": 2}),
        disposition=disposition,
        reason_codes=(disposition.value,),
        explanation=None,
        direction=direction,
        selected_rule_reason=None if direction is None else "accepted",
        matched_rule_reasons=() if direction is None else ("accepted",),
        strategy_features=(),
    )


def test_forward_returns_use_exact_exchange_session_indexes() -> None:
    dataset = make_dataset(("100", "102", "101", "104", "105"))
    labeler = ForwardReturnOutcomeLabeler(2)

    label = labeler.label(dataset, dataset.bars[1].session_date)

    assert label is not None
    assert label.outcome_session == dataset.bars[3].session_date
    assert label.values.reference_price == Decimal(102)
    assert label.values.outcome_price == Decimal(104)
    assert label.values.raw_return == Decimal("0.019607843137254901960784313725490")
    assert labeler.label(dataset, dataset.bars[-2].session_date) is None


def test_forward_return_skips_holiday_and_weekend_as_one_session() -> None:
    dataset = make_dataset(("100", "102", "101", "104"))

    label = ForwardReturnOutcomeLabeler(1).label(dataset, date(2024, 7, 3))

    assert label is not None
    assert label.outcome_session == date(2024, 7, 5)
    assert label.values.raw_return == Decimal("0.029702970297029702970297029702970")


def test_directional_mfe_and_mae_use_only_the_declared_horizon() -> None:
    dataset = make_dataset(
        ("100", "101", "99", "100"),
        highs=("100", "105", "103", "999"),
        lows=("100", "98", "95", "1"),
    )
    label = ExcursionOutcomeLabeler(2).label(dataset, dataset.bars[0].session_date)
    assert label is not None
    outcome = PredictionOutcome(
        "outcome",
        "excursion",
        "1",
        "config",
        "1",
        dataset.metadata.dataset_id,
        dataset.metadata.data_sha256,
        label.signal_session,
        label.outcome_session,
        label.values,
    )

    up = DirectionalExcursionEvaluator().evaluate(
        _candidate(label.signal_session, PredictionDirection.UP), outcome
    )
    down = DirectionalExcursionEvaluator().evaluate(
        _candidate(label.signal_session, PredictionDirection.DOWN), outcome
    )

    assert up.mfe_percentage == Decimal("0.05")
    assert up.mae_percentage == Decimal("-0.05")
    assert up.mfe_session == dataset.bars[1].session_date
    assert up.mae_session == dataset.bars[2].session_date
    assert down.mfe_percentage == Decimal("0.05")
    assert down.mae_percentage == Decimal("-0.05")
    assert down.mfe_session == dataset.bars[2].session_date
    assert down.mae_session == dataset.bars[1].session_date


def _target_stop_result(
    *,
    direction: PredictionDirection,
    highs: tuple[str, ...],
    lows: tuple[str, ...],
    policy: SameSessionConflictPolicy = SameSessionConflictPolicy.AMBIGUOUS,
):
    dataset = make_dataset(
        tuple("100" for _ in highs),
        highs=highs,
        lows=lows,
    )
    labeler = TargetStopOutcomeLabeler(
        len(highs) - 1, Decimal("0.01"), Decimal("0.005"), policy
    )
    label = labeler.label(dataset, dataset.bars[0].session_date)
    assert label is not None
    outcome = PredictionOutcome(
        "outcome",
        "target_stop",
        "1",
        "config",
        "1",
        dataset.metadata.dataset_id,
        dataset.metadata.data_sha256,
        label.signal_session,
        label.outcome_session,
        label.values,
    )
    result = TargetStopEvaluator(Decimal("0.01"), Decimal("0.005"), policy).evaluate(
        _candidate(label.signal_session, direction), outcome
    )
    return dataset, result


def test_target_stop_labels_target_stop_neither_and_down_direction() -> None:
    _, target = _target_stop_result(
        direction=PredictionDirection.UP,
        highs=("100", "101", "100"),
        lows=("100", "99.8", "99.8"),
    )
    _, stop = _target_stop_result(
        direction=PredictionDirection.UP,
        highs=("100", "100.2", "101.2"),
        lows=("100", "99.5", "99.8"),
    )
    _, neither = _target_stop_result(
        direction=PredictionDirection.UP,
        highs=("100", "100.9", "100.8"),
        lows=("100", "99.6", "99.7"),
    )
    _, down_target = _target_stop_result(
        direction=PredictionDirection.DOWN,
        highs=("100", "100.4", "100.4"),
        lows=("100", "99", "99.6"),
    )

    assert target.label is TargetStopLabel.TARGET_FIRST
    assert stop.label is TargetStopLabel.STOP_FIRST
    assert neither.label is TargetStopLabel.NEITHER
    assert down_target.label is TargetStopLabel.TARGET_FIRST


def test_same_daily_bar_touch_is_ambiguous_and_preserves_range() -> None:
    dataset, result = _target_stop_result(
        direction=PredictionDirection.UP,
        highs=("100", "101", "100"),
        lows=("100", "99.5", "100"),
    )

    assert result.label is TargetStopLabel.BOTH_SAME_SESSION
    assert result.ambiguous_session == dataset.bars[1].session_date
    assert result.ambiguous_high == Decimal(101)
    assert result.ambiguous_low == Decimal("99.5")
    assert result.target_level == Decimal(101)
    assert result.stop_level == Decimal("99.5")


def test_conservative_policy_retains_ambiguity_but_labels_stop_first() -> None:
    dataset, result = _target_stop_result(
        direction=PredictionDirection.DOWN,
        highs=("100", "100.5", "100"),
        lows=("100", "99", "100"),
        policy=SameSessionConflictPolicy.CONSERVATIVE_STOP_FIRST,
    )

    assert result.label is TargetStopLabel.STOP_FIRST
    assert result.ambiguous_session == dataset.bars[1].session_date
    assert result.ambiguous_high == Decimal("100.5")
    assert result.ambiguous_low == Decimal(99)


def test_direction_dependent_labels_are_unavailable_without_direction() -> None:
    dataset, directional = _target_stop_result(
        direction=PredictionDirection.UP,
        highs=("100", "101", "100"),
        lows=("100", "99.8", "100"),
    )
    label = TargetStopOutcomeLabeler(2, Decimal("0.01"), Decimal("0.005")).label(
        dataset, dataset.bars[0].session_date
    )
    assert label is not None
    outcome = PredictionOutcome(
        "outcome",
        "target_stop",
        "1",
        "config",
        "1",
        dataset.metadata.dataset_id,
        dataset.metadata.data_sha256,
        label.signal_session,
        label.outcome_session,
        label.values,
    )

    unavailable = TargetStopEvaluator(Decimal("0.01"), Decimal("0.005")).evaluate(
        _candidate(label.signal_session, None), outcome
    )

    assert directional.available
    assert not unavailable.available
    assert unavailable.label is TargetStopLabel.UNAVAILABLE
    assert unavailable.unavailable_reason == "candidate_direction_unavailable"


def test_threshold_equality_counts_as_a_touch() -> None:
    _, result = _target_stop_result(
        direction=PredictionDirection.UP,
        highs=("100", "101"),
        lows=("100", "99.6"),
    )

    assert result.label is TargetStopLabel.TARGET_FIRST


def test_candidate_copy_can_change_direction_without_changing_outcome_path() -> None:
    candidate = _candidate(date(2024, 7, 1), PredictionDirection.UP)

    changed = replace(candidate, direction=PredictionDirection.DOWN)

    assert candidate.features_primitive() == changed.features_primitive()
    assert candidate.signal_session == changed.signal_session
