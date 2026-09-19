"""Hand-calculated 60m/2m path fixtures and adversarial temporal boundaries."""

from dataclasses import replace
from datetime import date, datetime, timedelta
from decimal import Decimal, localcontext
from typing import cast

import pytest

from quantforge.configuration import (
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.data import IntradayBar, TimeframeBarSeries
from quantforge.prediction import (
    IntradayExcursionEvaluator,
    IntradayPathOutcomeLabeler,
    IntradayPathValues,
    IntradayTargetStopEvaluator,
    InvalidPredictionConfigurationError,
    InvalidPredictionDataError,
    OutcomeAnchor,
    OutcomeAnchorKind,
    OutcomeEvaluationRequest,
    OutcomeResolutionStatus,
    OutcomeTemporalConfiguration,
    PredictionDirection,
    PredictionOutcome,
    SameBarConflictPolicy,
    TargetStopLabel,
    evaluate_outcome_request,
    intraday_target_stop_outcome,
)
from quantforge.prediction.timestamp_execution import bounded_outcome_source
from quantforge.timeframes import (
    BarCompletion,
    DevelopingBarExposure,
    IntradayInterval,
    Timeframe,
)
from quantforge.validation import OutcomeProvenance, TemporalOffset
from tests.unit.helpers import make_dataset
from tests.unit.prediction.test_feature_outcomes import (
    _candidate,  # pyright: ignore[reportPrivateUsage]
)
from tests.unit.prediction.test_intraday_forward_return import (
    TWO_MINUTES,
    priced_source,
)
from tests.unit.prediction.test_outcome_resolution import SESSION, series, timestamp


def path_source(
    *,
    prices: dict[datetime, tuple[str, str]] | None = None,
    session: date = SESSION,
    timeframe: Timeframe = TWO_MINUTES,
) -> TimeframeBarSeries:
    original = priced_source(session=session, timeframe=timeframe)
    prices = {} if prices is None else prices
    bars = tuple(
        replace(
            cast(IntradayBar, bar),
            high=Decimal(prices.get(bar.end_timestamp, ("100", "100"))[0]),
            low=Decimal(prices.get(bar.end_timestamp, ("100", "100"))[1]),
        )
        for bar in original.bars
    )
    return TimeframeBarSeries._from_validated_artifact(  # pyright: ignore[reportPrivateUsage]
        original.dataset_reference,
        timeframe,
        bars,
        dataset_family_manifest_id="fixture-family",
    )


def path_label(
    source: TimeframeBarSeries,
    *,
    decision: datetime | None = None,
    minutes: int = 60,
    session: date = SESSION,
) -> PredictionOutcome[IntradayPathValues]:
    dataset = make_dataset(("999",))
    labeler = IntradayPathOutcomeLabeler(
        OutcomeTemporalConfiguration.elapsed_duration(
            timedelta(minutes=minutes), source.timeframe
        )
    )
    request = OutcomeEvaluationRequest(
        OutcomeAnchor(
            OutcomeAnchorKind.TIMESTAMP,
            session,
            decision or timestamp(11, 20, session=session),
        ),
        labeler.temporal_configuration,
        labeler.configuration_id,
        dataset.metadata.dataset_id,
        dataset.metadata.data_sha256,
        source.dataset_reference,
    )
    bounded, resolution = bounded_outcome_source(source, request)
    labeled = evaluate_outcome_request(
        labeler, dataset, request, source=bounded, resolution=resolution
    )
    assert labeled is not None
    assert labeled.values.resolution == resolution
    return PredictionOutcome(
        "fixture",
        labeler.name,
        labeler.implementation_version,
        labeler.configuration_id,
        labeler.result_schema_version,
        dataset.metadata.dataset_id,
        dataset.metadata.data_sha256,
        labeled.signal_session,
        labeled.outcome_session,
        labeled.values,
    )


@pytest.mark.parametrize("extreme", [timestamp(11, 22), timestamp(12, 20)])
@pytest.mark.parametrize(
    "direction", [PredictionDirection.UP, PredictionDirection.DOWN]
)
def test_excursions_include_first_and_final_bar_exclude_decision_and_later(
    extreme: datetime, direction: PredictionDirection
) -> None:
    source = path_source(
        prices={
            timestamp(11, 20): ("900", "1"),
            extreme: ("100.3", "99.8"),
            timestamp(12, 22): ("999", "0.5"),
        }
    )
    outcome = path_label(source)
    path = outcome.values
    assert path.available
    assert path.reference_price == Decimal(100)
    assert len(path.future_ranges) == 30
    assert path.future_ranges[0].start_timestamp == timestamp(11, 20)
    assert path.future_ranges[0].end_timestamp == timestamp(11, 22)
    assert path.future_ranges[-1].end_timestamp == timestamp(12, 20)
    result = IntradayExcursionEvaluator().evaluate(
        _candidate(SESSION, direction), outcome
    )
    assert result.mfe_percentage == Decimal(
        "0.003" if direction is PredictionDirection.UP else "0.002"
    )
    assert result.mae_percentage == Decimal(
        "-0.002" if direction is PredictionDirection.UP else "-0.003"
    )
    assert result.mfe_bar is not None
    assert result.mfe_bar.end_timestamp == extreme
    assert result.mae_bar is not None
    assert result.mae_bar.end_timestamp == extreme
    assert (
        PrimitiveMappingSnapshot.capture(result.to_primitive()).to_primitive()
        == result.to_primitive()
    )


@pytest.mark.parametrize("side", ["flat", "above", "below"])
def test_excursion_zero_ties_and_unclamped_daily_convention(side: str) -> None:
    source = path_source()
    bars = tuple(
        replace(
            cast(IntradayBar, bar),
            open=Decimal(
                "101" if side == "above" else "99" if side == "below" else "100"
            ),
            close=Decimal(
                "101" if side == "above" else "99" if side == "below" else "100"
            ),
            high=Decimal(
                "101" if side == "above" else "99" if side == "below" else "100"
            ),
            low=Decimal(
                "101" if side == "above" else "99" if side == "below" else "100"
            ),
        )
        if bar.end_timestamp > timestamp(11, 20)
        else cast(IntradayBar, bar)
        for bar in source.bars
    )
    result = IntradayExcursionEvaluator().evaluate(
        _candidate(SESSION, PredictionDirection.UP),
        path_label(series(timeframe=TWO_MINUTES, bars=bars)),
    )
    expected = Decimal(
        "0.01" if side == "above" else "-0.01" if side == "below" else "0"
    )
    assert result.mfe_percentage == result.mae_percentage == expected
    assert result.mfe_bar is not None
    assert result.mfe_bar.end_timestamp == timestamp(11, 22)
    assert result.mae_bar is not None
    assert result.mae_bar.end_timestamp == timestamp(11, 22)


@pytest.mark.parametrize(
    ("prices", "expected", "hit"),
    [
        (
            {timestamp(11, 22): ("100.3", "100"), timestamp(11, 24): ("100", "99.8")},
            TargetStopLabel.TARGET_FIRST,
            timestamp(11, 22),
        ),
        (
            {timestamp(11, 22): ("100", "99.8"), timestamp(11, 24): ("100.3", "100")},
            TargetStopLabel.STOP_FIRST,
            timestamp(11, 22),
        ),
        (
            {timestamp(12, 20): ("100.3", "100")},
            TargetStopLabel.TARGET_FIRST,
            timestamp(12, 20),
        ),
        (
            {timestamp(12, 20): ("100", "99.8")},
            TargetStopLabel.STOP_FIRST,
            timestamp(12, 20),
        ),
        ({timestamp(12, 22): ("999", "1")}, TargetStopLabel.NEITHER, None),
        ({timestamp(11, 20): ("999", "1")}, TargetStopLabel.NEITHER, None),
        ({timestamp(11, 22): ("100.3", "99.8")}, TargetStopLabel.BOTH_SAME_BAR, None),
        ({timestamp(11, 22): ("999", "1")}, TargetStopLabel.BOTH_SAME_BAR, None),
    ],
)
def test_inclusive_thresholds_first_bar_order_and_ambiguity(
    prices: dict[datetime, tuple[str, str]],
    expected: TargetStopLabel,
    hit: datetime | None,
) -> None:
    result = IntradayTargetStopEvaluator(Decimal("0.003"), Decimal("0.002")).evaluate(
        _candidate(SESSION, PredictionDirection.UP),
        path_label(path_source(prices=prices)),
    )
    assert result.label is expected
    assert result.target_level == Decimal("100.3")
    assert result.stop_level == Decimal("99.8")
    assert (None if result.event_bar is None else result.event_bar.end_timestamp) == hit
    if expected is TargetStopLabel.BOTH_SAME_BAR:
        assert result.ambiguous_bar is not None
        assert result.ambiguous_bar.end_timestamp == timestamp(11, 22)
        assert (
            result.to_primitive()["ambiguous_observation_id"]
            == result.ambiguous_bar.observation_id
        )
        assert result.to_primitive()["event_timestamp"] is None
    else:
        assert result.ambiguous_bar is None


@pytest.mark.parametrize(
    ("bounds", "expected"),
    [
        (("100", "99.7"), TargetStopLabel.TARGET_FIRST),
        (("100.2", "100"), TargetStopLabel.STOP_FIRST),
        (("100.2", "99.7"), TargetStopLabel.BOTH_SAME_BAR),
    ],
)
def test_down_thresholds_use_existing_positive_distance_units(
    bounds: tuple[str, str], expected: TargetStopLabel
) -> None:
    result = IntradayTargetStopEvaluator(Decimal("0.003"), Decimal("0.002")).evaluate(
        _candidate(SESSION, PredictionDirection.DOWN),
        path_label(path_source(prices={timestamp(11, 22): bounds})),
    )
    assert result.label is expected
    assert result.target_level == Decimal("99.7")
    assert result.stop_level == Decimal("100.2")


@pytest.mark.parametrize(
    "kind", ["interior_gap", "first_gap", "endpoint_gap", "end", "developing"]
)
def test_incomplete_path_never_certifies_extremes_or_early_hit(kind: str) -> None:
    timeframe = replace(
        TWO_MINUTES, developing_bar_exposure=DevelopingBarExposure.INCLUDE
    )
    full = path_source(timeframe=timeframe, prices={timestamp(11, 22): ("110", "100")})
    missing = (
        timestamp(12, 20)
        if kind == "endpoint_gap"
        else timestamp(11, 22)
        if kind == "first_gap"
        else timestamp(11, 40)
    )
    bars = tuple(
        replace(
            cast(IntradayBar, bar),
            end_timestamp=bar.end_timestamp - timedelta(minutes=1),
            completion=BarCompletion.DEVELOPING,
        )
        if kind == "developing" and bar.end_timestamp == missing
        else cast(IntradayBar, bar)
        for bar in full.bars
        if (
            bar.end_timestamp <= timestamp(12, 4)
            if kind == "end"
            else bar.end_timestamp != missing or kind == "developing"
        )
    )
    outcome = path_label(series(timeframe=timeframe, bars=bars))
    expected = (
        OutcomeResolutionStatus.DATASET_END
        if kind == "end"
        else OutcomeResolutionStatus.INCOMPLETE
        if kind == "developing"
        else OutcomeResolutionStatus.MISSING_OBSERVATION
    )
    assert outcome.values.status is expected
    assert not outcome.values.available
    assert outcome.values.future_ranges == ()
    if kind in ("interior_gap", "first_gap", "developing"):
        assert outcome.values.resolution.available
        assert outcome.values.missing_observation_timestamp == missing
    signal = _candidate(SESSION, PredictionDirection.UP)
    excursions = IntradayExcursionEvaluator().evaluate(signal, outcome)
    target = IntradayTargetStopEvaluator(Decimal("0.003"), Decimal("0.002")).evaluate(
        signal, outcome
    )
    assert excursions.mfe_percentage is excursions.mae_percentage is None
    assert target.label is TargetStopLabel.UNAVAILABLE
    assert excursions.to_primitive()["available"] is False
    assert target.to_primitive()["unavailable_reason"] == expected.value


@pytest.mark.parametrize("session", [SESSION, date(2024, 7, 3)])
@pytest.mark.parametrize("overflow", [False, True])
def test_actual_session_close_never_shortens_or_rolls(
    session: date, overflow: bool
) -> None:
    close = timestamp(13 if session.day == 3 else 16, 0, session=session)
    source = path_source(session=session)
    next_session = date(2024, 7, 2) if session == SESSION else date(2024, 7, 5)
    later = path_source(session=next_session)
    source = series(
        timeframe=TWO_MINUTES,
        bars=cast(tuple[IntradayBar, ...], source.bars + later.bars),
    )
    result = path_label(
        source,
        decision=close - timedelta(minutes=30 if overflow else 60),
        session=session,
    ).values
    assert result.available is (not overflow)
    assert result.status is (
        OutcomeResolutionStatus.SESSION_OVERFLOW
        if overflow
        else OutcomeResolutionStatus.AVAILABLE
    )
    assert len(result.future_ranges) == (0 if overflow else 30)


def test_ceiling_resolution_and_post_resolved_endpoint_exclusion() -> None:
    source = path_source(
        prices={timestamp(11, 52): ("101", "99"), timestamp(11, 54): ("999", "1")}
    )
    outcome = path_label(source, minutes=31)
    assert outcome.values.resolution.requested_target_timestamp == timestamp(11, 51)
    assert outcome.values.future_ranges[-1].end_timestamp == timestamp(11, 52)
    result = IntradayExcursionEvaluator().evaluate(
        _candidate(SESSION, PredictionDirection.UP), outcome
    )
    assert result.mfe_percentage == Decimal("0.01")
    assert result.mae_percentage == Decimal("-0.01")


@pytest.mark.parametrize("kind", ["missing", "between", "developing"])
def test_reference_must_be_exact_completed_decision(kind: str) -> None:
    timeframe = replace(
        TWO_MINUTES, developing_bar_exposure=DevelopingBarExposure.INCLUDE
    )
    source = path_source(timeframe=timeframe)
    decision = timestamp(11, 20)
    bars = tuple(
        replace(
            cast(IntradayBar, bar),
            end_timestamp=bar.end_timestamp - timedelta(minutes=1),
            completion=BarCompletion.DEVELOPING,
        )
        if kind == "developing" and bar.end_timestamp == decision
        else cast(IntradayBar, bar)
        for bar in source.bars
        if kind != "missing" or bar.end_timestamp != decision
    )
    with pytest.raises(InvalidPredictionDataError, match="exact completed decision"):
        path_label(
            series(timeframe=timeframe, bars=bars),
            decision=decision + timedelta(minutes=1) if kind == "between" else decision,
        )


@pytest.mark.parametrize(
    "boundary", [timestamp(11, 20), timestamp(11, 42), timestamp(12, 20)]
)
@pytest.mark.parametrize("mismatch", ["symbol", "adjustment basis"])
def test_each_used_bar_must_match_prediction_dataset(
    boundary: datetime, mismatch: str
) -> None:
    source = path_source()
    bars = tuple(
        (
            replace(cast(IntradayBar, bar), symbol="QQQ")
            if mismatch == "symbol"
            else replace(
                cast(IntradayBar, bar),
                provenance=replace(
                    cast(IntradayBar, bar).provenance,
                    adjustment_basis=replace(
                        cast(IntradayBar, bar).provenance.adjustment_basis,
                        adjusted_fields_used=True,
                    ),
                ),
            )
        )
        if bar.end_timestamp == boundary
        else cast(IntradayBar, bar)
        for bar in source.bars
    )
    with pytest.raises(InvalidPredictionDataError, match=mismatch):
        path_label(series(timeframe=TWO_MINUTES, bars=bars))


def test_future_append_and_post_horizon_mutation_leave_labels_stable() -> None:
    full = path_source(prices={timestamp(11, 22): ("100.3", "99.8")})
    prefix = series(
        timeframe=TWO_MINUTES,
        bars=tuple(
            cast(IntradayBar, bar)
            for bar in full.bars
            if bar.end_timestamp <= timestamp(12, 20)
        ),
    )
    changed = path_source(
        prices={timestamp(11, 22): ("100.3", "99.8"), timestamp(12, 22): ("999", "1")}
    )
    assert (
        path_label(prefix).values.to_primitive()
        == path_label(full).values.to_primitive()
        == path_label(changed).values.to_primitive()
    )


def test_directionless_candidates_are_explicitly_unavailable() -> None:
    outcome = path_label(path_source())
    signal = _candidate(SESSION, None)
    for evaluator in (
        IntradayExcursionEvaluator(),
        IntradayTargetStopEvaluator(Decimal("0.003"), Decimal("0.002")),
    ):
        values = evaluator.evaluate(signal, outcome).to_primitive()
        assert values["available"] is False
        assert values["unavailable_reason"] == "candidate_direction_unavailable"
        assert values["direction"] is None


def test_decimal_policy_is_independent_of_ambient_context() -> None:
    outcome = path_label(path_source(prices={timestamp(11, 22): ("100.321", "99.876")}))
    signal = _candidate(SESSION, PredictionDirection.UP)
    for evaluator in (
        IntradayExcursionEvaluator(),
        IntradayTargetStopEvaluator(Decimal("0.00321"), Decimal("0.00234")),
    ):
        expected = evaluator.evaluate(signal, outcome).to_primitive()
        with localcontext() as context:
            context.prec = 2
            assert evaluator.evaluate(signal, outcome).to_primitive() == expected


def test_identity_temporal_reach_and_policy_fail_closed() -> None:
    source = path_source()
    baseline = intraday_target_stop_outcome(
        timedelta(minutes=60), source, Decimal("0.003"), Decimal("0.002")
    )
    changes = (
        intraday_target_stop_outcome(
            timedelta(minutes=30), source, Decimal("0.003"), Decimal("0.002")
        ),
        intraday_target_stop_outcome(
            timedelta(minutes=60), source, Decimal("0.005"), Decimal("0.002")
        ),
        intraday_target_stop_outcome(
            timedelta(minutes=60), source, Decimal("0.003"), Decimal("0.003")
        ),
        intraday_target_stop_outcome(
            timedelta(minutes=60),
            path_source(
                timeframe=Timeframe.us_equity(IntradayInterval(timedelta(minutes=1)))
            ),
            Decimal("0.003"),
            Decimal("0.002"),
        ),
    )
    assert (
        len({baseline.configuration_id, *(item.configuration_id for item in changes)})
        == 5
    )
    labeler = cast(IntradayPathOutcomeLabeler, baseline.labeler)
    assert OutcomeProvenance.capture_timestamp(
        labeler
    ).future_horizon == TemporalOffset.duration(timedelta(minutes=62))
    temporal = OutcomeTemporalConfiguration.from_primitive(
        cast(PrimitiveMapping, labeler.configuration()["temporal_configuration"])
    )
    assert (
        IntradayPathOutcomeLabeler(temporal).configuration_id
        == labeler.configuration_id
    )
    evaluator = cast(IntradayTargetStopEvaluator, baseline.evaluator)
    config = evaluator.configuration()
    parameters = cast(PrimitiveMapping, config["parameters"])
    changed_policy: PrimitiveMapping = {
        **config,
        "parameters": {
            **parameters,
            "same_bar_conflict_policy": "conservative_stop_first",
        },
    }
    assert configuration_identity(changed_policy) != evaluator.configuration_id
    with pytest.raises(InvalidPredictionConfigurationError, match="ambiguity"):
        IntradayTargetStopEvaluator(
            Decimal("0.003"),
            Decimal("0.002"),
            cast(SameBarConflictPolicy, "conservative_stop_first"),
        )
    with pytest.raises(InvalidPredictionConfigurationError, match="elapsed"):
        IntradayPathOutcomeLabeler(OutcomeTemporalConfiguration.exchange_sessions(1))


@pytest.mark.parametrize("invalid", ["0", "-0.002", "1", "NaN", "Infinity"])
@pytest.mark.parametrize("field", ["target", "stop"])
def test_threshold_units_require_positive_finite_ratios(
    invalid: str, field: str
) -> None:
    with pytest.raises(InvalidPredictionConfigurationError, match="strictly between"):
        IntradayTargetStopEvaluator(
            Decimal(invalid if field == "target" else "0.003"),
            Decimal(invalid if field == "stop" else "0.002"),
        )


def test_reference_close_matches_qf49_and_scales_excursions() -> None:
    from tests.unit.prediction.test_intraday_forward_return import label

    source = path_source(prices={timestamp(11, 20): ("100", "50")})
    bars = tuple(
        replace(cast(IntradayBar, bar), close=Decimal("50"))
        if bar.end_timestamp == timestamp(11, 20)
        else cast(IntradayBar, bar)
        for bar in source.bars
    )
    source = series(timeframe=TWO_MINUTES, bars=bars)
    outcome = path_label(source)
    assert outcome.values.reference_price == label(source, minutes=60).reference_price
    result = IntradayExcursionEvaluator().evaluate(
        _candidate(SESSION, PredictionDirection.UP), outcome
    )
    assert result.mfe_percentage == result.mae_percentage == Decimal("1")


@pytest.mark.parametrize("developing_endpoint", [False, True])
def test_endpoint_failure_status_remains_qf46_authority(
    developing_endpoint: bool,
) -> None:
    timeframe = replace(
        TWO_MINUTES, developing_bar_exposure=DevelopingBarExposure.INCLUDE
    )
    source = path_source(timeframe=timeframe)
    endpoint = timestamp(12, 20)
    bars = tuple(
        replace(
            cast(IntradayBar, bar),
            end_timestamp=endpoint - timedelta(minutes=1),
            completion=BarCompletion.DEVELOPING,
        )
        if bar.end_timestamp == endpoint
        else cast(IntradayBar, bar)
        for bar in source.bars
        if bar.end_timestamp < endpoint
        or (developing_endpoint and bar.end_timestamp == endpoint)
    )
    result = path_label(series(timeframe=timeframe, bars=bars)).values
    assert result.status is (
        OutcomeResolutionStatus.INCOMPLETE
        if developing_endpoint
        else OutcomeResolutionStatus.DATASET_END
    )
    assert result.status is result.resolution.status
    assert result.future_ranges == ()


def test_completed_terminal_partial_bar_uses_actual_calendar_boundary() -> None:
    # 390-minute normal session has a final five-minute window on 7m bars.
    timeframe = Timeframe.us_equity(IntradayInterval(timedelta(minutes=7)))
    source = path_source(timeframe=timeframe, prices={timestamp(16, 0): ("101", "99")})
    outcome = path_label(source, decision=timestamp(14, 59), minutes=60)
    assert outcome.values.available
    assert outcome.values.resolution.requested_target_timestamp == timestamp(15, 59)
    final = outcome.values.future_ranges[-1]
    assert final.end_timestamp == timestamp(16, 0)
    assert final.end_timestamp - final.start_timestamp == timedelta(minutes=5)
    result = IntradayExcursionEvaluator().evaluate(
        _candidate(SESSION, PredictionDirection.UP), outcome
    )
    assert result.mfe_percentage == Decimal("0.01")
    assert result.mae_percentage == Decimal("-0.01")
