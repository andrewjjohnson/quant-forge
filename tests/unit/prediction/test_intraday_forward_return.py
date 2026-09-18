"""Concrete endpoint returns on deterministic canonical two-minute observations."""

from dataclasses import replace
from datetime import date, datetime, timedelta
from decimal import Decimal, localcontext
from typing import cast

import pytest

from quantforge.configuration import PrimitiveMapping, PrimitiveMappingSnapshot
from quantforge.data import IntradayBar, TimeframeBarSeries
from quantforge.prediction import (
    ForwardReturnOutcomeLabeler,
    IntradayForwardReturnOutcomeLabeler,
    IntradayForwardReturnValues,
    OutcomeAnchor,
    OutcomeAnchorKind,
    OutcomeEvaluationRequest,
    OutcomeResolution,
    OutcomeResolutionStatus,
    OutcomeTemporalConfiguration,
    evaluate_outcome_request,
)
from quantforge.prediction.errors import (
    InvalidPredictionConfigurationError,
    InvalidPredictionDataError,
)
from quantforge.prediction.timestamp_execution import bounded_outcome_source
from quantforge.timeframes import (
    BarCompletion,
    DevelopingBarExposure,
    IntradayInterval,
    Timeframe,
)
from quantforge.validation import OutcomeProvenance, TemporalOffset
from tests.unit.data.test_multi_timeframe import (
    _intraday_bar,  # pyright: ignore[reportPrivateUsage]
)
from tests.unit.helpers import make_dataset
from tests.unit.prediction.test_outcome_resolution import SESSION, series, timestamp

TWO_MINUTES = Timeframe.us_equity(IntradayInterval(timedelta(minutes=2)))


def priced_source(
    *,
    session: date = SESSION,
    prices: dict[datetime, str] | None = None,
    timeframe: Timeframe = TWO_MINUTES,
) -> TimeframeBarSeries:
    original = series(session=session, timeframe=timeframe)
    prices = {} if prices is None else prices
    bars = tuple(
        replace(
            cast(IntradayBar, bar),
            open=Decimal("100"),
            high=Decimal("200"),
            low=Decimal("1"),
            close=Decimal(prices.get(bar.end_timestamp, "100")),
        )
        for bar in original.bars
    )
    return series(session=session, timeframe=timeframe, bars=bars)


def label(
    source: TimeframeBarSeries,
    *,
    decision: datetime | None = None,
    minutes: int = 30,
    session: date = SESSION,
) -> IntradayForwardReturnValues:
    dataset = make_dataset(("999",))  # Daily prices must never supply the reference.
    labeler = IntradayForwardReturnOutcomeLabeler(
        OutcomeTemporalConfiguration.elapsed_duration(
            timedelta(minutes=minutes), source.timeframe
        )
    )
    labeler.validate_dataset(dataset)
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
    result = evaluate_outcome_request(
        labeler, dataset, request, source=bounded, resolution=resolution
    )
    assert result is not None
    assert result.signal_session == result.outcome_session == session
    assert result.values.resolution == resolution
    return result.values


@pytest.mark.parametrize("minutes", [10, 30, 60, 120])
@pytest.mark.parametrize(
    ("future_close", "expected_return"),
    [("100.3", "0.003"), ("99.8", "-0.002"), ("100", "0")],
)
def test_exact_endpoint_return_uses_normalized_decimals(
    minutes: int, future_close: str, expected_return: str
) -> None:
    decision = timestamp(11, 20)
    endpoint = decision + timedelta(minutes=minutes)
    source = priced_source(prices={endpoint: future_close})
    values = label(source, minutes=minutes)
    assert values.available
    assert values.status is OutcomeResolutionStatus.AVAILABLE
    assert values.reference_price == Decimal("100")
    assert values.outcome_price == Decimal(future_close)
    assert values.raw_return == Decimal(expected_return)
    assert values.resolution.requested_target_timestamp == endpoint
    assert values.resolution.resolved_observation_timestamp == endpoint
    primitive = values.to_primitive()
    assert Decimal(str(primitive["raw_return"])) == Decimal(expected_return)
    assert (
        primitive["reference_price_convention"]
        == "completed_decision_observation_close"
    )
    assert (
        primitive["future_price_convention"] == "resolved_completed_observation_close"
    )
    assert primitive["source_reference"] == source.dataset_reference.to_primitive(
        include_feed_scope=True
    )
    assert PrimitiveMappingSnapshot.capture(primitive).to_primitive() == primitive
    reference = next(bar for bar in source.bars if bar.end_timestamp == decision)
    assert values.reference_observation_id == reference.bar_id


def test_between_boundary_target_uses_qf46_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from quantforge.prediction import timestamp_execution

    resolver = timestamp_execution.resolve_future_observation
    calls: list[OutcomeEvaluationRequest] = []

    def tracked(
        request: OutcomeEvaluationRequest, source: TimeframeBarSeries
    ) -> OutcomeResolution:
        calls.append(request)
        return resolver(request, source)

    monkeypatch.setattr(timestamp_execution, "resolve_future_observation", tracked)
    values = label(priced_source(prices={timestamp(11, 52): "101"}), minutes=31)
    assert len(calls) == 1
    assert values.resolution.requested_target_timestamp == timestamp(11, 51)
    assert values.resolution.resolved_observation_timestamp == timestamp(11, 52)
    assert values.raw_return == Decimal("0.01")


@pytest.mark.parametrize("missing_kind", ["gap", "end", "developing"])
def test_future_unavailability_keeps_qf46_status_and_null_return(
    missing_kind: str,
) -> None:
    timeframe = replace(
        TWO_MINUTES, developing_bar_exposure=DevelopingBarExposure.INCLUDE
    )
    full = priced_source(timeframe=timeframe)
    endpoint = timestamp(11, 52)
    bars = tuple(
        cast(IntradayBar, bar)
        for bar in full.bars
        if (
            bar.end_timestamp != endpoint
            if missing_kind == "gap"
            else bar.end_timestamp < endpoint
        )
    )
    if missing_kind == "developing":
        bars += (
            _intraday_bar(
                timeframe,
                timestamp(11, 50),
                timestamp(11, 51),
                BarCompletion.DEVELOPING,
            ),
        )
    source = series(timeframe=timeframe, bars=bars)
    result = label(source, minutes=31)
    expected = {
        "gap": OutcomeResolutionStatus.MISSING_OBSERVATION,
        "end": OutcomeResolutionStatus.DATASET_END,
        "developing": OutcomeResolutionStatus.INCOMPLETE,
    }[missing_kind]
    assert result.status is expected
    assert not result.available
    assert result.raw_return is None
    assert result.outcome_price is None
    assert result.resolution.expected_observation_timestamp == endpoint
    assert result.resolution.resolved_observation_timestamp is None
    if missing_kind == "gap":
        assert any(bar.end_timestamp == timestamp(11, 54) for bar in source.bars)


@pytest.mark.parametrize(
    ("session", "hour", "minute"),
    [(SESSION, 15, 30), (date(2024, 7, 3), 12, 30)],
)
def test_overflow_never_shortens_or_rolls_to_next_session(
    session: date, hour: int, minute: int
) -> None:
    source = priced_source(session=session)
    next_session = date(2024, 7, 2) if session == SESSION else date(2024, 7, 5)
    later = priced_source(session=next_session)
    source = series(
        timeframe=TWO_MINUTES,
        bars=cast(tuple[IntradayBar, ...], source.bars + later.bars),
    )
    decision = timestamp(hour, minute, session=session)
    result = label(source, decision=decision, minutes=60, session=session)
    assert result.status is OutcomeResolutionStatus.SESSION_OVERFLOW
    assert result.resolution.requested_target_timestamp == decision + timedelta(hours=1)
    assert result.resolution.expected_observation_timestamp is None
    assert result.raw_return is None


@pytest.mark.parametrize("session", [SESSION, date(2024, 7, 3)])
def test_horizon_ending_at_actual_close_is_available(session: date) -> None:
    close = timestamp(13 if session.day == 3 else 16, 0, session=session)
    result = label(
        priced_source(session=session),
        decision=close - timedelta(minutes=30),
        session=session,
    )
    assert result.available
    assert result.resolution.resolved_observation_timestamp == close


@pytest.mark.parametrize("change", ["missing", "developing", "between"])
def test_exact_completed_reference_is_required(change: str) -> None:
    timeframe = replace(
        TWO_MINUTES, developing_bar_exposure=DevelopingBarExposure.INCLUDE
    )
    source = priced_source(timeframe=timeframe)
    decision = timestamp(11, 20)
    bars = cast(tuple[IntradayBar, ...], source.bars)
    if change == "missing":
        bars = tuple(bar for bar in bars if bar.end_timestamp != decision)
    elif change == "developing":
        bars = tuple(
            replace(
                bar,
                end_timestamp=decision - timedelta(minutes=1),
                completion=BarCompletion.DEVELOPING,
            )
            if bar.end_timestamp == decision
            else bar
            for bar in bars
        )
        decision -= timedelta(minutes=1)
    else:
        decision += timedelta(minutes=1)
    with pytest.raises(InvalidPredictionDataError, match="exact completed decision"):
        label(series(timeframe=timeframe, bars=bars), decision=decision)


def test_no_path_or_post_endpoint_price_can_change_historical_return() -> None:
    source = priced_source(prices={timestamp(11, 50): "101"})
    baseline = label(source)
    # Endpoint returns do not require intervening path coverage.
    endpoint_only = tuple(
        cast(IntradayBar, bar)
        for bar in source.bars
        if bar.end_timestamp in (timestamp(11, 20), timestamp(11, 50))
    )
    short = series(timeframe=TWO_MINUTES, bars=endpoint_only)
    assert label(short).to_primitive() == baseline.to_primitive()
    later = priced_source(session=date(2024, 7, 2))
    appended = series(
        timeframe=TWO_MINUTES,
        bars=cast(tuple[IntradayBar, ...], source.bars + later.bars),
    )
    assert label(appended).to_primitive() == baseline.to_primitive()
    changed = priced_source(prices={timestamp(11, 50): "101", timestamp(11, 52): "199"})
    assert label(changed).to_primitive() == baseline.to_primitive()


def test_decimal_policy_does_not_depend_on_callers_context() -> None:
    source = priced_source(prices={timestamp(11, 20): "99", timestamp(11, 50): "100"})
    expected = label(source).raw_return
    with localcontext() as context:
        context.prec = 2
        assert label(source).raw_return == expected
    assert expected == Decimal("0.010101010101010101010101010101010")


def test_identity_and_temporal_reach_reuse_existing_contract() -> None:
    labelers = tuple(
        IntradayForwardReturnOutcomeLabeler(
            OutcomeTemporalConfiguration.elapsed_duration(
                timedelta(minutes=m), TWO_MINUTES
            )
        )
        for m in (10, 30, 60, 120)
    )
    assert len({item.configuration_id for item in labelers}) == 4
    for labeler, minutes in zip(labelers, (10, 30, 60, 120), strict=True):
        assert OutcomeProvenance.capture_timestamp(
            labeler
        ).future_horizon == TemporalOffset.duration(timedelta(minutes=minutes + 2))
        config = labeler.configuration()
        temporal = OutcomeTemporalConfiguration.from_primitive(
            cast(PrimitiveMapping, config["temporal_configuration"])
        )
        assert (
            IntradayForwardReturnOutcomeLabeler(temporal).configuration_id
            == labeler.configuration_id
        )
        assert "required_future_sessions" not in config
    other = IntradayForwardReturnOutcomeLabeler(
        OutcomeTemporalConfiguration.elapsed_duration(
            timedelta(minutes=30),
            Timeframe.us_equity(IntradayInterval(timedelta(minutes=5))),
        )
    )
    assert other.configuration_id != labelers[1].configuration_id
    with pytest.raises(InvalidPredictionConfigurationError, match="elapsed"):
        IntradayForwardReturnOutcomeLabeler(
            OutcomeTemporalConfiguration.exchange_sessions(1)
        )


@pytest.mark.parametrize("sessions", [1, 5])
def test_legacy_session_values_and_configuration_remain_unchanged(
    sessions: int,
) -> None:
    dataset = make_dataset(("100", "101", "102", "103", "104", "105"))
    labeler = ForwardReturnOutcomeLabeler(sessions)
    labeler.validate_dataset(dataset)
    result = labeler.label(dataset, SESSION)
    assert result is not None
    assert result.values.raw_return == Decimal(sessions) / 100
    assert result.values.horizon_sessions == sessions
    assert result.outcome_session == dataset.bars[sessions].session_date
    assert labeler.configuration() == {
        "component_name": "forward_close_return",
        "component_type": "prediction_outcome_labeler",
        "contract_version": "1",
        "implementation_version": "3",
        "result_schema_version": "1",
        "required_market_fields": ["close"],
        "parameters": {
            "future_sessions": sessions,
            "outcome_field": "close",
            "reference_field": "completed_signal_session_close",
            "return_formula": "outcome_close / reference_close - 1",
        },
    }
