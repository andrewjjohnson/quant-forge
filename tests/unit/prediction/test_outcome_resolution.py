"""Expected-boundary alignment and future-only metadata fixtures."""

from dataclasses import replace
from datetime import UTC, date, datetime, time, timedelta
from typing import cast
from zoneinfo import ZoneInfo

import pytest

from quantforge.configuration import PrimitiveMappingSnapshot
from quantforge.data import IntradayBar, TimeframeBarSeries
from quantforge.data.intraday_aggregation import intraday_session_windows
from quantforge.prediction import (
    OutcomeAnchor,
    OutcomeAnchorKind,
    OutcomeEvaluationRequest,
    OutcomeResolution,
    OutcomeResolutionStatus,
    OutcomeTemporalConfiguration,
    OutcomeTemporalError,
    SchemaFieldCategory,
    outcome_resolution_fields,
    resolve_future_observation,
)
from quantforge.prediction.feature_dataset import (
    _schema_value_matches,  # pyright: ignore[reportPrivateUsage]
)
from quantforge.timeframes import (
    BarCompletion,
    DevelopingBarExposure,
    IntradayAnchor,
    IntradayInterval,
    Timeframe,
)
from tests.unit.data.test_multi_timeframe import (
    _family,  # pyright: ignore[reportPrivateUsage]
    _intraday_bar,  # pyright: ignore[reportPrivateUsage]
)
from tests.unit.prediction.test_outcome_temporal import TIMEFRAME

NEW_YORK = ZoneInfo("America/New_York")
SESSION = date(2024, 7, 1)


def timestamp(hour: int, minute: int, *, session: date = SESSION) -> datetime:
    return datetime.combine(session, time(hour, minute), NEW_YORK)


def series(
    *,
    session: date = SESSION,
    timeframe: Timeframe = TIMEFRAME,
    bars: tuple[IntradayBar, ...] | None = None,
) -> TimeframeBarSeries:
    family = _family()
    lineage = family.datasets[0]
    # Fixture-only source builder follows the existing isolated alignment tests.
    # Production callers use TimeframeBarSeries.from_*_dataset validation.
    reference = replace(
        family.reference(lineage.dataset_id),
        timeframe_configuration_id=timeframe.configuration_id,
    )
    windows = intraday_session_windows(session, timeframe)
    fixture_bars = (
        tuple(
            _intraday_bar(
                timeframe,
                window.start_timestamp,
                window.end_timestamp,
                window.completion,
            )
            for window in windows
        )
        if bars is None
        else bars
    )
    return TimeframeBarSeries._from_validated_artifact(  # pyright: ignore[reportPrivateUsage]
        reference, timeframe, fixture_bars
    )


def request(
    source: TimeframeBarSeries,
    *,
    decision: datetime | None = None,
    duration: timedelta = timedelta(minutes=30),
    session: date = SESSION,
) -> OutcomeEvaluationRequest:
    temporal = OutcomeTemporalConfiguration.elapsed_duration(duration, source.timeframe)
    return OutcomeEvaluationRequest(
        OutcomeAnchor(
            OutcomeAnchorKind.TIMESTAMP,
            session,
            decision or timestamp(11, 20, session=session),
        ),
        temporal,
        temporal.configuration_id,
        "prediction-dataset",
        "prediction-fingerprint",
        source.dataset_reference,
    )


@pytest.mark.parametrize(
    ("decision", "expected"),
    [
        (timestamp(11, 20), timestamp(11, 50)),
        (timestamp(11, 21), timestamp(11, 55)),
        (timestamp(15, 30), timestamp(16, 0)),
    ],
)
def test_target_uses_first_expected_completed_boundary(
    decision: datetime, expected: datetime
) -> None:
    source = series()
    evaluation = request(source, decision=decision)
    resolved = resolve_future_observation(evaluation, source)
    assert resolved.status is OutcomeResolutionStatus.AVAILABLE
    assert resolved.available
    assert resolved.requested_target_timestamp == decision + timedelta(minutes=30)
    assert resolved.expected_observation_timestamp == expected
    assert resolved.resolved_observation_timestamp == expected
    assert resolved.observation is not None
    assert resolved.observation.end_timestamp == expected
    assert (
        resolved.to_primitive()["requested_target_timestamp"]
        == (decision + timedelta(minutes=30)).astimezone(UTC).isoformat()
    )


@pytest.mark.parametrize("minutes", [10, 30, 60, 120])
def test_two_minute_endpoints_and_conservative_reach(minutes: int) -> None:
    source = series(
        timeframe=Timeframe.us_equity(IntradayInterval(timedelta(minutes=2)))
    )
    decision = timestamp(11, 21) + timedelta(microseconds=1)
    evaluation = request(source, decision=decision, duration=timedelta(minutes=minutes))
    resolved = resolve_future_observation(evaluation, source)
    assert resolved.resolved_observation_timestamp is not None
    assert resolved.resolved_observation_timestamp >= decision + timedelta(
        minutes=minutes
    )
    assert (
        resolved.resolved_observation_timestamp
        <= decision + evaluation.temporal_configuration.required_future_duration
    )


@pytest.mark.parametrize(
    ("session", "decision", "duration"),
    [
        (SESSION, timestamp(15, 45), timedelta(minutes=30)),
        (
            date(2024, 7, 3),
            timestamp(12, 45, session=date(2024, 7, 3)),
            timedelta(minutes=30),
        ),
        (SESSION, timestamp(11, 20), timedelta(hours=24)),
    ],
)
def test_overflow_is_unavailable_and_never_rolls_or_shortens(
    session: date, decision: datetime, duration: timedelta
) -> None:
    source = series(session=session)
    resolved = resolve_future_observation(
        request(source, session=session, decision=decision, duration=duration), source
    )
    assert resolved.status is OutcomeResolutionStatus.SESSION_OVERFLOW
    assert not resolved.available
    assert resolved.requested_target_timestamp == decision + duration
    assert resolved.expected_observation_timestamp is None
    assert resolved.resolved_observation_timestamp is None
    assert resolved.observation is None


def test_missing_required_observation_never_uses_later_completed_bar() -> None:
    complete = series()
    expected = timestamp(11, 50)
    incomplete = series(
        bars=tuple(
            cast(IntradayBar, bar)
            for bar in complete.bars
            if bar.end_timestamp != expected
        )
    )
    resolved = resolve_future_observation(request(incomplete), incomplete)
    assert resolved.status is OutcomeResolutionStatus.MISSING_OBSERVATION
    assert resolved.expected_observation_timestamp == expected
    assert resolved.resolved_observation_timestamp is None
    assert resolved.observation is None


@pytest.mark.parametrize("empty", [False, True])
def test_end_of_available_data_is_explicit(empty: bool) -> None:
    complete = series()
    truncated = series(
        bars=()
        if empty
        else tuple(
            cast(IntradayBar, bar)
            for bar in complete.bars
            if bar.end_timestamp <= timestamp(11, 45)
        )
    )
    resolved = resolve_future_observation(request(truncated), truncated)
    assert resolved.status is OutcomeResolutionStatus.DATASET_END
    assert resolved.expected_observation_timestamp == timestamp(11, 50)
    assert not resolved.available


def test_developing_required_observation_is_incomplete_not_available() -> None:
    timeframe = replace(
        TIMEFRAME, developing_bar_exposure=DevelopingBarExposure.INCLUDE
    )
    complete = series(timeframe=timeframe)
    bars = tuple(
        cast(IntradayBar, bar)
        for bar in complete.bars
        if bar.end_timestamp <= timestamp(11, 45)
    )
    developing = _intraday_bar(
        timeframe, timestamp(11, 45), timestamp(11, 48), BarCompletion.DEVELOPING
    )
    source = series(timeframe=timeframe, bars=(*bars, developing))
    resolved = resolve_future_observation(request(source), source)
    assert resolved.status is OutcomeResolutionStatus.INCOMPLETE
    assert not resolved.available
    assert resolved.observation is None


@pytest.mark.parametrize("session", [SESSION, date(2024, 7, 3)])
def test_completed_terminal_partial_bar_is_a_valid_endpoint(session: date) -> None:
    source = series(
        session=session,
        timeframe=Timeframe.us_equity(IntradayInterval(timedelta(hours=4))),
    )
    close = timestamp(13 if session.day == 3 else 16, 0, session=session)
    resolved = resolve_future_observation(
        request(source, session=session, decision=close - timedelta(minutes=45)), source
    )
    assert resolved.requested_target_timestamp == close - timedelta(minutes=15)
    assert resolved.resolved_observation_timestamp == close
    assert resolved.observation is not None
    assert (
        resolved.observation.completion
        is BarCompletion.COMPLETED_PARTIAL_DURATION_TERMINAL
    )


def test_clock_anchored_leading_partial_uses_shared_canonical_windows() -> None:
    timeframe = Timeframe(
        IntradayInterval(timedelta(minutes=5), IntradayAnchor.CLOCK, time(9, 32))
    )
    source = series(timeframe=timeframe)
    resolved = resolve_future_observation(
        request(source, decision=timestamp(9, 30), duration=timedelta(minutes=1)),
        source,
    )
    assert resolved.resolved_observation_timestamp == timestamp(9, 32)
    assert resolved.observation is not None
    assert (
        resolved.observation.completion
        is BarCompletion.COMPLETED_PARTIAL_DURATION_LEADING
    )


def test_request_requires_the_exact_source_reference_and_session() -> None:
    source = series()
    evaluation = request(source)
    for changed in (
        replace(evaluation, source_reference=None),
        replace(
            evaluation,
            source_reference=replace(source.dataset_reference, dataset_id="other"),
        ),
        replace(
            evaluation,
            anchor=OutcomeAnchor(
                OutcomeAnchorKind.TIMESTAMP, date(2024, 7, 2), timestamp(11, 20)
            ),
        ),
        replace(
            evaluation,
            anchor=OutcomeAnchor(OutcomeAnchorKind.TIMESTAMP, SESSION, timestamp(8, 0)),
        ),
    ):
        with pytest.raises(OutcomeTemporalError):
            resolve_future_observation(changed, source)


def test_future_append_preserves_resolved_endpoint_and_configuration() -> None:
    source = series()
    evaluation = request(source)
    historical = resolve_future_observation(evaluation, source)
    appended = series(session=date(2024, 7, 2))
    combined = series(bars=cast(tuple[IntradayBar, ...], source.bars + appended.bars))
    # Same immutable reference: a projection with unrelated later observations.
    assert (
        resolve_future_observation(evaluation, combined).to_primitive()
        == historical.to_primitive()
    )
    assert (
        request(combined).temporal_configuration.configuration_id
        == evaluation.temporal_configuration.configuration_id
    )


def test_metadata_schema_remains_future_only_and_serializable() -> None:
    source = series()
    available = resolve_future_observation(request(source), source)
    overflow = resolve_future_observation(
        request(source, duration=timedelta(hours=24)), source
    )
    fields = outcome_resolution_fields()
    assert all(field.category is SchemaFieldCategory.FUTURE_OUTCOME for field in fields)
    for resolved in (available, overflow):
        metadata = resolved.metadata_primitive()
        assert set(metadata) == {field.name for field in fields}
        assert all(
            _schema_value_matches(field, metadata[field.name]) for field in fields
        )
        assert PrimitiveMappingSnapshot.capture(metadata).to_primitive() == metadata
        assert metadata["available"] is resolved.available
        assert (
            metadata["outcome_configuration_id"]
            == resolved.request.outcome_configuration_id
        )
        assert metadata["status"] == resolved.status.value
    assert overflow.metadata_primitive()["resolved_observation_timestamp"] is None


def test_resolution_cannot_claim_available_without_observation_or_shift_target() -> (
    None
):
    source = series()
    result = resolve_future_observation(request(source), source)
    with pytest.raises(OutcomeTemporalError):
        replace(result, observation=None)
    with pytest.raises(OutcomeTemporalError):
        replace(result, status=OutcomeResolutionStatus.DATASET_END)
    with pytest.raises(OutcomeTemporalError):
        replace(
            result,
            requested_target_timestamp=result.requested_target_timestamp
            + timedelta(minutes=5),
        )
    assert isinstance(result, OutcomeResolution)


def test_resolution_canonicalizes_equivalent_aware_boundary_representations() -> None:
    source = series()
    result = resolve_future_observation(request(source), source)
    assert result.expected_observation_timestamp is not None
    equivalent = replace(
        result,
        requested_target_timestamp=result.requested_target_timestamp.astimezone(
            NEW_YORK
        ),
        expected_observation_timestamp=result.expected_observation_timestamp.astimezone(
            NEW_YORK
        ),
    )
    assert equivalent.to_primitive() == result.to_primitive()
    with pytest.raises(OutcomeTemporalError):
        replace(
            result,
            requested_target_timestamp=result.requested_target_timestamp.replace(
                tzinfo=None
            ),
        )
