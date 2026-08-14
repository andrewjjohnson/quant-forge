from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import cast
from zoneinfo import ZoneInfo

import pytest

from quantforge.data.developing_bars import DevelopingBar
from quantforge.data.intraday import IntradayBar, IntradayBarProvenance
from quantforge.data.lineage import (
    AdjustmentBasis,
    DatasetFamilyReference,
    FeedScope,
    SourceConsistencyMode,
    SourceConsistencyValidation,
)
from quantforge.data.models import AdjustmentMode
from quantforge.data.multi_timeframe import (
    ContextAvailability,
    ContextBar,
    ContextCompletionPolicy,
    ContextTimeframeRequirement,
    MultiTimeframeContext,
    TimeframeContext,
)
from quantforge.data.session_aggregation import AggregatedSessionBar
from quantforge.indicators import (
    BOLLINGER_BANDWIDTH_OUTPUT,
    BOLLINGER_LOWER_BAND_OUTPUT,
    BOLLINGER_MIDDLE_BAND_OUTPUT,
    BOLLINGER_UPPER_BAND_OUTPUT,
    EXPONENTIAL_MOVING_AVERAGE_OUTPUT,
    SIMPLE_MOVING_AVERAGE_OUTPUT,
    TALIB_INDICATOR_BACKEND,
    BollingerBands,
    BollingerBandsParameters,
    ConfiguredTimeframeIndicator,
    DevelopingBarSupport,
    ExponentialMovingAverage,
    ExponentialMovingAverageParameters,
    IndicatorSourceError,
    MarketField,
    SimpleMovingAverage,
    SimpleMovingAverageParameters,
    TimeframeNeutralIndicator,
    UnsupportedDevelopingBarError,
    WilderAverageTrueRange,
    WilderAverageTrueRangeParameters,
    WilderDirectionalMovement,
    WilderDirectionalMovementParameters,
    WilderRelativeStrengthIndex,
    WilderRelativeStrengthIndexParameters,
    bind_indicator,
    evaluate_indicator,
)
from quantforge.timeframes import (
    BarCompletion,
    IntradayInterval,
    SessionInterval,
    Timeframe,
    TradingWeekInterval,
    resolve_exchange_session,
    resolve_trading_week,
)

from ..helpers import make_dataset

NEW_YORK = ZoneInfo("America/New_York")
RETRIEVED_AT = datetime(2025, 1, 1, tzinfo=UTC)
FAMILY_ID = "fixture-family"
SOURCE_ID = "fixture-source"
CLOSES = ("10", "11", "10", "12", "13", "12", "14")
SESSIONS = (
    date(2024, 7, 1),
    date(2024, 7, 2),
    date(2024, 7, 5),
    date(2024, 7, 8),
    date(2024, 7, 9),
    date(2024, 7, 10),
    date(2024, 7, 11),
)
WEEKS = tuple(date(2024, 6, 3) + timedelta(days=7 * index) for index in range(7))


def _timeframes() -> tuple[Timeframe, Timeframe, Timeframe, Timeframe]:
    return (
        Timeframe.us_equity(IntradayInterval(timedelta(minutes=5))),
        Timeframe.us_equity(IntradayInterval(timedelta(hours=4))),
        Timeframe.us_equity(SessionInterval()),
        Timeframe.us_equity(TradingWeekInterval()),
    )


def _adjustment_basis() -> AdjustmentBasis:
    return AdjustmentBasis(
        adjustment_mode=AdjustmentMode.UNADJUSTED,
        ohlc_basis="raw_provider",
        volume_basis="raw_provider",
        corporate_action_policy="separate_provider_actions",
        adjusted_fields_used=False,
    )


def _provenance() -> IntradayBarProvenance:
    return IntradayBarProvenance(
        provider_name="fixture-provider",
        provider_symbol="SPY",
        adapter_version="fixture-1",
        retrieved_at=RETRIEVED_AT,
        source_request_id="fixture-request",
        source_snapshot_id=SOURCE_ID,
        feed_scope=FeedScope.consolidated(),
        adjustment_basis=_adjustment_basis(),
    )


def _intraday_bars(
    timeframe: Timeframe,
    *,
    closes: tuple[str, ...] = CLOSES,
    sessions: tuple[date, ...] = SESSIONS,
) -> tuple[IntradayBar, ...]:
    duration = cast(IntradayInterval, timeframe.interval).nominal_duration
    return tuple(
        IntradayBar(
            symbol="SPY",
            session_date=session_date,
            start_timestamp=(
                session := resolve_exchange_session(session_date)
            ).open_timestamp,
            end_timestamp=session.open_timestamp + duration,
            timeframe=timeframe,
            completion=BarCompletion.COMPLETED,
            open=(close := Decimal(close_text)),
            high=close + Decimal(1),
            low=close,
            close=close,
            volume=Decimal(1000 + index),
            provenance=_provenance(),
        )
        for index, (session_date, close_text) in enumerate(
            zip(sessions, closes, strict=True)
        )
    )


def _session_bars(
    timeframe: Timeframe,
    *,
    closes: tuple[str, ...] = CLOSES,
) -> tuple[AggregatedSessionBar, ...]:
    all_period_starts = (
        SESSIONS if isinstance(timeframe.interval, SessionInterval) else WEEKS
    )
    period_starts = all_period_starts[: len(closes)]
    bars: list[AggregatedSessionBar] = []
    for index, (period_start, close_text) in enumerate(
        zip(period_starts, closes, strict=True)
    ):
        sessions = (
            (resolve_exchange_session(period_start),)
            if isinstance(timeframe.interval, SessionInterval)
            else resolve_trading_week(period_start).sessions
        )
        close = Decimal(close_text)
        bars.append(
            AggregatedSessionBar(
                symbol="SPY",
                timeframe=timeframe,
                period_start_date=period_start,
                session_dates=tuple(item.session_date for item in sessions),
                start_timestamp=sessions[0].open_timestamp,
                end_timestamp=sessions[-1].close_timestamp,
                open=close,
                high=close + Decimal(1),
                low=close,
                close=close,
                volume=Decimal(1000 + index),
                source_bar_ids=tuple(
                    f"source-{period_start.isoformat()}-{source_index}"
                    for source_index, _ in enumerate(sessions)
                ),
                source_dataset_id=SOURCE_ID,
            )
        )
    return tuple(bars)


def _reference(
    timeframe: Timeframe, *, family_id: str = FAMILY_ID
) -> DatasetFamilyReference:
    dataset_id = (
        SOURCE_ID
        if timeframe == _timeframes()[0]
        else f"derived-{timeframe.configuration_id}"
    )
    return DatasetFamilyReference(
        family_id,
        dataset_id,
        SOURCE_ID,
        timeframe.configuration_id,
    )


def _context(
    bars_by_timeframe: dict[str, tuple[ContextBar, ...]],
    *,
    primary_timeframe: Timeframe | None = None,
    completion_policy: ContextCompletionPolicy = (
        ContextCompletionPolicy.COMPLETED_BARS_ONLY
    ),
    as_of: datetime | None = None,
    family_id: str = FAMILY_ID,
) -> MultiTimeframeContext:
    five_minute, _, _, _ = _timeframes()
    primary = primary_timeframe or five_minute
    timeframes = tuple(
        sorted(
            (
                timeframe
                for timeframe in _timeframes()
                if timeframe.configuration_id in bars_by_timeframe
                and timeframe != primary
            ),
            key=lambda item: item.configuration_id,
        )
    )
    required = tuple(ContextTimeframeRequirement(timeframe) for timeframe in timeframes)
    all_bars = tuple(
        bar for timeframe_bars in bars_by_timeframe.values() for bar in timeframe_bars
    )
    decision_timestamp = as_of or max(
        bar.end_timestamp for bar in all_bars
    ) + timedelta(days=1)
    aligned: list[TimeframeContext] = []
    for requirement in (ContextTimeframeRequirement(primary), *required):
        timeframe_bars = bars_by_timeframe[requirement.timeframe.configuration_id]
        completed = tuple(
            bar
            for bar in timeframe_bars
            if bar.completion is not BarCompletion.DEVELOPING
        )
        latest_completed = None if not completed else completed[-1].end_timestamp
        latest = timeframe_bars[-1]
        aligned.append(
            TimeframeContext._from_aligned_series(  # pyright: ignore[reportPrivateUsage]
                requirement=requirement,
                dataset_reference=_reference(
                    requirement.timeframe, family_id=family_id
                ),
                availability=ContextAvailability.AVAILABLE,
                bars=timeframe_bars,
                latest_completed_bar_timestamp=latest_completed,
                age=decision_timestamp - latest.end_timestamp,
            )
        )
    return MultiTimeframeContext._from_aligned_timeframes(  # pyright: ignore[reportPrivateUsage]
        as_of=decision_timestamp,
        primary_timeframe=primary,
        required_timeframes=required,
        completion_policy=completion_policy,
        source_consistency=SourceConsistencyValidation(
            SourceConsistencyMode.COMMON_DATASET_FAMILY,
            family_id,
            None,
        ),
        timeframes=tuple(aligned),
    )


def _all_completed_context(
    *, closes: tuple[str, ...] = CLOSES, family_id: str = FAMILY_ID
) -> MultiTimeframeContext:
    five_minute, four_hour, daily, weekly = _timeframes()
    return _context(
        {
            five_minute.configuration_id: _intraday_bars(
                five_minute, closes=closes, sessions=SESSIONS[: len(closes)]
            ),
            four_hour.configuration_id: _intraday_bars(
                four_hour, closes=closes, sessions=SESSIONS[: len(closes)]
            ),
            daily.configuration_id: _session_bars(daily, closes=closes),
            weekly.configuration_id: _session_bars(weekly, closes=closes),
        },
        family_id=family_id,
    )


@pytest.mark.parametrize(
    "indicator",
    [
        BollingerBands(BollingerBandsParameters(2)),
        ExponentialMovingAverage(ExponentialMovingAverageParameters(2)),
        SimpleMovingAverage(SimpleMovingAverageParameters(2)),
        WilderRelativeStrengthIndex(WilderRelativeStrengthIndexParameters(2)),
        WilderAverageTrueRange(WilderAverageTrueRangeParameters(2)),
        WilderDirectionalMovement(WilderDirectionalMovementParameters(2)),
    ],
)
def test_indicator_contracts_run_on_5m_4h_daily_and_weekly_bars(
    indicator: TimeframeNeutralIndicator,
) -> None:
    context = _all_completed_context()

    for timeframe in _timeframes():
        output = evaluate_indicator(indicator, context, timeframe)
        assert len(output.bar_ids) == len(CLOSES)
        assert tuple(field.name for field in output.fields) == indicator.output_fields
        assert output.source_timeframe == timeframe
        assert output.warm_up_bars == indicator.warm_up_observations
        assert output.developing_bar_support is DevelopingBarSupport.DEVELOPING_AS_OF


def test_ema_values_are_identical_on_intraday_daily_and_weekly_fixtures() -> None:
    closes = ("10", "11", "12", "13", "14")
    context = _all_completed_context(closes=closes)
    indicator = ExponentialMovingAverage(ExponentialMovingAverageParameters(3))
    expected = (None, None, Decimal(11), Decimal(12), Decimal(13))

    for timeframe in _timeframes():
        output = evaluate_indicator(indicator, context, timeframe)
        assert output.values_for(EXPONENTIAL_MOVING_AVERAGE_OUTPUT) == expected


def test_bollinger_values_are_identical_on_intraday_daily_and_weekly_fixtures() -> None:
    context = _all_completed_context(closes=("10", "12", "14"))
    indicator = BollingerBands(BollingerBandsParameters(2, Decimal(2)))
    expected = {
        BOLLINGER_MIDDLE_BAND_OUTPUT: (None, Decimal(11), Decimal(13)),
        BOLLINGER_UPPER_BAND_OUTPUT: (None, Decimal(13), Decimal(15)),
        BOLLINGER_LOWER_BAND_OUTPUT: (None, Decimal(9), Decimal(11)),
        BOLLINGER_BANDWIDTH_OUTPUT: (
            None,
            Decimal("0.3636363636363636363636363636363636"),
            Decimal("0.3076923076923076923076923076923077"),
        ),
    }

    for timeframe in _timeframes():
        output = evaluate_indicator(indicator, context, timeframe)
        assert output.aggregation_provenance == _reference(timeframe).to_primitive()
        for field_name, values in expected.items():
            assert output.values_for(field_name) == values


def test_daily_generic_evaluation_preserves_existing_numerical_results() -> None:
    daily = _timeframes()[2]
    indicator = WilderDirectionalMovement(WilderDirectionalMovementParameters(2))
    generic = evaluate_indicator(indicator, _all_completed_context(), daily)
    legacy = indicator.calculate(make_dataset(CLOSES, sessions=SESSIONS))

    for field_name in indicator.output_fields:
        assert generic.values_for(field_name) == legacy.values_for(field_name)


def test_configuration_identity_binds_timeframe_policy_fields_and_lineage() -> None:
    context = _all_completed_context()
    _, four_hour, daily, _ = _timeframes()
    indicator = SimpleMovingAverage(SimpleMovingAverageParameters(2))
    open_indicator = SimpleMovingAverage(
        SimpleMovingAverageParameters(2, source_field=MarketField.OPEN)
    )
    four_hour_bound = bind_indicator(indicator, context, four_hour)
    daily_bound = bind_indicator(indicator, context, daily)
    developing_context = _context(
        {
            timeframe.configuration_id: context.bars_for(timeframe)
            for timeframe in _timeframes()
        },
        completion_policy=ContextCompletionPolicy.DEVELOPING_BAR_AS_OF,
    )
    developing_bound = bind_indicator(indicator, developing_context, four_hour)
    open_bound = bind_indicator(open_indicator, context, four_hour)
    other_family_bound = bind_indicator(
        indicator,
        _all_completed_context(family_id="other-aggregation-family"),
        four_hour,
    )

    assert four_hour_bound.configuration_id != daily_bound.configuration_id
    assert four_hour_bound.configuration_id != developing_bound.configuration_id
    assert four_hour_bound.configuration_id != open_bound.configuration_id
    assert four_hour_bound.configuration_id != other_family_bound.configuration_id
    source = cast(dict[str, object], four_hour_bound.configuration()["source"])
    assert source["fields"] == ["close"]
    assert source["observation_unit"] == "bar"
    assert source["warm_up_bars"] == 2
    assert source["aggregation_provenance"] == _reference(four_hour).to_primitive()


def test_ema_identity_binds_period_field_timeframe_and_completion_policy() -> None:
    context = _all_completed_context()
    _, four_hour, daily, _ = _timeframes()
    close_two = ExponentialMovingAverage(ExponentialMovingAverageParameters(2))
    close_three = ExponentialMovingAverage(ExponentialMovingAverageParameters(3))
    open_two = ExponentialMovingAverage(
        ExponentialMovingAverageParameters(2, source_field=MarketField.OPEN)
    )
    developing_context = _context(
        {
            timeframe.configuration_id: context.bars_for(timeframe)
            for timeframe in _timeframes()
        },
        completion_policy=ContextCompletionPolicy.DEVELOPING_BAR_AS_OF,
    )

    identities = {
        bind_indicator(close_two, context, four_hour).configuration_id,
        bind_indicator(close_three, context, four_hour).configuration_id,
        bind_indicator(open_two, context, four_hour).configuration_id,
        bind_indicator(close_two, context, daily).configuration_id,
        bind_indicator(close_two, developing_context, four_hour).configuration_id,
    }

    assert len(identities) == 5


def test_talib_ema_preserves_timeframe_completion_and_lineage_metadata() -> None:
    context = _all_completed_context()
    _, four_hour, _, _ = _timeframes()
    indicator = ExponentialMovingAverage(
        ExponentialMovingAverageParameters(3),
        backend_id=TALIB_INDICATOR_BACKEND,
    )

    configured = bind_indicator(indicator, context, four_hour)
    output = configured.calculate(context)

    assert output.backend_identity == indicator.backend_identity
    assert output.source_timeframe == four_hour
    assert output.source_fields == (MarketField.CLOSE,)
    assert output.completion_policy is ContextCompletionPolicy.COMPLETED_BARS_ONLY
    assert output.dataset_reference == _reference(four_hour)
    assert output.aggregation_provenance == _reference(four_hour).to_primitive()
    assert output.configuration_id == configured.configuration_id
    assert output.values_for(EXPONENTIAL_MOVING_AVERAGE_OUTPUT) == (
        None,
        None,
        Decimal("10.333333333333334"),
        Decimal("11.166666666666668"),
        Decimal("12.083333333333334"),
        Decimal("12.041666666666668"),
        Decimal("13.020833333333334"),
    )


def test_bollinger_identity_binds_all_formula_and_source_parameters() -> None:
    context = _all_completed_context()
    _, four_hour, daily, _ = _timeframes()
    close_two = BollingerBands(BollingerBandsParameters(2, Decimal(2)))
    close_three = BollingerBands(BollingerBandsParameters(3, Decimal(2)))
    close_wider = BollingerBands(BollingerBandsParameters(2, Decimal(3)))
    open_two = BollingerBands(BollingerBandsParameters(2, Decimal(2), MarketField.OPEN))
    developing_context = _context(
        {
            timeframe.configuration_id: context.bars_for(timeframe)
            for timeframe in _timeframes()
        },
        completion_policy=ContextCompletionPolicy.DEVELOPING_BAR_AS_OF,
    )

    identities = {
        bind_indicator(close_two, context, four_hour).configuration_id,
        bind_indicator(close_three, context, four_hour).configuration_id,
        bind_indicator(close_wider, context, four_hour).configuration_id,
        bind_indicator(open_two, context, four_hour).configuration_id,
        bind_indicator(close_two, context, daily).configuration_id,
        bind_indicator(close_two, developing_context, four_hour).configuration_id,
    }

    assert len(identities) == 6


def test_four_hour_instance_rejects_a_daily_only_context() -> None:
    context = _all_completed_context()
    _, four_hour, daily, _ = _timeframes()
    configured = bind_indicator(
        SimpleMovingAverage(SimpleMovingAverageParameters(2)), context, four_hour
    )
    daily_only = _context(
        {daily.configuration_id: context.bars_for(daily)},
        primary_timeframe=daily,
    )

    with pytest.raises(IndicatorSourceError, match="not declared"):
        configured.calculate(daily_only)


@pytest.mark.parametrize(
    "indicator",
    [
        BollingerBands(BollingerBandsParameters(2)),
        ExponentialMovingAverage(ExponentialMovingAverageParameters(2)),
        WilderDirectionalMovement(WilderDirectionalMovementParameters(2)),
    ],
)
def test_appending_future_bars_does_not_change_historical_indicator_values(
    indicator: TimeframeNeutralIndicator,
) -> None:
    _, four_hour, _, _ = _timeframes()
    cutoff_context = _all_completed_context(closes=CLOSES[:5])
    extended_context = _all_completed_context()

    cutoff = evaluate_indicator(indicator, cutoff_context, four_hour)
    extended = evaluate_indicator(indicator, extended_context, four_hour)

    assert extended.bar_end_timestamps[:5] == cutoff.bar_end_timestamps
    for field_name in indicator.output_fields:
        assert extended.values_for(field_name)[:5] == cutoff.values_for(field_name)


def _developing_context() -> MultiTimeframeContext:
    five_minute, four_hour, _, _ = _timeframes()
    session = resolve_exchange_session(date(2024, 7, 9))
    as_of = datetime(2024, 7, 9, 10, tzinfo=NEW_YORK).astimezone(UTC)
    source_bars = tuple(
        IntradayBar(
            symbol="SPY",
            session_date=session.session_date,
            start_timestamp=session.open_timestamp + timedelta(minutes=5 * index),
            end_timestamp=session.open_timestamp + timedelta(minutes=5 * (index + 1)),
            timeframe=five_minute,
            completion=BarCompletion.COMPLETED,
            open=Decimal(10 + index),
            high=Decimal(11 + index),
            low=Decimal(10 + index),
            close=Decimal(10 + index),
            volume=Decimal(1000),
            provenance=_provenance(),
        )
        for index in range(6)
    )
    completed = _intraday_bars(
        four_hour,
        closes=CLOSES[:4],
        sessions=SESSIONS[:4],
    )
    developing = DevelopingBar(
        symbol="SPY",
        timeframe=four_hour,
        period_start_date=session.session_date,
        session_dates=(session.session_date,),
        as_of=as_of,
        observed_start_timestamp=session.open_timestamp,
        observed_end_timestamp=as_of,
        expected_completion_boundary=session.open_timestamp + timedelta(hours=4),
        open=Decimal(10),
        high=Decimal(16),
        low=Decimal(10),
        close=Decimal(15),
        volume=Decimal(6000),
        source_bar_ids=tuple(bar.bar_id for bar in source_bars),
        source_dataset_reference=_reference(five_minute),
        source_timeframe=five_minute,
    )
    return _context(
        {
            five_minute.configuration_id: source_bars,
            four_hour.configuration_id: (*completed, developing),
        },
        completion_policy=ContextCompletionPolicy.DEVELOPING_BAR_AS_OF,
        as_of=as_of,
    )


def test_developing_bar_evaluation_is_explicit_and_causal() -> None:
    _, four_hour, _, _ = _timeframes()
    context = _developing_context()
    indicator = SimpleMovingAverage(SimpleMovingAverageParameters(2))

    output = evaluate_indicator(indicator, context, four_hour)

    assert output.completion_policy is ContextCompletionPolicy.DEVELOPING_BAR_AS_OF
    assert output.completion_states[-1] is BarCompletion.DEVELOPING
    assert output.values_for(SIMPLE_MOVING_AVERAGE_OUTPUT)[-1] == Decimal("13.5")


def test_ema_developing_bar_uses_only_the_causal_as_of_close() -> None:
    _, four_hour, _, _ = _timeframes()
    output = evaluate_indicator(
        ExponentialMovingAverage(ExponentialMovingAverageParameters(1)),
        _developing_context(),
        four_hour,
    )

    assert output.completion_states[-1] is BarCompletion.DEVELOPING
    assert output.values_for(EXPONENTIAL_MOVING_AVERAGE_OUTPUT)[-1] == Decimal(15)


def test_bollinger_developing_bar_uses_only_the_causal_as_of_close() -> None:
    _, four_hour, _, _ = _timeframes()
    output = evaluate_indicator(
        BollingerBands(BollingerBandsParameters(1)),
        _developing_context(),
        four_hour,
    )

    assert output.completion_policy is ContextCompletionPolicy.DEVELOPING_BAR_AS_OF
    assert output.completion_states[-1] is BarCompletion.DEVELOPING
    assert output.values_for(BOLLINGER_MIDDLE_BAND_OUTPUT)[-1] == Decimal(15)
    assert output.values_for(BOLLINGER_UPPER_BAND_OUTPUT)[-1] == Decimal(15)
    assert output.values_for(BOLLINGER_LOWER_BAND_OUTPUT)[-1] == Decimal(15)
    assert output.values_for(BOLLINGER_BANDWIDTH_OUTPUT)[-1] == Decimal(0)


def test_completed_only_indicator_rejects_developing_bar() -> None:
    class CompletedOnlySimpleMovingAverage(SimpleMovingAverage):
        developing_bar_support = DevelopingBarSupport.COMPLETED_ONLY

    _, four_hour, _, _ = _timeframes()
    context = _developing_context()
    configured: ConfiguredTimeframeIndicator = bind_indicator(
        CompletedOnlySimpleMovingAverage(SimpleMovingAverageParameters(2)),
        context,
        four_hour,
    )

    with pytest.raises(UnsupportedDevelopingBarError, match="explicit context"):
        configured.calculate(context)
