"""Only structurally immutable source graphs qualify for historical sharing."""

from copy import deepcopy
from dataclasses import FrozenInstanceError, dataclass, field, fields, replace
from datetime import datetime, timedelta, tzinfo
from typing import Any, cast

import pytest
from pandas import Timestamp  # pyright: ignore[reportMissingTypeStubs]

from quantforge.data import IntradayBar, TimeframeBarSeries
from quantforge.prediction.source_sharing import prediction_source_copy_memo
from tests.unit.prediction.test_prediction_window import WindowProvider


def source() -> TimeframeBarSeries:
    provider = WindowProvider()
    return next(
        item for item in provider.series if isinstance(item.bars[0], IntradayBar)
    )


def test_shared_graph_includes_bars_provenance_and_timeframe() -> None:
    original = source()
    memo = prediction_source_copy_memo(original)
    assert memo[id(original)] == original
    assert len(memo) <= 2
    clone = deepcopy(original, memo.copy())
    assert clone is memo[id(original)]
    assert deepcopy(clone, memo.copy()) is clone
    assert prediction_source_copy_memo(clone) == {id(clone): clone}
    bar = cast(IntradayBar, clone.bars[0])
    assert bar.provenance is cast(IntradayBar, original.bars[0]).provenance
    assert type(bar.start_timestamp) is datetime
    assert type(bar.end_timestamp) is datetime
    with pytest.raises(AttributeError):
        setattr(bar.end_timestamp, "notes", [])
    for target, attribute, replacement in (
        (clone, "bars", ()),
        (bar, "close", bar.close + 1),
        (bar.provenance, "provider_name", "changed"),
        (clone.dataset_reference, "dataset_id", "changed"),
        (clone.timeframe.session_policy, "calendar_name", "changed"),
        (bar.provenance.adjustment_basis, "ohlc_basis", "changed"),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(target, attribute, replacement)
    with pytest.raises(TypeError):
        cast(Any, clone.bars)[0] = bar
    # Serializers expose detached mutable values, never the backing records.
    primitive = bar.provenance.to_primitive()
    primitive["provider_name"] = "changed"
    assert bar.provenance.provider_name != "changed"


@dataclass(frozen=True, slots=True)
class AnnotatedBar(IntradayBar):
    notes: list[str] = field(default_factory=list[str])


class AnnotatedText(str):
    def __new__(cls, text: str) -> "AnnotatedText":
        instance = super().__new__(cls, text)
        instance.notes = []
        return instance

    notes: list[str]


@pytest.mark.parametrize("location", ["bar", "provenance", "source"])
def test_unknown_subclasses_with_mutable_state_keep_detached_copies(
    location: str,
) -> None:
    original = source()
    bar = cast(IntradayBar, original.bars[0])
    if location == "bar":
        annotated = AnnotatedBar(
            **{item.name: getattr(bar, item.name) for item in fields(bar)}
        )
        bars = (annotated, *original.bars[1:])
        original = TimeframeBarSeries._from_validated_artifact(  # pyright: ignore[reportPrivateUsage]
            original.dataset_reference, original.timeframe, bars
        )

        def notes(series: TimeframeBarSeries) -> list[str]:
            return cast(AnnotatedBar, series.bars[0]).notes
    elif location == "provenance":
        annotated_text = AnnotatedText(bar.provenance.provider_name)
        bars = (
            replace(
                bar, provenance=replace(bar.provenance, provider_name=annotated_text)
            ),
            *original.bars[1:],
        )
        original = TimeframeBarSeries._from_validated_artifact(  # pyright: ignore[reportPrivateUsage]
            original.dataset_reference, original.timeframe, bars
        )

        def notes(series: TimeframeBarSeries) -> list[str]:
            return cast(
                AnnotatedText,
                cast(IntradayBar, series.bars[0]).provenance.provider_name,
            ).notes
    else:

        @dataclass(frozen=True, slots=True)
        class AnnotatedSeries(TimeframeBarSeries):
            notes: list[str] = field(default_factory=list[str])

        original = AnnotatedSeries._from_validated_artifact(  # pyright: ignore[reportPrivateUsage]
            original.dataset_reference, original.timeframe, original.bars
        )
        object.__setattr__(original, "notes", [])

        def notes(series: TimeframeBarSeries) -> list[str]:
            return cast(AnnotatedSeries, series).notes

    memo = prediction_source_copy_memo(original)
    assert memo == {}
    first, second = deepcopy(original, memo.copy()), deepcopy(original, memo.copy())
    notes(first).append("decision A")
    assert notes(second) == notes(original) == []


def test_mutable_container_inside_frozen_record_is_not_shared() -> None:
    original = deepcopy(source())
    # Defend against runtime values violating annotations, even in exact types.
    object.__setattr__(original.dataset_reference, "dataset_id", ["mutable"])
    assert prediction_source_copy_memo(original) == {}
    clone = deepcopy(original)
    cast(Any, clone.dataset_reference.dataset_id).append("changed")
    assert original.dataset_reference.dataset_id == ["mutable"]


@pytest.mark.parametrize("change", ["attributes", "nanoseconds", "subclass"])
def test_timestamp_normalization_never_drops_state_or_precision(change: str) -> None:
    original = deepcopy(source())
    timestamp = cast(Any, Timestamp("2024-07-01T13:30:00+00:00"))
    if change == "attributes":
        timestamp.notes = []
    elif change == "nanoseconds":
        timestamp = Timestamp("2024-07-01T13:30:00.000000001+00:00")
    else:

        class CustomTimestamp(datetime):
            pass

        timestamp = CustomTimestamp.fromisoformat(timestamp.isoformat())
    object.__setattr__(original.bars[0], "start_timestamp", timestamp)
    assert prediction_source_copy_memo(original) == {}


def test_all_existing_artifact_timeframes_support_immutable_backing() -> None:
    for original in WindowProvider().series:
        memo = prediction_source_copy_memo(original)
        assert memo[id(original)] == original
        assert deepcopy(original, memo.copy()) is memo[id(original)]


def test_custom_timezone_and_absent_source_are_not_shared() -> None:
    class MutableTimezone(tzinfo):
        def __init__(self) -> None:
            self.offset = timedelta(0)

        def utcoffset(self, dt: datetime | None) -> timedelta:
            return self.offset

    original = deepcopy(source())
    bar = original.bars[0]
    object.__setattr__(
        bar, "end_timestamp", bar.end_timestamp.replace(tzinfo=MutableTimezone())
    )
    assert prediction_source_copy_memo(original) == {}
    assert prediction_source_copy_memo(None) == {}
