"""Narrow copy boundary for already validated historical outcome sources."""

from dataclasses import fields
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from typing import cast
from zoneinfo import ZoneInfo

from pandas import Timestamp  # pyright: ignore[reportMissingTypeStubs]

from quantforge.data.intraday import IntradayBar, IntradayBarProvenance
from quantforge.data.intraday_validation import IntradayCoverageInterval
from quantforge.data.lineage import (
    AdjustmentBasis,
    DatasetFamilyReference,
    FeedCoverage,
    FeedScope,
)
from quantforge.data.models import AdjustmentMode
from quantforge.data.multi_timeframe import (
    TimeframeBarSeries,
    _DevelopingSourceEvidence,  # pyright: ignore[reportPrivateUsage]
)
from quantforge.data.session_aggregation import AggregatedSessionBar
from quantforge.timeframes import (
    BarCompletion,
    BarLabel,
    CrossSessionPolicy,
    DevelopingBarExposure,
    ExchangeSessionPolicy,
    IntradayAnchor,
    IntradayInterval,
    SessionInterval,
    SessionScope,
    Timeframe,
    TradingWeekInterval,
)

# Exact types, deliberately excluding subclasses with extra mutable state. These
# records are frozen and slotted. Inspect their actual fields too: annotations
# alone cannot exclude mutable containers or subclass instances supplied by a
# custom study. New source record types must be reviewed before sharing them.
_SOURCE_RECORD_TYPES = frozenset(
    {
        TimeframeBarSeries,
        IntradayBar,
        AggregatedSessionBar,
        IntradayBarProvenance,
        DatasetFamilyReference,
        FeedScope,
        AdjustmentBasis,
        Timeframe,
        ExchangeSessionPolicy,
        IntradayInterval,
        SessionInterval,
        TradingWeekInterval,
        _DevelopingSourceEvidence,
        IntradayCoverageInterval,
    }
)
_SOURCE_SCALAR_TYPES = frozenset(
    {
        type(None),
        str,
        bool,
        int,
        Decimal,
        date,
        timedelta,
        FeedCoverage,
        AdjustmentMode,
        BarCompletion,
        BarLabel,
        CrossSessionPolicy,
        DevelopingBarExposure,
        IntradayAnchor,
        SessionScope,
    }
)


def prediction_source_copy_memo(
    source: TimeframeBarSeries | None,
) -> dict[int, object]:
    """Verify deep immutability once, then share only this validated source root.

    This checks copy safety, not scientific validity. The existing artifact
    constructors and QF-11 source/ancestry checks remain authoritative. Unknown
    or mutable graphs retain ordinary deepcopy semantics. Never use this memo
    for QF-11's detached bounded labeler inputs: their mutation guards still need
    independent copies.

    Pass a fresh copy of the returned memo to each study deepcopy. Reusing a
    populated memo would also share previously copied mutable component state.
    Only the root is retained, so per-decision memo size is independent of bars.
    """
    if source is None or type(source) is not TimeframeBarSeries:
        return {}
    shared = _immutable_source_value(source, {})
    if shared is _MUTABLE:
        return {}
    # A calendar Timestamp needs one immutable backing copy. Both the caller's
    # source and the template's normalized source must resolve to that backing.
    return {id(source): shared, id(shared): shared}


_MUTABLE = object()


def _immutable_source_value(value: object, verified: dict[int, object]) -> object:
    value_type = type(value)
    if value_type in _SOURCE_SCALAR_TYPES:
        return value
    if value_type in (datetime, time):
        # datetime/time can otherwise enclose arbitrary mutable tzinfo objects.
        return value if _immutable_timezone(cast(datetime | time, value)) else _MUTABLE
    if id(value) in verified:
        return verified[id(value)]
    # A cycle cannot belong to the immutable source contract. Keep the sentinel
    # while visiting, and memoize completed values to avoid traversing common
    # provenance/timeframe subgraphs once per bar.
    verified[id(value)] = _MUTABLE
    if value_type is Timestamp:
        timestamp = cast(datetime, value)
        # exchange-calendars supplies this datetime subclass. Its numerical value
        # is immutable, but its __dict__ is not. Do not discard attached state or
        # silently round nanoseconds when creating the immutable backing.
        if vars(value) or not _immutable_timezone(timestamp):
            return _MUTABLE
        normalized = datetime(
            timestamp.year,
            timestamp.month,
            timestamp.day,
            timestamp.hour,
            timestamp.minute,
            timestamp.second,
            timestamp.microsecond,
            timestamp.tzinfo,
            fold=timestamp.fold,
        )
        if normalized != timestamp or normalized.isoformat() != timestamp.isoformat():
            return _MUTABLE
        result = normalized
    elif value_type is tuple:
        original = cast(tuple[object, ...], value)
        items = tuple(_immutable_source_value(item, verified) for item in original)
        if any(item is _MUTABLE for item in items):
            return _MUTABLE
        result = (
            value
            if all(a is b for a, b in zip(original, items, strict=True))
            else items
        )
    elif value_type in _SOURCE_RECORD_TYPES:
        record_fields = fields(value_type)
        original = tuple(getattr(value, item.name) for item in record_fields)
        items = tuple(_immutable_source_value(item, verified) for item in original)
        if any(item is _MUTABLE for item in items):
            return _MUTABLE
        result = value
        if any(a is not b for a, b in zip(original, items, strict=True)):
            # Structural copy of an already validated record; no constructors,
            # re-sorting, provenance reconstruction, or scientific rehashing.
            result = object.__new__(value_type)
            for item, child in zip(record_fields, items, strict=True):
                object.__setattr__(result, item.name, child)
    else:
        return _MUTABLE
    verified[id(value)] = result
    return result


def _immutable_timezone(value: datetime | time) -> bool:
    return type(value.tzinfo) in (type(None), timezone, ZoneInfo)
