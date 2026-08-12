"""Deterministic exchange-session daily and trading-week OHLCV aggregation."""

from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import cast

from quantforge.configuration import (
    PrimitiveMapping,
    configuration_identity,
    decimal_to_primitive,
)
from quantforge.data.exceptions import ValidationError
from quantforge.data.identity import canonical_json_bytes, sha256_hex
from quantforge.data.intraday import IntradayBar, IntradayBarBatch
from quantforge.data.intraday_aggregation import MissingConstituentPolicy
from quantforge.data.intraday_ingestion import IntradayDataset
from quantforge.data.intraday_validation import (
    IntradayCoverageInterval,
    IntradayCoverageReport,
    IntradayCoverageStatus,
    IntradayValidationMode,
    validate_intraday_coverage,
)
from quantforge.data.lineage import (
    AdjustmentBasis,
    AggregationPolicy,
    DatasetFamily,
    DatasetLineage,
    FeedScope,
)
from quantforge.timeframes import (
    BarCompletion,
    CrossSessionPolicy,
    DevelopingBarExposure,
    IntradayInterval,
    SessionInterval,
    Timeframe,
    TradingWeekInterval,
    resolve_exchange_session,
    resolve_trading_week,
)

SESSION_AGGREGATION_SCHEMA_VERSION = "1"
SESSION_AGGREGATION_POLICY_VERSION = "1"
_POLICY_NAME = "quantforge_exchange_session_ohlcv"
_PRODUCER_NAME = "quantforge"


class SessionAggregationValidationError(ValidationError):
    """Source data or target semantics cannot produce trusted session bars."""


class SessionAggregationQualityError(SessionAggregationValidationError):
    """Strict aggregation found incomplete source constituents."""

    def __init__(self, report: "SessionAggregationReport") -> None:
        super().__init__(
            "session aggregation failed strict constituent validation: "
            f"{report.source_missing_interval_count} missing and "
            f"{report.source_unexpected_interval_count} unexpected source intervals",
            missing_sessions=tuple(
                value.isoformat() for value in report.incomplete_sessions
            ),
        )
        self.report = report


@dataclass(frozen=True, slots=True)
class SessionAggregationPolicy:
    """Complete identity-bearing policy for daily and weekly construction."""

    missing_constituent_policy: MissingConstituentPolicy = (
        MissingConstituentPolicy.STRICT
    )
    schema_version: str = SESSION_AGGREGATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        missing_policy = cast(object, self.missing_constituent_policy)
        if not isinstance(missing_policy, MissingConstituentPolicy):
            raise SessionAggregationValidationError(
                "missing-constituent policy is invalid"
            )
        if self.schema_version != SESSION_AGGREGATION_SCHEMA_VERSION:
            raise SessionAggregationValidationError(
                f"session aggregation schema {SESSION_AGGREGATION_SCHEMA_VERSION} "
                "is required"
            )

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "schema_version": self.schema_version,
            "missing_constituent_policy": self.missing_constituent_policy.value,
            "source_requirement": "validated_canonical_intraday_dataset",
            "source_cross_session_policy": "prohibited",
            "daily_boundary": "one_complete_exchange_session",
            "weekly_boundary": "monday_sunday_exchange_trading_week",
            "partial_source_range_policy": "exclude_incomplete_target_period",
            "unexpected_source_interval_policy": "reject_or_disclose_and_exclude",
            "provider_native_eod_policy": "prohibited",
            "ohlcv": {
                "open": "first_open",
                "high": "maximum_high",
                "low": "minimum_low",
                "close": "final_close",
                "volume": "sum",
            },
        }

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.to_primitive())

    def to_lineage_policy(self) -> AggregationPolicy:
        return AggregationPolicy(
            _POLICY_NAME,
            SESSION_AGGREGATION_POLICY_VERSION,
            self.to_primitive(),
        )


@dataclass(frozen=True, slots=True)
class AggregatedSessionBar:
    """One completed exchange-session daily or exchange-week OHLCV bar."""

    symbol: str
    timeframe: Timeframe
    period_start_date: date
    session_dates: tuple[date, ...]
    start_timestamp: datetime
    end_timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    source_bar_ids: tuple[str, ...]
    source_dataset_id: str
    producer_name: str = _PRODUCER_NAME
    completion: BarCompletion = BarCompletion.COMPLETED
    schema_version: str = SESSION_AGGREGATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        interval = self.timeframe.interval
        if not isinstance(interval, (SessionInterval, TradingWeekInterval)):
            raise SessionAggregationValidationError(
                "aggregated session bar requires a daily or weekly timeframe"
            )
        if self.timeframe.developing_bar_exposure is not DevelopingBarExposure.EXCLUDE:
            raise SessionAggregationValidationError(
                "aggregated session bars must exclude developing values"
            )
        if not self.session_dates or self.session_dates != tuple(
            sorted(set(self.session_dates))
        ):
            raise SessionAggregationValidationError(
                "aggregated session dates must be sorted and unique"
            )
        if self.period_start_date > self.session_dates[0]:
            raise SessionAggregationValidationError(
                "period start cannot follow its first exchange session"
            )
        if isinstance(interval, SessionInterval):
            expected_sessions = (
                resolve_exchange_session(
                    self.period_start_date, self.timeframe.session_policy
                ),
            )
        else:
            trading_week = resolve_trading_week(
                self.period_start_date, self.timeframe.session_policy
            )
            expected_sessions = trading_week.sessions
            if self.period_start_date != trading_week.week_start:
                raise SessionAggregationValidationError(
                    "weekly period start must be its exchange-week Monday"
                )
        if self.session_dates != tuple(
            session.session_date for session in expected_sessions
        ):
            raise SessionAggregationValidationError(
                "bar sessions do not exactly match the target exchange period"
            )
        start = _utc_timestamp(self.start_timestamp, "bar start")
        end = _utc_timestamp(self.end_timestamp, "bar end")
        if start >= end:
            raise SessionAggregationValidationError("bar start must precede bar end")
        if (
            start != expected_sessions[0].open_timestamp
            or end != expected_sessions[-1].close_timestamp
        ):
            raise SessionAggregationValidationError(
                "bar boundaries must use actual exchange opens and closes"
            )
        if self.completion is not BarCompletion.COMPLETED:
            raise SessionAggregationValidationError(
                "daily and weekly aggregate bars must be completed"
            )
        if self.producer_name != _PRODUCER_NAME:
            raise SessionAggregationValidationError(
                "provider-native EOD bars cannot be session aggregates"
            )
        if not self.source_dataset_id:
            raise SessionAggregationValidationError("source dataset ID is required")
        if not self.symbol.strip():
            raise SessionAggregationValidationError("bar symbol is required")
        if not self.source_bar_ids or len(set(self.source_bar_ids)) != len(
            self.source_bar_ids
        ):
            raise SessionAggregationValidationError(
                "source bar IDs must be nonempty and unique"
            )
        prices = tuple(
            cast(object, value)
            for value in (self.open, self.high, self.low, self.close)
        )
        if any(
            not isinstance(value, Decimal) or not value.is_finite() for value in prices
        ):
            raise SessionAggregationValidationError(
                "aggregated OHLC must use finite Decimal values"
            )
        volume = cast(object, self.volume)
        if not isinstance(volume, Decimal) or not volume.is_finite():
            raise SessionAggregationValidationError(
                "aggregated volume must be a finite Decimal"
            )
        if any(cast(Decimal, value) <= 0 for value in prices) or volume < 0:
            raise SessionAggregationValidationError(
                "aggregated prices must be positive and volume nonnegative"
            )
        if self.high < max(self.open, self.low, self.close) or self.low > min(
            self.open, self.high, self.close
        ):
            raise SessionAggregationValidationError(
                "aggregated OHLC relationships are invalid"
            )
        object.__setattr__(self, "symbol", self.symbol.strip().upper())
        object.__setattr__(self, "start_timestamp", start)
        object.__setattr__(self, "end_timestamp", end)

    @property
    def period_identifier(self) -> str:
        prefix = (
            "session"
            if isinstance(self.timeframe.interval, SessionInterval)
            else "week"
        )
        return f"{prefix}:{self.period_start_date.isoformat()}"

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "schema_version": self.schema_version,
            "bar_type": "exchange_session_aggregate",
            "symbol": self.symbol,
            "period_identifier": self.period_identifier,
            "period_start_date": self.period_start_date.isoformat(),
            "session_dates": [value.isoformat() for value in self.session_dates],
            "start_timestamp": self.start_timestamp.isoformat(),
            "end_timestamp": self.end_timestamp.isoformat(),
            "timeframe": {
                "configuration_id": self.timeframe.configuration_id,
                "configuration": self.timeframe.to_primitive(),
            },
            "completion": self.completion.value,
            "open": decimal_to_primitive(self.open),
            "high": decimal_to_primitive(self.high),
            "low": decimal_to_primitive(self.low),
            "close": decimal_to_primitive(self.close),
            "volume": decimal_to_primitive(self.volume),
            "source_bar_ids": list(self.source_bar_ids),
            "source_dataset_id": self.source_dataset_id,
            "producer_name": self.producer_name,
        }

    @property
    def bar_id(self) -> str:
        return configuration_identity(self.to_primitive())


@dataclass(frozen=True, slots=True)
class SessionAggregationWindowQuality:
    """Constituent and session evidence for one daily or weekly period."""

    period_start_date: date
    session_dates: tuple[date, ...]
    start_timestamp: datetime
    end_timestamp: datetime
    expected_constituent_count: int
    observed_constituent_count: int
    missing_constituents: tuple[IntradayCoverageInterval, ...]
    source_bar_ids: tuple[str, ...]
    output_bar_id: str | None

    @property
    def is_complete(self) -> bool:
        return not self.missing_constituents

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "period_start_date": self.period_start_date.isoformat(),
            "session_dates": [value.isoformat() for value in self.session_dates],
            "start_timestamp": self.start_timestamp.isoformat(),
            "end_timestamp": self.end_timestamp.isoformat(),
            "expected_constituent_count": self.expected_constituent_count,
            "observed_constituent_count": self.observed_constituent_count,
            "is_complete": self.is_complete,
            "missing_constituents": [
                value.to_primitive() for value in self.missing_constituents
            ],
            "source_bar_ids": list(self.source_bar_ids),
            "output_bar_id": self.output_bar_id,
        }


@dataclass(frozen=True, slots=True)
class SessionAggregationReport:
    """Deterministic source-quality and per-period aggregation evidence."""

    source_dataset_id: str
    source_batch_id: str
    source_quality_report_id: str
    source_coverage_status: IntradayCoverageStatus
    source_missing_interval_count: int
    source_unexpected_interval_count: int
    source_developing_interval_count: int
    source_zero_volume_interval_count: int
    incomplete_sessions: tuple[date, ...]
    target_timeframe_configuration_id: str
    aggregation_policy_configuration_id: str
    excluded_partial_periods: tuple[date, ...]
    windows: tuple[SessionAggregationWindowQuality, ...]
    schema_version: str = SESSION_AGGREGATION_SCHEMA_VERSION

    @property
    def missing_constituent_count(self) -> int:
        return sum(len(window.missing_constituents) for window in self.windows)

    @property
    def status(self) -> IntradayCoverageStatus:
        if (
            self.source_coverage_status is IntradayCoverageStatus.INCOMPLETE
            or self.missing_constituent_count
        ):
            return IntradayCoverageStatus.INCOMPLETE
        return IntradayCoverageStatus.COMPLETE

    @property
    def is_complete(self) -> bool:
        return self.status is IntradayCoverageStatus.COMPLETE

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "schema_version": self.schema_version,
            "report_type": "session_aggregation_quality",
            "source_dataset_id": self.source_dataset_id,
            "source_batch_id": self.source_batch_id,
            "source_quality_report_id": self.source_quality_report_id,
            "source_coverage_status": self.source_coverage_status.value,
            "source_missing_interval_count": self.source_missing_interval_count,
            "source_unexpected_interval_count": self.source_unexpected_interval_count,
            "source_developing_interval_count": self.source_developing_interval_count,
            "source_zero_volume_interval_count": self.source_zero_volume_interval_count,
            "incomplete_sessions": [
                value.isoformat() for value in self.incomplete_sessions
            ],
            "target_timeframe_configuration_id": self.target_timeframe_configuration_id,
            "aggregation_policy_configuration_id": (
                self.aggregation_policy_configuration_id
            ),
            "excluded_partial_periods": [
                value.isoformat() for value in self.excluded_partial_periods
            ],
            "status": self.status.value,
            "is_complete": self.is_complete,
            "missing_constituent_count": self.missing_constituent_count,
            "windows": [window.to_primitive() for window in self.windows],
        }

    @property
    def report_id(self) -> str:
        return configuration_identity(self.to_primitive())


@dataclass(frozen=True, slots=True)
class AggregatedSessionDatasetMetadata:
    """Identity and immutable-manifest facts for one daily/weekly dataset."""

    dataset_id: str
    source_dataset_id: str
    source_request_id: str
    source_batch_id: str
    source_data_sha256: str
    source_raw_snapshot_ids: tuple[str, ...]
    source_quality_report_id: str
    source_quality_report: IntradayCoverageReport
    provider_name: str
    provider_symbol: str
    target_timeframe: Timeframe
    aggregation_policy: SessionAggregationPolicy
    family: DatasetFamily
    aggregation_report: SessionAggregationReport
    bar_count: int
    data_sha256: str
    normalized_location: str
    manifest_location: str
    schema_version: str = SESSION_AGGREGATION_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class AggregatedSessionDataset:
    """Completed daily or weekly bars bound to source evidence and lineage."""

    bars: tuple[AggregatedSessionBar, ...]
    metadata: AggregatedSessionDatasetMetadata

    @property
    def aggregation_report(self) -> SessionAggregationReport:
        return self.metadata.aggregation_report

    @property
    def dataset_family(self) -> DatasetFamily:
        return self.metadata.family

    def serialize_bars(self) -> bytes:
        return _serialize_bars(self.metadata.target_timeframe, self.bars)

    def to_manifest(self) -> PrimitiveMapping:
        metadata = self.metadata
        return {
            **_dataset_identity_primitive(
                source_dataset_id=metadata.source_dataset_id,
                source_request_id=metadata.source_request_id,
                source_batch_id=metadata.source_batch_id,
                source_data_sha256=metadata.source_data_sha256,
                source_raw_snapshot_ids=metadata.source_raw_snapshot_ids,
                source_quality_report=metadata.source_quality_report,
                provider_name=metadata.provider_name,
                provider_symbol=metadata.provider_symbol,
                target_timeframe=metadata.target_timeframe,
                aggregation_policy=metadata.aggregation_policy,
                family_id=metadata.family.family_id,
                aggregation_report=metadata.aggregation_report,
                bar_count=metadata.bar_count,
                data_sha256=metadata.data_sha256,
            ),
            "dataset_id": metadata.dataset_id,
            "normalized_location": metadata.normalized_location,
            "manifest_location": metadata.manifest_location,
            "dataset_family": metadata.family.to_manifest(),
        }

    def serialize_manifest(self) -> bytes:
        return canonical_json_bytes(self.to_manifest())

    def validate(self) -> None:
        """Recompute content, identity, paths, provenance, and family invariants."""
        bars_value = cast(object, self.bars)
        metadata_value = cast(object, self.metadata)
        if not isinstance(bars_value, tuple):
            raise SessionAggregationValidationError(
                "derived session bars must be a tuple"
            )
        if not isinstance(metadata_value, AggregatedSessionDatasetMetadata):
            raise SessionAggregationValidationError(
                "derived session metadata is invalid"
            )
        metadata = metadata_value
        if metadata.schema_version != SESSION_AGGREGATION_SCHEMA_VERSION:
            raise SessionAggregationValidationError(
                "derived session metadata schema is invalid"
            )
        if any(
            not isinstance(bar, AggregatedSessionBar)
            for bar in cast(tuple[object, ...], bars_value)
        ):
            raise SessionAggregationValidationError(
                "derived session dataset contains an invalid bar"
            )
        if tuple(bar.start_timestamp for bar in self.bars) != tuple(
            sorted(bar.start_timestamp for bar in self.bars)
        ):
            raise SessionAggregationValidationError(
                "derived session bars must be chronological"
            )
        source_ids = tuple(
            source_id for bar in self.bars for source_id in bar.source_bar_ids
        )
        if len(source_ids) != len(set(source_ids)):
            raise SessionAggregationValidationError(
                "the same source bar cannot be counted twice"
            )
        if any(
            bar.timeframe != metadata.target_timeframe
            or bar.source_dataset_id != metadata.source_dataset_id
            or bar.producer_name != _PRODUCER_NAME
            for bar in self.bars
        ):
            raise SessionAggregationValidationError(
                "derived session bar target or provenance mismatch"
            )
        normalized_bytes = self.serialize_bars()
        data_sha256 = sha256_hex(normalized_bytes)
        if metadata.bar_count != len(self.bars):
            raise SessionAggregationValidationError(
                "derived session bar count does not match its bars"
            )
        if metadata.data_sha256 != data_sha256:
            raise SessionAggregationValidationError(
                "derived session content digest does not match its bars"
            )
        report = metadata.aggregation_report
        source_quality = metadata.source_quality_report
        if metadata.source_quality_report_id != source_quality.report_id:
            raise SessionAggregationValidationError(
                "derived session source quality report identity mismatch"
            )
        if (
            metadata.source_request_id != source_quality.request_id
            or metadata.source_batch_id != source_quality.batch_id
        ):
            raise SessionAggregationValidationError(
                "derived session source quality bindings mismatch"
            )
        if (
            report.source_dataset_id != metadata.source_dataset_id
            or report.source_batch_id != metadata.source_batch_id
            or report.source_quality_report_id != source_quality.report_id
            or report.source_coverage_status is not source_quality.status
            or report.source_missing_interval_count
            != len(source_quality.missing_intervals)
            or report.source_unexpected_interval_count
            != len(source_quality.unexpected_intervals)
            or report.source_developing_interval_count
            != len(source_quality.developing_intervals)
            or report.source_zero_volume_interval_count
            != len(source_quality.zero_volume_intervals)
            or report.incomplete_sessions != source_quality.incomplete_sessions
            or report.target_timeframe_configuration_id
            != metadata.target_timeframe.configuration_id
            or report.aggregation_policy_configuration_id
            != metadata.aggregation_policy.configuration_id
        ):
            raise SessionAggregationValidationError(
                "derived session aggregation report bindings mismatch"
            )
        emitted_ids = tuple(
            window.output_bar_id
            for window in report.windows
            if window.output_bar_id is not None
        )
        if emitted_ids != tuple(bar.bar_id for bar in self.bars):
            raise SessionAggregationValidationError(
                "derived session windows do not match output bars"
            )
        identity = _dataset_identity_primitive(
            source_dataset_id=metadata.source_dataset_id,
            source_request_id=metadata.source_request_id,
            source_batch_id=metadata.source_batch_id,
            source_data_sha256=metadata.source_data_sha256,
            source_raw_snapshot_ids=metadata.source_raw_snapshot_ids,
            source_quality_report=source_quality,
            provider_name=metadata.provider_name,
            provider_symbol=metadata.provider_symbol,
            target_timeframe=metadata.target_timeframe,
            aggregation_policy=metadata.aggregation_policy,
            family_id=metadata.family.family_id,
            aggregation_report=report,
            bar_count=len(self.bars),
            data_sha256=data_sha256,
        )
        if metadata.dataset_id != configuration_identity(identity):
            raise SessionAggregationValidationError(
                "derived session dataset identity mismatch"
            )
        directory = f"session/derived/{metadata.dataset_id}"
        if metadata.normalized_location != f"{directory}/bars.json":
            raise SessionAggregationValidationError(
                "derived session normalized location is not canonical"
            )
        if metadata.manifest_location != f"{directory}/manifest.json":
            raise SessionAggregationValidationError(
                "derived session manifest location is not canonical"
            )
        expected_family = _family(
            source_dataset_id=metadata.source_dataset_id,
            source_timeframe=metadata.family.source_timeframe,
            canonical_symbol=self.bars[0].symbol
            if self.bars
            else metadata.family.canonical_symbol,
            provider_name=metadata.provider_name,
            feed_scope=metadata.family.feed_scope,
            adjustment_basis=metadata.family.adjustment_basis,
            target_timeframe=metadata.target_timeframe,
            policy=metadata.aggregation_policy,
            derived_dataset_id=metadata.dataset_id,
        )
        if metadata.family != expected_family:
            raise SessionAggregationValidationError(
                "derived session family lineage mismatch"
            )
        if (
            source_quality.timeframe_configuration_id
            != metadata.family.source_timeframe.configuration_id
        ):
            raise SessionAggregationValidationError(
                "derived session family source timeframe mismatch"
            )


def _utc_timestamp(value: datetime, field_name: str) -> datetime:
    untyped_value = cast(object, value)
    if not isinstance(untyped_value, datetime) or untyped_value.utcoffset() is None:
        raise SessionAggregationValidationError(
            f"{field_name} must be a timezone-aware datetime"
        )
    return untyped_value.astimezone(UTC)


def _serialize_bars(
    target_timeframe: Timeframe, bars: tuple[AggregatedSessionBar, ...]
) -> bytes:
    return canonical_json_bytes(
        {
            "schema_version": SESSION_AGGREGATION_SCHEMA_VERSION,
            "artifact_type": "exchange_session_aggregate_bars",
            "target_timeframe": {
                "configuration_id": target_timeframe.configuration_id,
                "configuration": target_timeframe.to_primitive(),
            },
            "bars": [{"bar_id": bar.bar_id, "bar": bar.to_primitive()} for bar in bars],
        }
    )


def _validate_source_dataset(source_dataset: IntradayDataset) -> IntradayBarBatch:
    source_value = cast(object, source_dataset)
    if not isinstance(source_value, IntradayDataset):
        raise TypeError("session aggregation requires an IntradayDataset")
    batch = IntradayBarBatch(source_dataset.request, source_dataset.bars)
    metadata = source_dataset.metadata
    if metadata.batch_id != batch.batch_id:
        raise SessionAggregationValidationError(
            "source dataset batch identity does not match its bars"
        )
    if metadata.bar_count != len(batch.bars):
        raise SessionAggregationValidationError(
            "source dataset bar count does not match its bars"
        )
    if metadata.data_sha256 != sha256_hex(batch.serialize()):
        raise SessionAggregationValidationError(
            "source dataset content digest does not match its bars"
        )
    report = validate_intraday_coverage(batch, mode=IntradayValidationMode.DIAGNOSTIC)
    if metadata.quality_report != report:
        raise SessionAggregationValidationError(
            "source dataset quality report does not match its bars"
        )
    return batch


def _validate_target(
    source_dataset: IntradayDataset, target_timeframe: Timeframe
) -> None:
    target_value = cast(object, target_timeframe)
    if not isinstance(target_value, Timeframe):
        raise SessionAggregationValidationError("target timeframe is invalid")
    interval = target_timeframe.interval
    if isinstance(interval, SessionInterval):
        if interval.session_count != 1:
            raise SessionAggregationValidationError(
                "QF-19 daily aggregation requires exactly one exchange session"
            )
    elif isinstance(interval, TradingWeekInterval):
        if interval.week_count != 1:
            raise SessionAggregationValidationError(
                "QF-19 weekly aggregation requires exactly one exchange trading week"
            )
    else:
        raise SessionAggregationValidationError(
            "target timeframe must be exchange-session daily or exchange-weekly"
        )
    if (
        target_timeframe.session_policy
        != source_dataset.request.timeframe.session_policy
    ):
        raise SessionAggregationValidationError(
            "source and target exchange-session policies must match"
        )
    if target_timeframe.developing_bar_exposure is not DevelopingBarExposure.EXCLUDE:
        raise SessionAggregationValidationError(
            "session aggregation emits completed bars only"
        )
    source_interval = cast(IntradayInterval, source_dataset.request.timeframe.interval)
    if source_interval.cross_session_policy is not CrossSessionPolicy.PROHIBITED:
        raise SessionAggregationValidationError(
            "session aggregation requires prohibited source cross-session continuation"
        )


def _expected_intervals_by_session(
    source_dataset: IntradayDataset,
) -> dict[date, tuple[IntradayCoverageInterval, ...]]:
    completed_by_session: dict[date, list[IntradayBar]] = {}
    for bar in source_dataset.bars:
        if bar.completion is not BarCompletion.DEVELOPING:
            completed_by_session.setdefault(bar.session_date, []).append(bar)
    expected: dict[date, tuple[IntradayCoverageInterval, ...]] = {}
    for session in source_dataset.quality_report.sessions:
        unexpected = {
            (value.start_timestamp, value.end_timestamp, value.completion)
            for value in session.unexpected_intervals
        }
        observed = tuple(
            IntradayCoverageInterval(
                bar.session_date,
                bar.start_timestamp,
                bar.end_timestamp,
                bar.completion,
            )
            for bar in completed_by_session.get(session.session_date, ())
            if (bar.start_timestamp, bar.end_timestamp, bar.completion)
            not in unexpected
        )
        values = tuple(
            sorted(
                (*observed, *session.missing_intervals),
                key=lambda item: (item.start_timestamp, item.end_timestamp),
            )
        )
        if len(values) != session.expected_interval_count:
            raise SessionAggregationValidationError(
                "source quality evidence cannot reconstruct expected intervals"
            )
        expected[session.session_date] = values
    return expected


def _complete_target_periods(
    source_dataset: IntradayDataset, target_timeframe: Timeframe
) -> tuple[tuple[tuple[date, tuple[date, ...]], ...], tuple[date, ...]]:
    sessions = source_dataset.quality_report.sessions
    full_session_dates = {
        item.session_date for item in sessions if item.request_covers_full_session
    }
    if isinstance(target_timeframe.interval, SessionInterval):
        daily_periods = tuple(
            (item.session_date, (item.session_date,))
            for item in sessions
            if item.session_date in full_session_dates
        )
        daily_excluded = tuple(
            item.session_date
            for item in sessions
            if item.session_date not in full_session_dates
        )
        return daily_periods, daily_excluded

    week_starts = sorted(
        {
            item.session_date - timedelta(days=item.session_date.weekday())
            for item in sessions
        }
    )
    periods: list[tuple[date, tuple[date, ...]]] = []
    excluded: list[date] = []
    represented = {item.session_date for item in sessions}
    for week_start in week_starts:
        trading_week = resolve_trading_week(week_start, target_timeframe.session_policy)
        expected_dates = tuple(item.session_date for item in trading_week.sessions)
        if (
            set(expected_dates) <= represented
            and set(expected_dates) <= full_session_dates
        ):
            periods.append((week_start, expected_dates))
        else:
            excluded.append(week_start)
    return tuple(periods), tuple(excluded)


def _aggregate_period(
    source_dataset: IntradayDataset,
    target_timeframe: Timeframe,
    period_start_date: date,
    session_dates: tuple[date, ...],
    expected_by_session: dict[date, tuple[IntradayCoverageInterval, ...]],
    observed_by_key: dict[tuple[datetime, datetime, BarCompletion], IntradayBar],
) -> tuple[AggregatedSessionBar | None, SessionAggregationWindowQuality]:
    expected = tuple(
        interval
        for session_date in session_dates
        for interval in expected_by_session[session_date]
    )
    observed = tuple(
        observed_by_key[key]
        for key in (
            (interval.start_timestamp, interval.end_timestamp, interval.completion)
            for interval in expected
        )
        if key in observed_by_key
    )
    missing = tuple(
        interval
        for interval in expected
        if (interval.start_timestamp, interval.end_timestamp, interval.completion)
        not in observed_by_key
    )
    if not expected:
        raise SessionAggregationValidationError(
            "target period contains no expected source intervals"
        )
    output: AggregatedSessionBar | None = None
    if observed:
        output = AggregatedSessionBar(
            symbol=source_dataset.request.symbol,
            timeframe=target_timeframe,
            period_start_date=period_start_date,
            session_dates=session_dates,
            start_timestamp=source_dataset.quality_report.sessions[
                next(
                    index
                    for index, value in enumerate(
                        source_dataset.quality_report.sessions
                    )
                    if value.session_date == session_dates[0]
                )
            ].session_open_timestamp,
            end_timestamp=source_dataset.quality_report.sessions[
                next(
                    index
                    for index, value in enumerate(
                        source_dataset.quality_report.sessions
                    )
                    if value.session_date == session_dates[-1]
                )
            ].session_close_timestamp,
            open=observed[0].open,
            high=max(bar.high for bar in observed),
            low=min(bar.low for bar in observed),
            close=observed[-1].close,
            volume=sum((bar.volume for bar in observed), start=Decimal(0)),
            source_bar_ids=tuple(bar.bar_id for bar in observed),
            source_dataset_id=source_dataset.metadata.dataset_id,
        )
    start_timestamp = next(
        value.session_open_timestamp
        for value in source_dataset.quality_report.sessions
        if value.session_date == session_dates[0]
    )
    end_timestamp = next(
        value.session_close_timestamp
        for value in source_dataset.quality_report.sessions
        if value.session_date == session_dates[-1]
    )
    quality = SessionAggregationWindowQuality(
        period_start_date=period_start_date,
        session_dates=session_dates,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
        expected_constituent_count=len(expected),
        observed_constituent_count=len(observed),
        missing_constituents=missing,
        source_bar_ids=tuple(bar.bar_id for bar in observed),
        output_bar_id=None if output is None else output.bar_id,
    )
    return output, quality


def _family(
    *,
    source_dataset_id: str,
    source_timeframe: Timeframe,
    canonical_symbol: str,
    provider_name: str,
    feed_scope: FeedScope,
    adjustment_basis: AdjustmentBasis,
    target_timeframe: Timeframe,
    policy: SessionAggregationPolicy,
    derived_dataset_id: str | None,
) -> DatasetFamily:
    children = () if derived_dataset_id is None else (derived_dataset_id,)
    datasets = [
        DatasetLineage(
            dataset_id=source_dataset_id,
            timeframe=source_timeframe,
            canonical_source_snapshot_id=source_dataset_id,
            parent_dataset_id=None,
            child_dataset_ids=children,
        )
    ]
    if derived_dataset_id is not None:
        datasets.append(
            DatasetLineage(
                dataset_id=derived_dataset_id,
                timeframe=target_timeframe,
                canonical_source_snapshot_id=source_dataset_id,
                parent_dataset_id=source_dataset_id,
            )
        )
    return DatasetFamily(
        canonical_symbol=canonical_symbol,
        provider_name=provider_name,
        feed_scope=feed_scope,
        adjustment_basis=adjustment_basis,
        aggregation_policy=policy.to_lineage_policy(),
        canonical_source_snapshot_id=source_dataset_id,
        datasets=tuple(datasets),
    )


def _dataset_identity_primitive(
    *,
    source_dataset_id: str,
    source_request_id: str,
    source_batch_id: str,
    source_data_sha256: str,
    source_raw_snapshot_ids: tuple[str, ...],
    source_quality_report: IntradayCoverageReport,
    provider_name: str,
    provider_symbol: str,
    target_timeframe: Timeframe,
    aggregation_policy: SessionAggregationPolicy,
    family_id: str,
    aggregation_report: SessionAggregationReport,
    bar_count: int,
    data_sha256: str,
) -> PrimitiveMapping:
    return {
        "schema_version": SESSION_AGGREGATION_SCHEMA_VERSION,
        "artifact_type": "derived_exchange_session_dataset_manifest",
        "producer": _PRODUCER_NAME,
        "source_dataset": {
            "dataset_id": source_dataset_id,
            "request_id": source_request_id,
            "batch_id": source_batch_id,
            "data_sha256": source_data_sha256,
            "raw_snapshot_ids": list(source_raw_snapshot_ids),
            "quality_report": {
                "report_id": source_quality_report.report_id,
                "report": source_quality_report.to_primitive(),
            },
            "provider_name": provider_name,
            "provider_symbol": provider_symbol,
        },
        "source_dataset_family": {
            "family_id": family_id,
            "canonical_source_snapshot_id": source_dataset_id,
        },
        "session_scope": target_timeframe.session_policy.to_primitive(),
        "target_timeframe": {
            "configuration_id": target_timeframe.configuration_id,
            "configuration": target_timeframe.to_primitive(),
        },
        "aggregation_policy": {
            "configuration_id": aggregation_policy.configuration_id,
            "configuration": aggregation_policy.to_primitive(),
        },
        "aggregation_report": {
            "report_id": aggregation_report.report_id,
            "report": aggregation_report.to_primitive(),
        },
        "bar_count": bar_count,
        "data_sha256": data_sha256,
    }


def aggregate_session_dataset(
    source_dataset: IntradayDataset,
    target_timeframe: Timeframe,
    *,
    policy: SessionAggregationPolicy | None = None,
) -> AggregatedSessionDataset:
    """Aggregate one immutable intraday source into completed daily or weekly bars."""
    source_batch = _validate_source_dataset(source_dataset)
    _validate_target(source_dataset, target_timeframe)
    aggregation_policy = policy or SessionAggregationPolicy()
    policy_value = cast(object, aggregation_policy)
    if not isinstance(policy_value, SessionAggregationPolicy):
        raise TypeError("session aggregation policy is invalid")
    source_report = source_dataset.quality_report
    expected_by_session = _expected_intervals_by_session(source_dataset)
    unexpected_keys = {
        (value.start_timestamp, value.end_timestamp, value.completion)
        for value in source_report.unexpected_intervals
    }
    observed_by_key = {
        (bar.start_timestamp, bar.end_timestamp, bar.completion): bar
        for bar in source_batch.bars
        if bar.completion is not BarCompletion.DEVELOPING
        and (bar.start_timestamp, bar.end_timestamp, bar.completion)
        not in unexpected_keys
    }
    periods, excluded_partial_periods = _complete_target_periods(
        source_dataset, target_timeframe
    )
    output_bars: list[AggregatedSessionBar] = []
    windows: list[SessionAggregationWindowQuality] = []
    for period_start, session_dates in periods:
        output, quality = _aggregate_period(
            source_dataset,
            target_timeframe,
            period_start,
            session_dates,
            expected_by_session,
            observed_by_key,
        )
        windows.append(quality)
        if output is not None:
            output_bars.append(output)
    report = SessionAggregationReport(
        source_dataset_id=source_dataset.metadata.dataset_id,
        source_batch_id=source_batch.batch_id,
        source_quality_report_id=source_report.report_id,
        source_coverage_status=source_report.status,
        source_missing_interval_count=len(source_report.missing_intervals),
        source_unexpected_interval_count=len(source_report.unexpected_intervals),
        source_developing_interval_count=len(source_report.developing_intervals),
        source_zero_volume_interval_count=len(source_report.zero_volume_intervals),
        incomplete_sessions=source_report.incomplete_sessions,
        target_timeframe_configuration_id=target_timeframe.configuration_id,
        aggregation_policy_configuration_id=aggregation_policy.configuration_id,
        excluded_partial_periods=excluded_partial_periods,
        windows=tuple(windows),
    )
    if (
        aggregation_policy.missing_constituent_policy is MissingConstituentPolicy.STRICT
        and not report.is_complete
    ):
        raise SessionAggregationQualityError(report)
    bars = tuple(output_bars)
    normalized_bytes = _serialize_bars(target_timeframe, bars)
    data_sha256 = sha256_hex(normalized_bytes)
    source_family = _family(
        source_dataset_id=source_dataset.metadata.dataset_id,
        source_timeframe=source_dataset.request.timeframe,
        canonical_symbol=source_dataset.request.symbol,
        provider_name=source_dataset.metadata.provider_name,
        feed_scope=source_dataset.request.feed_scope,
        adjustment_basis=source_dataset.request.adjustment_basis,
        target_timeframe=target_timeframe,
        policy=aggregation_policy,
        derived_dataset_id=None,
    )
    identity = _dataset_identity_primitive(
        source_dataset_id=source_dataset.metadata.dataset_id,
        source_request_id=source_dataset.request.request_id,
        source_batch_id=source_batch.batch_id,
        source_data_sha256=source_dataset.metadata.data_sha256,
        source_raw_snapshot_ids=source_dataset.metadata.raw_snapshot_ids,
        source_quality_report=source_report,
        provider_name=source_dataset.metadata.provider_name,
        provider_symbol=source_dataset.metadata.provider_symbol,
        target_timeframe=target_timeframe,
        aggregation_policy=aggregation_policy,
        family_id=source_family.family_id,
        aggregation_report=report,
        bar_count=len(bars),
        data_sha256=data_sha256,
    )
    dataset_id = configuration_identity(identity)
    family = _family(
        source_dataset_id=source_dataset.metadata.dataset_id,
        source_timeframe=source_dataset.request.timeframe,
        canonical_symbol=source_dataset.request.symbol,
        provider_name=source_dataset.metadata.provider_name,
        feed_scope=source_dataset.request.feed_scope,
        adjustment_basis=source_dataset.request.adjustment_basis,
        target_timeframe=target_timeframe,
        policy=aggregation_policy,
        derived_dataset_id=dataset_id,
    )
    directory = f"session/derived/{dataset_id}"
    metadata = AggregatedSessionDatasetMetadata(
        dataset_id=dataset_id,
        source_dataset_id=source_dataset.metadata.dataset_id,
        source_request_id=source_dataset.request.request_id,
        source_batch_id=source_batch.batch_id,
        source_data_sha256=source_dataset.metadata.data_sha256,
        source_raw_snapshot_ids=source_dataset.metadata.raw_snapshot_ids,
        source_quality_report_id=source_report.report_id,
        source_quality_report=source_report,
        provider_name=source_dataset.metadata.provider_name,
        provider_symbol=source_dataset.metadata.provider_symbol,
        target_timeframe=target_timeframe,
        aggregation_policy=aggregation_policy,
        family=family,
        aggregation_report=report,
        bar_count=len(bars),
        data_sha256=data_sha256,
        normalized_location=f"{directory}/bars.json",
        manifest_location=f"{directory}/manifest.json",
    )
    dataset = AggregatedSessionDataset(bars, metadata)
    dataset.validate()
    return dataset


__all__ = [
    "SESSION_AGGREGATION_POLICY_VERSION",
    "SESSION_AGGREGATION_SCHEMA_VERSION",
    "AggregatedSessionBar",
    "AggregatedSessionDataset",
    "AggregatedSessionDatasetMetadata",
    "SessionAggregationPolicy",
    "SessionAggregationQualityError",
    "SessionAggregationReport",
    "SessionAggregationValidationError",
    "SessionAggregationWindowQuality",
    "aggregate_session_dataset",
]
