"""Deterministic exchange-session-aware intraday OHLCV aggregation."""

from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from enum import StrEnum
from typing import cast
from zoneinfo import ZoneInfo

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.data.exceptions import ValidationError
from quantforge.data.identity import canonical_json_bytes, sha256_hex
from quantforge.data.intraday import (
    IntradayBar,
    IntradayBarBatch,
    IntradayBarProvenance,
    IntradayBarRequest,
)
from quantforge.data.intraday_ingestion import IntradayDataset
from quantforge.data.intraday_validation import (
    IntradayCoverageInterval,
    IntradayCoverageReport,
    IntradayCoverageStatus,
    IntradayValidationMode,
    validate_intraday_coverage,
)
from quantforge.data.lineage import (
    AggregationPolicy,
    DatasetFamily,
    DatasetLineage,
)
from quantforge.timeframes import (
    BarCompletion,
    CrossSessionPolicy,
    DevelopingBarExposure,
    IntradayAnchor,
    IntradayBarWindow,
    IntradayInterval,
    Timeframe,
    resolve_exchange_session,
)

INTRADAY_AGGREGATION_SCHEMA_VERSION = "1"
INTRADAY_AGGREGATION_POLICY_VERSION = "1"
_AGGREGATION_POLICY_NAME = "quantforge_intraday_ohlcv"
_AGGREGATION_PRODUCER_NAME = "quantforge"
_CLOCK_ANCHOR_EPOCH_DATE = date(1970, 1, 1)


class MissingConstituentPolicy(StrEnum):
    """How aggregation handles absent expected source intervals."""

    STRICT = "strict"
    DIAGNOSTIC = "diagnostic"


class IntradayAggregationValidationError(ValidationError):
    """Source data or aggregation configuration cannot produce trusted bars."""


class IntradayAggregationQualityError(IntradayAggregationValidationError):
    """Strict aggregation found incomplete source coverage."""

    def __init__(self, report: "IntradayAggregationReport") -> None:
        super().__init__(
            "intraday aggregation failed strict constituent validation: "
            f"{report.source_missing_interval_count} missing and "
            f"{report.source_unexpected_interval_count} unexpected source intervals",
            missing_sessions=tuple(
                session_date.isoformat() for session_date in report.incomplete_sessions
            ),
        )
        self.report = report


@dataclass(frozen=True, slots=True)
class IntradayAggregationPolicy:
    """Complete deterministic policy for larger intraday bar construction."""

    missing_constituent_policy: MissingConstituentPolicy = (
        MissingConstituentPolicy.STRICT
    )
    schema_version: str = INTRADAY_AGGREGATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        missing_policy = cast(object, self.missing_constituent_policy)
        if not isinstance(missing_policy, MissingConstituentPolicy):
            raise IntradayAggregationValidationError(
                "missing-constituent policy is invalid"
            )
        if self.schema_version != INTRADAY_AGGREGATION_SCHEMA_VERSION:
            raise IntradayAggregationValidationError(
                f"intraday aggregation schema {INTRADAY_AGGREGATION_SCHEMA_VERSION} "
                "is required"
            )

    def to_primitive(self) -> PrimitiveMapping:
        """Return every material aggregation choice in canonical form."""
        return {
            "schema_version": self.schema_version,
            "missing_constituent_policy": self.missing_constituent_policy.value,
            "source_interval_requirement": "strictly_smaller_exact_multiple",
            "session_boundary_policy": "never_cross",
            "terminal_partial_duration_policy": "emit_completed",
            "developing_bar_policy": "exclude",
            "unexpected_source_interval_policy": "reject_or_disclose_and_exclude",
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
        """Return the QF-14 policy reference bound into a dataset family."""
        return AggregationPolicy(
            _AGGREGATION_POLICY_NAME,
            INTRADAY_AGGREGATION_POLICY_VERSION,
            self.to_primitive(),
        )


@dataclass(frozen=True, slots=True)
class AggregatedWindowQuality:
    """Auditable constituent evidence for one target interval."""

    session_date: date
    start_timestamp: datetime
    end_timestamp: datetime
    completion: BarCompletion
    expected_constituent_count: int
    observed_constituent_count: int
    missing_constituents: tuple[IntradayCoverageInterval, ...]
    source_bar_ids: tuple[str, ...]
    output_bar_id: str | None

    @property
    def is_complete(self) -> bool:
        return not self.missing_constituents

    @property
    def was_emitted(self) -> bool:
        return self.output_bar_id is not None

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "session_date": self.session_date.isoformat(),
            "start_timestamp": self.start_timestamp.isoformat(),
            "end_timestamp": self.end_timestamp.isoformat(),
            "completion": self.completion.value,
            "expected_constituent_count": self.expected_constituent_count,
            "observed_constituent_count": self.observed_constituent_count,
            "is_complete": self.is_complete,
            "was_emitted": self.was_emitted,
            "missing_constituents": [
                interval.to_primitive() for interval in self.missing_constituents
            ],
            "source_bar_ids": list(self.source_bar_ids),
            "output_bar_id": self.output_bar_id,
        }


@dataclass(frozen=True, slots=True)
class IntradayAggregationReport:
    """Deterministic source-quality and per-window aggregation evidence."""

    source_dataset_id: str
    source_batch_id: str
    source_quality_report_id: str
    source_coverage_status: IntradayCoverageStatus
    source_missing_interval_count: int
    source_unexpected_interval_count: int
    source_developing_interval_count: int
    source_zero_volume_interval_count: int
    incomplete_sessions: tuple[date, ...]
    target_request_id: str
    target_timeframe_configuration_id: str
    aggregation_policy_configuration_id: str
    windows: tuple[AggregatedWindowQuality, ...]
    schema_version: str = INTRADAY_AGGREGATION_SCHEMA_VERSION

    @property
    def missing_constituent_count(self) -> int:
        return sum(len(window.missing_constituents) for window in self.windows)

    @property
    def skipped_window_count(self) -> int:
        return sum(not window.was_emitted for window in self.windows)

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
            "report_type": "intraday_aggregation_quality",
            "source_dataset_id": self.source_dataset_id,
            "source_batch_id": self.source_batch_id,
            "source_quality_report_id": self.source_quality_report_id,
            "source_coverage_status": self.source_coverage_status.value,
            "source_missing_interval_count": self.source_missing_interval_count,
            "source_unexpected_interval_count": (self.source_unexpected_interval_count),
            "source_developing_interval_count": (self.source_developing_interval_count),
            "source_zero_volume_interval_count": (
                self.source_zero_volume_interval_count
            ),
            "incomplete_sessions": [
                session_date.isoformat() for session_date in self.incomplete_sessions
            ],
            "target_request_id": self.target_request_id,
            "target_timeframe_configuration_id": (
                self.target_timeframe_configuration_id
            ),
            "aggregation_policy_configuration_id": (
                self.aggregation_policy_configuration_id
            ),
            "status": self.status.value,
            "is_complete": self.is_complete,
            "missing_constituent_count": self.missing_constituent_count,
            "skipped_window_count": self.skipped_window_count,
            "windows": [window.to_primitive() for window in self.windows],
        }

    @property
    def report_id(self) -> str:
        return configuration_identity(self.to_primitive())


@dataclass(frozen=True, slots=True)
class AggregatedIntradayDatasetMetadata:
    """Identity and immutable-manifest facts for one derived dataset."""

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
    target_request_id: str
    aggregation_policy: IntradayAggregationPolicy
    family: DatasetFamily
    aggregation_report: IntradayAggregationReport
    batch_id: str
    bar_count: int
    data_sha256: str
    normalized_location: str
    manifest_location: str
    schema_version: str = INTRADAY_AGGREGATION_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class AggregatedIntradayDataset:
    """Canonical derived bars bound to source evidence and family lineage."""

    request: IntradayBarRequest
    bars: tuple[IntradayBar, ...]
    metadata: AggregatedIntradayDatasetMetadata

    @property
    def aggregation_report(self) -> IntradayAggregationReport:
        return self.metadata.aggregation_report

    @property
    def dataset_family(self) -> DatasetFamily:
        return self.metadata.family

    def to_manifest(self) -> PrimitiveMapping:
        """Return the complete immutable derived-dataset manifest."""
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
                target_request=self.request,
                aggregation_policy=metadata.aggregation_policy,
                family_id=metadata.family.family_id,
                aggregation_report=metadata.aggregation_report,
                batch_id=metadata.batch_id,
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

    def serialize_bars(self) -> bytes:
        return IntradayBarBatch(self.request, self.bars).serialize()

    def validate(self) -> None:
        """Recompute every derived identity and canonical persistence invariant."""
        request_value = cast(object, self.request)
        bars_value = cast(object, self.bars)
        metadata_value = cast(object, self.metadata)
        if not isinstance(request_value, IntradayBarRequest):
            raise IntradayAggregationValidationError(
                "derived intraday request is invalid"
            )
        if not isinstance(bars_value, tuple):
            raise IntradayAggregationValidationError(
                "derived intraday bars must be a tuple"
            )
        if not isinstance(metadata_value, AggregatedIntradayDatasetMetadata):
            raise IntradayAggregationValidationError(
                "derived intraday metadata is invalid"
            )
        metadata = metadata_value
        if metadata.schema_version != INTRADAY_AGGREGATION_SCHEMA_VERSION:
            raise IntradayAggregationValidationError(
                "derived intraday metadata schema is invalid"
            )
        policy_value = cast(object, metadata.aggregation_policy)
        family_value = cast(object, metadata.family)
        report_value = cast(object, metadata.aggregation_report)
        source_quality_value = cast(object, metadata.source_quality_report)
        if not isinstance(policy_value, IntradayAggregationPolicy):
            raise IntradayAggregationValidationError(
                "derived intraday aggregation policy is invalid"
            )
        if not isinstance(family_value, DatasetFamily):
            raise IntradayAggregationValidationError(
                "derived intraday dataset family is invalid"
            )
        if not isinstance(report_value, IntradayAggregationReport):
            raise IntradayAggregationValidationError(
                "derived intraday aggregation report is invalid"
            )
        if not isinstance(source_quality_value, IntradayCoverageReport):
            raise IntradayAggregationValidationError(
                "derived intraday source quality report is invalid"
            )

        batch = IntradayBarBatch(self.request, self.bars)
        normalized_bytes = batch.serialize()
        data_sha256 = sha256_hex(normalized_bytes)
        if metadata.batch_id != batch.batch_id:
            raise IntradayAggregationValidationError(
                "derived intraday batch identity does not match its bars"
            )
        if metadata.bar_count != len(batch.bars):
            raise IntradayAggregationValidationError(
                "derived intraday bar count does not match its bars"
            )
        if metadata.data_sha256 != data_sha256:
            raise IntradayAggregationValidationError(
                "derived intraday content digest does not match its bars"
            )
        if metadata.target_request_id != self.request.request_id:
            raise IntradayAggregationValidationError(
                "derived intraday target request identity mismatch"
            )
        self._validate_source_and_report_bindings()
        self._validate_bar_provenance()

        identity = _dataset_identity_primitive(
            source_dataset_id=metadata.source_dataset_id,
            source_request_id=metadata.source_request_id,
            source_batch_id=metadata.source_batch_id,
            source_data_sha256=metadata.source_data_sha256,
            source_raw_snapshot_ids=metadata.source_raw_snapshot_ids,
            source_quality_report=metadata.source_quality_report,
            provider_name=metadata.provider_name,
            provider_symbol=metadata.provider_symbol,
            target_request=self.request,
            aggregation_policy=metadata.aggregation_policy,
            family_id=metadata.family.family_id,
            aggregation_report=metadata.aggregation_report,
            batch_id=batch.batch_id,
            bar_count=len(batch.bars),
            data_sha256=data_sha256,
        )
        if metadata.dataset_id != configuration_identity(identity):
            raise IntradayAggregationValidationError(
                "derived intraday dataset identity mismatch"
            )
        canonical_directory = f"intraday/derived/{metadata.dataset_id}"
        if metadata.normalized_location != f"{canonical_directory}/bars.json":
            raise IntradayAggregationValidationError(
                "derived intraday normalized location is not canonical"
            )
        if metadata.manifest_location != f"{canonical_directory}/manifest.json":
            raise IntradayAggregationValidationError(
                "derived intraday manifest location is not canonical"
            )
        self._validate_family()

    def _validate_source_and_report_bindings(self) -> None:
        metadata = self.metadata
        source_quality = metadata.source_quality_report
        report = metadata.aggregation_report
        if metadata.source_quality_report_id != source_quality.report_id:
            raise IntradayAggregationValidationError(
                "derived intraday source quality report identity mismatch"
            )
        if (
            metadata.source_request_id != source_quality.request_id
            or metadata.source_batch_id != source_quality.batch_id
        ):
            raise IntradayAggregationValidationError(
                "derived intraday source quality bindings mismatch"
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
        ):
            raise IntradayAggregationValidationError(
                "derived intraday aggregation report source bindings mismatch"
            )
        if (
            report.target_request_id != self.request.request_id
            or report.target_timeframe_configuration_id
            != self.request.timeframe.configuration_id
            or report.aggregation_policy_configuration_id
            != metadata.aggregation_policy.configuration_id
        ):
            raise IntradayAggregationValidationError(
                "derived intraday aggregation report target bindings mismatch"
            )
        emitted_ids = tuple(
            window.output_bar_id
            for window in report.windows
            if window.output_bar_id is not None
        )
        if emitted_ids != tuple(bar.bar_id for bar in self.bars):
            raise IntradayAggregationValidationError(
                "derived intraday aggregation windows do not match output bars"
            )

    def _validate_bar_provenance(self) -> None:
        metadata = self.metadata
        if any(
            bar.provenance.provider_name != _AGGREGATION_PRODUCER_NAME
            or bar.provenance.provider_symbol != metadata.provider_symbol
            or bar.provenance.source_snapshot_id != metadata.source_dataset_id
            for bar in self.bars
        ):
            raise IntradayAggregationValidationError(
                "derived intraday bar provenance mismatch"
            )

    def _validate_family(self) -> None:
        metadata = self.metadata
        family = metadata.family
        if family.canonical_source_snapshot_id != metadata.source_dataset_id:
            raise IntradayAggregationValidationError(
                "derived intraday family source identity mismatch"
            )
        if (
            family.canonical_symbol != self.request.symbol
            or family.provider_name != metadata.provider_name
            or family.feed_scope != self.request.feed_scope
            or family.adjustment_basis != self.request.adjustment_basis
            or family.aggregation_policy
            != metadata.aggregation_policy.to_lineage_policy()
        ):
            raise IntradayAggregationValidationError(
                "derived intraday family policy bindings mismatch"
            )
        expected_family = DatasetFamily(
            canonical_symbol=self.request.symbol,
            provider_name=metadata.provider_name,
            feed_scope=self.request.feed_scope,
            adjustment_basis=self.request.adjustment_basis,
            aggregation_policy=metadata.aggregation_policy.to_lineage_policy(),
            canonical_source_snapshot_id=metadata.source_dataset_id,
            datasets=(
                DatasetLineage(
                    dataset_id=metadata.source_dataset_id,
                    timeframe=family.source_timeframe,
                    canonical_source_snapshot_id=metadata.source_dataset_id,
                    parent_dataset_id=None,
                    child_dataset_ids=(metadata.dataset_id,),
                ),
                DatasetLineage(
                    dataset_id=metadata.dataset_id,
                    timeframe=self.request.timeframe,
                    canonical_source_snapshot_id=metadata.source_dataset_id,
                    parent_dataset_id=metadata.source_dataset_id,
                ),
            ),
        )
        if family != expected_family:
            raise IntradayAggregationValidationError(
                "derived intraday family lineage mismatch"
            )
        source_quality_timeframe_id = (
            metadata.source_quality_report.timeframe_configuration_id
        )
        if source_quality_timeframe_id != family.source_timeframe.configuration_id:
            raise IntradayAggregationValidationError(
                "derived intraday family source timeframe mismatch"
            )


def _utc_timestamp(timestamp: datetime) -> datetime:
    return timestamp.astimezone(UTC)


def _interval_key(
    interval: IntradayCoverageInterval,
) -> tuple[datetime, datetime, BarCompletion]:
    return (
        interval.start_timestamp,
        interval.end_timestamp,
        interval.completion,
    )


def _bar_key(bar: IntradayBar) -> tuple[datetime, datetime, BarCompletion]:
    return (bar.start_timestamp, bar.end_timestamp, bar.completion)


def _bar_interval(bar: IntradayBar) -> IntradayCoverageInterval:
    return IntradayCoverageInterval(
        bar.session_date,
        bar.start_timestamp,
        bar.end_timestamp,
        bar.completion,
    )


def _clock_bucket_end(
    session_open_timestamp: datetime,
    interval: IntradayInterval,
    timezone_name: str,
) -> datetime:
    timezone = ZoneInfo(timezone_name)
    anchor = datetime.combine(
        _CLOCK_ANCHOR_EPOCH_DATE,
        cast(time, interval.clock_anchor),
        timezone,
    ).astimezone(UTC)
    elapsed = session_open_timestamp - anchor
    bucket_start = anchor + (elapsed // interval.nominal_duration) * (
        interval.nominal_duration
    )
    return bucket_start + interval.nominal_duration


def _target_session_windows(
    session_date: date,
    timeframe: Timeframe,
) -> tuple[IntradayCoverageInterval, ...]:
    interval = cast(IntradayInterval, timeframe.interval)
    session = resolve_exchange_session(session_date, timeframe.session_policy)
    start_timestamp = session.open_timestamp
    next_clock_end = (
        _clock_bucket_end(
            session.open_timestamp,
            interval,
            timeframe.session_policy.timezone_name,
        )
        if interval.anchor is IntradayAnchor.CLOCK
        else None
    )
    windows: list[IntradayCoverageInterval] = []
    while start_timestamp < session.close_timestamp:
        anchored_end = (
            next_clock_end
            if next_clock_end is not None
            else start_timestamp + interval.nominal_duration
        )
        end_timestamp = min(anchored_end, session.close_timestamp)
        actual_duration = end_timestamp - start_timestamp
        if actual_duration == interval.nominal_duration:
            completion = BarCompletion.COMPLETED
        elif end_timestamp == session.close_timestamp:
            completion = BarCompletion.COMPLETED_PARTIAL_DURATION_TERMINAL
        else:
            completion = BarCompletion.COMPLETED_PARTIAL_DURATION_LEADING
        IntradayBarWindow(
            timeframe,
            session_date,
            start_timestamp,
            end_timestamp,
            completion,
        )
        windows.append(
            IntradayCoverageInterval(
                session_date,
                _utc_timestamp(start_timestamp),
                _utc_timestamp(end_timestamp),
                completion,
            )
        )
        start_timestamp = end_timestamp
        if next_clock_end is not None:
            next_clock_end += interval.nominal_duration
    return tuple(windows)


def _validate_source_dataset(source_dataset: IntradayDataset) -> IntradayBarBatch:
    source_value = cast(object, source_dataset)
    if not isinstance(source_value, IntradayDataset):
        raise TypeError("intraday aggregation requires an IntradayDataset")
    batch = IntradayBarBatch(source_dataset.request, source_dataset.bars)
    metadata = source_dataset.metadata
    if metadata.batch_id != batch.batch_id:
        raise IntradayAggregationValidationError(
            "source dataset batch identity does not match its bars"
        )
    if metadata.bar_count != len(batch.bars):
        raise IntradayAggregationValidationError(
            "source dataset bar count does not match its bars"
        )
    if metadata.data_sha256 != sha256_hex(batch.serialize()):
        raise IntradayAggregationValidationError(
            "source dataset content digest does not match its bars"
        )
    report = validate_intraday_coverage(batch, mode=IntradayValidationMode.DIAGNOSTIC)
    if metadata.quality_report != report:
        raise IntradayAggregationValidationError(
            "source dataset quality report does not match its bars"
        )
    return batch


def _validate_target_timeframe(
    source_timeframe: Timeframe,
    target_timeframe: Timeframe,
) -> tuple[IntradayInterval, IntradayInterval]:
    target_value = cast(object, target_timeframe)
    if not isinstance(target_value, Timeframe) or not isinstance(
        target_value.interval, IntradayInterval
    ):
        raise IntradayAggregationValidationError("target timeframe must be intraday")
    source_interval = cast(IntradayInterval, source_timeframe.interval)
    target_interval = cast(IntradayInterval, target_timeframe.interval)
    if target_interval.nominal_duration <= source_interval.nominal_duration or (
        target_interval.nominal_duration % source_interval.nominal_duration
    ):
        raise IntradayAggregationValidationError(
            "target duration must be a strictly larger exact multiple of the "
            "source duration"
        )
    if target_timeframe.session_policy != source_timeframe.session_policy:
        raise IntradayAggregationValidationError(
            "source and target exchange-session policies must match"
        )
    if (
        source_interval.cross_session_policy is not CrossSessionPolicy.PROHIBITED
        or target_interval.cross_session_policy is not CrossSessionPolicy.PROHIBITED
    ):
        raise IntradayAggregationValidationError(
            "intraday aggregation requires prohibited cross-session continuation"
        )
    if (
        target_interval.anchor is not source_interval.anchor
        or target_interval.clock_anchor != source_interval.clock_anchor
    ):
        raise IntradayAggregationValidationError(
            "source and target intraday anchors must match"
        )
    if target_timeframe.developing_bar_exposure is not DevelopingBarExposure.EXCLUDE:
        raise IntradayAggregationValidationError(
            "QF-18 aggregation emits completed bars only"
        )
    return source_interval, target_interval


def _expected_source_intervals(
    source_dataset: IntradayDataset,
) -> dict[date, tuple[IntradayCoverageInterval, ...]]:
    report = source_dataset.quality_report
    completed_by_session: dict[date, list[IntradayBar]] = {}
    for bar in source_dataset.bars:
        if bar.completion is not BarCompletion.DEVELOPING:
            completed_by_session.setdefault(bar.session_date, []).append(bar)
    expected: dict[date, tuple[IntradayCoverageInterval, ...]] = {}
    for session in report.sessions:
        unexpected_keys = {
            _interval_key(interval) for interval in session.unexpected_intervals
        }
        observed_expected = [
            _bar_interval(bar)
            for bar in completed_by_session.get(session.session_date, [])
            if _bar_key(bar) not in unexpected_keys
        ]
        intervals = tuple(
            sorted(
                (*observed_expected, *session.missing_intervals),
                key=lambda interval: (
                    interval.start_timestamp,
                    interval.end_timestamp,
                    interval.completion.value,
                ),
            )
        )
        if len(intervals) != session.expected_interval_count:
            raise IntradayAggregationValidationError(
                "source quality evidence cannot reconstruct expected intervals"
            )
        expected[session.session_date] = intervals
    return expected


def _aggregate_window(
    source_dataset: IntradayDataset,
    target_request: IntradayBarRequest,
    target_window: IntradayCoverageInterval,
    expected_intervals: tuple[IntradayCoverageInterval, ...],
    observed_by_key: dict[tuple[datetime, datetime, BarCompletion], IntradayBar],
) -> tuple[IntradayBar | None, AggregatedWindowQuality]:
    expected = tuple(
        interval
        for interval in expected_intervals
        if target_window.start_timestamp <= interval.start_timestamp
        and interval.end_timestamp <= target_window.end_timestamp
    )
    observed = tuple(
        observed_by_key[key]
        for key in (_interval_key(interval) for interval in expected)
        if key in observed_by_key
    )
    missing = tuple(
        interval
        for interval in expected
        if _interval_key(interval) not in observed_by_key
    )
    if not expected:
        raise IntradayAggregationValidationError(
            "target window contains no expected source intervals"
        )
    output_bar: IntradayBar | None = None
    if observed:
        provenance = IntradayBarProvenance(
            provider_name=_AGGREGATION_PRODUCER_NAME,
            provider_symbol=source_dataset.metadata.provider_symbol,
            adapter_version=(
                f"intraday-aggregation-{INTRADAY_AGGREGATION_POLICY_VERSION}"
            ),
            retrieved_at=source_dataset.metadata.retrieved_at,
            source_request_id=target_request.request_id,
            source_snapshot_id=source_dataset.metadata.dataset_id,
            feed_scope=target_request.feed_scope,
            adjustment_basis=target_request.adjustment_basis,
        )
        output_bar = IntradayBar(
            symbol=target_request.symbol,
            session_date=target_window.session_date,
            start_timestamp=target_window.start_timestamp,
            end_timestamp=target_window.end_timestamp,
            timeframe=target_request.timeframe,
            completion=target_window.completion,
            open=observed[0].open,
            high=max(bar.high for bar in observed),
            low=min(bar.low for bar in observed),
            close=observed[-1].close,
            volume=sum((bar.volume for bar in observed), start=observed[0].volume * 0),
            provenance=provenance,
        )
    quality = AggregatedWindowQuality(
        session_date=target_window.session_date,
        start_timestamp=target_window.start_timestamp,
        end_timestamp=target_window.end_timestamp,
        completion=target_window.completion,
        expected_constituent_count=len(expected),
        observed_constituent_count=len(observed),
        missing_constituents=missing,
        source_bar_ids=tuple(bar.bar_id for bar in observed),
        output_bar_id=None if output_bar is None else output_bar.bar_id,
    )
    return output_bar, quality


def _source_only_family(
    source_dataset: IntradayDataset,
    policy: IntradayAggregationPolicy,
) -> DatasetFamily:
    source_id = source_dataset.metadata.dataset_id
    return DatasetFamily(
        canonical_symbol=source_dataset.request.symbol,
        provider_name=source_dataset.metadata.provider_name,
        feed_scope=source_dataset.request.feed_scope,
        adjustment_basis=source_dataset.request.adjustment_basis,
        aggregation_policy=policy.to_lineage_policy(),
        canonical_source_snapshot_id=source_id,
        datasets=(
            DatasetLineage(
                dataset_id=source_id,
                timeframe=source_dataset.request.timeframe,
                canonical_source_snapshot_id=source_id,
                parent_dataset_id=None,
            ),
        ),
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
    target_request: IntradayBarRequest,
    aggregation_policy: IntradayAggregationPolicy,
    family_id: str,
    aggregation_report: IntradayAggregationReport,
    batch_id: str,
    bar_count: int,
    data_sha256: str,
) -> PrimitiveMapping:
    return {
        "schema_version": INTRADAY_AGGREGATION_SCHEMA_VERSION,
        "artifact_type": "derived_intraday_dataset_manifest",
        "producer": _AGGREGATION_PRODUCER_NAME,
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
        "target_request": {
            "request_id": target_request.request_id,
            "configuration": target_request.to_primitive(),
        },
        "aggregation_policy": {
            "configuration_id": aggregation_policy.configuration_id,
            "configuration": aggregation_policy.to_primitive(),
        },
        "dataset_family_id": family_id,
        "aggregation_report": {
            "report_id": aggregation_report.report_id,
            "report": aggregation_report.to_primitive(),
        },
        "batch_id": batch_id,
        "bar_count": bar_count,
        "data_sha256": data_sha256,
    }


def aggregate_intraday_dataset(
    source_dataset: IntradayDataset,
    target_timeframe: Timeframe,
    *,
    policy: IntradayAggregationPolicy | None = None,
) -> AggregatedIntradayDataset:
    """Aggregate one immutable source dataset into completed target bars.

    Strict policy is the default and raises before returning a derived dataset
    when source coverage is incomplete. Diagnostic policy excludes unexpected
    intervals, aggregates available expected constituents, and binds every gap
    into per-window quality evidence.
    """
    source_batch = _validate_source_dataset(source_dataset)
    _validate_target_timeframe(source_dataset.request.timeframe, target_timeframe)
    aggregation_policy = policy or IntradayAggregationPolicy()
    policy_value = cast(object, aggregation_policy)
    if not isinstance(policy_value, IntradayAggregationPolicy):
        raise TypeError("intraday aggregation policy is invalid")
    target_request = IntradayBarRequest(
        symbol=source_dataset.request.symbol,
        start_timestamp=source_dataset.request.start_timestamp,
        end_timestamp=source_dataset.request.end_timestamp,
        timeframe=target_timeframe,
        feed_scope=source_dataset.request.feed_scope,
        adjustment_basis=source_dataset.request.adjustment_basis,
    )
    source_report = source_dataset.quality_report
    expected_by_session = _expected_source_intervals(source_dataset)
    unexpected_keys = {
        _interval_key(interval) for interval in source_report.unexpected_intervals
    }
    observed_by_key = {
        _bar_key(bar): bar
        for bar in source_batch.bars
        if bar.completion is not BarCompletion.DEVELOPING
        and _bar_key(bar) not in unexpected_keys
    }
    output_bars: list[IntradayBar] = []
    window_quality: list[AggregatedWindowQuality] = []
    for session_report in source_report.sessions:
        for target_window in _target_session_windows(
            session_report.session_date, target_timeframe
        ):
            if not (
                target_request.start_timestamp <= target_window.start_timestamp
                and target_window.end_timestamp <= target_request.end_timestamp
            ):
                continue
            output_bar, quality = _aggregate_window(
                source_dataset,
                target_request,
                target_window,
                expected_by_session[session_report.session_date],
                observed_by_key,
            )
            window_quality.append(quality)
            if output_bar is not None:
                output_bars.append(output_bar)
    report = IntradayAggregationReport(
        source_dataset_id=source_dataset.metadata.dataset_id,
        source_batch_id=source_batch.batch_id,
        source_quality_report_id=source_report.report_id,
        source_coverage_status=source_report.status,
        source_missing_interval_count=len(source_report.missing_intervals),
        source_unexpected_interval_count=len(source_report.unexpected_intervals),
        source_developing_interval_count=len(source_report.developing_intervals),
        source_zero_volume_interval_count=len(source_report.zero_volume_intervals),
        incomplete_sessions=source_report.incomplete_sessions,
        target_request_id=target_request.request_id,
        target_timeframe_configuration_id=target_timeframe.configuration_id,
        aggregation_policy_configuration_id=aggregation_policy.configuration_id,
        windows=tuple(window_quality),
    )
    if (
        aggregation_policy.missing_constituent_policy is MissingConstituentPolicy.STRICT
        and not report.is_complete
    ):
        raise IntradayAggregationQualityError(report)
    batch = IntradayBarBatch(target_request, tuple(output_bars))
    normalized_bytes = batch.serialize()
    data_sha256 = sha256_hex(normalized_bytes)
    source_family = _source_only_family(source_dataset, aggregation_policy)
    identity = _dataset_identity_primitive(
        source_dataset_id=source_dataset.metadata.dataset_id,
        source_request_id=source_dataset.request.request_id,
        source_batch_id=source_batch.batch_id,
        source_data_sha256=source_dataset.metadata.data_sha256,
        source_raw_snapshot_ids=source_dataset.metadata.raw_snapshot_ids,
        source_quality_report=source_report,
        provider_name=source_dataset.metadata.provider_name,
        provider_symbol=source_dataset.metadata.provider_symbol,
        target_request=target_request,
        aggregation_policy=aggregation_policy,
        family_id=source_family.family_id,
        aggregation_report=report,
        batch_id=batch.batch_id,
        bar_count=len(batch.bars),
        data_sha256=data_sha256,
    )
    dataset_id = configuration_identity(identity)
    source_id = source_dataset.metadata.dataset_id
    family = DatasetFamily(
        canonical_symbol=source_dataset.request.symbol,
        provider_name=source_dataset.metadata.provider_name,
        feed_scope=source_dataset.request.feed_scope,
        adjustment_basis=source_dataset.request.adjustment_basis,
        aggregation_policy=aggregation_policy.to_lineage_policy(),
        canonical_source_snapshot_id=source_id,
        datasets=(
            DatasetLineage(
                dataset_id=source_id,
                timeframe=source_dataset.request.timeframe,
                canonical_source_snapshot_id=source_id,
                parent_dataset_id=None,
                child_dataset_ids=(dataset_id,),
            ),
            DatasetLineage(
                dataset_id=dataset_id,
                timeframe=target_timeframe,
                canonical_source_snapshot_id=source_id,
                parent_dataset_id=source_id,
            ),
        ),
    )
    directory = f"intraday/derived/{dataset_id}"
    metadata = AggregatedIntradayDatasetMetadata(
        dataset_id=dataset_id,
        source_dataset_id=source_id,
        source_request_id=source_dataset.request.request_id,
        source_batch_id=source_batch.batch_id,
        source_data_sha256=source_dataset.metadata.data_sha256,
        source_raw_snapshot_ids=source_dataset.metadata.raw_snapshot_ids,
        source_quality_report_id=source_report.report_id,
        source_quality_report=source_report,
        provider_name=source_dataset.metadata.provider_name,
        provider_symbol=source_dataset.metadata.provider_symbol,
        target_request_id=target_request.request_id,
        aggregation_policy=aggregation_policy,
        family=family,
        aggregation_report=report,
        batch_id=batch.batch_id,
        bar_count=len(batch.bars),
        data_sha256=data_sha256,
        normalized_location=f"{directory}/bars.json",
        manifest_location=f"{directory}/manifest.json",
    )
    return AggregatedIntradayDataset(target_request, batch.bars, metadata)


__all__ = [
    "INTRADAY_AGGREGATION_POLICY_VERSION",
    "INTRADAY_AGGREGATION_SCHEMA_VERSION",
    "AggregatedIntradayDataset",
    "AggregatedIntradayDatasetMetadata",
    "AggregatedWindowQuality",
    "IntradayAggregationPolicy",
    "IntradayAggregationQualityError",
    "IntradayAggregationReport",
    "IntradayAggregationValidationError",
    "MissingConstituentPolicy",
    "aggregate_intraday_dataset",
]
