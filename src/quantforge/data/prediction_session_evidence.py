"""Verify projected prices and their retained QF-19 constituent relationships."""

from collections import defaultdict
from datetime import date, datetime
from decimal import Decimal, InvalidOperation

from quantforge.configuration import (
    Primitive,
    PrimitiveMapping,
    configuration_identity,
    decimal_to_primitive,
)
from quantforge.data.exceptions import ValidationError
from quantforge.data.identity import serialize_bars_csv, sha256_hex
from quantforge.data.intraday import IntradayBar
from quantforge.data.intraday_aggregation import MissingConstituentPolicy
from quantforge.data.intraday_coverage_evidence import validate_retained_coverage_report
from quantforge.data.lineage import DatasetFamily
from quantforge.data.models import DailyBar, IntradayPredictionProvenance
from quantforge.data.multi_timeframe import TimeframeBarSeries
from quantforge.data.prediction_source_bars import validate_source_bar_evidence
from quantforge.data.session_aggregation import (
    AggregatedSessionBar,
    AggregatedSessionDataset,
    AggregatedSessionDatasetMetadata,
    SessionAggregationPolicy,
    SessionAggregationReport,
    SessionAggregationWindowQuality,
    session_constituent_ohlcv,
)
from quantforge.timeframes import SessionInterval, Timeframe


def _record(value: Primitive) -> PrimitiveMapping:
    if not isinstance(value, dict):
        raise ValueError("session evidence must contain records")
    return value


def _text(value: Primitive) -> str:
    if not isinstance(value, str):
        raise ValueError("session evidence must contain text")
    return value


def _texts(value: Primitive) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError("session evidence must contain arrays")
    return tuple(_text(item) for item in value)


def _bar(value: Primitive) -> AggregatedSessionBar:
    entry = _record(value)
    record = _record(entry["bar"])
    bar = AggregatedSessionBar(
        symbol=_text(record["symbol"]),
        timeframe=Timeframe.from_primitive(
            _record(_record(record["timeframe"])["configuration"])
        ),
        period_start_date=date.fromisoformat(_text(record["period_start_date"])),
        session_dates=tuple(
            date.fromisoformat(item) for item in _texts(record["session_dates"])
        ),
        start_timestamp=datetime.fromisoformat(_text(record["start_timestamp"])),
        end_timestamp=datetime.fromisoformat(_text(record["end_timestamp"])),
        open=Decimal(_text(record["open"])),
        high=Decimal(_text(record["high"])),
        low=Decimal(_text(record["low"])),
        close=Decimal(_text(record["close"])),
        volume=Decimal(_text(record["volume"])),
        source_bar_ids=_texts(record["source_bar_ids"]),
        source_dataset_id=_text(record["source_dataset_id"]),
    )
    if configuration_identity(entry) != configuration_identity(
        {"bar_id": bar.bar_id, "bar": bar.to_primitive()}
    ):
        raise ValueError("session bar identity or serialization is inconsistent")
    return bar


def session_projection_bars(
    bars: tuple[AggregatedSessionBar, ...],
) -> tuple[DailyBar, ...]:
    """Preserve exact OHLCV values with canonical, context-independent decimals."""
    return tuple(
        DailyBar(
            bar.symbol,
            bar.session_dates[0],
            *(
                Decimal(decimal_to_primitive(amount))
                for amount in (bar.open, bar.high, bar.low, bar.close, bar.volume)
            ),
        )
        for bar in bars
    )


def validate_session_projection_evidence(
    provenance: IntradayPredictionProvenance, record: PrimitiveMapping
) -> None:
    """Rebuild typed session evidence and bind its prices to the QF-3 fingerprint."""
    try:
        evidence = provenance.session_evidence.to_primitive()
        if set(evidence) != {"manifest", "bars"}:
            raise ValueError("session evidence fields are invalid")
        manifest = _record(evidence["manifest"])
        serialized = _record(evidence["bars"])
        entries = serialized["bars"]
        if not isinstance(entries, list):
            raise ValueError("session bars must be an array")
        source = provenance.source_manifest.to_primitive()
        source_batch = validate_source_bar_evidence(
            provenance.source_bar_evidence.to_primitive(), source
        )
        bars = tuple(_bar(entry) for entry in entries)
        by_session: defaultdict[date, list[IntradayBar]] = defaultdict(list)
        for source_bar in source_batch.bars:
            by_session[source_bar.session_date].append(source_bar)
        if any(
            bar.source_bar_ids
            != tuple(item.bar_id for item in by_session[bar.session_dates[0]])
            for bar in bars
        ):
            raise ValueError(
                "session constituent IDs differ from their canonical source session"
            )
        if any(
            (bar.open, bar.high, bar.low, bar.close, bar.volume)
            != session_constituent_ohlcv(tuple(by_session[bar.session_dates[0]]))
            for bar in bars
        ):
            raise ValueError("session OHLCV differs from its source constituents")
        coverage = validate_retained_coverage_report(source)
        full_sessions = tuple(
            session
            for session in coverage.sessions
            if session.request_covers_full_session
        )
        if not bars or tuple(bar.session_dates for bar in bars) != tuple(
            (session.session_date,) for session in full_sessions
        ):
            raise ValueError("session bars differ from full source sessions")
        target = Timeframe.from_primitive(
            _record(_record(manifest["target_timeframe"])["configuration"])
        )
        if (
            target.interval != SessionInterval(1)
            or target.configuration_id != provenance.session_timeframe_configuration_id
        ):
            raise ValueError("session bars require the declared one-session timeframe")
        policy = SessionAggregationPolicy(
            MissingConstituentPolicy(
                _text(
                    _record(_record(manifest["aggregation_policy"])["configuration"])[
                        "missing_constituent_policy"
                    ]
                )
            )
        )
        windows = tuple(
            SessionAggregationWindowQuality(
                period_start_date=bar.period_start_date,
                session_dates=bar.session_dates,
                start_timestamp=bar.start_timestamp,
                end_timestamp=bar.end_timestamp,
                expected_constituent_count=session.expected_interval_count,
                observed_constituent_count=len(bar.source_bar_ids),
                missing_constituents=(),
                source_bar_ids=bar.source_bar_ids,
                output_bar_id=bar.bar_id,
            )
            for bar, session in zip(bars, full_sessions, strict=True)
        )
        if any(
            window.expected_constituent_count != window.observed_constituent_count
            for window in windows
        ):
            raise ValueError("session bars have incomplete source constituents")
        report = SessionAggregationReport(
            source_dataset_id=provenance.source_dataset_id,
            source_batch_id=coverage.batch_id,
            source_quality_report_id=coverage.report_id,
            source_coverage_status=coverage.status,
            source_missing_interval_count=len(coverage.missing_intervals),
            source_unexpected_interval_count=len(coverage.unexpected_intervals),
            source_developing_interval_count=len(coverage.developing_intervals),
            source_zero_volume_interval_count=len(coverage.zero_volume_intervals),
            incomplete_sessions=coverage.incomplete_sessions,
            target_timeframe_configuration_id=target.configuration_id,
            aggregation_policy_configuration_id=policy.configuration_id,
            excluded_partial_periods=tuple(
                session.session_date
                for session in coverage.sessions
                if not session.request_covers_full_session
            ),
            windows=windows,
        )
        dataset_id = provenance.session_dataset_id
        family = DatasetFamily.from_manifest(_record(manifest["dataset_family"]))
        if any(bar.symbol != family.canonical_symbol for bar in bars):
            raise ValueError("session bars have inconsistent symbols")
        dataset = AggregatedSessionDataset(
            bars,
            AggregatedSessionDatasetMetadata(
                dataset_id=dataset_id,
                source_dataset_id=provenance.source_dataset_id,
                source_request_id=provenance.source_request_id,
                source_batch_id=coverage.batch_id,
                source_data_sha256=_text(source["data_sha256"]),
                source_raw_snapshot_ids=provenance.source_raw_snapshot_ids,
                source_quality_report_id=coverage.report_id,
                source_quality_report=coverage,
                provider_name=_text(source["provider_name"]),
                provider_symbol=_text(source["provider_symbol"]),
                target_timeframe=target,
                aggregation_policy=policy,
                family=family,
                aggregation_report=report,
                bar_count=len(bars),
                data_sha256=configuration_identity(serialized),
                normalized_location=f"session/derived/{dataset_id}/bars.json",
                manifest_location=f"session/derived/{dataset_id}/manifest.json",
            ),
        )
        # Reuse runtime artifact binding; generic family DAG checks are insufficient.
        TimeframeBarSeries.from_aggregated_session_dataset(
            dataset,
            family=DatasetFamily.from_manifest(
                provenance.family_manifest.to_primitive()
            ),
        )
        if configuration_identity(dataset.to_manifest()) != configuration_identity(
            manifest
        ):
            raise ValueError("session manifest differs from its retained evidence")
        fingerprint = sha256_hex(serialize_bars_csv(session_projection_bars(bars)))
        fields = [
            name for name in ("data_sha256", "bars_fingerprint") if name in record
        ]
        if not fields or any(record[name] != fingerprint for name in fields):
            raise ValueError("session projection bars fingerprint is inconsistent")
    except (
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        InvalidOperation,
        ValidationError,
    ) as error:
        raise ValidationError(
            f"session projection bars evidence is invalid: {error}"
        ) from error
