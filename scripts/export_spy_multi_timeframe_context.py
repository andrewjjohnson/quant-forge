#!/usr/bin/env python3
"""Export the deterministic, indicator-free QF-30 SPY context example.

The committed fixture is synthetic and redistributable. It is first persisted
through QuantForge's immutable intraday cache and then replayed from that cache;
no provider client, credential, prediction rule, or indicator backend is used.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import cast

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.data import (
    AdjustmentBasis,
    AdjustmentMode,
    AggregatedIntradayDataset,
    AggregatedSessionDataset,
    AggregationPolicy,
    ContextBar,
    ContextCompletionPolicy,
    ContextTimeframeRequirement,
    DatasetFamily,
    DatasetLineage,
    FeedScope,
    IntradayBar,
    IntradayBarBatch,
    IntradayBarProvenance,
    IntradayBarRequest,
    IntradayDataset,
    IntradayFetchResult,
    IntradayMarketDataCache,
    IntradayRawSnapshot,
    MultiTimeframeContext,
    TimeframeBarSeries,
    aggregate_intraday_dataset,
    aggregate_session_dataset,
    build_multi_timeframe_context,
)
from quantforge.data.identity import sha256_hex
from quantforge.data.models import ProviderRecord
from quantforge.timeframes import (
    BarCompletion,
    IntradayInterval,
    SessionInterval,
    Timeframe,
    TradingWeekInterval,
    resolve_exchange_session,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE_PATH = (
    REPOSITORY_ROOT / "examples" / "spy_multi_timeframe" / "fixture.json"
)
DEFAULT_CACHE_ROOT = REPOSITORY_ROOT / "data" / "qf30-spy-context-cache"
DEFAULT_OUTPUT_ROOT = REPOSITORY_ROOT / "examples" / "spy_multi_timeframe" / "exports"
EXAMPLE_SCHEMA_VERSION = "1"
CONTEXT_FAMILY_POLICY_NAME = "quantforge_context_artifact_set"
CONTEXT_FAMILY_POLICY_VERSION = "1"


class SpyContextExampleError(ValueError):
    """The QF-30 fixture, cache, or immutable export is inconsistent."""


@dataclass(frozen=True, slots=True)
class DecisionScenario:
    """One fixed historical decision timestamp used by the example."""

    name: str
    as_of: datetime
    purpose: str

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "name": self.name,
            "as_of": self.as_of.astimezone(UTC).isoformat(),
            "purpose": self.purpose,
        }


@dataclass(frozen=True, slots=True)
class PriceFormula:
    base_open: Decimal
    open_step_per_bar: Decimal
    high_offset: Decimal
    low_offset: Decimal
    close_offset: Decimal

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "base_open": str(self.base_open),
            "open_step_per_bar": str(self.open_step_per_bar),
            "high_offset": str(self.high_offset),
            "low_offset": str(self.low_offset),
            "close_offset": str(self.close_offset),
        }


@dataclass(frozen=True, slots=True)
class VolumeFormula:
    base_volume: Decimal
    step_per_bar: Decimal

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "base_volume": str(self.base_volume),
            "step_per_bar": str(self.step_per_bar),
        }


@dataclass(frozen=True, slots=True)
class ExampleFixture:
    """Validated local-only source recipe for the canonical SPY bars."""

    symbol: str
    provider_name: str
    provider_symbol: str
    adapter_version: str
    retrieved_at: datetime
    session_dates: tuple[date, ...]
    price_formula: PriceFormula
    volume_formula: VolumeFormula
    decision_scenarios: tuple[DecisionScenario, ...]

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "schema_version": EXAMPLE_SCHEMA_VERSION,
            "symbol": self.symbol,
            "provider_name": self.provider_name,
            "provider_symbol": self.provider_symbol,
            "adapter_version": self.adapter_version,
            "retrieved_at": self.retrieved_at.astimezone(UTC).isoformat(),
            "session_dates": [value.isoformat() for value in self.session_dates],
            "price_formula": self.price_formula.to_primitive(),
            "volume_formula": self.volume_formula.to_primitive(),
            "decision_scenarios": [
                scenario.to_primitive() for scenario in self.decision_scenarios
            ],
        }

    @property
    def fixture_id(self) -> str:
        return configuration_identity(self.to_primitive())


@dataclass(frozen=True, slots=True)
class ExampleDatasets:
    source: IntradayDataset
    four_hour: AggregatedIntradayDataset
    daily: AggregatedSessionDataset
    weekly: AggregatedSessionDataset
    family: DatasetFamily
    cache: IntradayMarketDataCache
    cache_status: str


@dataclass(frozen=True, slots=True)
class ExampleResult:
    fixture: ExampleFixture
    datasets: ExampleDatasets
    completed_contexts: tuple[tuple[DecisionScenario, MultiTimeframeContext], ...]
    developing_contexts: tuple[tuple[DecisionScenario, MultiTimeframeContext], ...]

    @property
    def example_id(self) -> str:
        return configuration_identity(_example_identity(self))


def _mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise SpyContextExampleError(f"{field_name} must be a JSON object")
    return cast(Mapping[str, object], value)


def _string(mapping: Mapping[str, object], key: str, field_name: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise SpyContextExampleError(f"{field_name}.{key} must be a nonempty string")
    return value


def _decimal(mapping: Mapping[str, object], key: str, field_name: str) -> Decimal:
    try:
        value = Decimal(_string(mapping, key, field_name))
    except InvalidOperation as error:
        raise SpyContextExampleError(
            f"{field_name}.{key} must be a finite decimal string"
        ) from error
    if not value.is_finite():
        raise SpyContextExampleError(
            f"{field_name}.{key} must be a finite decimal string"
        )
    return value


def _aware_datetime(value: str, field_name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise SpyContextExampleError(
            f"{field_name} must be an ISO-8601 timestamp"
        ) from error
    if parsed.utcoffset() is None:
        raise SpyContextExampleError(f"{field_name} must include a UTC offset")
    return parsed


def load_fixture(path: Path) -> ExampleFixture:
    """Load and validate the committed synthetic SPY fixture recipe."""
    try:
        root = _mapping(json.loads(path.read_text(encoding="utf-8")), "fixture")
    except OSError as error:
        raise SpyContextExampleError(f"cannot read fixture: {path}") from error
    except json.JSONDecodeError as error:
        raise SpyContextExampleError(f"fixture is not valid JSON: {path}") from error
    if _string(root, "schema_version", "fixture") != EXAMPLE_SCHEMA_VERSION:
        raise SpyContextExampleError(
            f"fixture schema {EXAMPLE_SCHEMA_VERSION} is required"
        )

    raw_session_dates = root.get("session_dates")
    if not isinstance(raw_session_dates, list) or not raw_session_dates:
        raise SpyContextExampleError("fixture.session_dates must be a nonempty array")
    parsed_session_dates: list[date] = []
    for index, value in enumerate(cast(list[object], raw_session_dates)):
        if not isinstance(value, str):
            raise SpyContextExampleError(
                f"fixture.session_dates[{index}] must be an ISO date string"
            )
        try:
            parsed_session_dates.append(date.fromisoformat(value))
        except ValueError as error:
            raise SpyContextExampleError(
                f"fixture.session_dates[{index}] must be an ISO exchange-session date"
            ) from error
    session_dates = tuple(parsed_session_dates)
    if session_dates != tuple(sorted(set(session_dates))):
        raise SpyContextExampleError(
            "fixture.session_dates must be chronologically unique"
        )
    for session_date in session_dates:
        resolve_exchange_session(session_date)

    price = _mapping(root.get("price_formula"), "fixture.price_formula")
    price_formula = PriceFormula(
        base_open=_decimal(price, "base_open", "fixture.price_formula"),
        open_step_per_bar=_decimal(price, "open_step_per_bar", "fixture.price_formula"),
        high_offset=_decimal(price, "high_offset", "fixture.price_formula"),
        low_offset=_decimal(price, "low_offset", "fixture.price_formula"),
        close_offset=_decimal(price, "close_offset", "fixture.price_formula"),
    )
    if (
        price_formula.base_open <= 0
        or price_formula.open_step_per_bar < 0
        or price_formula.high_offset < max(Decimal(0), price_formula.close_offset)
        or price_formula.low_offset < max(Decimal(0), -price_formula.close_offset)
    ):
        raise SpyContextExampleError("fixture price formula violates OHLC invariants")

    volume = _mapping(root.get("volume_formula"), "fixture.volume_formula")
    volume_formula = VolumeFormula(
        base_volume=_decimal(volume, "base_volume", "fixture.volume_formula"),
        step_per_bar=_decimal(volume, "step_per_bar", "fixture.volume_formula"),
    )
    if volume_formula.base_volume < 0 or volume_formula.step_per_bar < 0:
        raise SpyContextExampleError("fixture volume formula must be nonnegative")

    raw_scenarios = root.get("decision_scenarios")
    if not isinstance(raw_scenarios, list) or not raw_scenarios:
        raise SpyContextExampleError(
            "fixture.decision_scenarios must be a nonempty array"
        )
    scenarios: list[DecisionScenario] = []
    for index, raw_scenario in enumerate(cast(list[object], raw_scenarios)):
        scenario = _mapping(raw_scenario, f"fixture.decision_scenarios[{index}]")
        scenarios.append(
            DecisionScenario(
                name=_string(scenario, "name", "decision scenario"),
                as_of=_aware_datetime(
                    _string(scenario, "as_of", "decision scenario"),
                    "decision scenario as_of",
                ),
                purpose=_string(scenario, "purpose", "decision scenario"),
            )
        )
    if len({scenario.name for scenario in scenarios}) != len(scenarios):
        raise SpyContextExampleError("decision scenario names must be unique")

    fixture = ExampleFixture(
        symbol=_string(root, "symbol", "fixture").upper(),
        provider_name=_string(root, "provider_name", "fixture"),
        provider_symbol=_string(root, "provider_symbol", "fixture"),
        adapter_version=_string(root, "adapter_version", "fixture"),
        retrieved_at=_aware_datetime(
            _string(root, "retrieved_at", "fixture"), "fixture.retrieved_at"
        ),
        session_dates=session_dates,
        price_formula=price_formula,
        volume_formula=volume_formula,
        decision_scenarios=tuple(scenarios),
    )
    first_session = resolve_exchange_session(session_dates[0])
    last_session = resolve_exchange_session(session_dates[-1])
    for scenario in fixture.decision_scenarios:
        as_of = scenario.as_of.astimezone(UTC)
        if not first_session.open_timestamp <= as_of <= last_session.close_timestamp:
            raise SpyContextExampleError(
                f"decision scenario {scenario.name!r} falls outside source coverage"
            )
    return fixture


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
        corporate_action_policy="not_provided_for_synthetic_intraday_fixture",
        adjusted_fields_used=False,
    )


@dataclass(frozen=True, slots=True)
class _FixtureBar:
    session_date: date
    start_timestamp: datetime
    end_timestamp: datetime
    completion: BarCompletion
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal

    def raw_record(self) -> ProviderRecord:
        return {
            "session_date": self.session_date.isoformat(),
            "start_timestamp": self.start_timestamp.isoformat(),
            "end_timestamp": self.end_timestamp.isoformat(),
            "completion": self.completion.value,
            "open": str(self.open),
            "high": str(self.high),
            "low": str(self.low),
            "close": str(self.close),
            "volume": str(self.volume),
        }


def _fixture_bars(fixture: ExampleFixture) -> tuple[_FixtureBar, ...]:
    bars: list[_FixtureBar] = []
    interval = timedelta(minutes=5)
    for session_date in fixture.session_dates:
        session = resolve_exchange_session(session_date)
        start_timestamp = session.open_timestamp
        while start_timestamp < session.close_timestamp:
            end_timestamp = min(start_timestamp + interval, session.close_timestamp)
            sequence = Decimal(len(bars))
            open_price = (
                fixture.price_formula.base_open
                + fixture.price_formula.open_step_per_bar * sequence
            )
            bars.append(
                _FixtureBar(
                    session_date=session_date,
                    start_timestamp=start_timestamp,
                    end_timestamp=end_timestamp,
                    completion=(
                        BarCompletion.COMPLETED
                        if end_timestamp - start_timestamp == interval
                        else BarCompletion.COMPLETED_PARTIAL_DURATION_TERMINAL
                    ),
                    open=open_price,
                    high=open_price + fixture.price_formula.high_offset,
                    low=open_price - fixture.price_formula.low_offset,
                    close=open_price + fixture.price_formula.close_offset,
                    volume=(
                        fixture.volume_formula.base_volume
                        + fixture.volume_formula.step_per_bar * sequence
                    ),
                )
            )
            start_timestamp = end_timestamp
    return tuple(bars)


def _source_request(
    fixture: ExampleFixture, source_timeframe: Timeframe
) -> IntradayBarRequest:
    first = resolve_exchange_session(fixture.session_dates[0])
    last = resolve_exchange_session(fixture.session_dates[-1])
    return IntradayBarRequest(
        symbol=fixture.symbol,
        start_timestamp=first.open_timestamp,
        end_timestamp=last.close_timestamp,
        timeframe=source_timeframe,
        feed_scope=FeedScope.provider_defined("synthetic_regular_hours_fixture"),
        adjustment_basis=_adjustment_basis(),
    )


def _source_fetch_result(fixture: ExampleFixture) -> IntradayFetchResult:
    source_timeframe = _timeframes()[0]
    request = _source_request(fixture, source_timeframe)
    fixture_bars = _fixture_bars(fixture)
    records = tuple(bar.raw_record() for bar in fixture_bars)
    snapshot = IntradayRawSnapshot(
        provider_name=fixture.provider_name,
        provider_symbol=fixture.provider_symbol,
        adapter_version=fixture.adapter_version,
        endpoint="local-fixture://qf-30/spy-5-minute",
        source_request_id=request.request_id,
        chunk_start_timestamp=request.start_timestamp,
        chunk_end_timestamp=request.end_timestamp,
        retrieved_at=fixture.retrieved_at,
        request_parameters=(
            ("calendar", "XNYS"),
            ("fixture_id", fixture.fixture_id),
            ("interval", "5min"),
            ("session_scope", "regular_hours"),
        ),
        records=records,
    )
    provenance = IntradayBarProvenance(
        provider_name=fixture.provider_name,
        provider_symbol=fixture.provider_symbol,
        adapter_version=fixture.adapter_version,
        retrieved_at=fixture.retrieved_at,
        source_request_id=request.request_id,
        source_snapshot_id=snapshot.snapshot_id,
        feed_scope=request.feed_scope,
        adjustment_basis=request.adjustment_basis,
    )
    bars = tuple(
        IntradayBar(
            symbol=fixture.symbol,
            session_date=bar.session_date,
            start_timestamp=bar.start_timestamp,
            end_timestamp=bar.end_timestamp,
            timeframe=source_timeframe,
            completion=bar.completion,
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            volume=bar.volume,
            provenance=provenance,
        )
        for bar in fixture_bars
    )
    capabilities_id = configuration_identity(
        {
            "provider_name": fixture.provider_name,
            "adapter_version": fixture.adapter_version,
            "fixture_id": fixture.fixture_id,
            "supported_timeframe_id": source_timeframe.configuration_id,
            "feed_scope": request.feed_scope.to_primitive(),
        }
    )
    return IntradayFetchResult(
        IntradayBarBatch(request, bars), (snapshot,), capabilities_id
    )


def _load_or_seed_source(
    fixture: ExampleFixture, cache: IntradayMarketDataCache
) -> tuple[IntradayDataset, str]:
    expected = _source_fetch_result(fixture)
    cached = cache.find(fixture.provider_name, expected.batch.request)
    if cached is not None:
        if cached.bars != expected.batch.bars:
            raise SpyContextExampleError(
                "the cache request pointer names bars that differ from the committed "
                "fixture; use a fresh --cache-root"
            )
        return cached, "replayed_local_cache"
    return cache.persist(expected), "seeded_local_cache_from_committed_fixture"


def _context_family(
    source: IntradayDataset,
    derived: tuple[
        AggregatedIntradayDataset,
        AggregatedSessionDataset,
        AggregatedSessionDataset,
    ],
) -> DatasetFamily:
    artifact_family_ids = sorted(
        dataset.dataset_family.manifest_id for dataset in derived
    )
    children = tuple(sorted(dataset.metadata.dataset_id for dataset in derived))
    source_id = source.metadata.dataset_id
    return DatasetFamily(
        canonical_symbol=source.request.symbol,
        provider_name=source.metadata.provider_name,
        feed_scope=source.request.feed_scope,
        adjustment_basis=source.request.adjustment_basis,
        aggregation_policy=AggregationPolicy(
            CONTEXT_FAMILY_POLICY_NAME,
            CONTEXT_FAMILY_POLICY_VERSION,
            cast(
                PrimitiveMapping,
                {"artifact_family_manifest_ids": artifact_family_ids},
            ),
        ),
        canonical_source_snapshot_id=source_id,
        datasets=(
            DatasetLineage(
                dataset_id=source_id,
                timeframe=source.request.timeframe,
                canonical_source_snapshot_id=source_id,
                parent_dataset_id=None,
                child_dataset_ids=children,
            ),
            *(
                DatasetLineage(
                    dataset_id=dataset.metadata.dataset_id,
                    timeframe=(
                        dataset.request.timeframe
                        if isinstance(dataset, AggregatedIntradayDataset)
                        else dataset.metadata.target_timeframe
                    ),
                    canonical_source_snapshot_id=source_id,
                    parent_dataset_id=source_id,
                )
                for dataset in derived
            ),
        ),
    )


def build_datasets(fixture: ExampleFixture, cache_root: Path) -> ExampleDatasets:
    """Build every timeframe from one cache-validated canonical source."""
    cache = IntradayMarketDataCache(cache_root)
    source, cache_status = _load_or_seed_source(fixture, cache)
    _, four_hour_timeframe, daily_timeframe, weekly_timeframe = _timeframes()
    four_hour = aggregate_intraday_dataset(source, four_hour_timeframe)
    daily = aggregate_session_dataset(source, daily_timeframe)
    weekly = aggregate_session_dataset(source, weekly_timeframe)
    family = _context_family(source, (four_hour, daily, weekly))
    return ExampleDatasets(
        source=source,
        four_hour=four_hour,
        daily=daily,
        weekly=weekly,
        family=family,
        cache=cache,
        cache_status=cache_status,
    )


def _series(datasets: ExampleDatasets) -> tuple[TimeframeBarSeries, ...]:
    family = datasets.family
    return (
        TimeframeBarSeries.from_source_dataset(
            datasets.source, family=family, cache=datasets.cache
        ),
        TimeframeBarSeries.from_aggregated_intraday_dataset(
            datasets.four_hour, family=family
        ),
        TimeframeBarSeries.from_aggregated_session_dataset(
            datasets.daily, family=family
        ),
        TimeframeBarSeries.from_aggregated_session_dataset(
            datasets.weekly, family=family
        ),
    )


def _contexts(
    fixture: ExampleFixture,
    datasets: ExampleDatasets,
    completion_policy: ContextCompletionPolicy,
) -> tuple[tuple[DecisionScenario, MultiTimeframeContext], ...]:
    source_timeframe, four_hour, daily, weekly = _timeframes()
    series = _series(datasets)
    requirements = tuple(
        ContextTimeframeRequirement(timeframe)
        for timeframe in (four_hour, daily, weekly)
    )
    return tuple(
        (
            scenario,
            build_multi_timeframe_context(
                as_of=scenario.as_of,
                primary_timeframe=source_timeframe,
                required_timeframes=requirements,
                series=series,
                completion_policy=completion_policy,
            ),
        )
        for scenario in fixture.decision_scenarios
    )


def build_example(fixture_path: Path, cache_root: Path) -> ExampleResult:
    """Build completed and developing contexts without any indicator backend."""
    fixture = load_fixture(fixture_path)
    datasets = build_datasets(fixture, cache_root)
    return ExampleResult(
        fixture=fixture,
        datasets=datasets,
        completed_contexts=_contexts(
            fixture, datasets, ContextCompletionPolicy.COMPLETED_BARS_ONLY
        ),
        developing_contexts=_contexts(
            fixture, datasets, ContextCompletionPolicy.DEVELOPING_BAR_AS_OF
        ),
    )


def _timeframe_name(timeframe: Timeframe) -> str:
    interval = timeframe.interval
    if isinstance(interval, IntradayInterval):
        minutes = int(interval.nominal_duration.total_seconds() // 60)
        return f"{minutes}m" if minutes < 60 else f"{minutes // 60}h"
    if isinstance(interval, SessionInterval):
        return "daily"
    return "weekly"


def _csv_bytes(
    rows: Sequence[Mapping[str, object]], fieldnames: tuple[str, ...]
) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def _intraday_csv(
    dataset_id: str,
    bars: tuple[IntradayBar, ...],
    source_counts: Mapping[str, int],
) -> bytes:
    rows = [
        {
            "timeframe": _timeframe_name(bar.timeframe),
            "dataset_id": dataset_id,
            "bar_id": bar.bar_id,
            "session_date": bar.session_date.isoformat(),
            "start_timestamp": bar.start_timestamp.isoformat(),
            "end_timestamp": bar.end_timestamp.isoformat(),
            "completion": bar.completion.value,
            "actual_duration_minutes": int(bar.actual_duration.total_seconds() // 60),
            "open": str(bar.open),
            "high": str(bar.high),
            "low": str(bar.low),
            "close": str(bar.close),
            "volume": str(bar.volume),
            "source_bar_count": source_counts.get(bar.bar_id, 1),
        }
        for bar in bars
    ]
    return _csv_bytes(
        rows,
        (
            "timeframe",
            "dataset_id",
            "bar_id",
            "session_date",
            "start_timestamp",
            "end_timestamp",
            "completion",
            "actual_duration_minutes",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "source_bar_count",
        ),
    )


def _session_csv(dataset: AggregatedSessionDataset) -> bytes:
    rows = [
        {
            "timeframe": _timeframe_name(bar.timeframe),
            "dataset_id": dataset.metadata.dataset_id,
            "bar_id": bar.bar_id,
            "period_identifier": bar.period_identifier,
            "session_dates": ";".join(value.isoformat() for value in bar.session_dates),
            "start_timestamp": bar.start_timestamp.isoformat(),
            "end_timestamp": bar.end_timestamp.isoformat(),
            "completion": bar.completion.value,
            "open": str(bar.open),
            "high": str(bar.high),
            "low": str(bar.low),
            "close": str(bar.close),
            "volume": str(bar.volume),
            "source_bar_count": len(bar.source_bar_ids),
        }
        for bar in dataset.bars
    ]
    return _csv_bytes(
        rows,
        (
            "timeframe",
            "dataset_id",
            "bar_id",
            "period_identifier",
            "session_dates",
            "start_timestamp",
            "end_timestamp",
            "completion",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "source_bar_count",
        ),
    )


def _bar_payload(bar: ContextBar | None) -> PrimitiveMapping | None:
    if bar is None:
        return None
    return {"bar_id": bar.bar_id, "bar": bar.to_primitive()}


def _context_payload(
    values: tuple[tuple[DecisionScenario, MultiTimeframeContext], ...],
) -> PrimitiveMapping:
    return {
        "schema_version": EXAMPLE_SCHEMA_VERSION,
        "contexts": [
            {
                "scenario": scenario.to_primitive(),
                "context_id": context.context_id,
                "context": context.to_primitive(),
                "latest_bars": {
                    _timeframe_name(metadata.timeframe): _bar_payload(
                        metadata.latest_bar
                    )
                    for metadata in context.timeframes
                },
            }
            for scenario, context in values
        ],
    }


def _pretty_json_bytes(value: Mapping[str, object]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _context_rows(result: ExampleResult) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    policies = (
        ("completed", result.completed_contexts),
        ("developing", result.developing_contexts),
    )
    for policy_name, contexts in policies:
        for scenario, context in contexts:
            for metadata in context.timeframes:
                bar = metadata.latest_bar
                rows.append(
                    {
                        "scenario": scenario.name,
                        "policy": policy_name,
                        "as_of_utc": context.as_of.isoformat(),
                        "timeframe": _timeframe_name(metadata.timeframe),
                        "availability": metadata.availability.value,
                        "completion": "missing"
                        if bar is None
                        else bar.completion.value,
                        "start_utc": ""
                        if bar is None
                        else bar.start_timestamp.isoformat(),
                        "end_utc": "" if bar is None else bar.end_timestamp.isoformat(),
                        "open": "" if bar is None else str(bar.open),
                        "high": "" if bar is None else str(bar.high),
                        "low": "" if bar is None else str(bar.low),
                        "close": "" if bar is None else str(bar.close),
                        "volume": "" if bar is None else str(bar.volume),
                    }
                )
    return rows


def _context_table(result: ExampleResult) -> bytes:
    columns = (
        "scenario",
        "policy",
        "as_of_utc",
        "timeframe",
        "availability",
        "completion",
        "start_utc",
        "end_utc",
        "open",
        "high",
        "low",
        "close",
        "volume",
    )
    lines = [
        "# QF-30 synchronized SPY context table",
        "",
        "This table contains OHLCV context only. The fixture is synthetic, and no",
        "prediction, signal, order, fill, P&L, or future-performance claim is made.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in _context_rows(result):
        lines.append("| " + " | ".join(row[column] for column in columns) + " |")
    lines.append("")
    return "\n".join(lines).encode()


def _example_identity(result: ExampleResult) -> PrimitiveMapping:
    datasets = result.datasets
    return {
        "schema_version": EXAMPLE_SCHEMA_VERSION,
        "artifact_type": "spy_multi_timeframe_context_example",
        "fixture_id": result.fixture.fixture_id,
        "source_dataset_id": datasets.source.metadata.dataset_id,
        "source_quality_report_id": datasets.source.quality_report.report_id,
        "derived_dataset_ids": {
            "4h": datasets.four_hour.metadata.dataset_id,
            "daily": datasets.daily.metadata.dataset_id,
            "weekly": datasets.weekly.metadata.dataset_id,
        },
        "dataset_family_id": datasets.family.family_id,
        "dataset_family_manifest_id": datasets.family.manifest_id,
        "completed_context_ids": [
            context.context_id for _, context in result.completed_contexts
        ],
        "developing_context_ids": [
            context.context_id for _, context in result.developing_contexts
        ],
        "decision_scenarios": [
            scenario.to_primitive() for scenario in result.fixture.decision_scenarios
        ],
        "indicator_backend": None,
        "prediction_or_trading_result": None,
    }


def _artifact_bytes(result: ExampleResult) -> dict[str, bytes]:
    datasets = result.datasets
    four_hour_counts = {
        window.output_bar_id: window.observed_constituent_count
        for window in datasets.four_hour.aggregation_report.windows
        if window.output_bar_id is not None
    }
    artifacts = {
        "bars_5m.csv": _intraday_csv(
            datasets.source.metadata.dataset_id, datasets.source.bars, {}
        ),
        "bars_4h.csv": _intraday_csv(
            datasets.four_hour.metadata.dataset_id,
            datasets.four_hour.bars,
            four_hour_counts,
        ),
        "bars_daily.csv": _session_csv(datasets.daily),
        "bars_weekly.csv": _session_csv(datasets.weekly),
        "contexts_completed.json": _pretty_json_bytes(
            _context_payload(result.completed_contexts)
        ),
        "contexts_developing.json": _pretty_json_bytes(
            _context_payload(result.developing_contexts)
        ),
        "context_table.md": _context_table(result),
        "source_manifest.json": (
            datasets.cache.root
            / "intraday"
            / "datasets"
            / datasets.source.metadata.dataset_id
            / "manifest.json"
        ).read_bytes(),
        "derived_4h_manifest.json": datasets.four_hour.serialize_manifest(),
        "derived_daily_manifest.json": datasets.daily.serialize_manifest(),
        "derived_weekly_manifest.json": datasets.weekly.serialize_manifest(),
        "dataset_family_manifest.json": datasets.family.serialize_manifest(),
    }
    artifact_hashes = {
        name: {"sha256": sha256_hex(content), "size_bytes": len(content)}
        for name, content in sorted(artifacts.items())
    }
    manifest: dict[str, object] = {
        **_example_identity(result),
        "example_id": result.example_id,
        "source": {
            "kind": "synthetic_redistributable_fixture",
            "canonical_symbol": result.fixture.symbol,
            "provider_name": result.fixture.provider_name,
            "feed_scope": datasets.source.request.feed_scope.to_primitive(),
            "request": datasets.source.request.to_primitive(),
            "bar_count": len(datasets.source.bars),
        },
        "artifacts": artifact_hashes,
        "limitations": [
            (
                "OHLCV values are deterministic synthetic fixture values, not "
                "observed market prices."
            ),
            (
                "The example contains no indicators, predictions, signals, "
                "orders, fills, or P&L."
            ),
            (
                "The example validates aggregation, lineage, cache replay, and "
                "temporal alignment only."
            ),
        ],
    }
    artifacts["manifest.json"] = _pretty_json_bytes(manifest)
    return artifacts


def _validate_existing_export(path: Path, expected: Mapping[str, bytes]) -> None:
    if not path.is_dir():
        raise SpyContextExampleError(
            f"immutable export path is not a directory: {path}"
        )
    actual_names = {entry.name for entry in path.iterdir() if entry.is_file()}
    expected_names = set(expected)
    if actual_names != expected_names:
        raise SpyContextExampleError(
            f"immutable export file set differs at {path}: "
            f"expected {sorted(expected_names)}, found {sorted(actual_names)}"
        )
    for name, content in expected.items():
        if (path / name).read_bytes() != content:
            raise SpyContextExampleError(
                f"immutable export content differs for {path / name}"
            )


def export_example(result: ExampleResult, output_root: Path) -> tuple[Path, str]:
    """Create one content-addressed export or verify its exact prior copy."""
    expected = _artifact_bytes(result)
    destination = output_root / result.example_id
    if destination.exists():
        _validate_existing_export(destination, expected)
        return destination, "reused_immutable_export"
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".qf30-", dir=output_root) as temporary:
        staging = Path(temporary)
        for name, content in expected.items():
            (staging / name).write_bytes(content)
        try:
            os.rename(staging, destination)
        except OSError:
            if not destination.exists():
                raise
            _validate_existing_export(destination, expected)
            return destination, "reused_immutable_export"
    return destination, "created_immutable_export"


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixture",
        type=Path,
        default=DEFAULT_FIXTURE_PATH,
        help="committed deterministic SPY fixture recipe",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=DEFAULT_CACHE_ROOT,
        help="local immutable QF-16 intraday cache root",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="root for the content-addressed QF-30 export",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = parse_arguments(arguments)
    result = build_example(parsed.fixture, parsed.cache_root)
    destination, export_status = export_example(result, parsed.output_root)
    print(
        json.dumps(
            {
                "cache_status": result.datasets.cache_status,
                "completed_context_count": len(result.completed_contexts),
                "dataset_family_id": result.datasets.family.family_id,
                "developing_context_count": len(result.developing_contexts),
                "example_id": result.example_id,
                "export_status": export_status,
                "indicator_backend": None,
                "output": str(destination.resolve()),
                "source_bar_count": len(result.datasets.source.bars),
                "source_dataset_id": result.datasets.source.metadata.dataset_id,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SpyContextExampleError as error:
        raise SystemExit(f"SPY context example error: {error}") from None
