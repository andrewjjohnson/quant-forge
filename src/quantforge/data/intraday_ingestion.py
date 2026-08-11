"""Immutable intraday acquisition, cache, and orchestration contracts."""

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Protocol, cast

from quantforge.configuration import PrimitiveMapping
from quantforge.data.exceptions import CacheError, ProviderError, RequestError
from quantforge.data.identity import canonical_json_bytes, sha256_hex
from quantforge.data.intraday import (
    IntradayBar,
    IntradayBarBatch,
    IntradayBarProvenance,
    IntradayBarRequest,
    IntradayProviderCapabilities,
)
from quantforge.data.models import JsonValue, ProviderRecord
from quantforge.timeframes import BarCompletion

INTRADAY_DATASET_SCHEMA_VERSION = "1"


def _validated_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field_name} must be a nonempty trimmed string")
    return value


def _utc_timestamp(value: object, field_name: str) -> datetime:
    if not isinstance(value, datetime) or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be a timezone-aware datetime")
    return value.astimezone(UTC)


def _json_records(records: tuple[ProviderRecord, ...]) -> list[dict[str, JsonValue]]:
    return [dict(record) for record in records]


@dataclass(frozen=True, slots=True)
class IntradayRawSnapshot:
    """One lossless JSON-compatible provider response for a bounded chunk."""

    provider_name: str
    provider_symbol: str
    adapter_version: str
    endpoint: str
    source_request_id: str
    chunk_start_timestamp: datetime
    chunk_end_timestamp: datetime
    retrieved_at: datetime
    request_parameters: tuple[tuple[str, str], ...]
    records: tuple[ProviderRecord, ...]

    def __post_init__(self) -> None:
        _validated_text(self.provider_name, "raw provider name")
        _validated_text(self.provider_symbol, "raw provider symbol")
        _validated_text(self.adapter_version, "raw adapter version")
        _validated_text(self.endpoint, "raw endpoint")
        _validated_text(self.source_request_id, "raw source request ID")
        start = _utc_timestamp(self.chunk_start_timestamp, "raw chunk start")
        end = _utc_timestamp(self.chunk_end_timestamp, "raw chunk end")
        retrieved_at = _utc_timestamp(self.retrieved_at, "raw retrieval timestamp")
        if start >= end:
            raise ValueError("raw chunk start must be earlier than end")
        parameters = tuple(sorted(self.request_parameters))
        if len({name for name, _ in parameters}) != len(parameters):
            raise ValueError("raw request parameter names must be unique")
        for name, parameter_value in parameters:
            _validated_text(name, "raw request parameter name")
            _validated_text(parameter_value, "raw request parameter value")
        object.__setattr__(self, "chunk_start_timestamp", start)
        object.__setattr__(self, "chunk_end_timestamp", end)
        object.__setattr__(self, "retrieved_at", retrieved_at)
        object.__setattr__(self, "request_parameters", parameters)

    def to_primitive(self) -> PrimitiveMapping:
        """Return raw payload and non-secret request provenance."""
        return cast(
            PrimitiveMapping,
            {
                "schema_version": INTRADAY_DATASET_SCHEMA_VERSION,
                "artifact_type": "intraday_raw_snapshot",
                "provider_name": self.provider_name,
                "provider_symbol": self.provider_symbol,
                "adapter_version": self.adapter_version,
                "endpoint": self.endpoint,
                "source_request_id": self.source_request_id,
                "chunk_start_timestamp": self.chunk_start_timestamp.isoformat(),
                "chunk_end_timestamp": self.chunk_end_timestamp.isoformat(),
                "retrieved_at": self.retrieved_at.isoformat(),
                "request_parameters": dict(self.request_parameters),
                "records": _json_records(self.records),
            },
        )

    def serialize(self) -> bytes:
        return canonical_json_bytes(self.to_primitive())

    @property
    def snapshot_id(self) -> str:
        return sha256_hex(self.serialize())


@dataclass(frozen=True, slots=True)
class IntradayFetchResult:
    """Canonical bars paired with every immutable raw response used."""

    batch: IntradayBarBatch
    raw_snapshots: tuple[IntradayRawSnapshot, ...]
    capabilities_configuration_id: str

    def __post_init__(self) -> None:
        if not self.raw_snapshots:
            raise ValueError("intraday fetch result requires raw snapshots")
        _validated_text(
            self.capabilities_configuration_id,
            "intraday capabilities configuration ID",
        )
        request = self.batch.request
        snapshots = self.raw_snapshots
        if snapshots[0].chunk_start_timestamp != request.start_timestamp:
            raise ValueError("raw chunks must start at the request boundary")
        if snapshots[-1].chunk_end_timestamp != request.end_timestamp:
            raise ValueError("raw chunks must end at the request boundary")
        provider_identity = (
            snapshots[0].provider_name,
            snapshots[0].provider_symbol,
            snapshots[0].adapter_version,
            snapshots[0].endpoint,
        )
        for index, snapshot in enumerate(snapshots):
            if snapshot.source_request_id != request.request_id:
                raise ValueError("raw snapshot request identity mismatch")
            if (
                snapshot.provider_name,
                snapshot.provider_symbol,
                snapshot.adapter_version,
                snapshot.endpoint,
            ) != provider_identity:
                raise ValueError("raw chunks must use one provider endpoint revision")
            if index and snapshots[index - 1].chunk_end_timestamp != (
                snapshot.chunk_start_timestamp
            ):
                raise ValueError("raw chunks must be ordered and contiguous")
        snapshots_by_id = {snapshot.snapshot_id: snapshot for snapshot in snapshots}
        if len(snapshots_by_id) != len(snapshots):
            raise ValueError("raw snapshot identities must be unique")
        for bar in self.batch.bars:
            snapshot = snapshots_by_id.get(bar.provenance.source_snapshot_id)
            if snapshot is None:
                raise ValueError("bar provenance names an unknown raw snapshot")
            if not (
                snapshot.chunk_start_timestamp
                <= bar.start_timestamp
                < snapshot.chunk_end_timestamp
            ):
                raise ValueError(
                    "bar start falls outside its referenced raw snapshot chunk"
                )
            if (
                bar.provenance.provider_name,
                bar.provenance.provider_symbol,
                bar.provenance.adapter_version,
                bar.provenance.retrieved_at,
                bar.provenance.source_request_id,
            ) != (
                snapshot.provider_name,
                snapshot.provider_symbol,
                snapshot.adapter_version,
                snapshot.retrieved_at,
                snapshot.source_request_id,
            ):
                raise ValueError(
                    "bar provenance does not match its referenced raw snapshot"
                )


class IntradayIngestionProvider(Protocol):
    """Provider boundary required by the cache-aware ingestion service."""

    name: str
    intraday_capabilities: IntradayProviderCapabilities

    def fetch_intraday(self, request: IntradayBarRequest) -> IntradayFetchResult: ...


@dataclass(frozen=True, slots=True)
class IntradayDatasetMetadata:
    """Manifest facts for one immutable normalized intraday dataset."""

    dataset_id: str
    request_id: str
    provider_name: str
    provider_symbol: str
    adapter_version: str
    retrieved_at: datetime
    capabilities_configuration_id: str
    batch_id: str
    bar_count: int
    raw_snapshot_ids: tuple[str, ...]
    raw_locations: tuple[str, ...]
    normalized_location: str
    data_sha256: str
    schema_version: str = INTRADAY_DATASET_SCHEMA_VERSION


@dataclass(frozen=True, slots=True)
class IntradayDataset:
    """Canonical intraday bars permanently bound to an immutable manifest."""

    request: IntradayBarRequest
    bars: tuple[IntradayBar, ...]
    metadata: IntradayDatasetMetadata


class IntradayMarketDataCache:
    """Content-addressed intraday raw chunks, batches, and request pointers."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def find(
        self, provider_name: str, request: IntradayBarRequest
    ) -> IntradayDataset | None:
        index = self._request_index(provider_name, request.request_id)
        if not index.exists():
            return None
        try:
            value = _string_mapping(
                cast(object, json.loads(index.read_text())), "request index"
            )
            dataset_id = _json_string(value, "dataset_id", "request index")
        except (OSError, ValueError, TypeError) as error:
            raise CacheError(f"corrupt intraday cache index: {index}") from error
        return self.load(dataset_id, request)

    def persist(
        self,
        result: IntradayFetchResult,
        *,
        replace_request_index: bool = False,
    ) -> IntradayDataset:
        request = result.batch.request
        normalized_bytes = result.batch.serialize()
        data_sha256 = sha256_hex(normalized_bytes)
        identity = self._identity_value(result, data_sha256)
        dataset_id = sha256_hex(canonical_json_bytes(identity))
        normalized_location = f"intraday/datasets/{dataset_id}/bars.json"
        raw_locations = tuple(
            f"intraday/raw/{snapshot.snapshot_id}.json"
            for snapshot in result.raw_snapshots
        )
        manifest = {
            **identity,
            "dataset_id": dataset_id,
            "normalized_location": normalized_location,
            "chunks": [
                {
                    **cast(dict[str, object], chunk),
                    "raw_location": raw_location,
                }
                for chunk, raw_location in zip(
                    cast(list[object], identity["chunks"]),
                    raw_locations,
                    strict=True,
                )
            ],
        }
        for snapshot, raw_location in zip(
            result.raw_snapshots, raw_locations, strict=True
        ):
            self._write_once(self.root / raw_location, snapshot.serialize())
        self._write_once(self.root / normalized_location, normalized_bytes)
        self._write_once(
            self.root / "intraday" / "datasets" / dataset_id / "manifest.json",
            canonical_json_bytes(manifest),
        )
        request_index = self._request_index(
            result.raw_snapshots[0].provider_name, request.request_id
        )
        request_content = canonical_json_bytes({"dataset_id": dataset_id})
        if replace_request_index:
            self._replace(request_index, request_content)
        else:
            self._write_once(request_index, request_content)
        return self.load(dataset_id, request)

    def load(self, dataset_id: str, request: IntradayBarRequest) -> IntradayDataset:
        directory = self.root / "intraday" / "datasets" / dataset_id
        try:
            manifest_value = json.loads((directory / "manifest.json").read_text())
            manifest = _string_mapping(manifest_value, "intraday manifest")
            if manifest.get("dataset_id") != dataset_id:
                raise CacheError("intraday manifest dataset identifier mismatch")
            if manifest.get("schema_version") != INTRADAY_DATASET_SCHEMA_VERSION:
                raise CacheError("unsupported intraday dataset schema")
            request_value = _string_mapping(manifest["request"], "manifest request")
            if request_value.get("request_id") != request.request_id or (
                request_value.get("configuration") != request.to_primitive()
            ):
                raise CacheError("intraday manifest request mismatch")
            normalized_location = _json_string(
                manifest, "normalized_location", "intraday manifest"
            )
            if normalized_location != f"intraday/datasets/{dataset_id}/bars.json":
                raise CacheError("intraday normalized path is not canonical")
            normalized_bytes = (self.root / normalized_location).read_bytes()
            normalized_value = json.loads(normalized_bytes)
            chunks = _mapping_list(manifest["chunks"], "manifest chunks")
            raw_snapshot_ids: list[str] = []
            raw_locations: list[str] = []
            for chunk in chunks:
                snapshot_id = _json_string(chunk, "raw_snapshot_id", "chunk")
                raw_location = _json_string(chunk, "raw_location", "chunk")
                if raw_location != f"intraday/raw/{snapshot_id}.json":
                    raise CacheError("intraday raw path is not canonical")
                raw_bytes = (self.root / raw_location).read_bytes()
                if sha256_hex(raw_bytes) != snapshot_id:
                    raise CacheError("intraday raw artifact checksum mismatch")
                raw_snapshot_ids.append(snapshot_id)
                raw_locations.append(raw_location)
            if sha256_hex(normalized_bytes) != manifest.get("data_sha256"):
                raise CacheError("intraday normalized artifact checksum mismatch")
            identity = _manifest_identity(manifest)
            if sha256_hex(canonical_json_bytes(identity)) != dataset_id:
                raise CacheError("intraday dataset identity mismatch")
            batch = _batch_from_primitive(normalized_value, request)
            if batch.batch_id != manifest.get("batch_id"):
                raise CacheError("intraday batch identity mismatch")
            metadata = IntradayDatasetMetadata(
                dataset_id=dataset_id,
                request_id=request.request_id,
                provider_name=_json_string(manifest, "provider_name", "manifest"),
                provider_symbol=_json_string(manifest, "provider_symbol", "manifest"),
                adapter_version=_json_string(manifest, "adapter_version", "manifest"),
                retrieved_at=_parse_utc(
                    _json_string(manifest, "retrieved_at", "manifest")
                ),
                capabilities_configuration_id=_json_string(
                    manifest, "capabilities_configuration_id", "manifest"
                ),
                batch_id=batch.batch_id,
                bar_count=_json_integer(manifest, "bar_count", "manifest"),
                raw_snapshot_ids=tuple(raw_snapshot_ids),
                raw_locations=tuple(raw_locations),
                normalized_location=normalized_location,
                data_sha256=_json_string(manifest, "data_sha256", "manifest"),
            )
        except CacheError:
            raise
        except (KeyError, OSError, TypeError, ValueError) as error:
            raise CacheError(
                f"incomplete or corrupt intraday cache entry: {dataset_id}"
            ) from error
        if metadata.bar_count != len(batch.bars):
            raise CacheError("intraday manifest bar count mismatch")
        return IntradayDataset(request, batch.bars, metadata)

    @staticmethod
    def _identity_value(
        result: IntradayFetchResult, data_sha256: str
    ) -> dict[str, object]:
        batch = result.batch
        first = result.raw_snapshots[0]
        return {
            "schema_version": INTRADAY_DATASET_SCHEMA_VERSION,
            "artifact_type": "intraday_dataset_manifest",
            "provider_name": first.provider_name,
            "provider_symbol": first.provider_symbol,
            "adapter_version": first.adapter_version,
            "retrieved_at": max(
                snapshot.retrieved_at for snapshot in result.raw_snapshots
            ).isoformat(),
            "request": {
                "request_id": batch.request.request_id,
                "configuration": batch.request.to_primitive(),
            },
            "feed_scope": batch.request.feed_scope.to_primitive(),
            "source_interval": batch.request.source_interval.to_primitive(),
            "session_scope": batch.request.timeframe.session_policy.scope.value,
            "capabilities_configuration_id": (result.capabilities_configuration_id),
            "chunks": [
                {
                    "chunk_index": index,
                    "chunk_start_timestamp": (
                        snapshot.chunk_start_timestamp.isoformat()
                    ),
                    "chunk_end_timestamp": snapshot.chunk_end_timestamp.isoformat(),
                    "retrieved_at": snapshot.retrieved_at.isoformat(),
                    "endpoint": snapshot.endpoint,
                    "raw_snapshot_id": snapshot.snapshot_id,
                    "raw_sha256": snapshot.snapshot_id,
                }
                for index, snapshot in enumerate(result.raw_snapshots)
            ],
            "batch_id": batch.batch_id,
            "bar_count": len(batch.bars),
            "data_sha256": data_sha256,
        }

    def _request_index(self, provider_name: str, request_id: str) -> Path:
        provider = _validated_text(provider_name, "provider name")
        if not all(character.isalnum() or character in "_-" for character in provider):
            raise RequestError("provider name is not safe for cache lookup")
        return self.root / "intraday" / "requests" / provider / f"{request_id}.json"

    @staticmethod
    def _write_once(path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.read_bytes() != content:
                raise CacheError(f"immutable artifact collision: {path}")
            return
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}."
        )
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary_name, path)
            except FileExistsError:
                if path.read_bytes() != content:
                    raise CacheError(f"immutable artifact collision: {path}")
        finally:
            Path(temporary_name).unlink(missing_ok=True)

    @staticmethod
    def _replace(path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}."
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)


class IntradayMarketDataService:
    """Reuse cached intraday data before requiring a provider or credential."""

    def __init__(
        self,
        cache: IntradayMarketDataCache,
        *,
        provider: IntradayIngestionProvider | None = None,
        provider_name: str | None = None,
    ) -> None:
        if provider is None and provider_name is None:
            raise RequestError("provider or provider_name is required")
        if provider is not None and provider_name not in (None, provider.name):
            raise RequestError("provider_name does not match the configured provider")
        self.cache = cache
        self.provider = provider
        self.provider_name = (
            provider.name if provider is not None else cast(str, provider_name)
        )

    def get_intraday_bars(
        self, request: IntradayBarRequest, *, refresh: bool = False
    ) -> IntradayDataset:
        """Return an immutable dataset, using cache before provider access."""
        if not refresh:
            cached = self.cache.find(self.provider_name, request)
            if cached is not None:
                return cached
        if self.provider is None:
            raise RequestError(
                "intraday request is not cached and no provider is configured"
            )
        self.provider.intraday_capabilities.validate_request(request)
        try:
            result = self.provider.fetch_intraday(request)
        except (ProviderError, RequestError):
            raise
        except Exception as error:
            raise ProviderError(
                f"{self.provider.name} intraday provider failed"
            ) from error
        return self.cache.persist(result, replace_request_index=refresh)


def _string_mapping(value: object, field_name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    mapping = cast(dict[object, object], value)
    if any(not isinstance(key, str) for key in mapping):
        raise ValueError(f"{field_name} must be a JSON object")
    return cast(dict[str, object], mapping)


def _mapping_list(value: object, field_name: str) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a JSON array")
    return [_string_mapping(item, field_name) for item in cast(list[object], value)]


def _json_string(mapping: dict[str, object], key: str, field_name: str) -> str:
    value = mapping[key]
    if not isinstance(value, str):
        raise ValueError(f"{field_name} {key} must be a JSON string")
    return value


def _json_integer(mapping: dict[str, object], key: str, field_name: str) -> int:
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} {key} must be a JSON integer")
    return value


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.utcoffset() is None:
        raise ValueError("cached timestamp must include a UTC offset")
    return parsed.astimezone(UTC)


def _manifest_identity(manifest: dict[str, object]) -> dict[str, object]:
    identity = manifest.copy()
    del identity["dataset_id"]
    del identity["normalized_location"]
    identity["chunks"] = [
        {key: value for key, value in chunk.items() if key != "raw_location"}
        for chunk in _mapping_list(identity["chunks"], "manifest chunks")
    ]
    return identity


def _batch_from_primitive(
    value: object, request: IntradayBarRequest
) -> IntradayBarBatch:
    mapping = _string_mapping(value, "normalized batch")
    request_value = _string_mapping(mapping["request"], "normalized request")
    if request_value.get("request_id") != request.request_id or (
        request_value.get("configuration") != request.to_primitive()
    ):
        raise CacheError("normalized intraday request mismatch")
    bars: list[IntradayBar] = []
    for item in _mapping_list(mapping["bars"], "normalized bars"):
        primitive = _string_mapping(item["bar"], "normalized bar")
        provenance_value = _string_mapping(
            primitive["provenance"], "normalized provenance"
        )
        if provenance_value.get("feed_scope") != request.feed_scope.to_primitive():
            raise CacheError("normalized intraday feed scope mismatch")
        if (
            provenance_value.get("adjustment_basis")
            != request.adjustment_basis.to_primitive()
        ):
            raise CacheError("normalized intraday adjustment basis mismatch")
        timeframe_value = _string_mapping(
            primitive["timeframe"], "normalized timeframe"
        )
        if timeframe_value.get("configuration") != request.timeframe.to_primitive():
            raise CacheError("normalized intraday timeframe mismatch")
        provenance = IntradayBarProvenance(
            provider_name=_json_string(provenance_value, "provider_name", "provenance"),
            provider_symbol=_json_string(
                provenance_value, "provider_symbol", "provenance"
            ),
            adapter_version=_json_string(
                provenance_value, "adapter_version", "provenance"
            ),
            retrieved_at=_parse_utc(
                _json_string(provenance_value, "retrieved_at", "provenance")
            ),
            source_request_id=_json_string(
                provenance_value, "source_request_id", "provenance"
            ),
            source_snapshot_id=_json_string(
                provenance_value, "source_snapshot_id", "provenance"
            ),
            feed_scope=request.feed_scope,
            adjustment_basis=request.adjustment_basis,
        )
        bar = IntradayBar(
            symbol=_json_string(primitive, "symbol", "bar"),
            session_date=datetime.fromisoformat(
                _json_string(primitive, "session_identifier", "bar")
            ).date(),
            start_timestamp=_parse_utc(
                _json_string(primitive, "start_timestamp", "bar")
            ),
            end_timestamp=_parse_utc(_json_string(primitive, "end_timestamp", "bar")),
            timeframe=request.timeframe,
            completion=BarCompletion(_json_string(primitive, "completion", "bar")),
            open=Decimal(_json_string(primitive, "open", "bar")),
            high=Decimal(_json_string(primitive, "high", "bar")),
            low=Decimal(_json_string(primitive, "low", "bar")),
            close=Decimal(_json_string(primitive, "close", "bar")),
            volume=Decimal(_json_string(primitive, "volume", "bar")),
            provenance=provenance,
        )
        if item.get("bar_id") != bar.bar_id:
            raise CacheError("normalized intraday bar identity mismatch")
        bars.append(bar)
    return IntradayBarBatch(request, tuple(bars))


__all__ = [
    "INTRADAY_DATASET_SCHEMA_VERSION",
    "IntradayDataset",
    "IntradayDatasetMetadata",
    "IntradayFetchResult",
    "IntradayIngestionProvider",
    "IntradayMarketDataCache",
    "IntradayMarketDataService",
    "IntradayRawSnapshot",
]
