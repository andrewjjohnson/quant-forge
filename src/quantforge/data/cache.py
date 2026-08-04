"""Small immutable filesystem cache for daily datasets."""

import csv
import hashlib
import json
import os
import tempfile
from dataclasses import asdict
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

from quantforge.data.exceptions import CacheError
from quantforge.data.models import (
    SCHEMA_VERSION,
    AdjustmentMode,
    DailyBar,
    DatasetMetadata,
    MarketDataset,
    ProviderResponse,
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def request_key(
    provider: str,
    symbol: str,
    start: date,
    end: date,
    adjustment: AdjustmentMode,
    calendar: str,
    *,
    strict: bool,
) -> str:
    """Hash every input that materially controls normalized output."""
    return _sha256(
        _canonical_json(
            {
                "provider": provider,
                "symbol": symbol,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "adjustment": adjustment.value,
                "calendar": calendar,
                "schema_version": SCHEMA_VERSION,
                "strict": strict,
            }
        )
    )


class MarketDataCache:
    """Atomic, content-addressed raw/normalized artifacts and request indexes."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def find(self, key: str) -> MarketDataset | None:
        index = self.root / "requests" / f"{key}.json"
        if not index.exists():
            return None
        try:
            value = json.loads(index.read_text())
            dataset_id = str(value["dataset_id"])
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise CacheError(f"corrupt cache index: {index}") from error
        return self.load(dataset_id)

    def persist(
        self,
        response: ProviderResponse,
        bars: tuple[DailyBar, ...],
        metadata_values: dict[str, object],
        key: str,
    ) -> MarketDataset:
        """Persist a new entry without replacing any existing artifact."""
        raw_value = {
            "provider_name": response.provider_name,
            "provider_symbol": response.provider_symbol,
            "retrieved_at": response.retrieved_at.astimezone(UTC).isoformat(),
            "provider_timezone": response.provider_timezone,
            "adjustment_mode": response.adjustment_mode.value,
            "records": response.records,
            "metadata": response.metadata,
            "adapter_version": response.adapter_version,
        }
        raw_bytes = _canonical_json(raw_value)
        raw_hash = _sha256(raw_bytes)
        csv_text = "symbol,session_date,open,high,low,close,volume\n" + "".join(
            f"{bar.symbol},{bar.session_date.isoformat()},{bar.open},{bar.high},{bar.low},{bar.close},{bar.volume}\n"
            for bar in bars
        )
        data_bytes = csv_text.encode()
        data_hash = _sha256(data_bytes)
        identity = {
            **_serialize_metadata_values(metadata_values),
            "raw_sha256": raw_hash,
            "data_sha256": data_hash,
            "schema_version": SCHEMA_VERSION,
        }
        dataset_id = _sha256(_canonical_json(identity))
        relative_raw = f"raw/{raw_hash}.json"
        relative_data = f"datasets/{dataset_id}/bars.csv"
        metadata = DatasetMetadata(
            **cast(Any, metadata_values),
            raw_location=relative_raw,
            normalized_location=relative_data,
            dataset_id=dataset_id,
            schema_version=SCHEMA_VERSION,
        )
        manifest_value = _metadata_to_dict(metadata) | {
            "raw_sha256": raw_hash,
            "data_sha256": data_hash,
        }
        self._write_once(self.root / relative_raw, raw_bytes)
        self._write_once(self.root / relative_data, data_bytes)
        self._write_once(
            self.root / "datasets" / dataset_id / "manifest.json",
            _canonical_json(manifest_value),
        )
        self._write_once(
            self.root / "requests" / f"{key}.json",
            _canonical_json({"dataset_id": dataset_id}),
        )
        return MarketDataset(bars, metadata)

    def load(self, dataset_id: str) -> MarketDataset:
        """Reload a complete entry and verify paths and content hashes."""
        directory = self.root / "datasets" / dataset_id
        manifest_path = directory / "manifest.json"
        try:
            manifest: dict[str, Any] = json.loads(manifest_path.read_text())
            if manifest["dataset_id"] != dataset_id:
                raise CacheError("manifest dataset identifier mismatch")
            data_path = self.root / str(manifest["normalized_location"])
            raw_path = self.root / str(manifest["raw_location"])
            data_bytes, raw_bytes = data_path.read_bytes(), raw_path.read_bytes()
            if (
                _sha256(data_bytes) != manifest["data_sha256"]
                or _sha256(raw_bytes) != manifest["raw_sha256"]
            ):
                raise CacheError("cached artifact checksum mismatch")
            with data_path.open(newline="") as stream:
                bars = tuple(
                    DailyBar(
                        row["symbol"],
                        date.fromisoformat(row["session_date"]),
                        Decimal(row["open"]),
                        Decimal(row["high"]),
                        Decimal(row["low"]),
                        Decimal(row["close"]),
                        Decimal(row["volume"]),
                    )
                    for row in csv.DictReader(stream)
                )
            metadata = _metadata_from_dict(manifest)
        except CacheError:
            raise
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise CacheError(
                f"incomplete or corrupt cache entry: {dataset_id}"
            ) from error
        if len(bars) != metadata.bar_count:
            raise CacheError("cached bar count differs from manifest")
        return MarketDataset(bars, metadata)

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


def _metadata_to_dict(metadata: DatasetMetadata) -> dict[str, object]:
    return _serialize_metadata_values(asdict(metadata))


def _serialize_metadata_values(
    metadata_values: dict[str, object],
) -> dict[str, object]:
    """Return metadata in the canonical JSON-compatible representation."""
    value = metadata_values.copy()
    for field in (
        "requested_start",
        "requested_end",
        "actual_first_session",
        "actual_last_session",
    ):
        value[field] = cast(date, value[field]).isoformat()
    value["retrieved_at"] = (
        cast(datetime, value["retrieved_at"]).astimezone(UTC).isoformat()
    )
    value["adjustment_mode"] = cast(AdjustmentMode, value["adjustment_mode"]).value
    value["missing_sessions"] = [
        item.isoformat() for item in cast(tuple[date, ...], value["missing_sessions"])
    ]
    value["split_sessions"] = [
        item.isoformat() for item in cast(tuple[date, ...], value["split_sessions"])
    ]
    return value


def _metadata_from_dict(value: dict[str, Any]) -> DatasetMetadata:
    return DatasetMetadata(
        canonical_symbol=str(value["canonical_symbol"]),
        provider_name=str(value["provider_name"]),
        provider_symbol=str(value["provider_symbol"]),
        retrieved_at=datetime.fromisoformat(value["retrieved_at"]).astimezone(UTC),
        requested_start=date.fromisoformat(value["requested_start"]),
        requested_end=date.fromisoformat(value["requested_end"]),
        actual_first_session=date.fromisoformat(value["actual_first_session"]),
        actual_last_session=date.fromisoformat(value["actual_last_session"]),
        calendar=str(value["calendar"]),
        provider_timezone=value["provider_timezone"],
        adjustment_mode=AdjustmentMode(value["adjustment_mode"]),
        raw_location=str(value["raw_location"]),
        normalized_location=str(value["normalized_location"]),
        dataset_id=str(value["dataset_id"]),
        schema_version=str(value["schema_version"]),
        bar_count=int(value["bar_count"]),
        missing_sessions=tuple(
            date.fromisoformat(item) for item in value["missing_sessions"]
        ),
        split_sessions=tuple(
            date.fromisoformat(item) for item in value["split_sessions"]
        ),
        adapter_version=str(value["adapter_version"]),
    )
