"""Small immutable filesystem cache for daily datasets."""

import csv
import json
import os
import tempfile
from dataclasses import asdict
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

from quantforge.data.exceptions import CacheError
from quantforge.data.identity import (
    calculate_dataset_id,
    canonical_json_bytes,
    dataset_identity_matches,
    serialize_bars_csv,
    serialize_metadata_values,
    sha256_hex,
)
from quantforge.data.models import (
    SCHEMA_VERSION,
    AdjustmentMode,
    DailyBar,
    DatasetMetadata,
    MarketDataset,
    ProviderResponse,
)


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
    return sha256_hex(
        canonical_json_bytes(
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
        raw_bytes = canonical_json_bytes(raw_value)
        raw_hash = sha256_hex(raw_bytes)
        data_bytes = serialize_bars_csv(bars)
        data_hash = sha256_hex(data_bytes)
        dataset_id = calculate_dataset_id(
            metadata_values,
            raw_sha256=raw_hash,
            data_sha256=data_hash,
            schema_version=SCHEMA_VERSION,
        )
        relative_raw = f"raw/{raw_hash}.json"
        relative_data = f"datasets/{dataset_id}/bars.csv"
        metadata = DatasetMetadata(
            **cast(Any, metadata_values),
            raw_location=relative_raw,
            normalized_location=relative_data,
            raw_sha256=raw_hash,
            data_sha256=data_hash,
            dataset_id=dataset_id,
            schema_version=SCHEMA_VERSION,
        )
        manifest_value = _metadata_to_dict(metadata)
        self._write_once(self.root / relative_raw, raw_bytes)
        self._write_once(self.root / relative_data, data_bytes)
        self._write_once(
            self.root / "datasets" / dataset_id / "manifest.json",
            canonical_json_bytes(manifest_value),
        )
        self._write_once(
            self.root / "requests" / f"{key}.json",
            canonical_json_bytes({"dataset_id": dataset_id}),
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
                sha256_hex(data_bytes) != manifest["data_sha256"]
                or sha256_hex(raw_bytes) != manifest["raw_sha256"]
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
        dataset = MarketDataset(bars, metadata)
        if not dataset_identity_matches(dataset):
            raise CacheError("cached dataset identity mismatch")
        return dataset

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
    return serialize_metadata_values(asdict(metadata))


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
        raw_sha256=str(value["raw_sha256"]),
        data_sha256=str(value["data_sha256"]),
        dataset_id=str(value["dataset_id"]),
        schema_version=str(value["schema_version"]),
        bar_count=int(value["bar_count"]),
        missing_sessions=tuple(
            date.fromisoformat(item) for item in value["missing_sessions"]
        ),
        split_sessions=tuple(
            date.fromisoformat(item) for item in value["split_sessions"]
        ),
        dividend_sessions=tuple(
            date.fromisoformat(item) for item in value["dividend_sessions"]
        ),
        adapter_version=str(value["adapter_version"]),
    )
