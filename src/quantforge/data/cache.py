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

from quantforge.data.corporate_actions import (
    CORPORATE_ACTION_SCHEMA_VERSION,
    CorporateActionSeed,
    bind_corporate_actions,
    corporate_actions_primitive,
)
from quantforge.data.exceptions import CacheError, ValidationError
from quantforge.data.identity import (
    calculate_dataset_id,
    canonical_json_bytes,
    serialize_bars_csv,
    serialize_metadata_values,
    sha256_hex,
)
from quantforge.data.models import (
    SCHEMA_VERSION,
    AdjustmentMode,
    CashDividend,
    CorporateAction,
    DailyBar,
    DatasetMetadata,
    MarketDataset,
    ProviderResponse,
    StockSplit,
)
from quantforge.data.validate import validate_market_dataset


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
        corporate_action_seeds: tuple[CorporateActionSeed, ...],
        metadata_values: dict[str, object],
        key: str,
        *,
        replace_request_index: bool = False,
    ) -> MarketDataset:
        """Persist immutable artifacts and optionally advance the request index."""
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
        relative_actions = f"datasets/{dataset_id}/corporate_actions.json"
        snapshot_id = cast(str, metadata_values["corporate_action_snapshot_id"])
        corporate_actions = bind_corporate_actions(
            corporate_action_seeds,
            dataset_id=dataset_id,
            snapshot_id=snapshot_id,
        )
        corporate_actions_bytes = canonical_json_bytes(
            corporate_actions_primitive(corporate_actions, snapshot_id)
        )
        metadata = DatasetMetadata(
            **cast(Any, metadata_values),
            raw_location=relative_raw,
            normalized_location=relative_data,
            corporate_actions_location=relative_actions,
            raw_sha256=raw_hash,
            data_sha256=data_hash,
            dataset_id=dataset_id,
            schema_version=SCHEMA_VERSION,
        )
        dataset = MarketDataset(bars, metadata, corporate_actions)
        validate_market_dataset(dataset)
        manifest_value = _metadata_to_dict(metadata)
        self._write_once(self.root / relative_raw, raw_bytes)
        self._write_once(self.root / relative_data, data_bytes)
        self._write_once(self.root / relative_actions, corporate_actions_bytes)
        self._write_once(
            self.root / "datasets" / dataset_id / "manifest.json",
            canonical_json_bytes(manifest_value),
        )
        request_index = self.root / "requests" / f"{key}.json"
        request_content = canonical_json_bytes({"dataset_id": dataset_id})
        if replace_request_index:
            self._replace(request_index, request_content)
        else:
            self._write_once(request_index, request_content)
        return dataset

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
            actions_path = self.root / str(manifest["corporate_actions_location"])
            data_bytes, raw_bytes = data_path.read_bytes(), raw_path.read_bytes()
            actions_value: object = json.loads(actions_path.read_text())
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
            corporate_actions = _corporate_actions_from_dict(
                actions_value,
                expected_snapshot_id=metadata.corporate_action_snapshot_id,
            )
        except CacheError:
            raise
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise CacheError(
                f"incomplete or corrupt cache entry: {dataset_id}"
            ) from error
        dataset = MarketDataset(bars, metadata, corporate_actions)
        try:
            validate_market_dataset(dataset)
        except ValidationError as error:
            raise CacheError(f"cached dataset validation failed: {error}") from error
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

    @staticmethod
    def _replace(path: Path, content: bytes) -> None:
        """Atomically replace a mutable request pointer, never dataset artifacts."""
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
        corporate_actions_location=str(value["corporate_actions_location"]),
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
        corporate_actions_complete=bool(value["corporate_actions_complete"]),
        corporate_action_count=int(value["corporate_action_count"]),
        dividend_count=int(value["dividend_count"]),
        split_count=int(value["split_count"]),
        corporate_action_snapshot_id=str(value["corporate_action_snapshot_id"]),
        ohlc_basis=str(value["ohlc_basis"]),
        volume_basis=str(value["volume_basis"]),
        adjusted_fields_used=bool(value["adjusted_fields_used"]),
        corporate_action_policy=str(value["corporate_action_policy"]),
        adapter_version=str(value["adapter_version"]),
    )


def _corporate_actions_from_dict(
    value: object, *, expected_snapshot_id: str
) -> tuple[CorporateAction, ...]:
    if not isinstance(value, dict):
        raise ValueError("corporate-action artifact must be an object")
    mapping = cast(dict[object, object], value)
    if mapping.get("schema_version") != CORPORATE_ACTION_SCHEMA_VERSION:
        raise ValueError("unsupported corporate-action artifact schema")
    if mapping.get(
        "corporate_action_snapshot_id"
    ) != expected_snapshot_id or not isinstance(mapping.get("actions"), list):
        raise ValueError("corporate-action artifact identity mismatch")
    actions: list[CorporateAction] = []
    for item in cast(list[object], mapping["actions"]):
        if not isinstance(item, dict):
            raise ValueError("corporate-action record must be an object")
        record = cast(dict[object, object], item)
        action_type = record.get("action_type")
        common = {
            "action_id": str(record["action_id"]),
            "symbol": str(record["symbol"]),
            "provider_name": str(record["provider_name"]),
            "source_dataset_id": str(record["source_dataset_id"]),
        }
        if action_type == "cash_dividend":
            actions.append(
                CashDividend(
                    **common,
                    ex_dividend_session=date.fromisoformat(
                        str(record["ex_dividend_session"])
                    ),
                    amount_per_share=Decimal(str(record["amount_per_share"])),
                )
            )
        elif action_type == "stock_split":
            actions.append(
                StockSplit(
                    **common,
                    effective_session=date.fromisoformat(
                        str(record["effective_session"])
                    ),
                    split_factor=Decimal(str(record["split_factor"])),
                )
            )
        else:
            raise ValueError("unsupported corporate-action type")
    return tuple(actions)
