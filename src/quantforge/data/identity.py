"""Canonical QF-3 dataset serialization and identity verification."""

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, date, datetime
from typing import cast

from quantforge.data.models import AdjustmentMode, DailyBar, MarketDataset


def canonical_json_bytes(value: object) -> bytes:
    """Serialize a JSON-compatible value with the QF-3 canonical policy."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()


def sha256_hex(content: bytes) -> str:
    """Return a lowercase SHA-256 hexadecimal digest."""
    return hashlib.sha256(content).hexdigest()


def serialize_bars_csv(bars: tuple[DailyBar, ...]) -> bytes:
    """Serialize canonical bars exactly as the immutable cache artifact."""
    csv_text = "symbol,session_date,open,high,low,close,volume\n" + "".join(
        f"{bar.symbol},{bar.session_date.isoformat()},{bar.open},{bar.high},{bar.low},{bar.close},{bar.volume}\n"
        for bar in bars
    )
    return csv_text.encode()


def serialize_metadata_values(
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
    for field in ("missing_sessions", "split_sessions", "dividend_sessions"):
        value[field] = [
            item.isoformat() for item in cast(tuple[date, ...], value[field])
        ]
    return value


def calculate_dataset_id(
    metadata_values: dict[str, object],
    *,
    raw_sha256: str,
    data_sha256: str,
    schema_version: str,
) -> str:
    """Calculate the QF-3 identity from metadata and immutable content digests."""
    identity = {
        **serialize_metadata_values(metadata_values),
        "raw_sha256": raw_sha256,
        "data_sha256": data_sha256,
        "schema_version": schema_version,
    }
    return sha256_hex(canonical_json_bytes(identity))


def dataset_identity_matches(dataset: MarketDataset) -> bool:
    """Return whether current bars and metadata reproduce the QF-3 identity."""
    try:
        metadata = dataset.metadata
        metadata_values = asdict(metadata)
        for field in (
            "raw_location",
            "normalized_location",
            "raw_sha256",
            "data_sha256",
            "dataset_id",
            "schema_version",
        ):
            del metadata_values[field]
        actual_data_sha256 = sha256_hex(serialize_bars_csv(dataset.bars))
        expected_dataset_id = calculate_dataset_id(
            metadata_values,
            raw_sha256=metadata.raw_sha256,
            data_sha256=metadata.data_sha256,
            schema_version=metadata.schema_version,
        )
        digests_are_canonical = all(
            len(digest) == 64
            and digest == digest.lower()
            and all(character in "0123456789abcdef" for character in digest)
            for digest in (metadata.raw_sha256, metadata.data_sha256)
        )
        return (
            digests_are_canonical
            and actual_data_sha256 == metadata.data_sha256
            and expected_dataset_id == metadata.dataset_id
            and metadata.raw_location == f"raw/{metadata.raw_sha256}.json"
            and metadata.normalized_location
            == f"datasets/{metadata.dataset_id}/bars.csv"
        )
    except (AttributeError, KeyError, TypeError, ValueError):
        return False
