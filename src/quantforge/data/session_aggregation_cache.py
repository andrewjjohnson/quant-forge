"""Immutable persistence for derived exchange-session datasets."""

import json
import os
import tempfile
from pathlib import Path
from typing import cast

from quantforge.data.exceptions import CacheError
from quantforge.data.identity import sha256_hex
from quantforge.data.intraday_ingestion import IntradayDataset
from quantforge.data.session_aggregation import (
    AggregatedSessionDataset,
    SessionAggregationPolicy,
    aggregate_session_dataset,
)
from quantforge.timeframes import Timeframe


class SessionAggregationCache:
    """Immutable content-addressed persistence for daily and weekly datasets."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def persist(self, dataset: AggregatedSessionDataset) -> AggregatedSessionDataset:
        """Write one derived dataset once and verify existing bytes."""
        dataset_value = cast(object, dataset)
        if not isinstance(dataset_value, AggregatedSessionDataset):
            raise TypeError("session aggregation cache requires a derived dataset")
        dataset.validate()
        self._write_once(
            self.root / dataset.metadata.normalized_location,
            dataset.serialize_bars(),
        )
        self._write_once(
            self.root / dataset.metadata.manifest_location,
            dataset.serialize_manifest(),
        )
        return dataset

    def load(
        self,
        dataset_id: str,
        *,
        source_dataset: IntradayDataset,
        target_timeframe: Timeframe,
        policy: SessionAggregationPolicy | None = None,
    ) -> AggregatedSessionDataset:
        """Reload and rederive an artifact to verify all persisted evidence."""
        expected = aggregate_session_dataset(
            source_dataset, target_timeframe, policy=policy
        )
        if expected.metadata.dataset_id != dataset_id:
            raise CacheError("derived session dataset identifier mismatch")
        directory = self.root / "session" / "derived" / dataset_id
        try:
            normalized_bytes = (directory / "bars.json").read_bytes()
            manifest_bytes = (directory / "manifest.json").read_bytes()
            manifest = cast(dict[str, object], json.loads(manifest_bytes))
        except (OSError, TypeError, ValueError) as error:
            raise CacheError(
                f"incomplete or corrupt derived session cache entry: {dataset_id}"
            ) from error
        if normalized_bytes != expected.serialize_bars():
            raise CacheError("derived session normalized artifact mismatch")
        if sha256_hex(normalized_bytes) != expected.metadata.data_sha256:
            raise CacheError("derived session normalized checksum mismatch")
        if manifest_bytes != expected.serialize_manifest():
            raise CacheError("derived session manifest mismatch")
        if manifest.get("dataset_id") != dataset_id:
            raise CacheError("derived session manifest dataset identifier mismatch")
        return expected

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


__all__ = ["SessionAggregationCache"]
