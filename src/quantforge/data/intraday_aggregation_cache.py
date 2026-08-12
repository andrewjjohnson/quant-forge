"""Immutable persistence for deterministic derived intraday datasets."""

import json
import os
import tempfile
from pathlib import Path
from typing import cast

from quantforge.data.exceptions import CacheError
from quantforge.data.identity import sha256_hex
from quantforge.data.intraday_aggregation import (
    AggregatedIntradayDataset,
    IntradayAggregationPolicy,
    aggregate_intraday_dataset,
)
from quantforge.data.intraday_ingestion import IntradayDataset
from quantforge.timeframes import Timeframe


class IntradayAggregationCache:
    """Immutable content-addressed persistence for derived intraday datasets."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def persist(self, dataset: AggregatedIntradayDataset) -> AggregatedIntradayDataset:
        """Write one derived dataset once and verify any existing content."""
        dataset_value = cast(object, dataset)
        if not isinstance(dataset_value, AggregatedIntradayDataset):
            raise TypeError("aggregation cache requires an AggregatedIntradayDataset")
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
        policy: IntradayAggregationPolicy | None = None,
    ) -> AggregatedIntradayDataset:
        """Reload and rederive an artifact to verify its complete identity."""
        expected = aggregate_intraday_dataset(
            source_dataset, target_timeframe, policy=policy
        )
        if expected.metadata.dataset_id != dataset_id:
            raise CacheError("derived intraday dataset identifier mismatch")
        directory = self.root / "intraday" / "derived" / dataset_id
        try:
            normalized_bytes = (directory / "bars.json").read_bytes()
            manifest_bytes = (directory / "manifest.json").read_bytes()
            manifest = cast(dict[str, object], json.loads(manifest_bytes))
        except (OSError, TypeError, ValueError) as error:
            raise CacheError(
                f"incomplete or corrupt derived intraday cache entry: {dataset_id}"
            ) from error
        if normalized_bytes != expected.serialize_bars():
            raise CacheError("derived intraday normalized artifact mismatch")
        if sha256_hex(normalized_bytes) != expected.metadata.data_sha256:
            raise CacheError("derived intraday normalized checksum mismatch")
        if manifest_bytes != expected.serialize_manifest():
            raise CacheError("derived intraday manifest mismatch")
        if manifest.get("dataset_id") != dataset_id:
            raise CacheError("derived intraday manifest dataset identifier mismatch")
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


__all__ = ["IntradayAggregationCache"]
