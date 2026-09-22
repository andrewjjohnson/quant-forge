"""Lossless source-bar evidence authenticated by the existing canonical batch hash."""

from datetime import datetime
from decimal import InvalidOperation
from typing import cast

from quantforge.configuration import Primitive, PrimitiveMapping, configuration_identity
from quantforge.data.exceptions import CacheError
from quantforge.data.intraday import (
    INTRADAY_CONTRACT_SCHEMA_VERSION,
    IntradayBar,
    IntradayBarBatch,
)
from quantforge.data.intraday_coverage_evidence import intraday_request_from_primitive
from quantforge.data.intraday_ingestion import intraday_batch_from_primitive

# Only repeated metadata is shared. All observation fields remain exact primitives.
_OBSERVATION_FIELDS = frozenset(
    {
        "session_identifier",
        "start_timestamp",
        "end_timestamp",
        "actual_duration_microseconds",
        "completion",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }
)


def capture_source_bar_evidence(bars: tuple[IntradayBar, ...]) -> PrimitiveMapping:
    """Deduplicate repeated timeframe/provenance metadata without altering bars."""
    templates: list[Primitive] = []
    indexes: dict[str, int] = {}
    observations: list[Primitive] = []
    for bar in bars:
        record = bar.to_primitive()
        template = {
            key: value
            for key, value in record.items()
            if key not in _OBSERVATION_FIELDS
        }
        template_id = configuration_identity(template)
        if template_id not in indexes:
            indexes[template_id] = len(templates)
            templates.append(template)
        observations.append(
            {
                "template_index": indexes[template_id],
                **{key: record[key] for key in _OBSERVATION_FIELDS},
            }
        )
    return {"templates": templates, "observations": observations}


def _validate_raw_chunk_bindings(
    batch: IntradayBarBatch, source: PrimitiveMapping
) -> None:
    """Apply the fetch-result bar/raw provenance relationships to retained facts."""
    chunks = source["chunks"]
    if not isinstance(chunks, list) or any(
        not isinstance(chunk, dict) for chunk in chunks
    ):
        raise ValueError("source raw chunks must be records")
    by_snapshot: dict[str, PrimitiveMapping] = {}
    for chunk in cast(list[PrimitiveMapping], chunks):
        snapshot_id = chunk["raw_snapshot_id"]
        if not isinstance(snapshot_id, str) or snapshot_id in by_snapshot:
            raise ValueError("source raw snapshot identities must be unique strings")
        by_snapshot[snapshot_id] = chunk
    for bar in batch.bars:
        chunk = by_snapshot.get(bar.provenance.source_snapshot_id)
        if chunk is None:
            raise ValueError("bar provenance names an unknown raw snapshot")
        if not (
            datetime.fromisoformat(cast(str, chunk["chunk_start_timestamp"]))
            <= bar.start_timestamp
            < datetime.fromisoformat(cast(str, chunk["chunk_end_timestamp"]))
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
            source["provider_name"],
            source["provider_symbol"],
            source["adapter_version"],
            datetime.fromisoformat(cast(str, chunk["retrieved_at"])),
            batch.request.request_id,
        ):
            raise ValueError("bar provenance differs from its retained raw snapshot")


def validate_source_bar_evidence(
    evidence: PrimitiveMapping, source: PrimitiveMapping
) -> IntradayBarBatch:
    """Reproduce the source digest and validate its canonical typed bar batch."""
    templates = evidence.get("templates")
    observations = evidence.get("observations")
    if (
        set(evidence) != {"templates", "observations"}
        or not isinstance(templates, list)
        or not isinstance(observations, list)
        or type(source.get("bar_count")) is not int
        or len(observations) != source["bar_count"]
    ):
        raise ValueError("source bar evidence fields or count are invalid")
    entries: list[Primitive] = []
    for observation in observations:
        if not isinstance(observation, dict) or set(observation) != {
            "template_index",
            *_OBSERVATION_FIELDS,
        }:
            raise ValueError("source bar observation fields are invalid")
        index = observation["template_index"]
        if type(index) is not int or not 0 <= index < len(templates):
            raise ValueError("source bar template index is invalid")
        template = templates[index]
        if not isinstance(template, dict) or set(template) & _OBSERVATION_FIELDS:
            raise ValueError("source bar template is invalid")
        bar = {**template, **{key: observation[key] for key in _OBSERVATION_FIELDS}}
        bar_id = configuration_identity(bar)
        entries.append({"bar_id": bar_id, "bar": bar})
    batch: PrimitiveMapping = {
        "schema_version": INTRADAY_CONTRACT_SCHEMA_VERSION,
        "contract_type": "intraday_bar_batch",
        "request": source["request"],
        "bars": entries,
    }
    digest = configuration_identity(batch)
    if digest != source.get("batch_id") or digest != source.get("data_sha256"):
        raise ValueError("source bar evidence differs from the canonical source batch")
    try:
        request_record = source["request"]
        if not isinstance(request_record, dict):
            raise ValueError("source request must be a record")
        configuration = request_record["configuration"]
        if not isinstance(configuration, dict):
            raise ValueError("source request configuration must be a record")
        request = intraday_request_from_primitive(configuration)
        restored = intraday_batch_from_primitive(batch, request)
        if restored.batch_id != digest:
            raise ValueError("source batch serialization is not canonical")
        _validate_raw_chunk_bindings(restored, source)
    except (
        KeyError,
        TypeError,
        ValueError,
        OverflowError,
        InvalidOperation,
        CacheError,
    ) as error:
        raise ValueError(f"source bar evidence is invalid: {error}") from error
    return restored
