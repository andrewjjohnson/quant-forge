"""Retained session evidence binds projected prices to the original artifact ID."""

from dataclasses import replace
from decimal import Decimal, localcontext
from pathlib import Path
from typing import cast

import pytest

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.data import (
    DatasetFamily,
    IntradayMarketDataCache,
    MarketDataCache,
    aggregate_session_dataset,
)
from quantforge.data.exceptions import ValidationError
from quantforge.data.models import IntradayPredictionProvenance
from quantforge.data.prediction_inputs import (
    prediction_dataset_from_intraday,
    validate_prediction_provenance,
)
from quantforge.data.prediction_session_evidence import session_projection_bars
from quantforge.prediction import PredictionMarketData
from tests.integration.test_intraday_prediction_provenance import DAILY, Fixture
from tests.integration.test_intraday_prediction_provenance import fixture as fixture


def test_session_evidence_is_exact_and_deeply_immutable(fixture: Fixture) -> None:
    provenance = fixture.dataset.metadata.intraday_provenance
    assert provenance is not None
    serialized = provenance.to_primitive()
    restored = IntradayPredictionProvenance.from_primitive(serialized)
    evidence = cast(PrimitiveMapping, serialized["session_evidence"])
    sessions = aggregate_session_dataset(fixture.source, DAILY)
    assert evidence["manifest"] == sessions.to_manifest()
    assert configuration_identity(cast(PrimitiveMapping, evidence["bars"])) == (
        sessions.metadata.data_sha256
    )
    cast(PrimitiveMapping, evidence["manifest"])["dataset_id"] = "changed"
    assert restored == provenance
    assert restored.to_primitive() != serialized


@pytest.mark.parametrize(
    "change",
    [
        "manifest_id",
        "manifest_source",
        "manifest_digest",
        "manifest_count",
        "bar_id",
        "ohlcv",
        "rehash_ohlcv",
        "numeric_representation",
        "bar_schema",
        "bar_timeframe",
        "source_bar_ids",
        "missing_bar",
        "extra_evidence",
        "extra_bar_field",
    ],
)
def test_session_evidence_cannot_change_under_its_original_artifact_id(
    fixture: Fixture, change: str
) -> None:
    market = PredictionMarketData.from_qf3(fixture.dataset.metadata).to_primitive()
    provenance = cast(PrimitiveMapping, market["intraday_provenance"])
    evidence = cast(PrimitiveMapping, provenance["session_evidence"])
    manifest = cast(PrimitiveMapping, evidence["manifest"])
    serialized = cast(PrimitiveMapping, evidence["bars"])
    entries = cast(list[PrimitiveMapping], serialized["bars"])
    entry = entries[0]
    bar = cast(PrimitiveMapping, entry["bar"])
    if change == "manifest_id":
        manifest["dataset_id"] = "0" * 64
    elif change == "manifest_source":
        cast(PrimitiveMapping, manifest["source_dataset"])["dataset_id"] = "0" * 64
    elif change == "manifest_digest":
        manifest["data_sha256"] = "0" * 64
    elif change == "manifest_count":
        manifest["bar_count"] = 999
    elif change == "bar_id":
        entry["bar_id"] = "0" * 64
    elif change in {"ohlcv", "rehash_ohlcv"}:
        bar["high"] = "1000"
        if change == "rehash_ohlcv":
            entry["bar_id"] = configuration_identity(bar)
            manifest["data_sha256"] = configuration_identity(serialized)
            aggregation = cast(PrimitiveMapping, manifest["aggregation_report"])
            report = cast(PrimitiveMapping, aggregation["report"])
            cast(list[PrimitiveMapping], report["windows"])[0]["output_bar_id"] = entry[
                "bar_id"
            ]
            aggregation["report_id"] = configuration_identity(report)
    elif change == "numeric_representation":
        bar["open"] = "100.00"
    elif change == "bar_schema":
        bar["schema_version"] = "unknown"
    elif change == "bar_timeframe":
        cast(PrimitiveMapping, bar["timeframe"])["configuration_id"] = "0" * 64
    elif change == "source_bar_ids":
        cast(list[str], bar["source_bar_ids"]).pop()
    elif change == "missing_bar":
        entries.pop()
    elif change == "extra_evidence":
        evidence["uncommitted"] = True
    else:
        bar["uncommitted"] = True
    with pytest.raises(ValidationError, match=r"session.*bars"):
        validate_prediction_provenance(market)


@pytest.mark.parametrize("precision", [2, 28])
def test_projection_decimal_representation_is_canonical(
    fixture: Fixture, tmp_path: Path, precision: int
) -> None:
    sessions = aggregate_session_dataset(fixture.source, DAILY)
    sessions = replace(
        sessions,
        bars=tuple(
            replace(
                bar,
                **{
                    name: Decimal(f"{getattr(bar, name):.4f}")
                    for name in ("open", "high", "low", "close", "volume")
                },
            )
            for bar in sessions.bars
        ),
    )
    provenance = fixture.dataset.metadata.intraday_provenance
    assert provenance is not None
    projected = prediction_dataset_from_intraday(
        fixture.source,
        sessions,
        cache=MarketDataCache(tmp_path),
        intraday_cache=IntradayMarketDataCache(fixture.cache.root),
        family=DatasetFamily.from_manifest(provenance.family_manifest.to_primitive()),
    )
    assert projected.metadata.dataset_id == fixture.dataset.metadata.dataset_id
    assert projected.metadata.data_sha256 == fixture.dataset.metadata.data_sha256
    with localcontext() as context:
        context.prec = precision
        assert session_projection_bars(sessions.bars) == projected.bars
        assert (
            MarketDataCache(tmp_path).load(projected.metadata.dataset_id) == projected
        )
