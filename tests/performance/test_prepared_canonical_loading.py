"""QF-65 invocation counts for one QF-45-like input load, without timing thresholds.

The reference load (no preparation session) repeats immutable canonical work in
every consumer. One session authenticates the canonical source once and reuses
that proof, so repeated work collapses to the scientifically necessary minimum:
one trust-boundary decode plus one independent QF-51 evidence rebuild, one
calendar resolution per session and one validation per derived artifact.
"""

from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import quantforge.data.intraday_aggregation as intraday_aggregation
import quantforge.data.intraday_ingestion as intraday_ingestion
import quantforge.data.prediction_inputs as prediction_inputs
import quantforge.data.prediction_source_bars as prediction_source_bars
import quantforge.data.session_aggregation as session_aggregation
import quantforge.timeframes as timeframes
from quantforge.data import (
    AggregatedIntradayDataset,
    AggregatedSessionDataset,
    IntradayBar,
    IntradayBarBatch,
)
from quantforge.data.prepared_canonical import canonical_preparation
from tests.integration.test_prepared_canonical_loading import load, persist_source
from tests.unit.helpers import SESSIONS

# Includes the 2024-07-03 early close and the 2024-07-04 holiday gap.
FIVE_SESSIONS = SESSIONS[:5]


def _instrument(monkeypatch: pytest.MonkeyPatch, counts: Counter[str]) -> None:
    def counted(name: str, original: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            counts[name] += 1
            return original(*args, **kwargs)

        return wrapper

    def counted_property(owner: type, attribute: str, name: str) -> None:
        getter = getattr(owner, attribute).fget
        monkeypatch.setattr(owner, attribute, property(counted(name, getter)))

    counted_property(IntradayBar, "bar_id", "intraday_bar_ids")
    counted_property(IntradayBarBatch, "batch_id", "batch_ids")
    monkeypatch.setattr(
        IntradayBar,
        "__post_init__",
        counted("intraday_bar_constructions", IntradayBar.__post_init__),
    )
    monkeypatch.setattr(
        timeframes,
        "_resolve_exchange_session",
        counted("calendar_resolutions", timeframes._resolve_exchange_session),  # pyright: ignore[reportPrivateUsage]
    )
    monkeypatch.setattr(
        timeframes,
        "configuration_identity",
        counted("timeframe_identities", timeframes.configuration_identity),
    )
    decode = counted("full_1m_decodes", intraday_ingestion._decode_batch)  # pyright: ignore[reportPrivateUsage]
    monkeypatch.setattr(intraday_ingestion, "_decode_batch", decode)
    monkeypatch.setattr(prediction_source_bars, "_decode_batch", decode)
    monkeypatch.setattr(
        prediction_inputs,
        "validate_session_projection_evidence",
        counted(
            "qf51_evidence_rebuilds",
            prediction_inputs.validate_session_projection_evidence,
        ),
    )
    for module in (intraday_aggregation, session_aggregation):
        monkeypatch.setattr(
            module,
            "_validate_source_dataset",
            counted(
                "source_reauthentications",
                module._validate_source_dataset,  # pyright: ignore[reportPrivateUsage]
            ),
        )
    monkeypatch.setattr(
        session_aggregation,
        "_complete_target_periods",
        counted(
            "daily_derivations",
            session_aggregation._complete_target_periods,  # pyright: ignore[reportPrivateUsage]
        ),
    )
    for owner, name in (
        (AggregatedIntradayDataset, "two_minute_validations"),
        (AggregatedSessionDataset, "daily_validations"),
    ):
        monkeypatch.setattr(owner, "_validate", counted(name, owner._validate))  # pyright: ignore[reportPrivateUsage]


def test_session_collapses_repeated_canonical_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    record_property: Callable[[str, object], None],
) -> None:
    request = persist_source(tmp_path, FIVE_SESSIONS)
    reference_counts: Counter[str] = Counter()
    prepared_counts: Counter[str] = Counter()
    with monkeypatch.context() as patch:
        _instrument(patch, reference_counts)
        reference = load(tmp_path, request)
    with monkeypatch.context() as patch:
        _instrument(patch, prepared_counts)
        with canonical_preparation() as preparation:
            prepared = load(tmp_path, request)
            statistics = preparation.statistics()
    assert prepared == reference
    bars = len(reference.source.bars)
    record_property("source_bars", bars)
    record_property("reference_counts", dict(reference_counts))
    record_property("prepared_counts", dict(prepared_counts))
    record_property("preparation", statistics)

    # One trust-boundary decode of the cache plus one independent rebuild of
    # the retained QF-51 source evidence (the new prediction input's check).
    assert prepared_counts["full_1m_decodes"] == 2
    assert reference_counts["full_1m_decodes"] >= 7
    assert prepared_counts["qf51_evidence_rebuilds"] == 1
    assert reference_counts["qf51_evidence_rebuilds"] >= 5
    assert prepared_counts["source_reauthentications"] == 0
    assert reference_counts["source_reauthentications"] >= 3
    assert prepared_counts["daily_derivations"] == 1
    assert reference_counts["daily_derivations"] >= 2
    assert prepared_counts["two_minute_validations"] == 1
    assert prepared_counts["daily_validations"] == 1
    assert reference_counts["daily_validations"] > 1
    # Exchange sessions resolve once per session; timeframe identities once each.
    assert prepared_counts["calendar_resolutions"] == len(FIVE_SESSIONS)
    assert reference_counts["calendar_resolutions"] > 8 * bars
    # Remaining hashes are decoded-definition canonicality checks.
    assert prepared_counts["timeframe_identities"] <= 64
    assert reference_counts["timeframe_identities"] > 20 * bars
    # Bars are constructed only when decoded or derived: two 1m decodes and
    # the 2m output, instead of every repeated reconstruction.
    assert prepared_counts["intraday_bar_constructions"] == 2 * bars + bars // 2
    assert (
        reference_counts["intraday_bar_constructions"]
        >= 3 * prepared_counts["intraday_bar_constructions"]
    )
    assert prepared_counts["batch_ids"] * 4 < reference_counts["batch_ids"]
    # Remaining per-bar IDs belong to series ordering and the one 2m validation.
    assert prepared_counts["intraday_bar_ids"] == 2 * bars
    assert reference_counts["intraday_bar_ids"] > 5 * bars
