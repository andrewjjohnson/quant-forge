"""Prediction projections require complete sessions and the exact context graph."""

from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from quantforge.data import (
    DatasetFamily,
    DatasetLineage,
    IntradayBarBatch,
    IntradayFetchResult,
    IntradayMarketDataCache,
    IntradayRawSnapshot,
    MarketDataCache,
    MissingConstituentPolicy,
    SessionAggregationPolicy,
    TimeframeBarSeries,
    aggregate_intraday_dataset,
    aggregate_session_dataset,
    build_multi_timeframe_context,
)
from quantforge.data.exceptions import ValidationError
from quantforge.data.prediction_inputs import (
    prediction_dataset_from_intraday,
    validate_prediction_context_sources,
)
from quantforge.prediction import (
    InvalidPredictionDataError,
    PredictionStudy,
    SignalFeatureCandidate,
    SignalFeatureDatasetError,
    build_signal_feature_dataset,
    intraday_forward_return_outcome,
    run_prediction_study,
)
from tests.integration.test_intraday_prediction_provenance import (
    DAILY,
    DECISION,
    TWO_MINUTES,
    Fixture,
    study_inputs,
)
from tests.integration.test_intraday_prediction_provenance import fixture as fixture
from tests.unit.prediction.test_multi_timeframe_study import FixtureContextProvider


@pytest.mark.parametrize(
    "omitted_indices",
    [(0,), (195,), (389,), tuple(range(390))],
    ids=["opening-bar", "interior-bar", "closing-bar", "whole-session"],
)
def test_projection_rejects_incomplete_diagnostic_sessions(
    fixture: Fixture, tmp_path: Path, omitted_indices: tuple[int, ...]
) -> None:
    original = fixture.source
    metadata = original.metadata
    raw = IntradayRawSnapshot(
        metadata.provider_name,
        metadata.provider_symbol,
        metadata.adapter_version,
        "offline-fixture",
        original.request.request_id,
        original.request.start_timestamp,
        original.request.end_timestamp,
        metadata.retrieved_at,
        (),
        ({"omitted_indices": list(omitted_indices)},),
    )
    bars = tuple(
        replace(
            bar,
            provenance=replace(bar.provenance, source_snapshot_id=raw.snapshot_id),
        )
        for index, bar in enumerate(original.bars)
        if index not in omitted_indices
    )
    intraday_cache = IntradayMarketDataCache(tmp_path / "source")
    source = intraday_cache.persist(
        IntradayFetchResult(
            IntradayBarBatch(original.request, bars),
            (raw,),
            metadata.capabilities_configuration_id,
        )
    )
    assert intraday_cache.load(source.metadata.dataset_id, source.request) == source
    sessions = aggregate_session_dataset(
        source,
        DAILY,
        policy=SessionAggregationPolicy(MissingConstituentPolicy.DIAGNOSTIC),
    )
    assert not sessions.aggregation_report.is_complete
    assert sessions.aggregation_report.missing_constituent_count == len(omitted_indices)
    output_cache = MarketDataCache(tmp_path / "projection")
    with pytest.raises(ValidationError, match="complete session"):
        prediction_dataset_from_intraday(
            source, sessions, cache=output_cache, intraday_cache=intraday_cache
        )
    assert not output_cache.root.exists()


def test_complete_diagnostic_sessions_can_be_projected(
    fixture: Fixture, tmp_path: Path
) -> None:
    sessions = aggregate_session_dataset(
        fixture.source,
        DAILY,
        policy=SessionAggregationPolicy(MissingConstituentPolicy.DIAGNOSTIC),
    )
    assert sessions.aggregation_report.is_complete
    cache = MarketDataCache(tmp_path)
    projected = prediction_dataset_from_intraday(
        fixture.source,
        sessions,
        cache=cache,
        intraday_cache=IntradayMarketDataCache(fixture.cache.root),
    )
    assert projected.bars == fixture.dataset.bars
    assert cache.load(projected.metadata.dataset_id) == projected


@pytest.mark.parametrize("mixed_graphs", [False, True], ids=["changed", "missing"])
def test_prediction_context_requires_exact_family_manifest(
    fixture: Fixture, tmp_path: Path, mixed_graphs: bool
) -> None:
    provenance = fixture.dataset.metadata.intraday_provenance
    assert provenance is not None
    original_family = DatasetFamily.from_manifest(
        provenance.family_manifest.to_primitive()
    )
    source_id = fixture.source.metadata.dataset_id
    extra_id = "unused-derived-dataset"
    members = tuple(
        replace(entry, child_dataset_ids=(*entry.child_dataset_ids, extra_id))
        if entry.dataset_id == source_id
        else entry
        for entry in original_family.datasets
    )
    family = replace(
        original_family,
        datasets=(
            *members,
            DatasetLineage(extra_id, TWO_MINUTES, source_id, source_id),
        ),
    )
    assert family.family_id == original_family.family_id
    assert family.manifest_id != original_family.manifest_id
    primary = TimeframeBarSeries.from_aggregated_intraday_dataset(
        aggregate_intraday_dataset(fixture.source, TWO_MINUTES), family=family
    )
    daily = (
        fixture.daily
        if mixed_graphs
        else TimeframeBarSeries.from_aggregated_session_dataset(
            aggregate_session_dataset(fixture.source, DAILY), family=family
        )
    )
    assert primary.dataset_reference == fixture.primary.dataset_reference
    assert daily.dataset_reference == fixture.daily.dataset_reference
    rule, _ = study_inputs(fixture)
    context = build_multi_timeframe_context(
        series=(primary, daily),
        primary_timeframe=TWO_MINUTES,
        required_timeframes=rule.context_requirements.context_timeframe_requirements(),
        as_of=DECISION,
    )
    assert context.dataset_family_manifest_id == (
        None if mixed_graphs else family.manifest_id
    )
    with pytest.raises(ValidationError, match="context family manifest"):
        validate_prediction_context_sources(fixture.dataset, context)

    provider = FixtureContextProvider(context)
    outcome = intraday_forward_return_outcome(timedelta(minutes=30), fixture.primary)
    study = PredictionStudy[SignalFeatureCandidate, Any, Any].create(
        rule, outcome.labeler, outcome.evaluator, outcome_source=fixture.primary
    )
    with pytest.raises(InvalidPredictionDataError, match="context family manifest"):
        run_prediction_study(fixture.dataset, study, context_provider=provider)
    output_root = tmp_path / "features"
    with pytest.raises(SignalFeatureDatasetError, match="context family manifest"):
        build_signal_feature_dataset(
            dataset=fixture.dataset,
            prediction_study=study,
            contextual_features=(),
            outcomes=(outcome,),
            context_provider=provider,
            output_root=output_root,
        )
    assert not output_root.exists()
