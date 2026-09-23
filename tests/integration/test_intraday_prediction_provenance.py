"""Offline QF-45 provenance regression through canonical caches and predictions."""

import json
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest

from quantforge.configuration import (
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.data import (
    AdjustmentBasis,
    AdjustmentMode,
    AggregationPolicy,
    DatasetFamily,
    DatasetLineage,
    FeedScope,
    IntradayBar,
    IntradayBarBatch,
    IntradayBarProvenance,
    IntradayBarRequest,
    IntradayDataset,
    IntradayFetchResult,
    IntradayMarketDataCache,
    IntradayRawSnapshot,
    IntradayValidationMode,
    MarketDataCache,
    MarketDataset,
    TimeframeBarSeries,
    aggregate_intraday_dataset,
    aggregate_session_dataset,
    build_multi_timeframe_context,
    validate_intraday_coverage,
    validate_market_dataset,
)
from quantforge.data.exceptions import ValidationError
from quantforge.data.identity import (
    canonical_json_bytes,
    serialize_metadata_values,
    sha256_hex,
)
from quantforge.data.intraday_aggregation import intraday_session_windows
from quantforge.data.models import (
    CorporateActionAvailability,
    IntradayPredictionProvenance,
)
from quantforge.data.multi_timeframe import MultiTimeframeContextValidationError
from quantforge.data.prediction_inputs import (
    prediction_dataset_from_intraday,
    validate_prediction_source,
)
from quantforge.experiments import ManifestError, StudyType, inspect_study
from quantforge.experiments._producer_integrity import validate_prediction_manifest
from quantforge.prediction import (
    InvalidPredictionDataError,
    PredictionContextRequirements,
    PredictionStudy,
    PredictionTimeframeRequirement,
    SignalFeatureCandidate,
    build_signal_feature_dataset,
    intraday_excursion_outcome,
    intraday_forward_return_outcome,
    intraday_target_stop_outcome,
    run_prediction_study,
)
from quantforge.prediction.feature_dataset import (
    _fixed_candidate_population_id,  # pyright: ignore[reportPrivateUsage]
    _FixedCandidateRule,  # pyright: ignore[reportPrivateUsage]
    _SignalFeatureRule,  # pyright: ignore[reportPrivateUsage]
)
from quantforge.prediction.study import (
    prepare_prediction_study_dataset,
    run_prediction_study_in_session,
)
from quantforge.timeframes import (
    IntradayInterval,
    SessionInterval,
    Timeframe,
    resolve_exchange_session,
)
from tests.unit.helpers import make_dataset
from tests.unit.prediction.test_multi_timeframe_feature_dataset import (
    _FixtureCandidateRule,  # pyright: ignore[reportPrivateUsage]
)
from tests.unit.prediction.test_multi_timeframe_study import FixtureContextProvider

ONE_MINUTE = Timeframe.us_equity(IntradayInterval(timedelta(minutes=1)))
TWO_MINUTES = Timeframe.us_equity(IntradayInterval(timedelta(minutes=2)))
DAILY = Timeframe.us_equity(SessionInterval(1))
DECISION = datetime(2024, 1, 3, 16, tzinfo=UTC)
BASIS = AdjustmentBasis(
    AdjustmentMode.UNADJUSTED,
    "raw_provider",
    "raw_provider",
    "not_provided_for_intraday_bars",
    False,
)


@dataclass(frozen=True)
class Fixture:
    source: IntradayDataset
    dataset: MarketDataset
    primary: TimeframeBarSeries
    daily: TimeframeBarSeries
    cache: MarketDataCache


def cached_fixture(
    root: Path,
    provider: str = "tiingo",
    basis: AdjustmentBasis = BASIS,
    *,
    provider_symbol: str = "SPY",
    session_dates: tuple[date, ...] = (date(2024, 1, 2), date(2024, 1, 3)),
    primary_timeframe: Timeframe = TWO_MINUTES,
) -> Fixture:
    """Synthetic prices, same SPY/1m/unavailable contract as real QF-45 data."""
    request = IntradayBarRequest(
        "SPY",
        resolve_exchange_session(
            session_dates[0], ONE_MINUTE.session_policy
        ).open_timestamp,
        resolve_exchange_session(
            session_dates[-1], ONE_MINUTE.session_policy
        ).close_timestamp,
        ONE_MINUTE,
        FeedScope.consolidated(),
        basis,
    )
    retrieved = datetime.combine(
        session_dates[-1] + timedelta(days=1), datetime.min.time(), UTC
    )
    raw = IntradayRawSnapshot(
        provider,
        provider_symbol,
        "fixture-v1",
        "offline-fixture",
        request.request_id,
        request.start_timestamp,
        request.end_timestamp,
        retrieved,
        (),
        ({"fixture": "synthetic canonical SPY minutes"},),
    )
    provenance = IntradayBarProvenance(
        provider,
        provider_symbol,
        "fixture-v1",
        retrieved,
        request.request_id,
        raw.snapshot_id,
        request.feed_scope,
        basis,
    )
    bars = tuple(
        IntradayBar(
            "SPY",
            session,
            interval.start_timestamp,
            interval.end_timestamp,
            ONE_MINUTE,
            interval.completion,
            Decimal(100),
            Decimal("100.2"),
            Decimal("99.8"),
            Decimal(100),
            Decimal(1000),
            provenance,
        )
        for session in session_dates
        for interval in intraday_session_windows(session, ONE_MINUTE)
    )
    intraday_cache = IntradayMarketDataCache(root)
    source = intraday_cache.persist(
        IntradayFetchResult(
            IntradayBarBatch(request, bars), (raw,), "fixture-capabilities"
        )
    )
    # Core reproducer starts at the cache boundary; no provider is constructed.
    source = intraday_cache.load(source.metadata.dataset_id, request)
    primary = aggregate_intraday_dataset(source, primary_timeframe)
    daily = aggregate_session_dataset(source, DAILY)
    children = (primary.metadata.dataset_id, daily.metadata.dataset_id)
    source_id = source.metadata.dataset_id
    family = DatasetFamily(
        "SPY",
        provider,
        request.feed_scope,
        basis,
        AggregationPolicy(
            "quantforge_context_artifact_set",
            "1",
            cast(
                PrimitiveMapping,
                {
                    "artifact_family_manifest_ids": sorted(
                        [
                            primary.dataset_family.manifest_id,
                            daily.dataset_family.manifest_id,
                        ]
                    )
                },
            ),
        ),
        source_id,
        (
            DatasetLineage(source_id, ONE_MINUTE, source_id, None, children),
            DatasetLineage(children[0], primary_timeframe, source_id, source_id),
            DatasetLineage(children[1], DAILY, source_id, source_id),
        ),
    )
    cache = MarketDataCache(root)
    dataset = prediction_dataset_from_intraday(
        source, daily, cache=cache, intraday_cache=intraday_cache, family=family
    )
    return Fixture(
        source,
        cache.load(dataset.metadata.dataset_id),
        TimeframeBarSeries.from_aggregated_intraday_dataset(primary, family=family),
        TimeframeBarSeries.from_aggregated_session_dataset(daily, family=family),
        cache,
    )


@pytest.fixture(scope="module")
def fixture(tmp_path_factory: pytest.TempPathFactory) -> Fixture:
    return cached_fixture(tmp_path_factory.mktemp("intraday-provenance"))


@pytest.mark.parametrize("change", ["dataset_id", "raw_snapshot_ids", "bars"])
def test_projection_rejects_source_not_matching_immutable_cache(
    fixture: Fixture, tmp_path: Path, change: str
) -> None:
    source = fixture.source
    if change == "dataset_id":
        source = replace(
            source,
            metadata=replace(
                source.metadata,
                dataset_id="0" * 64,
                normalized_location=f"intraday/datasets/{'0' * 64}/bars.json",
            ),
        )
    elif change == "raw_snapshot_ids":
        source = replace(
            source, metadata=replace(source.metadata, raw_snapshot_ids=("0" * 64,))
        )
    else:
        bars = (replace(source.bars[0], high=Decimal("100.3")), *source.bars[1:])
        batch = IntradayBarBatch(source.request, bars)
        source = replace(
            source,
            bars=bars,
            metadata=replace(
                source.metadata,
                batch_id=batch.batch_id,
                data_sha256=sha256_hex(batch.serialize()),
                quality_report=validate_intraday_coverage(
                    batch, mode=IntradayValidationMode.DIAGNOSTIC
                ),
            ),
        )
    # Reaggregation and self-consistent in-memory hashes cannot establish that
    # this source came from the immutable cache recorded by its identity.
    sessions = aggregate_session_dataset(source, DAILY)
    output_cache = MarketDataCache(tmp_path / "projection")
    with pytest.raises(MultiTimeframeContextValidationError, match="immutable cache"):
        prediction_dataset_from_intraday(
            source,
            sessions,
            cache=output_cache,
            intraday_cache=IntradayMarketDataCache(fixture.cache.root),
        )
    assert not output_cache.root.exists()


def test_projection_rejects_corrupt_raw_cache_artifact(
    fixture: Fixture, tmp_path: Path
) -> None:
    import shutil

    shutil.copytree(fixture.cache.root / "intraday", tmp_path / "intraday")
    raw_path = tmp_path / fixture.source.metadata.raw_locations[0]
    raw_path.write_bytes(b"corrupt raw extract")
    sessions = aggregate_session_dataset(fixture.source, DAILY)
    output_cache = MarketDataCache(tmp_path / "projection")
    with pytest.raises(MultiTimeframeContextValidationError, match="immutable cache"):
        prediction_dataset_from_intraday(
            fixture.source,
            sessions,
            cache=output_cache,
            intraday_cache=IntradayMarketDataCache(tmp_path),
        )
    assert not output_cache.root.exists()


def test_projection_accepts_reloaded_source_with_default_session_family(
    fixture: Fixture, tmp_path: Path
) -> None:
    intraday_cache = IntradayMarketDataCache(fixture.cache.root)
    source = intraday_cache.load(
        fixture.source.metadata.dataset_id, fixture.source.request
    )
    sessions = aggregate_session_dataset(source, DAILY)
    output_cache = MarketDataCache(tmp_path)
    result = prediction_dataset_from_intraday(
        source, sessions, cache=output_cache, intraday_cache=intraday_cache
    )
    provenance = result.metadata.intraday_provenance
    assert provenance is not None
    assert provenance.family_id == sessions.dataset_family.family_id
    assert provenance.source_dataset_id == source.metadata.dataset_id
    assert output_cache.load(result.metadata.dataset_id) == result
    assert (
        prediction_dataset_from_intraday(
            source, sessions, cache=output_cache, intraday_cache=intraday_cache
        )
        == result
    )


def study_inputs(
    fixture: Fixture,
) -> tuple[_FixtureCandidateRule, FixtureContextProvider]:
    requirements = PredictionContextRequirements(
        PredictionTimeframeRequirement(TWO_MINUTES, FeedScope.consolidated()),
        (PredictionTimeframeRequirement(DAILY, FeedScope.consolidated()),),
    )
    context = build_multi_timeframe_context(
        series=(fixture.primary, fixture.daily),
        primary_timeframe=TWO_MINUTES,
        required_timeframes=requirements.context_timeframe_requirements(),
        as_of=DECISION,
    )
    return _FixtureCandidateRule(requirements), FixtureContextProvider(context)


def test_cache_only_intraday_input_context_and_all_outcomes(
    fixture: Fixture, tmp_path: Path
) -> None:
    dataset = fixture.dataset
    assert len(fixture.source.bars) == 780
    assert len(fixture.primary.bars) == 390
    assert len(dataset.bars) == 2
    prepare_prediction_study_dataset(dataset)
    metadata = dataset.metadata
    assert metadata.corporate_action_policy == BASIS.corporate_action_policy
    assert metadata.corporate_actions_complete is False
    assert metadata.intraday_provenance is not None
    assert (
        metadata.intraday_provenance.corporate_action_availability
        is CorporateActionAvailability.UNAVAILABLE
    )
    assert (
        metadata.intraday_provenance.source_dataset_id
        == fixture.source.metadata.dataset_id
    )
    assert fixture.cache.load(metadata.dataset_id) == dataset
    raw = json.loads((fixture.cache.root / metadata.raw_location).read_text())
    assert (
        raw["metadata"]["dataset_family"]["family_id"]
        == metadata.intraday_provenance.family_id
    )
    outcomes = (
        intraday_forward_return_outcome(timedelta(minutes=30), fixture.primary),
        intraday_excursion_outcome(timedelta(minutes=60), fixture.primary),
        intraday_target_stop_outcome(
            timedelta(minutes=60), fixture.primary, Decimal("0.003"), Decimal("0.002")
        ),
    )
    rule, provider = study_inputs(fixture)
    first = outcomes[0]
    study = PredictionStudy[SignalFeatureCandidate, Any, Any].create(
        rule, first.labeler, first.evaluator, outcome_source=fixture.primary
    )
    result = run_prediction_study(dataset, study, context_provider=provider)
    assert len(result.rows) == 1
    assert result.rows[0].outcome.values.raw_return == Decimal(0)
    validate_prediction_manifest(result.manifest_primitive())
    exported = build_signal_feature_dataset(
        dataset=dataset,
        prediction_study=study,
        contextual_features=(),
        outcomes=outcomes,
        context_provider=provider,
        output_root=tmp_path,
    )
    assert len(exported.rows) == 1
    values = exported.rows[0].to_primitive()
    for outcome in outcomes:
        assert values[f"outcome_{outcome.namespace}_available"] is True
    assert values[f"outcome_{outcomes[1].namespace}_mfe_percentage"] == "0.002"
    assert (
        inspect_study(
            StudyType.FEATURE_DATASET,
            tmp_path / exported.dataset_id,
            artifact_root=tmp_path,
        ).provenance.producer_study_id
        == exported.dataset_id
    )


def test_other_provider_uses_identical_contract(tmp_path: Path) -> None:
    other = cached_fixture(tmp_path, "future-provider")
    rule, provider = study_inputs(other)
    outcome = intraday_forward_return_outcome(timedelta(minutes=30), other.primary)
    result = run_prediction_study(
        other.dataset,
        PredictionStudy[SignalFeatureCandidate, Any, Any].create(
            rule, outcome.labeler, outcome.evaluator, outcome_source=other.primary
        ),
        context_provider=provider,
    )
    assert len(result.rows) == 1


@pytest.mark.parametrize("outcome_name", ["forward_return", "excursion", "target_stop"])
def test_reused_session_revalidates_changed_outcome_source(
    fixture: Fixture, tmp_path: Path, outcome_name: str
) -> None:
    other = cached_fixture(tmp_path, "other-provider")
    outcome = {
        "forward_return": intraday_forward_return_outcome(
            timedelta(minutes=30), fixture.primary
        ),
        "excursion": intraday_excursion_outcome(timedelta(minutes=60), fixture.primary),
        "target_stop": intraday_target_stop_outcome(
            timedelta(minutes=60), fixture.primary, Decimal("0.003"), Decimal("0.002")
        ),
    }[outcome_name]
    rule, provider = study_inputs(fixture)
    original = PredictionStudy[SignalFeatureCandidate, Any, Any].create(
        rule, outcome.labeler, outcome.evaluator, outcome_source=fixture.primary
    )
    signals = run_prediction_study(
        fixture.dataset, original, context_provider=provider
    ).signals
    # Replay fixed timestamp candidates without context, so only the outcome
    # source check can enforce the projection's family and snapshot lineage.
    replay = _FixedCandidateRule(
        cast(_SignalFeatureRule, rule),
        PrimitiveMappingSnapshot.capture(rule.configuration()),
        signals,
        _fixed_candidate_population_id(signals),
        len(signals),
    )
    study = PredictionStudy[SignalFeatureCandidate, Any, Any].create(
        replay, outcome.labeler, outcome.evaluator, outcome_source=fixture.primary
    )
    prepared = prepare_prediction_study_dataset(fixture.dataset)
    first = run_prediction_study_in_session(prepared, study)
    assert len(first.rows) == 1
    assert first.unavailable_outcome_count == 0
    assert (
        run_prediction_study_in_session(prepared, study).to_primitive()
        == first.to_primitive()
    )

    # Keep the labeler object/configuration, symbol, and price basis unchanged.
    # The replacement source is independently canonical but has another lineage.
    incompatible = replace(study, outcome_source=other.primary)
    with pytest.raises(
        InvalidPredictionDataError, match="source lineage is incompatible"
    ):
        run_prediction_study_in_session(prepared, incompatible)
    assert (
        run_prediction_study_in_session(prepared, study).to_primitive()
        == first.to_primitive()
    )


@pytest.mark.parametrize(
    "mismatch", ["adjustment", "policy", "family", "snapshot", "symbol", "feed"]
)
def test_incompatible_source_rejected(fixture: Fixture, mismatch: str) -> None:
    reference = fixture.primary.dataset_reference
    bars = tuple(cast(IntradayBar, bar) for bar in fixture.primary.bars)
    if mismatch in {"family", "snapshot", "feed"}:
        reference = replace(
            reference,
            **{
                "family": {"family_id": "different-family"},
                "snapshot": {"canonical_source_snapshot_id": "different-source"},
                "feed": {"feed_scope": FeedScope.iex_only()},
            }[mismatch],
        )
    else:
        basis = (
            replace(
                BASIS,
                adjustment_mode=AdjustmentMode.SPLIT_ADJUSTED,
                ohlc_basis="split_adjusted",
                volume_basis="split_adjusted",
            )
            if mismatch == "adjustment"
            else replace(
                BASIS,
                corporate_action_policy="separate_provider_reported_cash_dividends_and_splits",
            )
            if mismatch == "policy"
            else BASIS
        )
        bars = tuple(
            replace(
                bar,
                symbol="QQQ" if mismatch == "symbol" else bar.symbol,
                provenance=replace(bar.provenance, adjustment_basis=basis),
            )
            for bar in bars
        )
    source = TimeframeBarSeries._from_validated_artifact(  # pyright: ignore[reportPrivateUsage]
        reference,
        TWO_MINUTES,
        bars,
        dataset_family_manifest_id=fixture.primary.dataset_family_manifest_id,
    )
    with pytest.raises(ValidationError, match="incompatible"):
        validate_prediction_source(fixture.dataset, source)
    rule, _ = study_inputs(fixture)
    outcome = intraday_forward_return_outcome(timedelta(minutes=30), source)
    # All future labelers pass through this shared generic validation.
    with pytest.raises(InvalidPredictionDataError, match="incompatible"):
        run_prediction_study(
            fixture.dataset,
            PredictionStudy[SignalFeatureCandidate, Any, Any].create(
                rule, outcome.labeler, outcome.evaluator, outcome_source=source
            ),
            context_provider=study_inputs(fixture)[1],
        )


@pytest.mark.parametrize("change", ["complete", "policy", "availability", "missing"])
def test_contradictory_event_claims_rejected(fixture: Fixture, change: str) -> None:
    metadata = fixture.dataset.metadata
    assert metadata.intraday_provenance is not None
    changes: dict[str, Any] = {
        "complete": {"corporate_actions_complete": True},
        "policy": {
            "corporate_action_policy": (
                "separate_provider_reported_cash_dividends_and_splits"
            )
        },
        "availability": {
            "intraday_provenance": replace(
                metadata.intraday_provenance,
                corporate_action_availability=CorporateActionAvailability.AVAILABLE,
            )
        },
        "missing": {"intraday_provenance": None},
    }[change]
    with pytest.raises(ValidationError, match="corporate-action"):
        validate_market_dataset(
            replace(fixture.dataset, metadata=replace(metadata, **changes))
        )


def test_provenance_roundtrip_and_material_identity(fixture: Fixture) -> None:
    provenance = fixture.dataset.metadata.intraday_provenance
    assert provenance is not None
    primitive = provenance.to_primitive()
    assert (
        IntradayPredictionProvenance.from_primitive(
            json.loads(canonical_json_bytes(primitive))
        )
        == provenance
    )
    identities = {
        configuration_identity(replace(provenance, **change).to_primitive())
        for change in (
            {},
            {"family_id": "another-family"},
            {"corporate_action_availability": CorporateActionAvailability.AVAILABLE},
        )
    }
    assert len(identities) == 3
    with pytest.raises(ValueError, match="fields"):
        IntradayPredictionProvenance.from_primitive({**primitive, "unknown": True})


def test_legacy_daily_serialization_and_context_contract_unchanged() -> None:
    from dataclasses import asdict

    from quantforge.prediction.models import PredictionMarketData

    dataset = make_dataset(("100", "101", "102"))
    prepare_prediction_study_dataset(dataset)
    # Captured from unchanged main, before the optional intraday reference existed.
    assert dataset.metadata.dataset_id == (
        "bf5acd09e0b0ccceecbcfd2cd87f7b9e90c68912e3f5f82e1b35b01769e6f6c3"
    )
    assert (
        configuration_identity(
            PredictionMarketData.from_qf3(dataset.metadata).to_primitive()
        )
        == "24942349b8307f41a28597479cf31615ebc0b90b9042f5ead2fce0a8d731eb5c"
    )
    assert (
        dataset.metadata.corporate_action_availability
        is CorporateActionAvailability.AVAILABLE
    )
    assert "intraday_provenance" not in serialize_metadata_values(
        asdict(dataset.metadata)
    )
    assert (
        "intraday_provenance"
        not in PredictionMarketData.from_qf3(dataset.metadata).to_primitive()
    )


def test_qf9_rejects_rehashed_contradictory_provenance(fixture: Fixture) -> None:
    rule, provider = study_inputs(fixture)
    outcome = intraday_forward_return_outcome(timedelta(minutes=30), fixture.primary)
    result = run_prediction_study(
        fixture.dataset,
        PredictionStudy[SignalFeatureCandidate, Any, Any].create(
            rule, outcome.labeler, outcome.evaluator, outcome_source=fixture.primary
        ),
        context_provider=provider,
    )
    manifest = result.manifest_primitive()
    market = cast(PrimitiveMapping, manifest["market_data"])
    market["corporate_actions_complete"] = True
    manifest["study_id"] = configuration_identity(
        {
            "component": manifest["component"],
            "engine_version": manifest["engine_version"],
            "market_data": market,
            "study_configuration": manifest["configuration"],
            "prediction_context": manifest["prediction_context"],
        }
    )
    with pytest.raises(ManifestError, match="corporate-action"):
        validate_prediction_manifest(manifest)


def test_completed_resume_reuses_identical_intraday_provenance(
    fixture: Fixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from quantforge.prediction import IntradayForwardReturnOutcomeLabeler

    rule, provider = study_inputs(fixture)
    outcome = intraday_forward_return_outcome(timedelta(minutes=30), fixture.primary)
    study = PredictionStudy[SignalFeatureCandidate, Any, Any].create(
        rule, outcome.labeler, outcome.evaluator, outcome_source=fixture.primary
    )
    first = build_signal_feature_dataset(
        dataset=fixture.dataset,
        prediction_study=study,
        contextual_features=(),
        outcomes=(outcome,),
        context_provider=provider,
        output_root=tmp_path,
    )

    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("completed compatible results must be reused")

    monkeypatch.setattr(IntradayForwardReturnOutcomeLabeler, "label_request", forbidden)
    second = build_signal_feature_dataset(
        dataset=fixture.cache.load(fixture.dataset.metadata.dataset_id),
        prediction_study=study,
        contextual_features=(),
        outcomes=(outcome,),
        context_provider=provider,
        output_root=tmp_path,
    )
    assert second == first


def test_changed_adjustment_cannot_reuse_checkpoint(
    fixture: Fixture, tmp_path: Path
) -> None:
    import shutil

    from quantforge.prediction import SignalFeaturePersistenceError

    adjusted = cached_fixture(
        tmp_path / "adjusted",
        basis=replace(
            BASIS,
            adjustment_mode=AdjustmentMode.SPLIT_ADJUSTED,
            ohlc_basis="split_adjusted",
            volume_basis="split_adjusted",
            adjusted_fields_used=True,
        ),
    )
    assert adjusted.dataset.metadata.dataset_id != fixture.dataset.metadata.dataset_id
    assert (
        adjusted.dataset.metadata.intraday_provenance
        != fixture.dataset.metadata.intraday_provenance
    )
    result_ids: list[str] = []
    for inputs in (fixture, adjusted):
        rule, provider = study_inputs(inputs)
        outcome = intraday_forward_return_outcome(timedelta(minutes=30), inputs.primary)
        study = PredictionStudy[SignalFeatureCandidate, Any, Any].create(
            rule, outcome.labeler, outcome.evaluator, outcome_source=inputs.primary
        )
        result = build_signal_feature_dataset(
            dataset=inputs.dataset,
            prediction_study=study,
            contextual_features=(),
            outcomes=(outcome,),
            context_provider=provider,
            output_root=tmp_path / "originals",
        )
        result_ids.append(result.dataset_id)
    assert result_ids[0] != result_ids[1]
    shutil.copytree(
        tmp_path / "originals" / result_ids[0],
        tmp_path / "resume" / result_ids[1],
    )
    with pytest.raises(SignalFeaturePersistenceError):
        build_signal_feature_dataset(
            dataset=adjusted.dataset,
            prediction_study=study,
            contextual_features=(),
            outcomes=(outcome,),
            context_provider=provider,
            output_root=tmp_path / "resume",
        )


def test_changed_event_provenance_cannot_reuse_cache(
    fixture: Fixture, tmp_path: Path
) -> None:
    import shutil

    from quantforge.data import CacheError
    from quantforge.prediction.models import PredictionMarketData

    source = fixture.dataset.metadata
    changed = replace(
        source,
        corporate_actions_complete=True,
        corporate_action_policy="separate_provider_reported_cash_dividends_and_splits",
        intraday_provenance=None,
    )
    assert configuration_identity(
        PredictionMarketData.from_qf3(changed).to_primitive()
    ) != configuration_identity(PredictionMarketData.from_qf3(source).to_primitive())
    shutil.copytree(fixture.cache.root, tmp_path / "cache")
    manifest_path = (
        tmp_path / "cache" / "datasets" / source.dataset_id / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())
    manifest["corporate_action_policy"] = changed.corporate_action_policy
    manifest["corporate_actions_complete"] = True
    del manifest["intraday_provenance"]
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    with pytest.raises(CacheError, match="identity"):
        MarketDataCache(tmp_path / "cache").load(source.dataset_id)


@pytest.mark.parametrize("skip", [False, True])
def test_incompatible_context_is_rejected_or_recorded_as_skipped(
    fixture: Fixture, tmp_path: Path, skip: bool
) -> None:
    from quantforge.prediction import PredictionContextFailurePolicy

    other = cached_fixture(tmp_path, "different-source")
    rule, _ = study_inputs(fixture)
    if skip:
        rule.context_requirements = replace(
            rule.context_requirements,
            failure_policy=PredictionContextFailurePolicy.SKIP,
        )
    _, other_provider = study_inputs(other)
    outcome = intraday_forward_return_outcome(timedelta(minutes=30), fixture.primary)
    study = PredictionStudy[SignalFeatureCandidate, Any, Any].create(
        rule, outcome.labeler, outcome.evaluator, outcome_source=fixture.primary
    )
    if not skip:
        with pytest.raises(InvalidPredictionDataError, match="context family manifest"):
            run_prediction_study(
                fixture.dataset, study, context_provider=other_provider
            )
    else:
        result = run_prediction_study(
            fixture.dataset, study, context_provider=other_provider
        )
        assert not result.signals
        assert not result.rows
        assert (
            cast(PrimitiveMapping, result.manifest_primitive()["prediction_context"])[
                "status"
            ]
            == "skipped"
        )
        validate_prediction_manifest(result.manifest_primitive())


def test_qf9_rejects_rehashed_source_family(fixture: Fixture) -> None:
    rule, provider = study_inputs(fixture)
    outcome = intraday_forward_return_outcome(timedelta(minutes=30), fixture.primary)
    result = run_prediction_study(
        fixture.dataset,
        PredictionStudy[SignalFeatureCandidate, Any, Any].create(
            rule, outcome.labeler, outcome.evaluator, outcome_source=fixture.primary
        ),
        context_provider=provider,
    )
    manifest = result.manifest_primitive()
    market = cast(PrimitiveMapping, manifest["market_data"])
    cast(PrimitiveMapping, market["intraday_provenance"])["family_id"] = (
        "incompatible-family"
    )
    manifest["study_id"] = configuration_identity(
        {
            "component": manifest["component"],
            "engine_version": manifest["engine_version"],
            "market_data": market,
            "study_configuration": manifest["configuration"],
            "prediction_context": manifest["prediction_context"],
        }
    )
    with pytest.raises(ManifestError, match="source lineage"):
        validate_prediction_manifest(manifest)
