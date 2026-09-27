"""QF-60 authenticated projection preparation and explicit scope boundaries."""

from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, cast

import pytest

import quantforge.data.prepared_prediction_views as preparation
from quantforge.configuration import PrimitiveMapping, PrimitiveMappingSnapshot
from quantforge.data import MarketDataset
from quantforge.data.exceptions import ValidationError
from quantforge.data.models import (
    BoundedPredictionProvenance,
    IntradayPredictionProvenance,
)
from quantforge.data.prediction_views import bounded_prediction_view
from quantforge.data.prepared_prediction_views import PreparedProjectionRegistry
from quantforge.prediction.models import PredictionMarketData
from tests.integration.test_intraday_prediction_provenance import (
    Fixture,
    cached_fixture,
)
from tests.unit.helpers import SESSIONS

SCOPE = PrimitiveMappingSnapshot.capture(
    {"plan_id": "plan", "fold_id": "fold-0", "role": "selection"}
)


@pytest.fixture(scope="module")
def fixture(tmp_path_factory: pytest.TempPathFactory) -> Fixture:
    return cached_fixture(
        tmp_path_factory.mktemp("projection-preparation"), session_dates=SESSIONS[:7]
    )


def test_projection_and_lineage_share_one_preparation(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = PreparedProjectionRegistry()
    cutoff = fixture.daily.bars[2].end_timestamp
    builds: list[int] = []
    original = preparation.bounded_prediction_view

    def build(*args: Any, **kwargs: Any) -> MarketDataset:
        builds.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(preparation, "bounded_prediction_view", build)
    view = registry.project(fixture.dataset, cutoff, scope=SCOPE)
    assert view == bounded_prediction_view(fixture.dataset, cutoff)
    assert registry.project(deepcopy(fixture.dataset), cutoff, scope=SCOPE) is view
    record = PredictionMarketData.from_qf3(view.metadata).to_primitive()
    for _ in range(3):
        registry.verify_lineage(
            record,
            deepcopy(fixture.dataset.metadata),
            cutoff,
            scope=SCOPE,
            start=view.metadata.actual_first_session,
        )
    assert builds == [1]
    assert registry.retained_count == 1
    assert (
        registry.prepare_lineage(fixture.dataset.metadata, cutoff, scope=SCOPE) is view
    )
    assert not hasattr(view, "canonical")
    registry.clear()
    assert registry.retained_count == 0
    # Restart can reconstruct solely from canonical plan evidence.
    assert (
        registry.prepare_lineage(fixture.dataset.metadata, cutoff, scope=SCOPE) == view
    )
    assert builds == [1, 1]


@pytest.mark.parametrize(
    "dimension", ["plan_id", "fold_id", "role", "warmup", "evaluation_bounds"]
)
def test_scopes_never_alias(fixture: Fixture, dimension: str) -> None:
    registry = PreparedProjectionRegistry()
    cutoff = fixture.daily.bars[2].end_timestamp
    first = registry.project(fixture.dataset, cutoff, scope=SCOPE)
    scope = PrimitiveMappingSnapshot.capture(
        {**SCOPE.to_primitive(), dimension: "different"}
    )
    second = registry.project(fixture.dataset, cutoff, scope=scope)
    assert second == first  # Identical bars do not authorize cross-role reuse.
    assert second is not first
    assert registry.retained_count == 2


def test_range_and_policy_and_memory_bound(
    fixture: Fixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = PreparedProjectionRegistry()
    cutoff = fixture.daily.bars[2].end_timestamp
    first = registry.project(fixture.dataset, cutoff, scope=SCOPE)
    later = registry.project(
        fixture.dataset, cutoff + timedelta(minutes=1), scope=SCOPE
    )
    assert first.metadata.dataset_id != later.metadata.dataset_id
    narrower = registry.project(fixture.dataset, cutoff, start=SESSIONS[1], scope=SCOPE)
    assert narrower.bars == first.bars[1:]
    assert registry.retained_count == 2
    monkeypatch.setattr(
        preparation, "BOUNDED_PREDICTION_ADAPTER_VERSION", "future-policy"
    )
    changed = registry.project(fixture.dataset, cutoff, start=SESSIONS[1], scope=SCOPE)
    assert changed == narrower
    assert changed is not narrower
    assert registry.retained_count == 2


@pytest.mark.parametrize(
    "dimension",
    [
        "source_dataset_id",
        "session_policy_id",
        "session_timeframe_configuration_id",
        "family_id",
    ],
)
def test_changed_ancestry_cannot_hide_behind_original_dataset_id(
    fixture: Fixture, dimension: str
) -> None:
    registry = PreparedProjectionRegistry()
    cutoff = fixture.daily.bars[2].end_timestamp
    registry.project(fixture.dataset, cutoff, scope=SCOPE)
    provenance = fixture.dataset.metadata.intraday_provenance
    assert isinstance(provenance, IntradayPredictionProvenance)
    changed = replace(
        fixture.dataset,
        metadata=replace(
            fixture.dataset.metadata,
            intraday_provenance=replace(provenance, **{dimension: "0" * 64}),
        ),
    )
    with pytest.raises(ValidationError):
        registry.project(changed, cutoff, scope=SCOPE)
    assert registry.retained_count == 1


def test_changed_adjustment_and_source_bars_fail_closed(fixture: Fixture) -> None:
    registry = PreparedProjectionRegistry()
    cutoff = fixture.daily.bars[2].end_timestamp
    registry.project(fixture.dataset, cutoff, scope=SCOPE)
    changed = replace(
        fixture.dataset,
        metadata=replace(fixture.dataset.metadata, ohlc_basis="foreign"),
    )
    with pytest.raises(ValidationError):
        registry.project(changed, cutoff, scope=SCOPE)
    changed = replace(
        fixture.dataset,
        bars=(
            replace(fixture.dataset.bars[0], close=Decimal("999")),
            *fixture.dataset.bars[1:],
        ),
    )
    with pytest.raises(ValidationError, match="source content changed"):
        registry.project(changed, cutoff, scope=SCOPE)


def test_equal_prices_from_foreign_source_are_separate(
    fixture: Fixture, tmp_path: Path
) -> None:
    foreign = cached_fixture(tmp_path, provider="other", session_dates=SESSIONS[:7])
    registry = PreparedProjectionRegistry()
    cutoff = fixture.daily.bars[2].end_timestamp
    first = registry.project(fixture.dataset, cutoff, scope=SCOPE)
    second = registry.project(foreign.dataset, cutoff, scope=SCOPE)
    assert first.bars == second.bars
    assert first.metadata.dataset_id != second.metadata.dataset_id
    record = PredictionMarketData.from_qf3(second.metadata).to_primitive()
    with pytest.raises(ValidationError, match="canonical plan ancestry"):
        registry.verify_lineage(record, fixture.dataset.metadata, cutoff, scope=SCOPE)


@pytest.mark.parametrize(
    "dimension", ["canonical_input_id", "canonical_provenance_id", "causal_cutoff"]
)
def test_rehashed_foreign_lineage_rejected_on_every_hit(
    fixture: Fixture, dimension: str
) -> None:
    registry = PreparedProjectionRegistry()
    cutoff = fixture.daily.bars[2].end_timestamp
    view = registry.project(fixture.dataset, cutoff, scope=SCOPE)
    record = PredictionMarketData.from_qf3(view.metadata).to_primitive()
    registry.verify_lineage(record, fixture.dataset.metadata, cutoff, scope=SCOPE)
    provenance = cast(PrimitiveMapping, record["intraday_provenance"])
    provenance[dimension] = (
        (cutoff + timedelta(minutes=1)).isoformat()
        if dimension == "causal_cutoff"
        else "intraday-projection-" + "0" * 64
        if dimension == "canonical_input_id"
        else "0" * 64
    )
    record["dataset_id"] = BoundedPredictionProvenance.from_primitive(
        provenance
    ).view_id
    with pytest.raises(ValidationError, match="canonical plan ancestry"):
        registry.verify_lineage(record, fixture.dataset.metadata, cutoff, scope=SCOPE)


def test_qf39_cannot_prepare_reserved_holdout(fixture: Fixture, tmp_path: Path) -> None:
    from quantforge.validation import PartitionRole
    from quantforge.walk_forward import WalkForwardError
    from quantforge.walk_forward.partitions import partition
    from tests.integration.test_bounded_prediction_workflow import workflow_fixture

    config, _ = workflow_fixture(tmp_path, fixture)
    registry = PreparedProjectionRegistry()
    with pytest.raises(WalkForwardError, match="never consumes"):
        partition(
            fixture.dataset,
            config.plan,
            0,
            PartitionRole.FINAL_HOLDOUT,
            minimum_observations=1,
            projection_registry=registry,
        )
    assert registry.retained_count == 0
    cutoff = fixture.daily.bars[2].end_timestamp
    selection = registry.project(fixture.dataset, cutoff, scope=SCOPE)
    # Even identical synthetic prices/cutoffs cannot share role-specific state.
    holdout_scope = PrimitiveMappingSnapshot.capture(
        {**SCOPE.to_primitive(), "role": "final_holdout"}
    )
    holdout = registry.project(fixture.dataset, cutoff, scope=holdout_scope)
    assert holdout == selection
    assert holdout is not selection


@pytest.mark.parametrize(
    "dimension",
    ["bars", "bar_price", "adjustment_mode", "retrieved_at", "missing_sessions"],
)
def test_json_equivalent_mutable_or_untyped_inputs_cannot_reuse(
    fixture: Fixture, dimension: str
) -> None:
    registry = PreparedProjectionRegistry()
    cutoff = fixture.daily.bars[2].end_timestamp
    registry.project(fixture.dataset, cutoff, scope=SCOPE)
    if dimension == "bars":
        changed = replace(fixture.dataset, bars=cast(Any, list(fixture.dataset.bars)))
    elif dimension == "bar_price":
        changed = replace(
            fixture.dataset,
            bars=(
                replace(
                    fixture.dataset.bars[0],
                    close=cast(Any, str(fixture.dataset.bars[0].close)),
                ),
                *fixture.dataset.bars[1:],
            ),
        )
    else:
        metadata = fixture.dataset.metadata
        altered: Any = (
            metadata.adjustment_mode.value
            if dimension == "adjustment_mode"
            else metadata.retrieved_at.replace(tzinfo=None)
            if dimension == "retrieved_at"
            else list(metadata.missing_sessions)
        )
        changed = replace(
            fixture.dataset, metadata=replace(metadata, **{dimension: altered})
        )
    with pytest.raises(ValidationError, match="deeply immutable"):
        registry.project(changed, cutoff, scope=SCOPE)
