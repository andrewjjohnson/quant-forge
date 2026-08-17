from dataclasses import replace
from datetime import time, timedelta
from typing import cast

import pytest

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.data import (
    AdjustmentBasis,
    AdjustmentMode,
    AggregationPolicy,
    DatasetFamily,
    DatasetFamilyValidationError,
    DatasetLineage,
    FeedScope,
    MixedDatasetFamilyError,
    SourceConsistencyMode,
    validate_source_consistency,
)
from quantforge.timeframes import (
    ExchangeSessionPolicy,
    IntradayInterval,
    SessionInterval,
    SessionScope,
    Timeframe,
)

SOURCE_ID = "source-5m-snapshot"
FIFTEEN_MINUTE_ID = "derived-15m"
HOURLY_ID = "derived-1h"
DAILY_ID = "derived-daily"


def _aggregation_policy(
    configuration: PrimitiveMapping | None = None,
) -> AggregationPolicy:
    return AggregationPolicy(
        "quantforge_ohlcv",
        "1",
        {
            "missing_source_bars": "reject",
            "partial_terminal_bars": "retain",
        }
        if configuration is None
        else configuration,
    )


def _adjustment_basis(
    adjustment_mode: AdjustmentMode = AdjustmentMode.UNADJUSTED,
) -> AdjustmentBasis:
    return AdjustmentBasis(
        adjustment_mode,
        "raw_provider"
        if adjustment_mode is AdjustmentMode.UNADJUSTED
        else "split_adjusted",
        "raw_provider"
        if adjustment_mode is AdjustmentMode.UNADJUSTED
        else "split_adjusted",
        "separate_provider_reported_cash_dividends_and_splits",
        adjustment_mode is not AdjustmentMode.UNADJUSTED,
    )


def _family(
    *,
    provider_name: str = "tiingo",
    feed_scope: FeedScope | None = None,
    source_timeframe: Timeframe | None = None,
    adjustment_basis: AdjustmentBasis | None = None,
    aggregation_policy: AggregationPolicy | None = None,
    reverse_order: bool = False,
) -> DatasetFamily:
    source = source_timeframe or Timeframe.us_equity(
        IntradayInterval(timedelta(minutes=5))
    )
    lineage = (
        DatasetLineage(
            SOURCE_ID,
            source,
            SOURCE_ID,
            None,
            (HOURLY_ID, FIFTEEN_MINUTE_ID),
        ),
        DatasetLineage(
            FIFTEEN_MINUTE_ID,
            Timeframe.us_equity(IntradayInterval(timedelta(minutes=15))),
            SOURCE_ID,
            SOURCE_ID,
        ),
        DatasetLineage(
            HOURLY_ID,
            Timeframe.us_equity(IntradayInterval(timedelta(hours=1))),
            SOURCE_ID,
            SOURCE_ID,
            (DAILY_ID,),
        ),
        DatasetLineage(
            DAILY_ID,
            Timeframe.us_equity(SessionInterval()),
            SOURCE_ID,
            HOURLY_ID,
        ),
    )
    return DatasetFamily(
        canonical_symbol="SPY",
        provider_name=provider_name,
        feed_scope=feed_scope or FeedScope.consolidated(),
        adjustment_basis=adjustment_basis or _adjustment_basis(),
        aggregation_policy=aggregation_policy or _aggregation_policy(),
        canonical_source_snapshot_id=SOURCE_ID,
        datasets=tuple(reversed(lineage)) if reverse_order else lineage,
    )


def test_valid_family_manifest_records_source_and_complete_lineage() -> None:
    family = _family()
    manifest = family.to_manifest()
    reference = family.reference(HOURLY_ID)
    source = cast(PrimitiveMapping, manifest["canonical_source"])
    lineage = cast(list[PrimitiveMapping], manifest["lineage"])

    assert manifest["family_id"] == family.family_id
    assert manifest["manifest_id"] == family.manifest_id
    assert source["symbol"] == "SPY"
    assert source["provider"] == "tiingo"
    assert source["feed_scope"] == {
        "coverage": "consolidated",
        "market_center": None,
        "provider_scope": None,
    }
    assert reference.feed_scope == FeedScope.consolidated()
    assert reference.to_primitive()["feed_scope"] == source["feed_scope"]
    assert source["source_interval"] == {
        "kind": "intraday",
        "nominal_duration_microseconds": 300_000_000,
        "anchor": "exchange_session_open",
        "clock_anchor": None,
        "clock_anchor_epoch_date": None,
        "cross_session_policy": "prohibited",
    }
    assert source["session_scope"] == "regular_hours"
    assert source["exchange_calendar"] == "XNYS"
    assert source["exchange_timezone"] == "America/New_York"
    assert {cast(str, item["dataset_id"]) for item in lineage} == {
        SOURCE_ID,
        FIFTEEN_MINUTE_ID,
        HOURLY_ID,
        DAILY_ID,
    }
    assert all(item["canonical_source_snapshot_id"] == SOURCE_ID for item in lineage)
    source_lineage = next(item for item in lineage if item["dataset_id"] == SOURCE_ID)
    assert source_lineage["parent_dataset_id"] is None
    assert source_lineage["child_dataset_ids"] == [FIFTEEN_MINUTE_ID, HOURLY_ID]
    assert manifest["source_consistency"] == {
        "required_policy": "common_canonical_source",
        "external_bar_validation_policy": None,
    }


def test_equivalent_families_have_equal_identity_and_serialization() -> None:
    first_configuration: PrimitiveMapping = {
        "missing_source_bars": "reject",
        "partial_terminal_bars": "retain",
    }
    second_configuration: PrimitiveMapping = {
        "partial_terminal_bars": "retain",
        "missing_source_bars": "reject",
    }
    first = _family(aggregation_policy=_aggregation_policy(first_configuration))
    second = _family(
        aggregation_policy=_aggregation_policy(second_configuration),
        reverse_order=True,
    )

    assert first == second
    assert first.family_id == second.family_id
    assert first.manifest_id == second.manifest_id
    assert first.serialize_manifest() == second.serialize_manifest()


def test_adding_a_derived_dataset_preserves_family_but_changes_manifest_identity() -> (
    None
):
    complete = _family()
    without_daily = tuple(
        replace(item, child_dataset_ids=()) if item.dataset_id == HOURLY_ID else item
        for item in complete.datasets
        if item.dataset_id != DAILY_ID
    )
    partial = replace(complete, datasets=without_daily)

    assert partial.family_id == complete.family_id
    assert partial.manifest_id != complete.manifest_id


@pytest.mark.parametrize(
    "variant",
    [
        _family(provider_name="alpha_vantage"),
        _family(feed_scope=FeedScope.single_venue("IEX")),
        _family(
            source_timeframe=Timeframe.us_equity(
                IntradayInterval(timedelta(minutes=15))
            )
        ),
        _family(
            source_timeframe=Timeframe(
                IntradayInterval(timedelta(minutes=5)),
                session_policy=ExchangeSessionPolicy(
                    scope=SessionScope.EXTENDED_HOURS,
                    extended_hours_start=time(4),
                    extended_hours_end=time(20),
                ),
            )
        ),
        _family(adjustment_basis=_adjustment_basis(AdjustmentMode.SPLIT_ADJUSTED)),
        _family(
            aggregation_policy=_aggregation_policy(
                {
                    "missing_source_bars": "reject",
                    "partial_terminal_bars": "discard",
                }
            )
        ),
    ],
)
def test_material_source_policy_changes_family_identity(variant: DatasetFamily) -> None:
    assert variant.family_id != _family().family_id


def test_aggregation_configuration_is_snapshotted_before_identity() -> None:
    mutable_configuration: PrimitiveMapping = {"missing_source_bars": "reject"}
    policy = _aggregation_policy(mutable_configuration)
    family = _family(aggregation_policy=policy)
    family_id = family.family_id

    mutable_configuration["missing_source_bars"] = "fill"

    assert family.family_id == family_id
    assert cast(PrimitiveMapping, policy.to_primitive()["configuration"]) == {
        "missing_source_bars": "reject"
    }


@pytest.mark.parametrize(
    ("adjustment_mode", "ohlc_basis", "volume_basis"),
    [
        (AdjustmentMode.UNADJUSTED, "split_adjusted", "raw_provider"),
        (AdjustmentMode.UNADJUSTED, "raw_provider", "split_adjusted"),
        (AdjustmentMode.SPLIT_ADJUSTED, "raw_provider", "split_adjusted"),
        (AdjustmentMode.SPLIT_ADJUSTED, "split_adjusted", "raw_provider"),
    ],
)
def test_adjustment_mode_rejects_contradictory_price_or_volume_basis(
    adjustment_mode: AdjustmentMode,
    ohlc_basis: str,
    volume_basis: str,
) -> None:
    with pytest.raises(DatasetFamilyValidationError, match="inconsistent"):
        AdjustmentBasis(
            adjustment_mode,
            ohlc_basis,
            volume_basis,
            "separate_provider_reported_cash_dividends_and_splits",
            False,
        )


def test_valid_multi_timeframe_references_share_one_family() -> None:
    family = _family()

    validation = validate_source_consistency(
        (
            family.reference(SOURCE_ID),
            family.reference(HOURLY_ID),
            family.reference(DAILY_ID),
        )
    )

    assert validation.mode is SourceConsistencyMode.COMMON_DATASET_FAMILY
    assert validation.family_id == family.family_id
    assert validation.external_validation_policy_id is None


def test_iex_only_and_consolidated_references_cannot_mix_silently() -> None:
    consolidated = _family()
    iex_only = _family(feed_scope=FeedScope.single_venue("IEX"))

    with pytest.raises(MixedDatasetFamilyError, match="mixed dataset families"):
        validate_source_consistency(
            (
                consolidated.reference(HOURLY_ID),
                iex_only.reference(DAILY_ID),
            )
        )


def test_incompatible_source_intervals_cannot_mix_silently() -> None:
    five_minute = _family()
    fifteen_minute = _family(
        source_timeframe=Timeframe.us_equity(IntradayInterval(timedelta(minutes=15)))
    )

    with pytest.raises(MixedDatasetFamilyError, match="external-bar policy"):
        validate_source_consistency(
            (
                five_minute.reference(HOURLY_ID),
                fifteen_minute.reference(DAILY_ID),
            )
        )


def test_every_derived_dataset_must_point_to_canonical_source() -> None:
    family = _family()
    inconsistent = tuple(
        replace(item, canonical_source_snapshot_id="different-source")
        if item.dataset_id == DAILY_ID
        else item
        for item in family.datasets
    )

    with pytest.raises(DatasetFamilyValidationError, match="every dataset must point"):
        replace(family, datasets=inconsistent)


def test_parent_and_child_links_must_agree() -> None:
    family = _family()
    inconsistent = tuple(
        replace(item, child_dataset_ids=()) if item.dataset_id == SOURCE_ID else item
        for item in family.datasets
    )

    with pytest.raises(DatasetFamilyValidationError, match="child dataset IDs"):
        replace(family, datasets=inconsistent)


def test_lineage_cycles_are_rejected() -> None:
    source_timeframe = Timeframe.us_equity(IntradayInterval(timedelta(minutes=5)))
    cyclic_lineage = (
        DatasetLineage(SOURCE_ID, source_timeframe, SOURCE_ID, None),
        DatasetLineage(
            "derived-a",
            Timeframe.us_equity(IntradayInterval(timedelta(minutes=15))),
            SOURCE_ID,
            "derived-b",
            ("derived-b",),
        ),
        DatasetLineage(
            "derived-b",
            Timeframe.us_equity(IntradayInterval(timedelta(hours=1))),
            SOURCE_ID,
            "derived-a",
            ("derived-a",),
        ),
    )

    with pytest.raises(DatasetFamilyValidationError, match="contains a cycle"):
        DatasetFamily(
            "SPY",
            "tiingo",
            FeedScope.consolidated(),
            _adjustment_basis(),
            _aggregation_policy(),
            SOURCE_ID,
            cyclic_lineage,
        )


class _ExplicitExternalValidationPolicy:
    def to_primitive(self) -> PrimitiveMapping:
        return {"policy": "test_cross_provider_bar_validation", "version": "1"}

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.to_primitive())

    def validate(self, references: tuple[object, ...]) -> None:
        assert len(references) == 2


class _MutatingExternalValidationPolicy:
    def __init__(self) -> None:
        self.configuration: PrimitiveMapping = {"policy": "mutable", "version": "1"}

    def to_primitive(self) -> PrimitiveMapping:
        return self.configuration

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.configuration)

    def validate(self, references: tuple[object, ...]) -> None:
        self.configuration["version"] = "2"


def test_future_external_validation_requires_an_explicit_identity_bearing_policy() -> (
    None
):
    consolidated = _family()
    iex_only = _family(feed_scope=FeedScope.single_venue("IEX"))
    policy = _ExplicitExternalValidationPolicy()

    validation = validate_source_consistency(
        (
            consolidated.reference(HOURLY_ID),
            iex_only.reference(DAILY_ID),
        ),
        external_bar_validation_policy=policy,
    )

    assert validation.mode is SourceConsistencyMode.EXTERNALLY_VALIDATED
    assert validation.family_id is None
    assert validation.external_validation_policy_id == policy.configuration_id


def test_external_validation_policy_cannot_mutate_while_authorizing_mixed_data() -> (
    None
):
    consolidated = _family()
    iex_only = _family(feed_scope=FeedScope.single_venue("IEX"))

    with pytest.raises(DatasetFamilyValidationError, match="mutated"):
        validate_source_consistency(
            (
                consolidated.reference(HOURLY_ID),
                iex_only.reference(DAILY_ID),
            ),
            external_bar_validation_policy=_MutatingExternalValidationPolicy(),
        )


def test_family_references_only_record_manifested_datasets() -> None:
    with pytest.raises(DatasetFamilyValidationError, match="not recorded"):
        _family().reference("provider-native-daily")
