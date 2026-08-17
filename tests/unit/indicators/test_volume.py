from copy import deepcopy
from decimal import Decimal
from typing import cast

import pytest

from quantforge.configuration import PrimitiveMapping
from quantforge.data.lineage import FeedScope
from quantforge.indicators import (
    RELATIVE_VOLUME_OUTPUT,
    VOLUME_MOVING_AVERAGE_OUTPUT,
    InvalidIndicatorParametersError,
    RelativeVolume,
    RelativeVolumeDenominatorPolicy,
    RelativeVolumeParameters,
    VolumeMovingAverage,
    VolumeMovingAverageParameters,
)
from quantforge.prediction import VolumeRatioContext

from ..helpers import make_dataset

VOLUMES = ("100", "200", "300", "400", "500", "600")


@pytest.mark.parametrize("lookback", [0, -1])
def test_volume_parameters_reject_nonpositive_lookbacks(lookback: int) -> None:
    with pytest.raises(InvalidIndicatorParametersError, match="greater than zero"):
        VolumeMovingAverageParameters(lookback, FeedScope.consolidated())
    with pytest.raises(InvalidIndicatorParametersError, match="greater than zero"):
        RelativeVolumeParameters(lookback, FeedScope.consolidated())


def test_volume_parameters_require_typed_feed_and_denominator_policies() -> None:
    with pytest.raises(InvalidIndicatorParametersError, match="feed_scope"):
        VolumeMovingAverageParameters(3, cast(FeedScope, "consolidated"))
    with pytest.raises(InvalidIndicatorParametersError, match="denominator_policy"):
        RelativeVolumeParameters(
            3,
            FeedScope.consolidated(),
            cast(RelativeVolumeDenominatorPolicy, "inclusive"),
        )


def test_volume_average_uses_only_full_trailing_windows() -> None:
    dataset = make_dataset(("10",) * len(VOLUMES), volumes=VOLUMES)
    indicator = VolumeMovingAverage(
        VolumeMovingAverageParameters(3, FeedScope.consolidated())
    )

    output = indicator.calculate(dataset)

    assert output.values_for(VOLUME_MOVING_AVERAGE_OUTPUT) == (
        None,
        None,
        Decimal(200),
        Decimal(300),
        Decimal(400),
        Decimal(500),
    )
    assert indicator.warm_up_observations == 3
    assert output.session_dates == tuple(bar.session_date for bar in dataset.bars)


def test_volume_average_retains_zero_and_masks_nonfinite_windows() -> None:
    dataset = make_dataset(
        ("10",) * 5,
        volumes=("0", "0", "100", "NaN", "300"),
    )

    values = (
        VolumeMovingAverage(VolumeMovingAverageParameters(2, FeedScope.consolidated()))
        .calculate(dataset)
        .values_for(VOLUME_MOVING_AVERAGE_OUTPUT)
    )

    assert values == (None, Decimal(0), Decimal(50), None, None)


def test_relative_volume_denominator_conventions_are_explicit() -> None:
    dataset = make_dataset(("10",) * len(VOLUMES), volumes=VOLUMES)
    consolidated = FeedScope.consolidated()
    inclusive = RelativeVolume(
        RelativeVolumeParameters(
            3,
            consolidated,
            RelativeVolumeDenominatorPolicy.INCLUDE_CURRENT_BAR,
        )
    )
    prior_only = RelativeVolume(
        RelativeVolumeParameters(
            3,
            consolidated,
            RelativeVolumeDenominatorPolicy.EXCLUDE_CURRENT_BAR,
        )
    )

    inclusive_values = inclusive.calculate(dataset).values_for(RELATIVE_VOLUME_OUTPUT)
    prior_values = prior_only.calculate(dataset).values_for(RELATIVE_VOLUME_OUTPUT)

    assert inclusive_values == (
        None,
        None,
        Decimal("1.5"),
        Decimal("1.333333333333333333333333333333333"),
        Decimal("1.25"),
        Decimal("1.2"),
    )
    assert prior_values == (
        None,
        None,
        None,
        Decimal(2),
        Decimal("1.666666666666666666666666666666667"),
        Decimal("1.5"),
    )
    assert inclusive.warm_up_observations == 3
    assert prior_only.warm_up_observations == 4
    assert (
        inclusive.configuration()["parameters"]
        != prior_only.configuration()["parameters"]
    )


@pytest.mark.parametrize(
    ("volumes", "policy", "unavailable_indices"),
    [
        (
            ("0", "0", "0", "100", "200"),
            RelativeVolumeDenominatorPolicy.INCLUDE_CURRENT_BAR,
            (0, 1, 2),
        ),
        (
            ("0", "0", "100", "200", "300"),
            RelativeVolumeDenominatorPolicy.EXCLUDE_CURRENT_BAR,
            (0, 1, 2),
        ),
        (
            ("100", "NaN", "300", "400", "500"),
            RelativeVolumeDenominatorPolicy.INCLUDE_CURRENT_BAR,
            (0, 1, 2),
        ),
    ],
)
def test_zero_missing_and_nonfinite_denominators_emit_none(
    volumes: tuple[str, ...],
    policy: RelativeVolumeDenominatorPolicy,
    unavailable_indices: tuple[int, ...],
) -> None:
    dataset = make_dataset(("10",) * len(volumes), volumes=volumes)
    output = RelativeVolume(
        RelativeVolumeParameters(2, FeedScope.consolidated(), policy)
    ).calculate(dataset)

    values = output.values_for(RELATIVE_VOLUME_OUTPUT)

    for index in unavailable_indices:
        assert values[index] is None
    assert all(value is None or value.is_finite() for value in values)


def test_feed_scope_and_formula_parameters_participate_in_identity() -> None:
    indicators = (
        RelativeVolume(RelativeVolumeParameters(2, FeedScope.consolidated())),
        RelativeVolume(RelativeVolumeParameters(3, FeedScope.consolidated())),
        RelativeVolume(RelativeVolumeParameters(2, FeedScope.iex_only())),
        RelativeVolume(
            RelativeVolumeParameters(
                2,
                FeedScope.consolidated(),
                RelativeVolumeDenominatorPolicy.EXCLUDE_CURRENT_BAR,
            )
        ),
    )

    assert len({indicator.configuration_id for indicator in indicators}) == 4
    assert indicators[0].configuration()["parameters"] == {
        "denominator_policy": "include_current_bar",
        "feed_scope": {
            "coverage": "consolidated",
            "market_center": None,
            "provider_scope": None,
        },
        "lookback": 2,
    }


def test_volume_configurations_round_trip_without_semantic_drift() -> None:
    average = VolumeMovingAverage(
        VolumeMovingAverageParameters(3, FeedScope.provider_defined("sip-equivalent"))
    )
    relative = RelativeVolume(
        RelativeVolumeParameters(
            3,
            FeedScope.iex_only(),
            RelativeVolumeDenominatorPolicy.EXCLUDE_CURRENT_BAR,
        )
    )

    restored_average = VolumeMovingAverage.from_configuration(average.configuration())
    restored_relative = RelativeVolume.from_configuration(relative.configuration())

    assert restored_average.configuration() == average.configuration()
    assert restored_average.configuration_id == average.configuration_id
    assert restored_relative.configuration() == relative.configuration()
    assert restored_relative.configuration_id == relative.configuration_id

    changed: PrimitiveMapping = {
        **relative.configuration(),
        "implementation_version": "2",
    }
    with pytest.raises(InvalidIndicatorParametersError, match="unsupported"):
        RelativeVolume.from_configuration(changed)


def test_volume_indicators_do_not_mutate_inputs_or_read_future_bars() -> None:
    dataset = make_dataset(("10",) * len(VOLUMES), volumes=VOLUMES)
    original = deepcopy(dataset)
    relative = RelativeVolume(RelativeVolumeParameters(3, FeedScope.consolidated()))
    average = VolumeMovingAverage(
        VolumeMovingAverageParameters(3, FeedScope.consolidated())
    )

    cutoff_dataset = make_dataset(("10",) * 4, volumes=VOLUMES[:4])
    cutoff_relative = relative.calculate(cutoff_dataset)
    extended_relative = relative.calculate(dataset)
    cutoff_average = average.calculate(cutoff_dataset)
    extended_average = average.calculate(dataset)

    assert dataset == original
    assert extended_relative.session_dates[:4] == cutoff_relative.session_dates
    assert extended_relative.values_for(RELATIVE_VOLUME_OUTPUT)[
        :4
    ] == cutoff_relative.values_for(RELATIVE_VOLUME_OUTPUT)
    assert extended_average.values_for(VOLUME_MOVING_AVERAGE_OUTPUT)[
        :4
    ] == cutoff_average.values_for(VOLUME_MOVING_AVERAGE_OUTPUT)


def test_qf7_volume_ratio_keeps_the_inclusive_numerical_convention() -> None:
    dataset = make_dataset(("10",) * len(VOLUMES), volumes=VOLUMES)
    context = VolumeRatioContext(3)
    expected = RelativeVolume(
        RelativeVolumeParameters(
            3,
            FeedScope.unknown(),
            RelativeVolumeDenominatorPolicy.INCLUDE_CURRENT_BAR,
        )
    ).calculate(dataset)

    assert context.values_for_dataset(dataset) == expected.values_for(
        RELATIVE_VOLUME_OUTPUT
    )
    configuration = context.configuration()
    assert configuration["implementation_version"] == "3"
    assert cast(dict[str, object], configuration["indicator"])["parameters"] == {
        "denominator_policy": "include_current_bar",
        "feed_scope": {
            "coverage": "unknown",
            "market_center": None,
            "provider_scope": None,
        },
        "lookback": 3,
    }
