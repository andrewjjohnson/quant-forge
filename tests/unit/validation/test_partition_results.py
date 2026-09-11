"""Construction-time immutability of exported partition membership records."""

from datetime import date
from typing import cast

import pytest

from quantforge.data.identity import canonical_json_bytes
from quantforge.validation import (
    ExchangeSessionBoundary,
    PurgedPartitionObservations,
    ValidationBoundary,
    ValidationPlanError,
    WindowObservationSelection,
)

COLLECTION_FIELDS = ("retained", "purged", "warm_up_context", "study_observations")


def _result(
    field_name: str, observations: tuple[ValidationBoundary, ...]
) -> PurgedPartitionObservations | WindowObservationSelection:
    identity = "a" * 64
    if field_name in {"retained", "purged"}:
        return PurgedPartitionObservations(
            plan_id=identity,
            fold_id=identity,
            source_window_id=identity,
            protected_window_id=identity,
            retained=observations if field_name == "retained" else (),
            purged=observations if field_name == "purged" else (),
            source_dataset_id=identity,
            source_timeframe_configuration_id=identity,
        )
    return WindowObservationSelection(
        window_id=identity,
        warm_up_context=observations if field_name == "warm_up_context" else (),
        study_observations=observations if field_name == "study_observations" else (),
        source_dataset_id=identity,
        source_timeframe_configuration_id=identity,
    )


@pytest.mark.parametrize("field_name", COLLECTION_FIELDS)
@pytest.mark.parametrize(
    "collection",
    [[], [ExchangeSessionBoundary(date(2024, 1, 2))], ({"mutable": []},)],
    ids=["empty-list", "boundary-list", "mutable-tuple-member"],
)
def test_result_rejects_mutable_membership(field_name: str, collection: object) -> None:
    with pytest.raises(ValidationPlanError, match="tuple of validation boundaries"):
        _result(field_name, cast(tuple[ValidationBoundary, ...], collection))


@pytest.mark.parametrize("field_name", COLLECTION_FIELDS)
def test_result_tuple_capture_preserves_identity_after_source_list_edits(
    field_name: str,
) -> None:
    caller_keys: list[ValidationBoundary] = [ExchangeSessionBoundary(date(2024, 1, 2))]
    captured_keys = tuple(caller_keys)
    result = _result(field_name, captured_keys)
    original = canonical_json_bytes(result.to_primitive())
    caller_keys.append(ExchangeSessionBoundary(date(2024, 1, 3)))
    caller_keys.pop(0)
    assert getattr(result, field_name) == captured_keys
    assert canonical_json_bytes(result.to_primitive()) == original
    caller_keys.clear()
    assert canonical_json_bytes(result.to_primitive()) == original
    assert (
        canonical_json_bytes(_result(field_name, captured_keys).to_primitive())
        == original
    )
