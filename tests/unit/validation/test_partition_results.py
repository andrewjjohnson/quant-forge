"""Construction-time immutability of exported partition membership records."""

from dataclasses import replace
from datetime import UTC, date, datetime
from typing import cast

import pytest

from quantforge.data.identity import canonical_json_bytes
from quantforge.timeframes import ExchangeSessionPolicy
from quantforge.validation import (
    ExchangeSessionBoundary,
    PurgedPartitionObservations,
    TimestampBoundary,
    ValidationBoundary,
    ValidationPlanError,
    WindowObservationSelection,
)

COLLECTION_FIELDS = ("retained", "purged", "warm_up_context", "study_observations")
SESSION_A = ExchangeSessionBoundary(date(2024, 1, 2))
SESSION_B = ExchangeSessionBoundary(date(2024, 1, 3))
SESSION_C = ExchangeSessionBoundary(date(2024, 1, 4))
TIMESTAMP_A = TimestampBoundary(datetime(2024, 1, 2, 15, tzinfo=UTC))
TIMESTAMP_B = TimestampBoundary(datetime(2024, 1, 3, 15, tzinfo=UTC))
LONDON_B = ExchangeSessionBoundary(
    date(2024, 1, 3), ExchangeSessionPolicy("XLON", "Europe/London")
)


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


def _paired_result(
    kind: str,
    earlier: tuple[ValidationBoundary, ...],
    later: tuple[ValidationBoundary, ...],
) -> PurgedPartitionObservations | WindowObservationSelection:
    result = _result("retained" if kind == "purge" else "warm_up_context", earlier)
    if isinstance(result, PurgedPartitionObservations):
        return replace(result, purged=later)
    return replace(result, study_observations=later)


@pytest.mark.parametrize("field_name", COLLECTION_FIELDS)
@pytest.mark.parametrize(
    "observations",
    [
        (SESSION_A, SESSION_A),
        (SESSION_B, SESSION_A),
        (SESSION_A, TIMESTAMP_B),
        (SESSION_A, LONDON_B),
    ],
    ids=["duplicate", "reversed", "mixed-axes", "mixed-session-policies"],
)
def test_result_rejects_invalid_collection_chronology(
    field_name: str,
    observations: tuple[ValidationBoundary, ...],
) -> None:
    with pytest.raises(ValidationPlanError):
        _result(field_name, observations)


@pytest.mark.parametrize("kind", ["purge", "selection"])
@pytest.mark.parametrize(
    ("earlier", "later"),
    [
        ((SESSION_A, SESSION_B), (SESSION_B, SESSION_C)),
        ((SESSION_B,), (SESSION_A,)),
        ((SESSION_A, SESSION_C), (SESSION_B,)),
        ((SESSION_A,), (TIMESTAMP_B,)),
        ((SESSION_A,), (LONDON_B,)),
    ],
    ids=["overlap", "reversed", "interleaved", "mixed-axes", "mixed-session-policies"],
)
def test_result_rejects_incompatible_cross_field_membership(
    kind: str,
    earlier: tuple[ValidationBoundary, ...],
    later: tuple[ValidationBoundary, ...],
) -> None:
    with pytest.raises(ValidationPlanError):
        _paired_result(kind, earlier, later)


@pytest.mark.parametrize("kind", ["purge", "selection"])
@pytest.mark.parametrize("keys", [(SESSION_A, SESSION_B), (TIMESTAMP_A, TIMESTAMP_B)])
def test_result_preserves_valid_ordered_and_empty_membership(
    kind: str,
    keys: tuple[ValidationBoundary, ...],
) -> None:
    for earlier, later in ((keys[:1], keys[1:]), ((), keys), (keys, ()), ((), ())):
        result = _paired_result(kind, earlier, later)
        assert canonical_json_bytes(result.to_primitive()) == canonical_json_bytes(
            _paired_result(kind, earlier, later).to_primitive()
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
