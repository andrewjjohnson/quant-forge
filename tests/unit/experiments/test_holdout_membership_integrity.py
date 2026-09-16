"""A consumed request retains complete partition evidence even without a result."""

from copy import deepcopy
from typing import cast

import pytest

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.experiments._json import mapping
from tests.unit.experiments.test_holdout_request_integrity import (
    CapturedRequest,
    inspect_request,
)
from tests.unit.experiments.test_holdout_request_integrity import (
    captured as captured,
)
from tests.unit.experiments.test_holdout_request_integrity import (
    no_research as no_research,
)
from tests.unit.experiments.test_holdout_request_integrity import (
    prediction as prediction,
)


def refresh_selection(partition: PrimitiveMapping) -> None:
    member = mapping(partition["membership"])
    if "selection_id" in member:
        member["selection_id"] = configuration_identity(
            {key: value for key, value in member.items() if key != "selection_id"}
        )


@pytest.mark.parametrize("nested", [False, True], ids=["partition", "selection"])
@pytest.mark.parametrize("change", ["missing", "null", "extra"])
def test_membership_requires_complete_producer_records(
    captured: CapturedRequest, nested: bool, change: str
) -> None:
    original = mapping(captured.marker["request"])
    partition = mapping(original["evaluation_membership"])
    target = mapping(partition["membership"]) if nested else partition
    for field in target if change != "extra" else ("undeclared",):
        if change == "null" and target[field] is None:
            continue
        request = deepcopy(original)
        changed_partition = mapping(request["evaluation_membership"])
        changed = (
            mapping(changed_partition["membership"]) if nested else changed_partition
        )
        if change == "missing":
            del changed[field]
        else:
            changed[field] = None
        if nested and field != "selection_id":
            refresh_selection(changed_partition)
        inspect_request(captured, request)


@pytest.mark.parametrize(
    "change",
    [
        "empty",
        "window",
        "purge",
        "bounded_dataset_id",
        "bounded_data_sha256",
        "empty_sessions",
        "duplicate_sessions",
        "reversed_sessions",
        "foreign_sessions",
        "invalid_session",
        "source_dataset_id",
        "source_timeframe_configuration_id",
        "source_family_manifest_id",
        "window_id",
        "selection_id",
        "eligible_warm_up",
        "missing_observation",
        "duplicate_observation",
        "reversed_observations",
        "boundary_field",
        "missing_warm_up",
    ],
)
def test_rehashed_membership_requires_valid_domains_and_captured_boundaries(
    captured: CapturedRequest, change: str
) -> None:
    request = deepcopy(mapping(captured.marker["request"]))
    partition = mapping(request["evaluation_membership"])
    member = mapping(partition["membership"])
    observations = cast(list[PrimitiveMapping], member["study_observations"])
    sessions = cast(list[str], partition["evaluation_sessions"])
    if change == "empty":
        request["evaluation_membership"] = {}
    elif change == "window":
        partition["window"] = captured.completed.source.plan.folds[
            0
        ].test.to_primitive()
    elif change == "purge":
        partition["purge"] = {}
    elif change in {"bounded_dataset_id", "bounded_data_sha256"}:
        partition[change] = "not-a-digest"
    elif change == "empty_sessions":
        partition["evaluation_sessions"] = []
    elif change == "duplicate_sessions":
        sessions.append(sessions[-1])
    elif change == "reversed_sessions":
        # The small native holdout has one retained session, so add an earlier one.
        sessions.append("2024-01-02")
    elif change == "foreign_sessions":
        sessions[0] = "2024-01-02"
    elif change == "invalid_session":
        sessions[0] = "2024-01-01T00:00:00"
    elif change in {
        "source_dataset_id",
        "source_timeframe_configuration_id",
        "source_family_manifest_id",
        "window_id",
        "selection_id",
    }:
        member[change] = "0" * 64
    elif change == "eligible_warm_up":
        member["warm_up_eligible_for_selection"] = True
    elif change == "missing_observation":
        observations.pop()
    elif change == "duplicate_observation":
        observations.append(deepcopy(observations[-1]))
    elif change == "reversed_observations":
        observations.reverse()
    elif change == "boundary_field":
        observations[0]["undeclared"] = None
    else:
        member["warm_up_context"] = []
    if change != "selection_id":
        refresh_selection(partition)
    inspect_request(captured, request)
