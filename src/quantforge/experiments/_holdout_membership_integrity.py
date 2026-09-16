"""Validate captured holdout partition metadata without constructing a partition."""

from datetime import date, datetime
from typing import cast

from quantforge.configuration import configuration_identity
from quantforge.experiments._aggregate_schema import record, records
from quantforge.experiments._json import ManifestError, digest, mapping, text
from quantforge.experiments._prediction_sessions import session_text
from quantforge.oos.models import OOSSource
from quantforge.timeframes import resolve_exchange_session
from quantforge.validation import (
    ExchangeSessionBoundary,
    TimestampBoundary,
    ValidationBoundary,
    WindowObservationSelection,
)
from quantforge.validation.models import boundary_value


def _boundaries(
    value: object, reference: ValidationBoundary
) -> tuple[ValidationBoundary, ...]:
    result: list[ValidationBoundary] = []
    for item in records(value):
        boundary = (
            ExchangeSessionBoundary(
                date.fromisoformat(session_text(item.get("session"))),
                reference.session_policy,
            )
            if isinstance(reference, ExchangeSessionBoundary)
            else TimestampBoundary(datetime.fromisoformat(text(item.get("timestamp"))))
        )
        if configuration_identity(item) != configuration_identity(
            boundary.to_primitive()
        ):
            raise ManifestError("holdout membership boundary is inconsistent")
        result.append(boundary)
    return tuple(result)


def validate_holdout_membership(source: OOSSource, value: object) -> None:
    """Read schema, identities, chronology and retained-session evidence only."""
    try:
        partition = record(
            value,
            {
                "window",
                "membership",
                "purge",
                "evaluation_sessions",
                "bounded_dataset_id",
                "bounded_data_sha256",
            },
            "holdout membership",
        )
        window = source.plan.final_holdout.window
        timeframe = source.plan.environment.outcome_dataset.standalone_timeframe
        if timeframe is None:
            raise ManifestError("holdout membership requires a source timeframe")
        member = mapping(partition["membership"])
        captured = WindowObservationSelection(
            window.window_id,
            _boundaries(member.get("warm_up_context"), window.interval.start),
            _boundaries(member.get("study_observations"), window.interval.start),
            source.plan.environment.outcome_dataset.dataset_ids[0],
            timeframe.configuration_id,
        )
        if (
            configuration_identity(member)
            != configuration_identity(captured.to_primitive())
            or configuration_identity(mapping(partition["window"]))
            != configuration_identity(window.to_primitive())
            or partition["purge"] is not None
            or len(captured.warm_up_context)
            != window.warm_up_observations_for(timeframe)
            or not captured.study_observations
            or captured.study_observations[0] != window.interval.start
            or captured.study_observations[-1] != window.interval.end
            or any(
                not window.interval.contains(item)
                for item in captured.study_observations
            )
            or any(
                boundary_value(item) >= boundary_value(window.interval.start)
                for item in captured.warm_up_context
            )
        ):
            raise ManifestError("holdout membership differs from its source window")
        digest(partition["bounded_dataset_id"])
        digest(partition["bounded_data_sha256"])
        sessions = partition["evaluation_sessions"]
        if not isinstance(sessions, list) or not sessions:
            raise ManifestError("holdout evaluation sessions must be a nonempty array")
        labels = [date.fromisoformat(session_text(item)) for item in sessions]
        if labels != sorted(set(labels)):
            raise ManifestError(
                "holdout evaluation sessions must be ordered and unique"
            )
        horizon = source.plan.purge_policy.label_horizon
        retained = (
            tuple(
                item
                for item in captured.study_observations
                if cast(TimestampBoundary, item).timestamp
                <= cast(TimestampBoundary, window.interval.end).timestamp
                - horizon.elapsed
            )
            if horizon.elapsed is not None
            else captured.study_observations[: -horizon.exchange_sessions]
            if horizon.exchange_sessions
            else captured.study_observations
        )
        evidence = tuple(
            ExchangeSessionBoundary(label, timeframe.session_policy)
            if isinstance(window.interval.start, ExchangeSessionBoundary)
            else TimestampBoundary(
                resolve_exchange_session(
                    label, timeframe.session_policy
                ).close_timestamp
            )
            for label in labels
        )
        minimum = mapping(source.definition.to_primitive()["configuration"])[
            "minimum_test_observations"
        ]
        if evidence != retained or type(minimum) is not int or len(evidence) < minimum:
            raise ManifestError(
                "holdout evaluation sessions differ from captured membership"
            )
    except (ValueError, KeyError, TypeError) as error:
        raise ManifestError("holdout evaluation membership is invalid") from error
