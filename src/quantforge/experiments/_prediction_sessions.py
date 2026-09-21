"""Calendar-only checks for the contiguous daily input required by QF-11."""

from datetime import date

from quantforge.configuration import PrimitiveMapping
from quantforge.data.calendar import expected_sessions
from quantforge.data.exceptions import ValidationError
from quantforge.data.prediction_inputs import validate_prediction_provenance
from quantforge.experiments._json import ManifestError, text


def session_text(value: object) -> str:
    result = text(value)
    try:
        if date.fromisoformat(result).isoformat() != result:
            raise ValueError
    except ValueError as error:
        raise ManifestError("prediction session must be an ISO date") from error
    return result


def recorded_session_indexes(market: PrimitiveMapping) -> dict[str, int]:
    """Verify declared daily coverage; never read prices or create outcomes."""
    if (
        "intraday_provenance" in market
        or market.get("corporate_action_policy") == "not_provided_for_intraday_bars"
    ):
        try:
            validate_prediction_provenance(market)
        except (TypeError, ValueError, ValidationError) as error:
            raise ManifestError(str(error)) from error
    first = date.fromisoformat(session_text(market.get("actual_first_session")))
    last = date.fromisoformat(session_text(market.get("actual_last_session")))
    try:
        sessions = expected_sessions(first, last, text(market.get("calendar")))
    except ValueError as error:
        raise ManifestError("prediction calendar coverage is invalid") from error
    if (
        not sessions
        or sessions[0] != first
        or sessions[-1] != last
        or type(market.get("bar_count")) is not int
        or market["bar_count"] != len(sessions)
    ):
        raise ManifestError("prediction dataset sessions differ from declared coverage")
    indexes = {session.isoformat(): index for index, session in enumerate(sessions)}
    missing = market.get("missing_sessions")
    if not isinstance(missing, list) or any(
        session_text(item) in indexes for item in missing
    ):
        raise ManifestError("prediction dataset has missing observed sessions")
    return indexes


def validate_signal_session(
    prediction: PrimitiveMapping,
    rule: PrimitiveMapping,
    indexes: dict[str, int],
    *,
    decision_session: object = None,
    context_warm_up_observations: int | None = None,
) -> str:
    session = session_text(prediction.get("signal_session"))
    warm_up = rule.get("warm_up_observations")
    observations = (
        context_warm_up_observations
        if context_warm_up_observations is not None
        else indexes.get(session, -1) + 1
    )
    if (
        type(warm_up) is not int
        or warm_up < 1
        or observations < warm_up
        or (decision_session is not None and session != decision_session)
    ):
        raise ManifestError("prediction signal session or warm-up is incompatible")
    return session


def validate_outcome_session(
    signal: str, outcome: object, horizon: object, indexes: dict[str, int]
) -> None:
    outcome_session = session_text(outcome)
    if (
        type(horizon) is not int
        or horizon < 1
        or signal not in indexes
        or outcome_session not in indexes
        or indexes[outcome_session] != indexes[signal] + horizon
    ):
        raise ManifestError(
            "prediction outcome session differs from configured horizon"
        )
