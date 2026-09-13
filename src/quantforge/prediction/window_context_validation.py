"""Validate persisted available QF-28 context metadata without component execution."""

from datetime import UTC, datetime, timedelta
from typing import cast

from quantforge.configuration import Primitive, PrimitiveMapping, configuration_identity
from quantforge.data.multi_timeframe import ContextCompletionPolicy
from quantforge.prediction.errors import InvalidPredictionOutputError
from quantforge.timeframes import BarCompletion


def _mapping(value: Primitive, label: str) -> PrimitiveMapping:
    if not isinstance(value, dict):
        raise InvalidPredictionOutputError(f"{label} must be an object")
    return value


def _records(value: Primitive, label: str) -> list[PrimitiveMapping]:
    if not isinstance(value, list):
        raise InvalidPredictionOutputError(f"{label} must be an ordered collection")
    return [_mapping(item, label) for item in value]


def _equal(actual: Primitive, expected: Primitive, label: str) -> None:
    if configuration_identity({"record": actual}) != configuration_identity(
        {"record": expected}
    ):
        raise InvalidPredictionOutputError(f"{label} is inconsistent")


def _timestamp(value: Primitive) -> datetime:
    if not isinstance(value, str):
        raise InvalidPredictionOutputError("context bar timestamp is missing")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise InvalidPredictionOutputError(
            "context bar timestamp is invalid"
        ) from error
    if parsed.utcoffset() is None:
        raise InvalidPredictionOutputError(
            "context bar timestamp must be timezone-aware"
        )
    return parsed.astimezone(UTC)


def _bar_ids(value: Primitive) -> list[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise InvalidPredictionOutputError("context bar IDs are invalid")
    result = cast(list[str], value)
    if len(set(result)) != len(result):
        raise InvalidPredictionOutputError("context bar IDs must be unique")
    return result


def _developing_end(
    aligned: PrimitiveMapping,
    timeframe: PrimitiveMapping,
    visible_ids: list[str],
    decision_timestamp: datetime,
    *,
    family_id: Primitive,
    symbol: Primitive,
) -> datetime:
    developing = _mapping(aligned.get("developing_bar"), "developing bar evidence")
    bar = _mapping(developing.get("bar"), "developing bar")
    bar_id = configuration_identity(bar)
    _equal(developing.get("bar_id"), bar_id, "developing bar identity")
    _equal(visible_ids[-1], bar_id, "latest developing bar reference")
    _equal(bar.get("timeframe"), timeframe, "developing timeframe")
    _equal(bar.get("symbol"), symbol, "developing symbol")
    _equal(bar.get("complete"), False, "developing completion flag")
    _equal(
        bar.get("completion"), BarCompletion.DEVELOPING.value, "developing completion"
    )
    reference = _mapping(
        bar.get("source_dataset_reference"), "developing source reference"
    )
    _equal(reference.get("family_id"), family_id, "developing source family")
    observed_start = _timestamp(bar.get("observed_start_timestamp"))
    observed_end = _timestamp(bar.get("observed_end_timestamp"))
    if (
        _timestamp(bar.get("as_of")) != decision_timestamp
        or not observed_start < observed_end <= decision_timestamp
        or _timestamp(bar.get("expected_completion_boundary")) <= decision_timestamp
    ):
        raise InvalidPredictionOutputError(
            "developing bar has incompatible temporal boundaries"
        )
    return observed_end


def _validate_timeframe(
    aligned: PrimitiveMapping,
    selected: PrimitiveMapping,
    requirement: PrimitiveMapping,
    decision_timestamp: datetime,
    *,
    primary: bool,
    source_completion_policy: Primitive,
    family_id: Primitive,
    symbol: Primitive,
) -> str:
    timeframe = _mapping(requirement.get("timeframe"), "declared timeframe")
    timeframe_configuration = _mapping(
        timeframe.get("configuration"), "timeframe configuration"
    )
    _equal(
        aligned.get("requirement"),
        {
            "timeframe": timeframe,
            "maximum_age_microseconds": None
            if primary
            else requirement.get("maximum_age_microseconds"),
        },
        "source timeframe requirement",
    )
    _equal(selected.get("requirement"), requirement, "rule timeframe requirement")
    _equal(
        aligned.get("bar_interval"),
        timeframe_configuration.get("interval"),
        "source bar interval",
    )
    reference = _mapping(
        aligned.get("dataset_reference"), "timeframe dataset reference"
    )
    dataset_id, source_id = (
        reference.get("dataset_id"),
        reference.get("canonical_source_snapshot_id"),
    )
    if (
        not isinstance(dataset_id, str)
        or not dataset_id
        or not isinstance(source_id, str)
        or not source_id
    ):
        raise InvalidPredictionOutputError("timeframe dataset identity is missing")
    _equal(aligned.get("dataset_id"), dataset_id, "timeframe dataset ID")
    _equal(reference.get("family_id"), family_id, "timeframe family")
    _equal(
        reference.get("timeframe_configuration_id"),
        timeframe.get("configuration_id"),
        "dataset timeframe",
    )
    if aligned.get("availability") != "available":
        raise InvalidPredictionOutputError(
            "required context timeframe is missing or stale"
        )
    visible_ids = _bar_ids(aligned.get("visible_bar_ids"))
    if not visible_ids:
        raise InvalidPredictionOutputError("required context timeframe has no bars")
    completed_end = (
        None
        if aligned.get("latest_completed_bar_timestamp") is None
        else _timestamp(aligned["latest_completed_bar_timestamp"])
    )
    completion = aligned.get("latest_completion_state")
    if completion == BarCompletion.DEVELOPING.value:
        if (
            primary
            or source_completion_policy
            != ContextCompletionPolicy.DEVELOPING_BAR_AS_OF.value
        ):
            raise InvalidPredictionOutputError(
                "context exposes an undeclared developing bar"
            )
        latest_end = _developing_end(
            aligned,
            timeframe,
            visible_ids,
            decision_timestamp,
            family_id=family_id,
            symbol=symbol,
        )
        completed_ids = visible_ids[:-1]
    elif isinstance(completion, str) and completion in {
        item.value for item in BarCompletion if item is not BarCompletion.DEVELOPING
    }:
        if "developing_bar" in aligned or completed_end is None:
            raise InvalidPredictionOutputError(
                "completed timeframe metadata is inconsistent"
            )
        latest_end, completed_ids = completed_end, visible_ids
    else:
        raise InvalidPredictionOutputError("context completion state is invalid")
    if (
        bool(completed_ids) != (completed_end is not None)
        or (completed_end is not None and completed_end > latest_end)
        or latest_end > decision_timestamp
        or (primary and latest_end != decision_timestamp)
    ):
        raise InvalidPredictionOutputError(
            "context timeframe exposes incompatible bar timestamps"
        )
    _equal(
        aligned.get("age_microseconds"),
        (decision_timestamp - latest_end) // timedelta(microseconds=1),
        "timeframe age",
    )
    if (
        requirement.get("completion_policy")
        == ContextCompletionPolicy.COMPLETED_BARS_ONLY.value
    ):
        selected_ids, selected_end = completed_ids, completed_end
    else:
        selected_ids, selected_end = visible_ids, latest_end
    if not selected_ids or selected_end is None:
        raise InvalidPredictionOutputError("rule timeframe has no permitted bars")
    maximum_age = requirement.get("maximum_age_microseconds")
    if maximum_age is not None and (
        not isinstance(maximum_age, int)
        or isinstance(maximum_age, bool)
        or maximum_age <= 0
        or (decision_timestamp - selected_end) // timedelta(microseconds=1)
        > maximum_age
    ):
        raise InvalidPredictionOutputError("rule timeframe exceeds its maximum age")
    _equal(
        selected.get("visible_bar_ids"),
        cast(list[Primitive], selected_ids),
        "rule selected bar IDs",
    )
    return source_id


def validate_window_context_snapshot(
    *,
    context: PrimitiveMapping,
    source: PrimitiveMapping,
    requirements: PrimitiveMapping,
    market_data: PrimitiveMapping,
    primary_timeframe: PrimitiveMapping,
    timestamp: str,
) -> None:
    """Validate available rule/source context metadata against fixed window inputs."""
    _equal(
        {
            "prediction_dataset_id": context.get("prediction_dataset_id"),
            "symbol": context.get("symbol"),
            "adjustment_basis": context.get("adjustment_basis"),
        },
        {
            "prediction_dataset_id": market_data["dataset_id"],
            "symbol": market_data["symbol"],
            "adjustment_basis": {
                key: market_data[key]
                for key in (
                    "adjustment_mode",
                    "ohlc_basis",
                    "volume_basis",
                    "corporate_action_policy",
                    "adjusted_fields_used",
                )
            },
        },
        "rule context dataset provenance",
    )
    primary = _mapping(requirements.get("primary"), "primary requirement")
    contextual = _records(requirements.get("contextual"), "contextual requirements")
    declared = [primary, *contextual]
    _equal(
        source.get("primary_timeframe"), primary_timeframe, "source primary timeframe"
    )
    _equal(primary.get("timeframe"), primary_timeframe, "declared primary timeframe")
    source_policy = requirements.get("context_completion_policy")
    _equal(source.get("completion_policy"), source_policy, "context completion policy")
    _equal(
        source.get("required_timeframes"),
        [
            {
                "timeframe": item.get("timeframe"),
                "maximum_age_microseconds": item.get("maximum_age_microseconds"),
            }
            for item in contextual
        ],
        "source contextual requirements",
    )
    aligned = _records(source.get("timeframes"), "source timeframes")
    selected = _records(context.get("timeframes"), "rule timeframes")
    if len(aligned) != len(declared) or len(selected) != len(declared):
        raise InvalidPredictionOutputError(
            "context timeframe coverage is incomplete or duplicated"
        )
    consistency = _mapping(source.get("source_consistency"), "source consistency")
    decision_timestamp = _timestamp(timestamp)
    source_ids = {
        _validate_timeframe(
            source_entry,
            rule_entry,
            requirement,
            decision_timestamp,
            primary=index == 0,
            source_completion_policy=source_policy,
            family_id=consistency.get("family_id"),
            symbol=market_data["symbol"],
        )
        for index, (source_entry, rule_entry, requirement) in enumerate(
            zip(aligned, selected, declared, strict=True)
        )
    }
    if len(source_ids) != 1:
        raise InvalidPredictionOutputError(
            "context timeframes have different canonical sources"
        )
