"""Validate persisted QF-20/QF-28 context metadata without component execution."""

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import cast

from quantforge.configuration import Primitive, PrimitiveMapping, configuration_identity
from quantforge.data.lineage import SourceConsistencyMode
from quantforge.data.multi_timeframe import (
    MULTI_TIMEFRAME_CONTEXT_SCHEMA_VERSION,
    ContextCompletionPolicy,
)
from quantforge.indicators.timeframe import TIMEFRAME_INDICATOR_CONTRACT_VERSION
from quantforge.prediction.errors import InvalidPredictionOutputError
from quantforge.prediction.window_timeframe_validation import (
    validate_source_timeframe_definition,
)
from quantforge.timeframes import BarCompletion


@dataclass(frozen=True)
class _SourceTimeframe:
    snapshot: PrimitiveMapping
    visible_ids: list[str]
    completed_ids: list[str]
    latest_end: datetime | None
    completed_end: datetime | None


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
) -> datetime:
    developing = _mapping(aligned.get("developing_bar"), "developing bar evidence")
    bar = _mapping(developing.get("bar"), "developing bar")
    bar_id = configuration_identity(bar)
    _equal(developing.get("bar_id"), bar_id, "developing bar identity")
    _equal(visible_ids[-1], bar_id, "latest developing bar reference")
    _equal(bar.get("timeframe"), timeframe, "developing timeframe")
    if not isinstance(bar.get("symbol"), str) or not bar["symbol"]:
        raise InvalidPredictionOutputError("developing symbol is missing")
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


def _validate_source_timeframe(
    aligned: PrimitiveMapping,
    requirement: PrimitiveMapping,
    decision_timestamp: datetime,
    *,
    primary: bool,
    source_completion_policy: Primitive,
    family_id: Primitive,
) -> _SourceTimeframe:
    timeframe = _mapping(requirement.get("timeframe"), "declared timeframe")
    timeframe_configuration = _mapping(
        timeframe.get("configuration"), "timeframe configuration"
    )
    _equal(
        aligned.get("requirement"),
        requirement,
        "source timeframe requirement",
    )
    _equal(
        aligned.get("bar_interval"),
        timeframe_configuration.get("interval"),
        "source bar interval",
    )
    reference = aligned.get("dataset_reference")
    if reference is None:
        _equal(aligned.get("dataset_id"), None, "absent timeframe dataset")
    else:
        reference = _mapping(reference, "timeframe dataset reference")
        if any(
            not isinstance(reference.get(key), str) or not reference[key]
            for key in (
                "dataset_id",
                "canonical_source_snapshot_id",
                "family_id",
                "timeframe_configuration_id",
            )
        ):
            raise InvalidPredictionOutputError("timeframe dataset identity is missing")
        _equal(
            aligned.get("dataset_id"), reference["dataset_id"], "timeframe dataset ID"
        )
        _equal(reference["family_id"], family_id, "timeframe family")
        _equal(
            reference["timeframe_configuration_id"],
            timeframe.get("configuration_id"),
            "dataset timeframe",
        )
    maximum_age = requirement.get("maximum_age_microseconds")
    if maximum_age is not None and (
        not isinstance(maximum_age, int)
        or isinstance(maximum_age, bool)
        or maximum_age <= 0
    ):
        raise InvalidPredictionOutputError("source maximum age must be positive")
    visible_ids = _bar_ids(aligned.get("visible_bar_ids"))
    if not visible_ids:
        _equal(
            {
                key: aligned.get(key)
                for key in (
                    "availability",
                    "latest_completed_bar_timestamp",
                    "latest_completion_state",
                    "age_microseconds",
                )
            },
            {
                "availability": "missing",
                "latest_completed_bar_timestamp": None,
                "latest_completion_state": None,
                "age_microseconds": None,
            },
            "missing timeframe metadata",
        )
        if "developing_bar" in aligned:
            raise InvalidPredictionOutputError(
                "missing timeframe exposes developing evidence"
            )
        return _SourceTimeframe(aligned, [], [], None, None)
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
    ):
        raise InvalidPredictionOutputError(
            "context timeframe exposes incompatible bar timestamps"
        )
    age = (decision_timestamp - latest_end) // timedelta(microseconds=1)
    _equal(aligned.get("age_microseconds"), age, "timeframe age")
    _equal(
        aligned.get("availability"),
        "stale" if isinstance(maximum_age, int) and age > maximum_age else "available",
        "timeframe availability",
    )
    return _SourceTimeframe(
        aligned, visible_ids, completed_ids, latest_end, completed_end
    )


def _validate_rule_timeframe(
    aligned: _SourceTimeframe,
    selected: PrimitiveMapping,
    requirement: PrimitiveMapping,
    decision_timestamp: datetime,
    *,
    primary: bool,
    symbol: Primitive,
) -> None:
    _equal(selected.get("requirement"), requirement, "rule timeframe requirement")
    if aligned.snapshot.get("availability") != "available":
        raise InvalidPredictionOutputError(
            "required context timeframe is missing or stale"
        )
    if primary and aligned.latest_end != decision_timestamp:
        raise InvalidPredictionOutputError(
            "primary bar must end at the scheduled timestamp"
        )
    if "developing_bar" in aligned.snapshot:
        developing = _mapping(aligned.snapshot["developing_bar"], "developing evidence")
        bar = _mapping(developing.get("bar"), "developing bar")
        _equal(bar.get("symbol"), symbol, "developing symbol")
    if (
        requirement.get("completion_policy")
        == ContextCompletionPolicy.COMPLETED_BARS_ONLY.value
    ):
        selected_ids, selected_end = aligned.completed_ids, aligned.completed_end
    else:
        selected_ids, selected_end = aligned.visible_ids, aligned.latest_end
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
    _validate_indicators(selected, requirement, aligned.snapshot)


def _validate_indicators(
    selected: PrimitiveMapping,
    requirement: PrimitiveMapping,
    source: PrimitiveMapping,
) -> None:
    reference = _mapping(source.get("dataset_reference"), "rule dataset reference")
    expanded_reference: PrimitiveMapping = {
        **reference,
        "feed_scope": requirement.get("feed_scope"),
    }
    expected: list[Primitive] = []
    for declaration in _records(requirement.get("indicators"), "declared indicators"):
        indicator = _mapping(declaration.get("indicator"), "declared indicator")
        # Rebuild QF-22's bound configuration from the fixed declaration and
        # validated source metadata; no indicator implementation is invoked.
        configuration: PrimitiveMapping = {
            "component_type": "timeframe_indicator",
            "contract_version": TIMEFRAME_INDICATOR_CONTRACT_VERSION,
            "indicator": {
                "configuration_id": indicator.get("configuration_id"),
                "configuration": indicator.get("configuration"),
            },
            "source": {
                "timeframe": requirement.get("timeframe"),
                "fields": indicator.get("source_fields"),
                "completion_policy": requirement.get("completion_policy"),
                "developing_bar_support": indicator.get("developing_bar_support"),
                "observation_unit": "bar",
                "warm_up_bars": indicator.get("warm_up_bars"),
                "aggregation_provenance": expanded_reference,
                "feed_scope": requirement.get("feed_scope"),
            },
        }
        expected.append(
            {
                "alias": declaration.get("alias"),
                "indicator_name": indicator.get("name"),
                "configuration_id": configuration_identity(configuration),
                "backend": indicator.get("backend"),
                "source_timeframe": requirement.get("timeframe"),
                "source_fields": indicator.get("source_fields"),
                "completion_policy": requirement.get("completion_policy"),
                "dataset_reference": expanded_reference,
                "warm_up_bars": indicator.get("warm_up_bars"),
                "visible_bar_ids": selected.get("visible_bar_ids"),
                "output_fields": indicator.get("output_fields"),
            }
        )
    _equal(selected.get("indicators"), expected, "rule indicator manifests")


def validate_window_source_snapshot(
    source: PrimitiveMapping,
) -> tuple[_SourceTimeframe, ...]:
    """Check QF-20's internal contract, including evidence retained by SKIP."""
    _equal(
        source.get("schema_version"),
        MULTI_TIMEFRAME_CONTEXT_SCHEMA_VERSION,
        "source schema",
    )
    _equal(
        source.get("artifact_type"), "multi_timeframe_context", "source artifact type"
    )
    source_policy = source.get("completion_policy")
    if not isinstance(source_policy, str) or source_policy not in {
        item.value for item in ContextCompletionPolicy
    }:
        raise InvalidPredictionOutputError("source completion policy is invalid")
    primary = _mapping(source.get("primary_timeframe"), "source primary timeframe")
    contextual = _records(source.get("required_timeframes"), "source requirements")
    declared: list[PrimitiveMapping] = [
        {"timeframe": primary, "maximum_age_microseconds": None},
        *contextual,
    ]
    timeframe_ids: list[str] = []
    session_policy: Primitive = None
    for requirement in declared:
        timeframe = _mapping(
            requirement.get("timeframe"), "source requirement timeframe"
        )
        validate_source_timeframe_definition(timeframe)
        configuration = _mapping(
            timeframe.get("configuration"), "source timeframe configuration"
        )
        configuration_id = configuration_identity(configuration)
        _equal(
            timeframe.get("configuration_id"),
            configuration_id,
            "source timeframe identity",
        )
        _equal(
            configuration.get("developing_bar_exposure"),
            "completed_only",
            "source timeframe exposure",
        )
        if not timeframe_ids:
            session_policy = _mapping(
                configuration.get("session_policy"), "source session policy"
            )
        _equal(
            configuration.get("session_policy"), session_policy, "source session policy"
        )
        timeframe_ids.append(configuration_id)
    if len(set(timeframe_ids)) != len(timeframe_ids) or timeframe_ids[1:] != sorted(
        timeframe_ids[1:]
    ):
        raise InvalidPredictionOutputError(
            "source requirements must be unique and ordered"
        )
    aligned = _records(source.get("timeframes"), "source timeframes")
    if len(aligned) != len(declared):
        raise InvalidPredictionOutputError(
            "source timeframe coverage is incomplete or duplicated"
        )
    consistency = _mapping(source.get("source_consistency"), "source consistency")
    _equal(
        consistency,
        {
            "mode": SourceConsistencyMode.COMMON_DATASET_FAMILY.value,
            "family_id": consistency.get("family_id"),
            "external_validation_policy_id": None,
        },
        "source consistency",
    )
    source_ids: set[str] = set()
    for item in aligned:
        reference = item.get("dataset_reference")
        if isinstance(reference, dict):
            source_id = reference.get("canonical_source_snapshot_id")
            if isinstance(source_id, str):
                source_ids.add(source_id)
    if len(source_ids) != 1:
        raise InvalidPredictionOutputError(
            "context requires one common canonical source"
        )
    as_of = _timestamp(source.get("as_of"))
    return tuple(
        _validate_source_timeframe(
            entry,
            requirement,
            as_of,
            primary=index == 0,
            source_completion_policy=source_policy,
            family_id=consistency.get("family_id"),
        )
        for index, (entry, requirement) in enumerate(
            zip(aligned, declared, strict=True)
        )
    )


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
    aligned = validate_window_source_snapshot(source)
    selected = _records(context.get("timeframes"), "rule timeframes")
    if len(aligned) != len(declared) or len(selected) != len(declared):
        raise InvalidPredictionOutputError(
            "context timeframe coverage is incomplete or duplicated"
        )
    decision_timestamp = _timestamp(timestamp)
    for index, (source_entry, rule_entry, requirement) in enumerate(
        zip(aligned, selected, declared, strict=True)
    ):
        _validate_rule_timeframe(
            source_entry,
            rule_entry,
            requirement,
            decision_timestamp,
            primary=index == 0,
            symbol=market_data["symbol"],
        )
