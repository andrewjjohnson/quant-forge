"""QF-67 explicit causal feature schemas: allowlist, types, missingness, safety."""

from datetime import timedelta
from typing import Any, cast

import pytest

from quantforge.configuration import Primitive
from quantforge.examples.spy_ema import DAILY, TWO_MINUTES
from quantforge.examples.spy_ema_ml_dataset import ema_event_feature_schema
from quantforge.ml import (
    FORBIDDEN_FEATURE_NAMES,
    EventFeatureDefinition,
    EventFeatureSchema,
    EventFeatureSchemaError,
    FeatureValueType,
    MissingValuePolicy,
)
from quantforge.timeframes import IntradayInterval, Timeframe


def definition(**overrides: Any) -> EventFeatureDefinition:
    values: dict[str, Any] = {
        "name": "ema_fast",
        "source_field": "current_fast",
        "value_type": FeatureValueType.DECIMAL,
        "missing_values": MissingValuePolicy.REJECT,
        "unit": "price",
        "description": "current completed 2m fast EMA",
        "timeframe": TWO_MINUTES,
    }
    values.update(overrides)
    return EventFeatureDefinition(**values)


@pytest.mark.parametrize(
    "name",
    [
        # QF-49/QF-46 outcome and resolution fields.
        "raw_return",
        "outcome_price",
        "reference_price",
        "resolved_observation_timestamp",
        "requested_target_timestamp",
        "available",
        "status",
        # QF-47 path, excursion and target/stop fields.
        "mfe_percentage",
        "mae_timestamp",
        "label",
        "target_level",
        "unavailable_reason",
        "future_ranges",
        # Prefixed future/outcome names.
        "forward_return_5m",
        "future_high",
        "outcome_session",
        "evaluation_id",
        "target_hit",
        # Partition, holdout, experiment and provenance identifiers/hashes.
        "fold_id",
        "fold_index",
        "partition_role",
        "holdout_state",
        "selection_id",
        "experiment_id",
        "result_id",
        "lineage_id",
        "row_id",
        "context_id",
        "dataset_fingerprint",
        "bars_sha256",
        "window_hash",
        # Reserved dataset columns.
        "row_index",
        "decision_timestamp",
        "target",
        "direction",
    ],
)
def test_future_outcome_provenance_and_partition_names_are_refused(name: str) -> None:
    with pytest.raises(EventFeatureSchemaError):
        definition(name=name)
    with pytest.raises(EventFeatureSchemaError):
        definition(source_field=name)


@pytest.mark.parametrize(
    "name",
    ["previous_fast", "current_slow", "daily_ema50", "minutes_into_session", "rsi_14"],
)
def test_ordinary_causal_names_are_accepted(name: str) -> None:
    assert definition(name=name, source_field=name).name == name
    assert name not in FORBIDDEN_FEATURE_NAMES


@pytest.mark.parametrize("name", ["", "CamelCase", "1abc", "a__b", "x-y", "a" * 65])
def test_feature_names_must_be_canonical_snake_case(name: str) -> None:
    with pytest.raises(EventFeatureSchemaError):
        definition(name=name)


@pytest.mark.parametrize(
    "overrides",
    [
        {"value_type": "float"},
        {"value_type": "decimal"},
        {"missing_values": "impute_mean"},
        {"availability": "known_after_outcome"},
        {"source": "row_outcome"},
        {"timeframe": "2m"},
        {"unit": ""},
        {"description": " "},
        {"version": ""},
    ],
)
def test_unsupported_types_and_policies_are_refused(overrides: dict[str, Any]) -> None:
    with pytest.raises(EventFeatureSchemaError):
        definition(**overrides)


def test_schema_rejects_duplicates_and_non_definitions() -> None:
    with pytest.raises(EventFeatureSchemaError, match="unique"):
        EventFeatureSchema("s", "1", (definition(), definition()))
    with pytest.raises(EventFeatureSchemaError, match="only once"):
        EventFeatureSchema("s", "1", (definition(), definition(name="ema_fast_copy")))
    with pytest.raises(EventFeatureSchemaError, match="nonempty"):
        EventFeatureSchema("s", "1", ())
    with pytest.raises(EventFeatureSchemaError, match="nonempty"):
        EventFeatureSchema("s", "1", [definition()])  # type: ignore[arg-type]
    with pytest.raises(EventFeatureSchemaError, match="only EventFeatureDefinition"):
        EventFeatureSchema(
            "s",
            "1",
            cast(tuple[EventFeatureDefinition, ...], ({"name": "ema_fast"},)),
        )


def test_schema_identity_is_versioned_ordered_and_round_trips() -> None:
    schema = ema_event_feature_schema()
    assert schema.column_names == (
        "ema_fast_previous",
        "ema_slow_previous",
        "ema_fast",
        "ema_slow",
        "daily_close",
        "daily_ema50",
    )
    primitive = schema.to_primitive()
    assert EventFeatureSchema.from_primitive(primitive) == schema
    assert EventFeatureSchema.from_primitive(primitive).schema_id == schema.schema_id
    reordered = EventFeatureSchema(schema.name, schema.version, schema.features[::-1])
    assert reordered.schema_id != schema.schema_id
    renamed = definition(version="2")
    assert renamed.definition_id != definition().definition_id
    assert definition(timeframe=DAILY).definition_id != definition().definition_id
    tampered = schema.to_primitive()
    features = cast(list[dict[str, Primitive]], tampered["features"])
    features[0]["definition_id"] = "0" * 64
    with pytest.raises(EventFeatureSchemaError, match="identity"):
        EventFeatureSchema.from_primitive(tampered)
    extra = schema.to_primitive()
    cast(list[dict[str, Primitive]], extra["features"])[0]["feature_raw_return"] = "1"
    with pytest.raises(EventFeatureSchemaError, match="fields"):
        EventFeatureSchema.from_primitive(extra)


def test_values_are_exact_typed_scalars_and_nulls_are_never_imputed() -> None:
    decimal = definition()
    assert decimal.value({"current_fast": "593.4127"}) == "593.4127"
    invalid_values: tuple[Primitive, ...] = (
        593.4127,
        "593.41270",
        "1E+3",
        "NaN",
        "Infinity",
        {},
        [],
        True,
    )
    for invalid in invalid_values:
        with pytest.raises(EventFeatureSchemaError):
            decimal.value({"current_fast": invalid})
    with pytest.raises(EventFeatureSchemaError, match="missing"):
        decimal.value({"current_fast": None})
    with pytest.raises(EventFeatureSchemaError, match="does not match"):
        decimal.value({"previous_fast": "1"})
    nullable = definition(missing_values=MissingValuePolicy.PRESERVE_NULL)
    assert nullable.value({"current_fast": None}) is None
    integer = definition(value_type=FeatureValueType.INTEGER)
    assert integer.value({"current_fast": 3}) == 3
    for invalid in (True, "3", 3.0):
        with pytest.raises(EventFeatureSchemaError):
            integer.value({"current_fast": cast(Primitive, invalid)})
    boolean = definition(value_type=FeatureValueType.BOOLEAN)
    assert boolean.value({"current_fast": False}) is False
    with pytest.raises(EventFeatureSchemaError):
        boolean.value({"current_fast": 0})


def test_source_timeframes_must_be_declared_by_the_plan() -> None:
    schema = EventFeatureSchema("s", "1", (definition(),))
    schema.require_plan_timeframes((TWO_MINUTES, DAILY))
    five = Timeframe.us_equity(IntradayInterval(timedelta(minutes=5)))
    undeclared = EventFeatureSchema("s", "1", (definition(timeframe=five),))
    with pytest.raises(EventFeatureSchemaError, match="timeframe"):
        undeclared.require_plan_timeframes((TWO_MINUTES, DAILY))
