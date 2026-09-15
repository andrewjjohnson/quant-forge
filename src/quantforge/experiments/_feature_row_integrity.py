"""Verify QF-7/QF-29 row contracts without feature or outcome execution."""

from dataclasses import replace
from typing import cast

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.experiments._json import ManifestError, digest, mapping, snapshot, text
from quantforge.experiments._prediction_sessions import (
    recorded_session_indexes,
    session_text,
)
from quantforge.prediction.feature_dataset import (
    _disposition_fields,  # pyright: ignore[reportPrivateUsage]
    _identity_fields,  # pyright: ignore[reportPrivateUsage]
    _row_id,  # pyright: ignore[reportPrivateUsage]
    _schema_value_matches,  # pyright: ignore[reportPrivateUsage]
)
from quantforge.prediction.signal_feature_models import SchemaField, SchemaFieldCategory


def _records(value: object) -> list[PrimitiveMapping]:
    if not isinstance(value, list):
        raise ManifestError("feature metadata records must be arrays")
    return [mapping(item) for item in cast(list[object], value)]


def _field(record: PrimitiveMapping) -> SchemaField:
    if type(record.get("nullable")) is not bool:
        raise ManifestError("feature schema nullability must be boolean")
    try:
        return SchemaField(
            text(record.get("field_name")),
            SchemaFieldCategory(text(record.get("category"))),
            text(record.get("data_type")),
            text(record.get("unit")),
            cast(bool, record["nullable"]),
            text(record.get("calculation_or_source")),
            text(record.get("temporal_availability")),
            snapshot(mapping(record["provenance"])) if "provenance" in record else None,
        )
    except ValueError as error:
        raise ManifestError("feature schema field is invalid") from error


def validate_feature_schema(
    manifest: PrimitiveMapping, schema: PrimitiveMapping
) -> tuple[SchemaField, ...]:
    """Bind the schema to persisted definitions, retaining their metadata verbatim."""
    configuration = mapping(manifest.get("configuration"))
    if manifest.get("market_data") != configuration.get("source_data") or manifest.get(
        "engine_version"
    ) != configuration.get("engine_version"):
        raise ManifestError("feature dataset provenance differs from its configuration")
    features = [
        *(
            _field(field)
            for field in _records(configuration.get("strategy_feature_fields"))
        ),
        *(
            _field(mapping(item.get("definition")))
            for item in _records(configuration.get("contextual_features"))
        ),
    ]
    fields = (
        *_identity_fields(),
        *_disposition_fields(),
        *(
            replace(field, name="feature_" + field.name)
            for field in sorted(features, key=lambda field: field.name)
        ),
        *(
            replace(
                _field(field),
                name="outcome_"
                + text(outcome.get("namespace"))
                + "_"
                + text(field.get("field_name")),
            )
            for outcome in _records(configuration.get("outcomes"))
            for field in _records(outcome.get("fields"))
        ),
    )
    expected = {
        "feature_schema_version": configuration.get("feature_schema_version"),
        "outcome_schema_version": configuration.get("outcome_schema_version"),
        "fields": [field.to_primitive() for field in fields],
    }
    if schema != expected or len({field.name for field in fields}) != len(fields):
        raise ManifestError("feature schema differs from its configured definitions")
    return fields


def feature_prediction_studies(manifest: PrimitiveMapping) -> PrimitiveMapping:
    """Validate contributing-study references without loading feature rows."""
    configuration = mapping(manifest.get("configuration"))
    namespaces = [
        text(outcome.get("namespace"))
        for outcome in _records(configuration.get("outcomes"))
    ]
    study_ids = manifest.get("prediction_study_ids")
    if not isinstance(study_ids, list) or len(study_ids) != len(namespaces):
        raise ManifestError("feature prediction-study references are incomplete")
    return {
        name: digest(value) for name, value in zip(namespaces, study_ids, strict=True)
    }


def validate_feature_row_provenance(
    manifest: PrimitiveMapping, result: PrimitiveMapping, rows: list[PrimitiveMapping]
) -> None:
    fields = validate_feature_schema(manifest, mapping(result.get("schema")))
    configuration = mapping(manifest.get("configuration"))
    market = mapping(manifest.get("market_data"))
    indexes = recorded_session_indexes(market)
    strategy = mapping(
        mapping(configuration.get("prediction_study_template")).get("strategy")
    )
    expected_studies = feature_prediction_studies(manifest)
    expected: PrimitiveMapping = {
        "study_id": manifest.get("dataset_id"),
        "source_dataset_id": market.get("dataset_id"),
        "dataset_fingerprint": market.get("bars_fingerprint"),
        "candidate_rule_id": strategy.get("component_name"),
        "candidate_rule_configuration_id": configuration_identity(strategy),
        "prediction_study_ids": expected_studies,
        **{
            name: market.get(name)
            for name in (
                "symbol",
                "provider_name",
                "adjustment_mode",
                "ohlc_basis",
                "volume_basis",
            )
        },
        **{
            name: configuration.get(name)
            for name in ("feature_schema_version", "outcome_schema_version")
        },
    }
    if "parameters" in strategy:
        expected["strategy_parameters"] = strategy["parameters"]
    source_rule = strategy.get("source_prediction_rule")
    if source_rule is not None:
        source_rule = mapping(source_rule)
        expected.update(
            strategy_id=source_rule.get("component_name"),
            strategy_implementation_version=source_rule.get("implementation_version"),
            strategy_configuration_id=configuration_identity(source_rule),
        )
    candidates: set[str] = set()
    sessions: list[str] = []
    for row in rows:
        if set(row) != {field.name for field in fields} or any(
            not _schema_value_matches(field, row[field.name]) for field in fields
        ):
            raise ManifestError("feature row differs from its schema")
        if configuration_identity(
            {key: row.get(key) for key in expected}
        ) != configuration_identity(expected):
            raise ManifestError("feature row provenance differs from its dataset")
        session = session_text(row.get("signal_session"))
        if session not in indexes:
            raise ManifestError("feature row session is outside its source dataset")
        sessions.append(session)
        parameters = mapping(row.get("strategy_parameters"))
        if row.get("strategy_parameters_id") != configuration_identity(parameters):
            raise ManifestError("feature parameter identity is inconsistent")
        candidate_id = configuration_identity(
            {
                "record_type": "signal_feature_candidate",
                "candidate_rule_configuration_id": row[
                    "candidate_rule_configuration_id"
                ],
                "dataset_fingerprint": row["dataset_fingerprint"],
                "parameters": parameters,
                "signal_session": session,
                "source_rule_configuration_id": digest(
                    row.get("strategy_configuration_id")
                ),
                "source_rule_id": text(row.get("strategy_id")),
                "source_rule_implementation_version": text(
                    row.get("strategy_implementation_version")
                ),
                "symbol": row["symbol"],
            }
        )
        if row.get("candidate_id") != candidate_id or row.get("row_id") != _row_id(
            text(manifest.get("dataset_id")), row
        ):
            raise ManifestError("feature candidate or row identity is inconsistent")
        if candidate_id in candidates:
            raise ManifestError("duplicate feature candidate identity")
        candidates.add(candidate_id)
        reasons = [
            text(reason)
            for reason in cast(list[object], row["disposition_reason_codes"])
        ]
        matched = [
            text(reason) for reason in cast(list[object], row["matched_rule_reasons"])
        ]
        selected = row["selected_rule_reason"]
        if (
            not reasons
            or len(reasons) != len(set(reasons))
            or row["direction"] not in (None, "up", "down")
            or (
                row["signal_disposition"] == "accepted"
                and (row["direction"] is None or selected is None)
            )
            or (selected is not None and (not matched or matched[0] != selected))
        ):
            raise ManifestError("feature row disposition evidence is inconsistent")
    if sessions != sorted(set(sessions)):
        raise ManifestError("feature rows must have ordered unique sessions")
