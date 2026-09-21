"""QF-7/QF-29 consistency checks over existing feature records."""

from collections import Counter
from typing import cast

from quantforge.configuration import PrimitiveMapping, configuration_identity
from quantforge.experiments._feature_row_integrity import (
    validate_feature_row_provenance,
)
from quantforge.experiments._json import ManifestError, mapping, text
from quantforge.experiments._prediction_input_integrity import (
    validate_prediction_input_sources,
)
from quantforge.prediction.signal_feature_models import SignalDisposition


def validate_feature_manifest(manifest: PrimitiveMapping) -> None:
    """Require the complete producer envelope, including unhashed declarations."""
    if set(manifest) != {
        "component",
        "configuration",
        "dataset_id",
        "engine_version",
        "feature_outcome_boundary",
        "limitations",
        "market_data",
        "prediction_study_ids",
        "record_counts",
        "status",
    }:
        raise ManifestError("feature manifest fields differ from producer schema")
    if (
        manifest["component"] != "quantforge_signal_feature_dataset"
        or manifest["status"] != "complete"
    ):
        raise ManifestError("expected a completed QF-7/QF-29 dataset")
    if (
        configuration_identity(mapping(manifest["configuration"]))
        != manifest["dataset_id"]
    ):
        raise ManifestError("feature dataset identity is inconsistent")
    text(manifest["engine_version"])
    if manifest["feature_outcome_boundary"] != (
        "candidate dispositions and causal features are fixed before any "
        "QF-11 outcome labeler is invoked"
    ):
        raise ManifestError("feature dataset feature/outcome boundary is invalid")
    limitations = manifest["limitations"]
    if not isinstance(limitations, list) or any(
        not isinstance(item, str) for item in limitations
    ):
        raise ManifestError("feature dataset limitations must be an array of strings")
    configuration = mapping(manifest["configuration"])
    outcome_sources = [configuration.get("outcome_source")]
    outcomes = configuration.get("outcomes")
    if not isinstance(outcomes, list):
        raise ManifestError("feature outcome configurations must be an array")
    for outcome in outcomes:
        component = mapping(outcome).get("component_configuration")
        if not isinstance(component, dict):
            raise ManifestError("feature outcome configuration must be an object")
        outcome_sources.append(component.get("outcome_source"))
    validate_prediction_input_sources(
        mapping(manifest["market_data"]),
        outcome_sources=outcome_sources,
        context=configuration.get("prediction_context"),
    )


def validate_feature_summary(
    manifest: PrimitiveMapping, summary: PrimitiveMapping
) -> None:
    """Reconcile producer count declarations without reading CSV or Parquet rows."""
    counts = mapping(manifest.get("record_counts"))
    dispositions = {f"{item.value}_count" for item in SignalDisposition}
    for record in (counts, summary):
        if (
            set(record) != {"candidate_count", *dispositions}
            or any(type(count) is not int or count < 0 for count in record.values())
            or record["candidate_count"]
            != sum(cast(int, record[key]) for key in dispositions)
        ):
            raise ManifestError("feature record counts are invalid or inconsistent")
    if configuration_identity(counts) != configuration_identity(summary):
        raise ManifestError("feature record counts differ between summary and manifest")


def validate_feature_rows(manifest: PrimitiveMapping, result: PrimitiveMapping) -> None:
    """Reconcile candidate/disposition counts without generating features or labels."""
    rows = result.get("rows")
    if not isinstance(rows, list):
        raise ManifestError("feature result rows must be an array of records")
    dispositions = Counter(text(mapping(row).get("signal_disposition")) for row in rows)
    if set(dispositions) - {item.value for item in SignalDisposition}:
        raise ManifestError("feature row has an invalid signal disposition")
    expected = {
        "candidate_count": len(rows),
        **{
            f"{item.value}_count": dispositions[item.value]
            for item in SignalDisposition
        },
    }
    if "summary" in result:
        validate_feature_summary(manifest, mapping(result["summary"]))
    counts = mapping(manifest.get("record_counts"))
    if any(
        type(counts.get(key)) is not int or cast(int, counts[key]) != count
        for key, count in expected.items()
    ):
        raise ManifestError("feature record counts are inconsistent with result rows")
    validate_feature_row_provenance(manifest, result, [mapping(row) for row in rows])
