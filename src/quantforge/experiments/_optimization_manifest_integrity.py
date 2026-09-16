"""Validate QF-6's saved operational metadata without constructing a study."""

from math import prod

from quantforge.configuration import PrimitiveMapping
from quantforge.experiments._aggregate_schema import counter, record, records
from quantforge.experiments._json import ManifestError, mapping, text


def validate_optimization_manifest(manifest: PrimitiveMapping) -> None:
    """Require complete execution, persistence and grid-safeguard declarations."""
    record(
        manifest,
        {
            "study_id",
            "study_schema_version",
            "identity_inputs",
            "execution_configuration",
            "persistence_configuration",
            "scale_safeguard",
            "combination_counts",
            "operational_timestamp_policy",
            "warnings",
            "limitations",
        },
        "optimization manifest",
    )
    execution = record(
        manifest["execution_configuration"],
        {
            "mode",
            "maximum_workers",
            "parallelism",
            "retry_failed",
            "stale_running_policy",
            "fail_fast",
        },
        "optimization execution",
    )
    parallelism = {
        "sequential": "none",
        "process": "bounded_standard_library_process_pool",
    }
    mode = text(execution["mode"])
    if (
        mode not in parallelism
        or execution["parallelism"] != parallelism[mode]
        or counter(execution["maximum_workers"]) == 0
        or type(execution["retry_failed"]) is not bool
        or type(execution["fail_fast"]) is not bool
        or execution["stale_running_policy"] != "retry"
    ):
        raise ManifestError("optimization execution configuration is invalid")
    persistence = record(
        manifest["persistence_configuration"],
        {
            "store",
            "output_root",
            "successful_trial_overwrite",
        },
        "optimization persistence",
    )
    text(persistence["output_root"])
    if (
        persistence["store"] != "atomic_local_json_files"
        or persistence["successful_trial_overwrite"] != "forbidden"
    ):
        raise ManifestError("optimization persistence configuration is invalid")
    safeguard = record(
        manifest["scale_safeguard"],
        {
            "maximum_combinations",
            "allow_large_grid",
            "count_expression",
        },
        "optimization scale safeguard",
    )
    maximum = counter(safeguard["maximum_combinations"])
    counts = record(
        manifest["combination_counts"],
        {
            "total_cartesian",
            "valid",
            "excluded",
        },
        "optimization combination counts",
    )
    if any(type(value) is not int or value < 0 for value in counts.values()):
        raise ManifestError("Cartesian manifest counts are invalid")
    axes = records(
        mapping(mapping(manifest["identity_inputs"])["search_space"])["parameters"]
    )
    sizes: list[int] = []
    for axis in axes:
        values = axis.get("values")
        if not isinstance(values, list) or not values:
            raise ManifestError("optimization search axis values are invalid")
        sizes.append(len(values))
    total = prod(sizes)
    expression = " x ".join(str(size) for size in sizes) + f" = {total:,}"
    if (
        not sizes
        or maximum == 0
        or type(safeguard["allow_large_grid"]) is not bool
        or (total > maximum and not safeguard["allow_large_grid"])
        or safeguard["count_expression"] != expression
        or counter(counts["total_cartesian"]) != total
        or counter(counts["valid"]) + counter(counts["excluded"]) != total
    ):
        raise ManifestError(
            "Cartesian manifest counts or optimization scale safeguard are inconsistent"
        )
    if manifest["operational_timestamp_policy"] != (
        "trial timestamps are diagnostic and excluded from deterministic identities"
    ):
        raise ManifestError("optimization timestamp policy is invalid")
    for name in ("warnings", "limitations"):
        disclosures = manifest[name]
        if not isinstance(disclosures, list) or any(
            not isinstance(item, str) for item in disclosures
        ):
            raise ManifestError("optimization disclosures must be arrays of strings")
