"""QF-7/QF-29 consistency checks over existing feature records."""

from collections import Counter
from typing import cast

from quantforge.configuration import PrimitiveMapping
from quantforge.experiments._json import ManifestError, mapping, text
from quantforge.prediction.signal_feature_models import SignalDisposition


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
    declarations = [mapping(manifest.get("record_counts"))]
    if "summary" in result:
        declarations.append(mapping(result["summary"]))
    if any(
        type(counts.get(key)) is not int or cast(int, counts[key]) != count
        for counts in declarations
        for key, count in expected.items()
    ):
        raise ManifestError("feature record counts are inconsistent with result rows")
