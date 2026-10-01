"""Optional lightweight JSON export of one rapid scan (QF-72). NON-AUTHORITATIVE.

Exports use the reserved ``.rapid.json`` suffix and declare
``"authoritative": false``; QF-9 refuses both, so an export can never be indexed,
reported or published as research evidence. Exports are also refused inside a
holdout ledger or an authoritative experiment/study directory.
"""

import json
import os
from pathlib import Path

from quantforge.experiments.artifacts import NON_AUTHORITATIVE_SUFFIX
from quantforge.rapid.models import RapidScanError, RapidScanResult

RAPID_EXPORT_SUFFIX = NON_AUTHORITATIVE_SUFFIX
# Markers of authoritative research stores: the holdout ledger, QF-9/QF-32/QF-39
# study manifests and the QF-45 execution root.
_AUTHORITATIVE_MARKERS = (
    "store.json",
    "manifest.json",
    "execution.json",
    "pre-holdout-complete.json",
)


def _refuse_authoritative_location(directory: Path) -> None:
    for ancestor in (directory, *directory.parents):
        if ancestor.name.lower() == "holdout-ledger" or any(
            (ancestor / marker).is_file() for marker in _AUTHORITATIVE_MARKERS
        ):
            raise RapidScanError(
                "rapid exports cannot be written inside a holdout ledger or an "
                f"authoritative study directory: {ancestor}"
            )
        if (ancestor / ".git").exists():
            return


def export_rapid_scan(result: RapidScanResult, path: Path) -> Path:
    """Atomically write one exploratory scan as ``*.rapid.json``."""
    if type(result) is not RapidScanResult:
        raise RapidScanError("only rapid scan results can be exported")
    destination = Path(path).resolve()
    if not destination.name.lower().endswith(RAPID_EXPORT_SUFFIX):
        raise RapidScanError(
            f"rapid exports must use the {RAPID_EXPORT_SUFFIX!r} suffix"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    _refuse_authoritative_location(destination.parent)
    payload = json.dumps(result.to_primitive(), indent=2, sort_keys=True) + "\n"
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, destination)
    return destination
