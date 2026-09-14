"""QF-39-local atomic records, following the existing file-study conventions."""

import json
import os
import tempfile
from pathlib import Path
from typing import cast

from quantforge.configuration import (
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
)
from quantforge.walk_forward.models import WalkForwardPersistenceError


def write_record(
    path: Path, payload: PrimitiveMapping, *, immutable: bool = False
) -> None:
    if immutable and path.exists():
        if read_record(path) != payload:
            raise WalkForwardPersistenceError("immutable walk-forward artifact differs")
        return
    envelope: PrimitiveMapping = {
        "payload": payload,
        "fingerprint": configuration_identity(payload),
    }
    temporary: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, filename = tempfile.mkstemp(
            prefix=f".{path.name}.", dir=path.parent
        )
        temporary = Path(filename)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(
                PrimitiveMappingSnapshot.capture(envelope).canonical_json + "\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError as error:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        raise WalkForwardPersistenceError(
            "failed to persist walk-forward state"
        ) from error


def read_record(path: Path) -> PrimitiveMapping:
    try:
        envelope = cast(PrimitiveMapping, json.loads(path.read_text(encoding="utf-8")))
        payload = envelope["payload"]
        if (
            not isinstance(payload, dict)
            or configuration_identity(payload) != envelope["fingerprint"]
        ):
            raise ValueError("fingerprint mismatch")
        return payload
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise WalkForwardPersistenceError(
            "invalid or corrupt walk-forward artifact"
        ) from error
