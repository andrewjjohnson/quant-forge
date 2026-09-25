"""Canonical JSON/SHA-256 primitives for normalized historical windows."""

import hashlib
import json
from collections.abc import Iterable
from copy import copy
from typing import cast

from quantforge.configuration import (
    Primitive,
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
)
from quantforge.prediction.errors import InvalidPredictionOutputError


def mapping(value: object) -> PrimitiveMapping:
    if not isinstance(value, dict):
        raise InvalidPredictionOutputError("window record must be an object")
    return cast(PrimitiveMapping, value)


def canonical(record: PrimitiveMapping) -> bytes:
    return PrimitiveMappingSnapshot.capture(record).canonical_json.encode("utf-8")


def _pairs(pairs: list[tuple[str, Primitive]]) -> PrimitiveMapping:
    result: PrimitiveMapping = {}
    for key, value in pairs:
        if key in result:
            raise InvalidPredictionOutputError("duplicate window JSON key")
        result[key] = value
    return result


def _constant(value: str) -> Primitive:
    raise InvalidPredictionOutputError(f"non-finite window JSON value: {value}")


def decode(encoded: bytes, *, canonical_line: bool = False) -> PrimitiveMapping:
    try:
        record = mapping(
            json.loads(encoded, object_pairs_hook=_pairs, parse_constant=_constant)
        )
        if canonical_line and encoded != canonical(record) + b"\n":
            raise InvalidPredictionOutputError("noncanonical or truncated window line")
        return record
    except (ValueError, TypeError, UnicodeError) as error:
        raise InvalidPredictionOutputError(f"invalid window JSON: {error}") from error


def ordered_result_identity(
    window_id: str, decisions: Iterable[PrimitiveMapping]
) -> str:
    """Exactly configuration_identity({decisions: [...], window_id: ...}).

    Encode the canonical array one record at a time, without a list or an
    expanded market/configuration copy. This is ordinary canonical JSON hashing,
    not a new hash chain or an alternative identity algorithm.
    """
    digest = WindowResultIdentity(window_id)
    for decision in decisions:
        digest.update(decision)
    return digest.hexdigest()


class WindowResultIdentity:
    """Constant-state hash of the canonical ordered normalized collection."""

    def __init__(self, window_id: str) -> None:
        self._digest = hashlib.sha256(b'{"decisions":[')
        self._separator = b""
        self._suffix = b"]," + canonical({"window_id": window_id})[1:]

    def update(self, decision: PrimitiveMapping) -> None:
        self._digest.update(self._separator)
        self._digest.update(canonical(decision))
        self._separator = b","

    def copy(self) -> "WindowResultIdentity":
        """Copy constant hash state for transactional persistence, unchanged bytes."""
        duplicate = copy(self)
        duplicate._digest = self._digest.copy()
        return duplicate

    def hexdigest(self) -> str:
        digest = self._digest.copy()
        digest.update(self._suffix)
        return digest.hexdigest()


class StudyIdentity:
    """Reuse the canonical QF-11 SHA-256 prefix for immutable market evidence.

    Existing study identities place market_data before prediction_context in
    sorted JSON. Copying that hash state authenticates original QF-11 identities
    without encoding or hashing the shared market record for every decision.
    """

    def __init__(self, identity: PrimitiveMapping) -> None:
        prefix = canonical(
            {
                "component": "quantforge_prediction_study",
                "engine_version": identity["prediction_engine_version"],
                "market_data": identity["market_data"],
            }
        )
        self._prefix = hashlib.sha256(prefix[:-1] + b',"prediction_context":')
        self._suffix = (
            b',"study_configuration":'
            + canonical(mapping(identity["configuration"]))
            + b"}"
        )

    def for_context(self, context: PrimitiveMapping) -> str:
        digest = self._prefix.copy()
        digest.update(canonical(context))
        digest.update(self._suffix)
        return digest.hexdigest()
