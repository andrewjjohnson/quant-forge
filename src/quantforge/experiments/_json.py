"""Strict, credential-free primitive snapshots for experiment metadata."""

import json
import re
from pathlib import Path
from typing import cast
from urllib.parse import urlsplit

from quantforge.configuration import (
    Primitive,
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
)


class ManifestError(ValueError):
    """Invalid, incompatible, unsafe, or corrupt experiment metadata."""


def safe_metadata(value: Primitive) -> None:
    """Reject credential-bearing metadata, never echoing its keys or values.

    Only explicit primitive research snapshots are accepted, never provider
    objects, environment dictionaries, or object __dict__ representations.
    This is a defensive credential-field guard, not arbitrary secret discovery.
    """
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = re.sub(r"[^a-z0-9]", "", key.lower())
            # QF-5 uses these fixed labels for simulated corporate-action
            # accounting. No other account fields or identifiers are safe.
            simulation_account = (
                key == "account_id"
                and isinstance(item, str)
                and item in {"benchmark", "strategy"}
            )
            if any(
                word in normalized
                for word in (
                    "apikey",
                    "token",
                    "password",
                    "passwd",
                    "secret",
                    "credential",
                    "authorization",
                    "privatekey",
                    "connectionstring",
                )
            ) or (
                re.search(
                    r"account(?:id|number|num|no|name|holder|code|ref)|accounts?$",
                    normalized,
                )
                and not simulation_account
            ):
                raise ManifestError("credential-bearing metadata is prohibited")
            safe_metadata(item)
    elif isinstance(value, list):
        for item in value:
            safe_metadata(item)
    elif isinstance(value, str):
        if re.search(
            r"(?i)(\bbearer\s+|-----BEGIN .*PRIVATE KEY|"
            r"(?:api[_-]?key|token|password)=)",
            value,
        ):
            raise ManifestError("credential-bearing metadata is prohibited")
        if "://" in value:
            try:
                parsed = urlsplit(value)
                if parsed.username or parsed.password or parsed.query:
                    raise ManifestError(
                        "authenticated or query-bearing URLs are prohibited"
                    )
            except ValueError as error:
                raise ManifestError("unsafe metadata URL") from error


def snapshot(payload: PrimitiveMapping) -> PrimitiveMappingSnapshot:
    safe_metadata(payload)
    try:
        return PrimitiveMappingSnapshot.capture(payload)
    except (TypeError, ValueError) as error:
        raise ManifestError("metadata must contain finite JSON primitives") from error


def mapping(value: object) -> PrimitiveMapping:
    if not isinstance(value, dict):
        raise ManifestError("expected a metadata object")
    return cast(PrimitiveMapping, value)


def text(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError("expected a nonempty metadata string")
    return value


def digest(value: object) -> str:
    result = text(value)
    if re.fullmatch(r"[0-9a-f]{64}", result) is None:
        raise ManifestError("expected a lowercase SHA-256 digest")
    return result


def _pairs(pairs: list[tuple[str, Primitive]]) -> PrimitiveMapping:
    result: PrimitiveMapping = {}
    for key, value in pairs:
        if key in result:
            raise ManifestError("duplicate JSON object key")
        result[key] = value
    return result


def read_json(path: Path) -> PrimitiveMapping:
    try:
        result = mapping(json.loads(path.read_bytes(), object_pairs_hook=_pairs))
        snapshot(result)
        return result
    except (OSError, ValueError, TypeError) as error:
        raise ManifestError("invalid, unsafe, or unreadable JSON artifact") from error


def pointer(document: PrimitiveMapping, location: str) -> Primitive:
    """Read an RFC 6901 JSON pointer; it never invokes a producer serializer."""
    result: Primitive = document
    if location and not location.startswith("/"):
        raise ManifestError("JSON pointer must be empty or start with slash")
    try:
        for part in location.split("/")[1:] if location else ():
            if re.search(r"~(?![01])", part):
                raise ManifestError("invalid JSON pointer escape")
            key = part.replace("~1", "/").replace("~0", "~")
            if isinstance(result, dict):
                result = result[key]
            elif isinstance(result, list) and re.fullmatch(r"0|[1-9][0-9]*", key):
                result = result[int(key)]
            else:
                raise ManifestError("JSON pointer does not resolve")
        return result
    except (KeyError, IndexError) as error:
        raise ManifestError("JSON pointer does not resolve") from error


EMPTY_SNAPSHOT = PrimitiveMappingSnapshot("{}")
