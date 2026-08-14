"""Backend-neutral contracts for standard indicator computation."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol, cast

from quantforge.configuration import PrimitiveMapping, PrimitiveMappingSnapshot
from quantforge.indicators.base import IndicatorBar
from quantforge.indicators.exceptions import (
    InvalidIndicatorBackendError,
    MisalignedIndicatorOutputError,
    UnsupportedIndicatorBackendError,
)
from quantforge.indicators.models import IndicatorFieldOutput, MarketField

INDICATOR_BACKEND_CONTRACT_VERSION = "1"
NATIVE_INDICATOR_BACKEND = "native_v1"
TALIB_INDICATOR_BACKEND = "talib_v1"


@dataclass(frozen=True, slots=True)
class StandardIndicatorDefinition:
    """Canonical standard-indicator name, inputs, parameters, and outputs."""

    name: str
    parameters: PrimitiveMappingSnapshot
    input_fields: tuple[MarketField, ...]
    output_fields: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise InvalidIndicatorBackendError(
                "standard indicator name must be non-empty"
            )
        if (
            not self.input_fields
            or len(self.input_fields) != len(set(self.input_fields))
            or not self.output_fields
            or any(not name for name in self.output_fields)
            or len(self.output_fields) != len(set(self.output_fields))
        ):
            raise InvalidIndicatorBackendError(
                "standard indicator inputs and outputs must be non-empty and unique"
            )

    def to_primitive(self) -> PrimitiveMapping:
        """Return the normalized backend-neutral definition."""
        return {
            "name": self.name,
            "parameters": self.parameters.to_primitive(),
            "input_fields": [field.value for field in self.input_fields],
            "output_fields": list(self.output_fields),
        }


@dataclass(frozen=True, slots=True)
class IndicatorBackendIdentity:
    """Stable backend and exact library identity for deterministic studies."""

    backend_id: str
    library_name: str
    library_version: str
    function_name: str
    contract_version: str = INDICATOR_BACKEND_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if not all(
            (
                self.backend_id,
                self.library_name,
                self.library_version,
                self.function_name,
                self.contract_version,
            )
        ):
            raise InvalidIndicatorBackendError(
                "indicator backend identity fields must be non-empty"
            )

    def to_primitive(self) -> PrimitiveMapping:
        """Return the complete deterministic backend identity."""
        return {
            "backend_id": self.backend_id,
            "contract_version": self.contract_version,
            "library_name": self.library_name,
            "library_version": self.library_version,
            "function_name": self.function_name,
        }


@dataclass(frozen=True, slots=True)
class IndicatorComputationRequest:
    """Canonical bars plus one backend-neutral standard indicator definition."""

    definition: StandardIndicatorDefinition
    bars: tuple[IndicatorBar, ...]


@dataclass(frozen=True, slots=True)
class IndicatorComputationResult:
    """Backend output normalized to QuantForge names, values, and metadata."""

    definition_name: str
    backend_identity: IndicatorBackendIdentity
    normalized_parameters: PrimitiveMappingSnapshot
    normalized_input_fields: tuple[MarketField, ...]
    fields: tuple[IndicatorFieldOutput, ...]
    observation_count: int

    def __post_init__(self) -> None:
        field_names = tuple(field.name for field in self.fields)
        if not self.definition_name:
            raise MisalignedIndicatorOutputError(
                "indicator backend definition name must be non-empty"
            )
        if isinstance(self.observation_count, bool) or self.observation_count < 0:
            raise MisalignedIndicatorOutputError(
                "indicator backend observation count cannot be negative"
            )
        if (
            not self.normalized_input_fields
            or len(self.normalized_input_fields)
            != len(set(self.normalized_input_fields))
            or not field_names
            or any(not name for name in field_names)
            or len(field_names) != len(set(field_names))
            or any(len(field.values) != self.observation_count for field in self.fields)
        ):
            raise MisalignedIndicatorOutputError(
                "indicator backend output must align with every canonical input bar"
            )
        if any(
            value is not None and not value.is_finite()
            for field in self.fields
            for value in field.values
        ):
            raise MisalignedIndicatorOutputError(
                "indicator backend values must be finite decimals or None"
            )

    def metadata(self) -> PrimitiveMapping:
        """Serialize normalized request and backend provenance without value rows."""
        return {
            "definition_name": self.definition_name,
            "backend": self.backend_identity.to_primitive(),
            "normalized_parameters": self.normalized_parameters.to_primitive(),
            "normalized_input_fields": [
                field.value for field in self.normalized_input_fields
            ],
            "normalized_output_fields": [field.name for field in self.fields],
            "observation_count": self.observation_count,
        }


class IndicatorBackend(Protocol):
    """Adapter that translates one standard definition to a calculation library."""

    @property
    def backend_id(self) -> str: ...

    def identity_for(
        self, definition: StandardIndicatorDefinition
    ) -> IndicatorBackendIdentity: ...

    def compute(
        self, request: IndicatorComputationRequest
    ) -> IndicatorComputationResult: ...


class IndicatorBackendRegistry:
    """Explicit stable-id resolver for standard-indicator backends."""

    def __init__(self, backends: Iterable[IndicatorBackend]) -> None:
        resolved: dict[str, IndicatorBackend] = {}
        for backend in backends:
            backend_id = cast(object, backend.backend_id)
            if not isinstance(backend_id, str) or not backend_id:
                raise InvalidIndicatorBackendError(
                    "indicator backend id must be a non-empty string"
                )
            if backend_id in resolved:
                raise InvalidIndicatorBackendError(
                    f"duplicate indicator backend id: {backend_id}"
                )
            resolved[backend_id] = backend
        if not resolved:
            raise InvalidIndicatorBackendError(
                "indicator backend registry must not be empty"
            )
        self._backends = resolved

    @property
    def backend_ids(self) -> tuple[str, ...]:
        """Return stable backend identifiers in deterministic order."""
        return tuple(sorted(self._backends))

    def resolve(self, backend_id: str) -> IndicatorBackend:
        """Resolve one configured backend or fail with a domain error."""
        try:
            return self._backends[backend_id]
        except KeyError as error:
            raise UnsupportedIndicatorBackendError(
                f"unsupported indicator backend: {backend_id}"
            ) from error


__all__ = [
    "INDICATOR_BACKEND_CONTRACT_VERSION",
    "NATIVE_INDICATOR_BACKEND",
    "TALIB_INDICATOR_BACKEND",
    "IndicatorBackend",
    "IndicatorBackendIdentity",
    "IndicatorBackendRegistry",
    "IndicatorComputationRequest",
    "IndicatorComputationResult",
    "StandardIndicatorDefinition",
]
