"""Immutable indicator fields and aligned output records."""

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from enum import StrEnum

from quantforge.configuration import PrimitiveMapping, decimal_to_primitive
from quantforge.indicators.exceptions import MisalignedIndicatorOutputError

type IndicatorValue = Decimal | None
_SESSION_DATE_FIELD = "session_date"


class MarketField(StrEnum):
    """Provider-independent fields in the QF-3 normalized daily-bar schema."""

    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"
    VOLUME = "volume"


@dataclass(frozen=True, slots=True)
class IndicatorFieldOutput:
    """One named indicator field aligned with every input session."""

    name: str
    values: tuple[IndicatorValue, ...]


@dataclass(frozen=True, slots=True)
class IndicatorOutput:
    """Aligned, immutable output from one configured indicator.

    ``None`` is the explicit unavailable value. It is used for warm-up rows and
    for full windows containing a missing/non-finite source observation.
    """

    indicator_name: str
    configuration_id: str
    session_dates: tuple[date, ...]
    fields: tuple[IndicatorFieldOutput, ...]

    def __post_init__(self) -> None:
        names = tuple(field.name for field in self.fields)
        if not self.indicator_name or not self.configuration_id:
            raise MisalignedIndicatorOutputError(
                "indicator name and configuration identity are required"
            )
        if (
            not names
            or any(not name for name in names)
            or len(names) != len(set(names))
        ):
            raise MisalignedIndicatorOutputError(
                "indicator output field names must be non-empty and unique"
            )
        if _SESSION_DATE_FIELD in names:
            raise MisalignedIndicatorOutputError(
                f"indicator output field name is reserved: {_SESSION_DATE_FIELD}"
            )
        if any(len(field.values) != len(self.session_dates) for field in self.fields):
            raise MisalignedIndicatorOutputError(
                "every indicator field must align with every input session"
            )
        if any(
            value is not None and not value.is_finite()
            for field in self.fields
            for value in field.values
        ):
            raise MisalignedIndicatorOutputError(
                "indicator output values must be finite decimals or None"
            )

    def values_for(self, field_name: str) -> tuple[IndicatorValue, ...]:
        """Return a named output series or raise a domain alignment error."""
        for field in self.fields:
            if field.name == field_name:
                return field.values
        raise MisalignedIndicatorOutputError(
            f"indicator output does not contain field: {field_name}"
        )

    def to_rows(self) -> list[PrimitiveMapping]:
        """Return a deterministic JSON-compatible row representation."""
        rows: list[PrimitiveMapping] = []
        for index, session_date in enumerate(self.session_dates):
            row: PrimitiveMapping = {_SESSION_DATE_FIELD: session_date.isoformat()}
            for field in self.fields:
                value = field.values[index]
                row[field.name] = None if value is None else decimal_to_primitive(value)
            rows.append(row)
        return rows
