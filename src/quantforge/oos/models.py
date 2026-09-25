"""Typed QF-40 outputs; prediction and trading results remain separate."""

from dataclasses import dataclass, field
from decimal import Decimal

from quantforge.configuration import (
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
    decimal_to_primitive,
)
from quantforge.prediction.window_reader import PredictionWindowReader
from quantforge.validation import ValidationPlan
from quantforge.walk_forward.models import FoldResult


def optional_decimal(value: Decimal | None) -> str | None:
    return None if value is None else decimal_to_primitive(value)


@dataclass(frozen=True, slots=True)
class MetricSummary:
    """Explicit denominator and missingness; all numeric units are ratios."""

    sample_count: int
    unavailable_count: int
    mean: Decimal | None
    median: Decimal | None = None
    minimum: Decimal | None = None
    maximum: Decimal | None = None

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "status": "unavailable"
            if not self.sample_count
            else "partial"
            if self.unavailable_count
            else "available",
            "sample_count": self.sample_count,
            "unavailable_count": self.unavailable_count,
            "mean": optional_decimal(self.mean),
            "median": optional_decimal(self.median),
            "minimum": optional_decimal(self.minimum),
            "maximum": optional_decimal(self.maximum),
        }


@dataclass(frozen=True, slots=True)
class ConfigurationStabilitySummary:
    windows: tuple[PrimitiveMappingSnapshot, ...]
    comparable_transitions: int
    configuration_changes: int
    repeat_selections: int
    frequencies: PrimitiveMappingSnapshot
    parameter_changes: PrimitiveMappingSnapshot

    def to_primitive(self) -> PrimitiveMapping:
        from quantforge.backtesting._arithmetic import arithmetic

        with arithmetic():
            changes = (
                Decimal(self.configuration_changes) / self.comparable_transitions
                if self.comparable_transitions
                else None
            )
            repeats = (
                Decimal(self.repeat_selections) / self.comparable_transitions
                if self.comparable_transitions
                else None
            )
        return {
            "windows": [window.to_primitive() for window in self.windows],
            "comparable_transitions": self.comparable_transitions,
            "configuration_changes": self.configuration_changes,
            "configuration_change_frequency": optional_decimal(changes),
            "repeat_selections": self.repeat_selections,
            "repeat_selection_frequency": optional_decimal(repeats),
            "selection_counts": self.frequencies.to_primitive(),
            "parameter_change_counts": self.parameter_changes.to_primitive(),
            "interpretation": "descriptive; no automatic quality judgement",
        }


@dataclass(frozen=True, slots=True)
class OOSSource:
    """Verified QF-39 snapshots in QF-8 order, including incomplete folds."""

    plan: ValidationPlan
    study_id: str
    definition: PrimitiveMappingSnapshot
    lineage: PrimitiveMappingSnapshot
    folds: tuple[FoldResult, ...]
    references: tuple[PrimitiveMappingSnapshot, ...]
    prediction_windows: tuple[PredictionWindowReader | None, ...] = field(
        default=(), compare=False, repr=False
    )

    @property
    def lineage_id(self) -> str:
        return configuration_identity(self.lineage.to_primitive())


@dataclass(frozen=True, slots=True)
class PredictionOOSAggregate:
    summary: PrimitiveMappingSnapshot
    stability: ConfigurationStabilitySummary
    observations: tuple[PrimitiveMappingSnapshot, ...]
    provenance: PrimitiveMappingSnapshot
    window_sources: tuple[PrimitiveMappingSnapshot, ...] | None = None

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "schema_version": "1" if self.window_sources is None else "2",
            "kind": "prediction_oos_aggregate",
            "summary": self.summary.to_primitive(),
            "stability": self.stability.to_primitive(),
            **(
                {"observations": [item.to_primitive() for item in self.observations]}
                if self.window_sources is None
                else {
                    "window_sources": [
                        item.to_primitive() for item in self.window_sources
                    ]
                }
            ),
            "provenance": self.provenance.to_primitive(),
        }

    @property
    def aggregate_id(self) -> str:
        return configuration_identity(self.to_primitive())


@dataclass(frozen=True, slots=True)
class BacktestOOSAggregate:
    summary: PrimitiveMappingSnapshot
    stability: ConfigurationStabilitySummary
    normalized_equity: tuple[PrimitiveMappingSnapshot, ...]
    native_windows: tuple[PrimitiveMappingSnapshot, ...]
    provenance: PrimitiveMappingSnapshot

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "schema_version": "1",
            "kind": "backtest_oos_aggregate",
            "summary": self.summary.to_primitive(),
            "stability": self.stability.to_primitive(),
            "normalized_equity": [
                item.to_primitive() for item in self.normalized_equity
            ],
            "native_windows": [item.to_primitive() for item in self.native_windows],
            "provenance": self.provenance.to_primitive(),
        }

    @property
    def aggregate_id(self) -> str:
        return configuration_identity(self.to_primitive())
