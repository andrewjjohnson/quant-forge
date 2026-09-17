"""Immutable presentation snapshots with optional read-only holdout authority."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from typing import TYPE_CHECKING, cast

from quantforge.configuration import PrimitiveMapping, PrimitiveMappingSnapshot
from quantforge.experiments import ArtifactEntry

if TYPE_CHECKING:
    from quantforge.oos import HoldoutLedger, OOSSource

REPORT_VERSION = "1"


class ResearchReportError(ValueError):
    """Unsafe, inconsistent, or incompatible report inputs/output."""


class ReportPhase(StrEnum):
    PROVENANCE = "PROVENANCE"
    IN_SAMPLE = "IN-SAMPLE"
    SELECTION = "VALIDATION / SELECTION"
    OOS = "WALK-FORWARD OOS"
    HOLDOUT = "FINAL HOLDOUT"


@dataclass(frozen=True, slots=True)
class ResearchReportConfig:
    """Thresholds are opt-in; preview limits never change research values."""

    maximum_preview_rows: int = 20
    minimum_sample_size: int | None = None
    high_trial_count: int | None = None
    maximum_configuration_change_frequency: Decimal | None = None

    def __post_init__(self) -> None:
        for number in (
            self.maximum_preview_rows,
            self.minimum_sample_size,
            self.high_trial_count,
        ):
            if number is not None and (type(number) is not int or number < 1):
                raise ResearchReportError("report counts must be positive integers")
        frequency = self.maximum_configuration_change_frequency
        if frequency is not None and (
            not isinstance(cast(object, frequency), Decimal)
            or not frequency.is_finite()
            or not 0 <= frequency <= 1
        ):
            raise ResearchReportError("turnover threshold must be a Decimal in [0, 1]")

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "maximum_preview_rows": self.maximum_preview_rows,
            "minimum_sample_size": self.minimum_sample_size,
            "high_trial_count": self.high_trial_count,
            "maximum_configuration_change_frequency": (
                None
                if self.maximum_configuration_change_frequency is None
                else str(self.maximum_configuration_change_frequency)
            ),
        }


@dataclass(frozen=True, slots=True)
class ReportArtifact:
    entry: ArtifactEntry
    status: str
    # The wrapper permits JSON pointers to arrays/scalars as well as objects.
    content: PrimitiveMappingSnapshot | None = None


@dataclass(frozen=True, slots=True)
class ReportSection:
    title: str
    phase: ReportPhase
    content: PrimitiveMappingSnapshot
    artifact_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ResearchWarning:
    code: str
    message: str
    source: str


@dataclass(frozen=True, slots=True)
class ResearchReport:
    """A render-time snapshot, including the observed holdout authority state."""

    report_id: str
    manifest_id: str
    manifest_path: str
    artifact_root: str
    sections: tuple[ReportSection, ...]
    warnings: tuple[ResearchWarning, ...]
    artifacts: tuple[ReportArtifact, ...]
    config: ResearchReportConfig
    header: PrimitiveMappingSnapshot
    # Authority handles are excluded from presentation identity and equality.
    # Export rechecks them so delayed publication cannot assert stale unseen state.
    holdout_source: OOSSource | None = field(default=None, repr=False, compare=False)
    holdout_ledger: HoldoutLedger | None = field(
        default=None, repr=False, compare=False
    )
