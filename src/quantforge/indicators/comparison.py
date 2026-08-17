"""Deterministic, backend-neutral standard-indicator comparisons."""

import csv
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import date, datetime
from decimal import ROUND_HALF_EVEN, Context, Decimal, DecimalException, localcontext
from pathlib import Path
from typing import Protocol, cast

from quantforge.configuration import (
    Primitive,
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
    decimal_to_primitive,
)
from quantforge.data import dataset_identity_matches, validate_market_dataset
from quantforge.data.exceptions import ValidationError as MarketDataValidationError
from quantforge.data.models import MarketDataset
from quantforge.indicators.backends import (
    NATIVE_INDICATOR_BACKEND,
    TALIB_INDICATOR_BACKEND,
    IndicatorBackendIdentity,
    IndicatorBackendRegistry,
    IndicatorComputationRequest,
    IndicatorComputationResult,
    StandardIndicatorDefinition,
    default_indicator_backend_registry,
)
from quantforge.indicators.base import IndicatorBar
from quantforge.indicators.exceptions import IndicatorComparisonError
from quantforge.timeframes import (
    DEFAULT_US_EQUITY_TIMEFRAME,
    ExchangeSessionPolicy,
    SessionInterval,
    Timeframe,
)

INDICATOR_COMPARISON_ENGINE_VERSION = "1"
INDICATOR_COMPARISON_SCHEMA_VERSION = "1"
INDICATOR_COMPARISON_ARTIFACT_FILENAMES = (
    "comparison.json",
    "field_summary.csv",
    "divergences.csv",
    "summary.txt",
)
_ARITHMETIC_PRECISION = 34
_ARITHMETIC_CONTEXT = Context(prec=_ARITHMETIC_PRECISION, rounding=ROUND_HALF_EVEN)
type ComparisonTimestamp = date | datetime


class BackendNeutralStandardIndicator(Protocol):
    """Standard indicator exposing the QF-35 normalized definition."""

    @property
    def standard_definition(self) -> StandardIndicatorDefinition: ...


@dataclass(frozen=True, slots=True)
class IndicatorComparisonTolerances:
    """Symmetric absolute-plus-relative divergence tolerances."""

    absolute: Decimal = Decimal(0)
    relative: Decimal = Decimal(0)

    def __post_init__(self) -> None:
        for name, value in (("absolute", self.absolute), ("relative", self.relative)):
            if not value.is_finite() or value < 0:
                raise IndicatorComparisonError(
                    f"indicator comparison {name} tolerance must be a finite "
                    "non-negative Decimal"
                )

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "absolute": decimal_to_primitive(self.absolute),
            "relative": decimal_to_primitive(self.relative),
            "rule": (
                "absolute_difference > absolute + relative * "
                "max(abs(backend_a), abs(backend_b))"
            ),
        }


@dataclass(frozen=True, slots=True)
class IndicatorComparisonSource:
    """Canonical bars, alignment keys, dataset identity, and timeframe semantics."""

    bars: tuple[IndicatorBar, ...]
    timestamps: tuple[ComparisonTimestamp, ...]
    dataset_id: str
    dataset_fingerprint: str
    timeframe: Timeframe

    def __post_init__(self) -> None:
        if (
            not self.bars
            or len(self.bars) != len(self.timestamps)
            or not self.dataset_id
            or not self.dataset_fingerprint
        ):
            raise IndicatorComparisonError(
                "comparison source requires aligned bars and dataset identity"
            )
        timestamp_type = type(self.timestamps[0])
        if timestamp_type not in (date, datetime) or any(
            type(value) is not timestamp_type for value in self.timestamps
        ):
            raise IndicatorComparisonError(
                "comparison timestamps must use one consistent date or datetime type"
            )
        if timestamp_type is datetime and any(
            cast(datetime, value).utcoffset() is None for value in self.timestamps
        ):
            raise IndicatorComparisonError(
                "comparison datetime timestamps must be timezone-aware"
            )
        if any(
            current >= following
            for current, following in zip(
                self.timestamps, self.timestamps[1:], strict=False
            )
        ):
            raise IndicatorComparisonError(
                "comparison timestamps must be unique and strictly chronological"
            )

    @classmethod
    def from_market_dataset(
        cls,
        dataset: MarketDataset,
        *,
        timeframe: Timeframe = DEFAULT_US_EQUITY_TIMEFRAME,
    ) -> "IndicatorComparisonSource":
        """Bind a QF-3 daily dataset to an explicit canonical timeframe."""
        try:
            validate_market_dataset(dataset)
        except MarketDataValidationError as error:
            raise IndicatorComparisonError(str(error)) from error
        if not dataset_identity_matches(dataset):
            raise IndicatorComparisonError(
                "comparison bars and provenance do not reproduce the dataset identity"
            )
        canonical_daily_timeframe = Timeframe(
            interval=SessionInterval(),
            session_policy=ExchangeSessionPolicy(
                calendar_name=timeframe.session_policy.calendar_name,
                timezone_name=timeframe.session_policy.timezone_name,
            ),
        )
        if (
            timeframe.session_policy.calendar_name != dataset.metadata.calendar
            or timeframe != canonical_daily_timeframe
        ):
            raise IndicatorComparisonError(
                "QF-3 daily comparison requires its matching canonical one-session "
                "regular-hours, bar-start, completed-only timeframe"
            )
        return cls(
            bars=cast(tuple[IndicatorBar, ...], dataset.bars),
            timestamps=tuple(bar.session_date for bar in dataset.bars),
            dataset_id=dataset.metadata.dataset_id,
            dataset_fingerprint=dataset.metadata.data_sha256,
            timeframe=timeframe,
        )

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "dataset_id": self.dataset_id,
            "dataset_fingerprint": self.dataset_fingerprint,
            "observation_count": len(self.bars),
            "first_timestamp": self.timestamps[0].isoformat(),
            "last_timestamp": self.timestamps[-1].isoformat(),
            "timeframe": {
                "configuration_id": self.timeframe.configuration_id,
                "configuration": self.timeframe.to_primitive(),
            },
        }


@dataclass(frozen=True, slots=True)
class IndicatorDivergence:
    """One overlapping finite value pair beyond the configured tolerance."""

    output_name: str
    timestamp: str
    backend_a_value: Decimal
    backend_b_value: Decimal
    absolute_difference: Decimal
    relative_difference: Decimal | None

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "output_name": self.output_name,
            "timestamp": self.timestamp,
            "backend_a_value": decimal_to_primitive(self.backend_a_value),
            "backend_b_value": decimal_to_primitive(self.backend_b_value),
            "absolute_difference": decimal_to_primitive(self.absolute_difference),
            "relative_difference": _optional_decimal(self.relative_difference),
        }


@dataclass(frozen=True, slots=True)
class IndicatorFieldComparison:
    """Warm-up, availability, and overlapping numerical statistics for one field."""

    output_name: str
    backend_a_first_valid_timestamp: str | None
    backend_b_first_valid_timestamp: str | None
    backend_a_leading_unavailable_count: int
    backend_b_leading_unavailable_count: int
    backend_a_valid_count: int
    backend_b_valid_count: int
    overlapping_valid_count: int
    backend_a_only_valid_timestamps: tuple[str, ...]
    backend_b_only_valid_timestamps: tuple[str, ...]
    maximum_absolute_difference: Decimal | None
    mean_absolute_difference: Decimal | None
    median_absolute_difference: Decimal | None
    maximum_relative_difference: Decimal | None
    mean_relative_difference: Decimal | None
    median_relative_difference: Decimal | None
    divergences: tuple[IndicatorDivergence, ...]

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "output_name": self.output_name,
            "availability": {
                "backend_a_first_valid_timestamp": (
                    self.backend_a_first_valid_timestamp
                ),
                "backend_b_first_valid_timestamp": (
                    self.backend_b_first_valid_timestamp
                ),
                "backend_a_leading_unavailable_count": (
                    self.backend_a_leading_unavailable_count
                ),
                "backend_b_leading_unavailable_count": (
                    self.backend_b_leading_unavailable_count
                ),
                "backend_a_valid_count": self.backend_a_valid_count,
                "backend_b_valid_count": self.backend_b_valid_count,
                "overlapping_valid_count": self.overlapping_valid_count,
                "backend_a_only_valid_timestamps": list(
                    self.backend_a_only_valid_timestamps
                ),
                "backend_b_only_valid_timestamps": list(
                    self.backend_b_only_valid_timestamps
                ),
                "availability_differences_are_formula_divergences": False,
            },
            "differences": {
                "maximum_absolute_difference": _optional_decimal(
                    self.maximum_absolute_difference
                ),
                "mean_absolute_difference": _optional_decimal(
                    self.mean_absolute_difference
                ),
                "median_absolute_difference": _optional_decimal(
                    self.median_absolute_difference
                ),
                "maximum_relative_difference": _optional_decimal(
                    self.maximum_relative_difference
                ),
                "mean_relative_difference": _optional_decimal(
                    self.mean_relative_difference
                ),
                "median_relative_difference": _optional_decimal(
                    self.median_relative_difference
                ),
                "divergence_count": len(self.divergences),
            },
            "divergences": cast(
                list[Primitive], [item.to_primitive() for item in self.divergences]
            ),
        }

    def summary_row(self) -> PrimitiveMapping:
        return {
            "output_name": self.output_name,
            "backend_a_first_valid_timestamp": self.backend_a_first_valid_timestamp,
            "backend_b_first_valid_timestamp": self.backend_b_first_valid_timestamp,
            "backend_a_leading_unavailable_count": (
                self.backend_a_leading_unavailable_count
            ),
            "backend_b_leading_unavailable_count": (
                self.backend_b_leading_unavailable_count
            ),
            "backend_a_valid_count": self.backend_a_valid_count,
            "backend_b_valid_count": self.backend_b_valid_count,
            "overlapping_valid_count": self.overlapping_valid_count,
            "backend_a_only_valid_count": len(self.backend_a_only_valid_timestamps),
            "backend_b_only_valid_count": len(self.backend_b_only_valid_timestamps),
            "maximum_absolute_difference": _optional_decimal(
                self.maximum_absolute_difference
            ),
            "mean_absolute_difference": _optional_decimal(
                self.mean_absolute_difference
            ),
            "median_absolute_difference": _optional_decimal(
                self.median_absolute_difference
            ),
            "maximum_relative_difference": _optional_decimal(
                self.maximum_relative_difference
            ),
            "mean_relative_difference": _optional_decimal(
                self.mean_relative_difference
            ),
            "median_relative_difference": _optional_decimal(
                self.median_relative_difference
            ),
            "divergence_count": len(self.divergences),
        }


@dataclass(frozen=True, slots=True)
class IndicatorBackendComparisonResult:
    """Complete deterministic comparison for one normalized indicator definition."""

    comparison_id: str
    source_snapshot: PrimitiveMappingSnapshot
    definition: StandardIndicatorDefinition
    backend_a_identity: IndicatorBackendIdentity
    backend_b_identity: IndicatorBackendIdentity
    backend_a_computation: IndicatorComputationResult
    backend_b_computation: IndicatorComputationResult
    tolerances: IndicatorComparisonTolerances
    field_comparisons: tuple[IndicatorFieldComparison, ...]
    engine_version: str = INDICATOR_COMPARISON_ENGINE_VERSION
    schema_version: str = INDICATOR_COMPARISON_SCHEMA_VERSION

    @property
    def source(self) -> PrimitiveMapping:
        return self.source_snapshot.to_primitive()

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "comparison_id": self.comparison_id,
            "component": "quantforge_indicator_backend_comparison",
            "engine_version": self.engine_version,
            "schema_version": self.schema_version,
            "source": self.source,
            "indicator_definition": self.definition.to_primitive(),
            "backend_a": self.backend_a_identity.to_primitive(),
            "backend_b": self.backend_b_identity.to_primitive(),
            "tolerances": self.tolerances.to_primitive(),
            "arithmetic": {
                "decimal_precision": _ARITHMETIC_PRECISION,
                "rounding": "ROUND_HALF_EVEN",
                "relative_denominator": (
                    "max(abs(backend_a), abs(backend_b)); unavailable when both zero"
                ),
            },
            "field_comparisons": cast(
                list[Primitive],
                [item.to_primitive() for item in self.field_comparisons],
            ),
            "interpretation": (
                "comparison only; backend ordering does not imply ranking or migration"
            ),
        }

    def human_summary(self) -> str:
        lines = [
            f"Indicator backend comparison {self.comparison_id}",
            f"Indicator: {self.definition.name}",
            (
                "Backends: "
                f"{self.backend_a_identity.backend_id} vs "
                f"{self.backend_b_identity.backend_id}"
            ),
            (
                "Tolerances: absolute="
                f"{decimal_to_primitive(self.tolerances.absolute)}, relative="
                f"{decimal_to_primitive(self.tolerances.relative)}"
            ),
        ]
        for field in self.field_comparisons:
            lines.append(
                f"{field.output_name}: first valid "
                f"{field.backend_a_first_valid_timestamp or 'none'} / "
                f"{field.backend_b_first_valid_timestamp or 'none'}; "
                f"overlap={field.overlapping_valid_count}; "
                f"divergences={len(field.divergences)}; "
                "max_abs="
                f"{_optional_decimal(field.maximum_absolute_difference) or 'none'}"
            )
        lines.append("Comparison is descriptive; no backend is selected or migrated.")
        return "\n".join(lines) + "\n"


def compare_indicator_backends(
    dataset: MarketDataset,
    indicator: BackendNeutralStandardIndicator,
    *,
    timeframe: Timeframe = DEFAULT_US_EQUITY_TIMEFRAME,
    backend_a_id: str = NATIVE_INDICATOR_BACKEND,
    backend_b_id: str = TALIB_INDICATOR_BACKEND,
    tolerances: IndicatorComparisonTolerances | None = None,
    backend_registry: IndicatorBackendRegistry | None = None,
) -> IndicatorBackendComparisonResult:
    """Compare one QF-35 definition on the same QF-3 dataset and timeframe."""
    try:
        definition = indicator.standard_definition
    except AttributeError as error:
        raise IndicatorComparisonError(
            "indicator does not expose a backend-neutral standard definition"
        ) from error
    source = IndicatorComparisonSource.from_market_dataset(dataset, timeframe=timeframe)
    return compare_standard_indicator_backends(
        source,
        definition,
        backend_a_id=backend_a_id,
        backend_b_id=backend_b_id,
        tolerances=tolerances,
        backend_registry=backend_registry,
    )


def compare_standard_indicator_backends(
    source: IndicatorComparisonSource,
    definition: StandardIndicatorDefinition,
    *,
    backend_a_id: str = NATIVE_INDICATOR_BACKEND,
    backend_b_id: str = TALIB_INDICATOR_BACKEND,
    tolerances: IndicatorComparisonTolerances | None = None,
    backend_registry: IndicatorBackendRegistry | None = None,
) -> IndicatorBackendComparisonResult:
    """Run one normalized QF-35 request through two explicitly named backends."""
    if not backend_a_id or not backend_b_id or backend_a_id == backend_b_id:
        raise IndicatorComparisonError(
            "comparison requires two distinct non-empty backend ids"
        )
    selected_tolerances = tolerances or IndicatorComparisonTolerances()
    registry = backend_registry or default_indicator_backend_registry()
    backend_a = registry.resolve(backend_a_id)
    backend_b = registry.resolve(backend_b_id)
    backend_a_identity = backend_a.identity_for(definition)
    backend_b_identity = backend_b.identity_for(definition)
    if (
        backend_a_identity.backend_id != backend_a_id
        or backend_b_identity.backend_id != backend_b_id
    ):
        raise IndicatorComparisonError(
            "resolved backend identity does not match the requested registry id"
        )
    request = IndicatorComputationRequest(definition, source.bars)
    result_a = backend_a.compute(request)
    result_b = backend_b.compute(request)
    _validate_backend_result(result_a, definition, backend_a_identity, len(source.bars))
    _validate_backend_result(result_b, definition, backend_b_identity, len(source.bars))
    fields_a = {field.name: field.values for field in result_a.fields}
    fields_b = {field.name: field.values for field in result_b.fields}
    field_comparisons = tuple(
        _compare_field(
            output_name,
            source.timestamps,
            fields_a[output_name],
            fields_b[output_name],
            selected_tolerances,
        )
        for output_name in definition.output_fields
    )
    source_snapshot = PrimitiveMappingSnapshot.capture(source.to_primitive())
    identity_values: PrimitiveMapping = {
        "component": "quantforge_indicator_backend_comparison",
        "engine_version": INDICATOR_COMPARISON_ENGINE_VERSION,
        "schema_version": INDICATOR_COMPARISON_SCHEMA_VERSION,
        "source": source_snapshot.to_primitive(),
        "indicator_definition": definition.to_primitive(),
        "backend_a": result_a.backend_identity.to_primitive(),
        "backend_b": result_b.backend_identity.to_primitive(),
        "tolerances": selected_tolerances.to_primitive(),
        "field_comparisons": cast(
            list[Primitive], [item.to_primitive() for item in field_comparisons]
        ),
    }
    return IndicatorBackendComparisonResult(
        comparison_id=configuration_identity(identity_values),
        source_snapshot=source_snapshot,
        definition=definition,
        backend_a_identity=result_a.backend_identity,
        backend_b_identity=result_b.backend_identity,
        backend_a_computation=result_a,
        backend_b_computation=result_b,
        tolerances=selected_tolerances,
        field_comparisons=field_comparisons,
    )


def export_indicator_backend_comparison(
    result: IndicatorBackendComparisonResult, output_root: Path
) -> Path:
    """Atomically create deterministic JSON, CSV, and text artifacts."""
    destination = output_root / result.comparison_id
    if destination.exists():
        raise IndicatorComparisonError(
            f"indicator backend comparison already exists: {destination}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{result.comparison_id}.", dir=str(output_root))
    )
    try:
        _write_json(temporary / "comparison.json", result.to_primitive())
        _write_rows(
            temporary / "field_summary.csv",
            [item.summary_row() for item in result.field_comparisons],
            fieldnames=_FIELD_SUMMARY_FIELDNAMES,
        )
        _write_rows(
            temporary / "divergences.csv",
            [
                divergence.to_primitive()
                for field in result.field_comparisons
                for divergence in field.divergences
            ],
            fieldnames=_DIVERGENCE_FIELDNAMES,
        )
        _write_text(temporary / "summary.txt", result.human_summary())
        os.rename(temporary, destination)
    except IndicatorComparisonError:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    except (OSError, TypeError, ValueError) as error:
        shutil.rmtree(temporary, ignore_errors=True)
        raise IndicatorComparisonError(
            "failed to export immutable indicator backend comparison"
        ) from error
    return destination


def validate_indicator_backend_comparison_export(
    result: IndicatorBackendComparisonResult, path: Path
) -> Path:
    """Require an existing export to match regenerated bytes exactly."""
    try:
        if path.name != result.comparison_id or not path.is_dir():
            raise IndicatorComparisonError(
                "indicator comparison export does not match the expected result"
            )
        entries = {entry.name: entry for entry in path.iterdir()}
        if set(entries) != set(INDICATOR_COMPARISON_ARTIFACT_FILENAMES):
            raise IndicatorComparisonError(
                "indicator comparison export does not match the expected result"
            )
        with tempfile.TemporaryDirectory(
            prefix="quantforge-indicator-comparison-validation-"
        ) as temporary_root:
            expected = export_indicator_backend_comparison(result, Path(temporary_root))
            if any(
                entries[name].read_bytes() != (expected / name).read_bytes()
                for name in INDICATOR_COMPARISON_ARTIFACT_FILENAMES
            ):
                raise IndicatorComparisonError(
                    "indicator comparison export does not match the expected result"
                )
    except IndicatorComparisonError:
        raise
    except OSError as error:
        raise IndicatorComparisonError(
            "failed to validate immutable indicator comparison export"
        ) from error
    return path


def _validate_backend_result(
    result: IndicatorComputationResult,
    definition: StandardIndicatorDefinition,
    expected_backend_identity: IndicatorBackendIdentity,
    observation_count: int,
) -> None:
    if (
        result.definition_name != definition.name
        or result.backend_identity != expected_backend_identity
        or result.normalized_parameters != definition.parameters
        or result.normalized_input_fields != definition.input_fields
        or {field.name for field in result.fields} != set(definition.output_fields)
        or result.observation_count != observation_count
    ):
        raise IndicatorComparisonError(
            "backend comparison result violates the normalized contract: "
            f"{expected_backend_identity.backend_id}"
        )


def _compare_field(
    output_name: str,
    timestamps: tuple[ComparisonTimestamp, ...],
    values_a: tuple[Decimal | None, ...],
    values_b: tuple[Decimal | None, ...],
    tolerances: IndicatorComparisonTolerances,
) -> IndicatorFieldComparison:
    rendered_timestamps = tuple(value.isoformat() for value in timestamps)
    overlapping_absolute: list[Decimal] = []
    overlapping_relative: list[Decimal] = []
    only_a: list[str] = []
    only_b: list[str] = []
    divergences: list[IndicatorDivergence] = []
    try:
        with localcontext(_ARITHMETIC_CONTEXT):
            for timestamp, value_a, value_b in zip(
                rendered_timestamps, values_a, values_b, strict=True
            ):
                if value_a is None:
                    if value_b is not None:
                        only_b.append(timestamp)
                    continue
                if value_b is None:
                    only_a.append(timestamp)
                    continue
                absolute_difference = abs(value_a - value_b)
                scale = max(abs(value_a), abs(value_b))
                relative_difference = (
                    None if scale == 0 else absolute_difference / scale
                )
                overlapping_absolute.append(absolute_difference)
                if relative_difference is not None:
                    overlapping_relative.append(relative_difference)
                threshold = tolerances.absolute + tolerances.relative * scale
                if absolute_difference > threshold:
                    divergences.append(
                        IndicatorDivergence(
                            output_name,
                            timestamp,
                            value_a,
                            value_b,
                            absolute_difference,
                            relative_difference,
                        )
                    )
    except DecimalException as error:
        raise IndicatorComparisonError(
            "indicator comparison arithmetic failed under its configured policy"
        ) from error
    return IndicatorFieldComparison(
        output_name=output_name,
        backend_a_first_valid_timestamp=_first_valid_timestamp(
            rendered_timestamps, values_a
        ),
        backend_b_first_valid_timestamp=_first_valid_timestamp(
            rendered_timestamps, values_b
        ),
        backend_a_leading_unavailable_count=_leading_unavailable_count(values_a),
        backend_b_leading_unavailable_count=_leading_unavailable_count(values_b),
        backend_a_valid_count=sum(value is not None for value in values_a),
        backend_b_valid_count=sum(value is not None for value in values_b),
        overlapping_valid_count=len(overlapping_absolute),
        backend_a_only_valid_timestamps=tuple(only_a),
        backend_b_only_valid_timestamps=tuple(only_b),
        maximum_absolute_difference=_maximum(overlapping_absolute),
        mean_absolute_difference=_mean(overlapping_absolute),
        median_absolute_difference=_median(overlapping_absolute),
        maximum_relative_difference=_maximum(overlapping_relative),
        mean_relative_difference=_mean(overlapping_relative),
        median_relative_difference=_median(overlapping_relative),
        divergences=tuple(divergences),
    )


def _first_valid_timestamp(
    timestamps: tuple[str, ...], values: tuple[Decimal | None, ...]
) -> str | None:
    return next(
        (
            timestamp
            for timestamp, value in zip(timestamps, values, strict=True)
            if value is not None
        ),
        None,
    )


def _leading_unavailable_count(values: tuple[Decimal | None, ...]) -> int:
    return next(
        (index for index, value in enumerate(values) if value is not None), len(values)
    )


def _maximum(values: list[Decimal]) -> Decimal | None:
    return None if not values else max(values)


def _mean(values: list[Decimal]) -> Decimal | None:
    if not values:
        return None
    with localcontext(_ARITHMETIC_CONTEXT):
        return sum(values, Decimal(0)) / Decimal(len(values))


def _median(values: list[Decimal]) -> Decimal | None:
    if not values:
        return None
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    with localcontext(_ARITHMETIC_CONTEXT):
        return (ordered[midpoint - 1] + ordered[midpoint]) / Decimal(2)


def _optional_decimal(value: Decimal | None) -> str | None:
    return None if value is None else decimal_to_primitive(value)


_FIELD_SUMMARY_FIELDNAMES = (
    "output_name",
    "backend_a_first_valid_timestamp",
    "backend_b_first_valid_timestamp",
    "backend_a_leading_unavailable_count",
    "backend_b_leading_unavailable_count",
    "backend_a_valid_count",
    "backend_b_valid_count",
    "overlapping_valid_count",
    "backend_a_only_valid_count",
    "backend_b_only_valid_count",
    "maximum_absolute_difference",
    "mean_absolute_difference",
    "median_absolute_difference",
    "maximum_relative_difference",
    "mean_relative_difference",
    "median_relative_difference",
    "divergence_count",
)
_DIVERGENCE_FIELDNAMES = (
    "output_name",
    "timestamp",
    "backend_a_value",
    "backend_b_value",
    "absolute_difference",
    "relative_difference",
)


def _write_json(path: Path, value: PrimitiveMapping) -> None:
    _write_text(
        path,
        json.dumps(value, ensure_ascii=True, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
    )


def _write_rows(
    path: Path,
    rows: list[PrimitiveMapping],
    *,
    fieldnames: tuple[str, ...],
) -> None:
    if any(tuple(row) != fieldnames for row in rows):
        raise IndicatorComparisonError(
            f"indicator comparison rows have inconsistent schema: {path.name}"
        )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        stream.flush()
        os.fsync(stream.fileno())


def _write_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


__all__ = [
    "INDICATOR_COMPARISON_ARTIFACT_FILENAMES",
    "INDICATOR_COMPARISON_ENGINE_VERSION",
    "INDICATOR_COMPARISON_SCHEMA_VERSION",
    "BackendNeutralStandardIndicator",
    "IndicatorBackendComparisonResult",
    "IndicatorComparisonSource",
    "IndicatorComparisonTolerances",
    "IndicatorDivergence",
    "IndicatorFieldComparison",
    "compare_indicator_backends",
    "compare_standard_indicator_backends",
    "export_indicator_backend_comparison",
    "validate_indicator_backend_comparison_export",
]
