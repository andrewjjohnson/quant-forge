"""Indicator and signal impact comparisons for explicit prediction backends."""

import csv
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import date
from decimal import ROUND_HALF_EVEN, Context, Decimal, localcontext
from enum import StrEnum
from pathlib import Path
from typing import cast

from quantforge.configuration import (
    Primitive,
    PrimitiveMapping,
    PrimitiveMappingSnapshot,
    configuration_identity,
    decimal_to_primitive,
)
from quantforge.data.models import MarketDataset
from quantforge.indicators import (
    NATIVE_INDICATOR_BACKEND,
    TALIB_INDICATOR_BACKEND,
    IndicatorBackendComparisonResult,
    IndicatorBackendRegistry,
    IndicatorComparisonTolerances,
    WilderDirectionalMovement,
    WilderDirectionalMovementParameters,
    WilderRelativeStrengthIndex,
    WilderRelativeStrengthIndexParameters,
    compare_indicator_backends,
)
from quantforge.prediction.errors import (
    InvalidPredictionOutputError,
    PredictionExportError,
)
from quantforge.prediction.models import (
    PredictionAnalysisResult,
    PredictionDirection,
    PredictionRow,
    PredictionSignal,
)
from quantforge.prediction.outcomes.overnight_gap import (
    create_overnight_gap_prediction_study,
)
from quantforge.prediction.overnight_gap import (
    OvernightGapPredictionParameters,
    OvernightGapPredictionStrategy,
)
from quantforge.prediction.runner import run_prediction_analysis
from quantforge.prediction.study import run_prediction_study
from quantforge.timeframes import DEFAULT_US_EQUITY_TIMEFRAME, Timeframe

PREDICTION_BACKEND_COMPARISON_ENGINE_VERSION = "1"
PREDICTION_BACKEND_COMPARISON_SCHEMA_VERSION = "1"
PREDICTION_BACKEND_COMPARISON_ARTIFACT_FILENAMES = (
    "comparison.json",
    "indicator_fields.csv",
    "indicator_divergences.csv",
    "prediction_changes.csv",
    "prediction_metrics.csv",
    "summary.txt",
)
_ARITHMETIC_CONTEXT = Context(prec=34, rounding=ROUND_HALF_EVEN)


class PredictionSignalComparisonStatus(StrEnum):
    """Relationship between backend predictions on one signal session."""

    BACKEND_A_ONLY = "backend_a_only"
    BACKEND_B_ONLY = "backend_b_only"
    MATCHED = "matched"
    DIRECTION_CHANGED = "direction_changed"


@dataclass(frozen=True, slots=True)
class PredictionSignalComparison:
    """One date-level prediction presence or direction comparison."""

    signal_session: date
    status: PredictionSignalComparisonStatus
    backend_a_direction: PredictionDirection | None
    backend_b_direction: PredictionDirection | None
    backend_a_reason: str | None
    backend_b_reason: str | None

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "signal_session": self.signal_session.isoformat(),
            "status": self.status.value,
            "backend_a_direction": _optional_direction(self.backend_a_direction),
            "backend_b_direction": _optional_direction(self.backend_b_direction),
            "backend_a_reason": self.backend_a_reason,
            "backend_b_reason": self.backend_b_reason,
        }


@dataclass(frozen=True, slots=True)
class PredictionMetricDifference:
    """One descriptive metric under each backend plus backend-b-minus-a delta."""

    metric_name: str
    backend_a_value: Decimal | None
    backend_b_value: Decimal | None
    difference: Decimal | None

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "metric_name": self.metric_name,
            "backend_a_value": _optional_decimal(self.backend_a_value),
            "backend_b_value": _optional_decimal(self.backend_b_value),
            "difference_backend_b_minus_a": _optional_decimal(self.difference),
        }


@dataclass(frozen=True, slots=True)
class PredictionBackendComparison:
    """Presence, direction, and summary-metric impact for one prediction rule."""

    backend_a_id: str
    backend_b_id: str
    backend_a_prediction_count: int
    backend_b_prediction_count: int
    backend_a_metric_sample_count: int
    backend_b_metric_sample_count: int
    signal_comparisons: tuple[PredictionSignalComparison, ...]
    metrics: tuple[PredictionMetricDifference, ...]

    @property
    def backend_a_only_prediction_dates(self) -> tuple[date, ...]:
        return tuple(
            item.signal_session
            for item in self.signal_comparisons
            if item.status is PredictionSignalComparisonStatus.BACKEND_A_ONLY
        )

    @property
    def backend_b_only_prediction_dates(self) -> tuple[date, ...]:
        return tuple(
            item.signal_session
            for item in self.signal_comparisons
            if item.status is PredictionSignalComparisonStatus.BACKEND_B_ONLY
        )

    @property
    def changed_direction_count(self) -> int:
        return sum(
            item.status is PredictionSignalComparisonStatus.DIRECTION_CHANGED
            for item in self.signal_comparisons
        )

    @property
    def matched_prediction_count(self) -> int:
        return sum(
            item.status is PredictionSignalComparisonStatus.MATCHED
            for item in self.signal_comparisons
        )

    @property
    def shared_prediction_date_count(self) -> int:
        return self.changed_direction_count + self.matched_prediction_count

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "backend_a_id": self.backend_a_id,
            "backend_b_id": self.backend_b_id,
            "counts": {
                "backend_a_predictions": self.backend_a_prediction_count,
                "backend_b_predictions": self.backend_b_prediction_count,
                "backend_a_metric_sample": self.backend_a_metric_sample_count,
                "backend_b_metric_sample": self.backend_b_metric_sample_count,
                "backend_a_only_prediction_dates": len(
                    self.backend_a_only_prediction_dates
                ),
                "backend_b_only_prediction_dates": len(
                    self.backend_b_only_prediction_dates
                ),
                "shared_prediction_dates": self.shared_prediction_date_count,
                "matched_predictions": self.matched_prediction_count,
                "changed_directions": self.changed_direction_count,
            },
            "backend_a_only_prediction_dates": [
                value.isoformat() for value in self.backend_a_only_prediction_dates
            ],
            "backend_b_only_prediction_dates": [
                value.isoformat() for value in self.backend_b_only_prediction_dates
            ],
            "signal_comparisons": cast(
                list[Primitive],
                [item.to_primitive() for item in self.signal_comparisons],
            ),
            "metrics": cast(
                list[Primitive], [item.to_primitive() for item in self.metrics]
            ),
        }


@dataclass(frozen=True, slots=True)
class OvernightGapBackendComparisonResult:
    """End-to-end indicator and prediction impact report for the QF-11 baseline."""

    comparison_id: str
    source_snapshot: PrimitiveMappingSnapshot
    parameters_snapshot: PrimitiveMappingSnapshot
    indicator_comparisons: tuple[IndicatorBackendComparisonResult, ...]
    prediction_comparison: PredictionBackendComparison
    engine_version: str = PREDICTION_BACKEND_COMPARISON_ENGINE_VERSION
    schema_version: str = PREDICTION_BACKEND_COMPARISON_SCHEMA_VERSION

    @property
    def source(self) -> PrimitiveMapping:
        return self.source_snapshot.to_primitive()

    @property
    def parameters(self) -> PrimitiveMapping:
        return self.parameters_snapshot.to_primitive()

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "comparison_id": self.comparison_id,
            "component": "quantforge_overnight_gap_backend_comparison",
            "engine_version": self.engine_version,
            "schema_version": self.schema_version,
            "source": self.source,
            "prediction_rule": {
                "name": OvernightGapPredictionStrategy.name,
                "implementation_version": (
                    OvernightGapPredictionStrategy.implementation_version
                ),
                "parameters": self.parameters,
            },
            "indicator_comparisons": cast(
                list[Primitive],
                [item.to_primitive() for item in self.indicator_comparisons],
            ),
            "prediction_comparison": self.prediction_comparison.to_primitive(),
            "limitations": [
                "comparison is descriptive and does not rank or select a backend",
                "no existing study is migrated automatically",
                "accuracy and signed returns are prediction-study metrics, not "
                "executable trading performance",
            ],
        }

    def human_summary(self) -> str:
        prediction = self.prediction_comparison
        lines = [
            f"Overnight-gap backend comparison {self.comparison_id}",
            f"Backends: {prediction.backend_a_id} vs {prediction.backend_b_id}",
        ]
        for indicator in self.indicator_comparisons:
            divergence_count = sum(
                len(field.divergences) for field in indicator.field_comparisons
            )
            lines.append(
                f"{indicator.definition.name}: "
                f"{divergence_count} value divergences beyond tolerance"
            )
        lines.extend(
            (
                "Predictions: "
                f"{prediction.backend_a_prediction_count} / "
                f"{prediction.backend_b_prediction_count}",
                "Prediction dates only in backend A/B: "
                f"{len(prediction.backend_a_only_prediction_dates)} / "
                f"{len(prediction.backend_b_only_prediction_dates)}",
                f"Matched predictions: {prediction.matched_prediction_count}",
                f"Changed directions: {prediction.changed_direction_count}",
            )
        )
        for metric in prediction.metrics:
            lines.append(
                f"{metric.metric_name}: "
                f"{_optional_decimal(metric.backend_a_value) or 'none'} / "
                f"{_optional_decimal(metric.backend_b_value) or 'none'}; "
                "backend_b_minus_a="
                f"{_optional_decimal(metric.difference) or 'none'}"
            )
        lines.append("Comparison only; native studies remain unchanged.")
        return "\n".join(lines) + "\n"


def compare_prediction_backends(
    backend_a_result: PredictionAnalysisResult,
    backend_b_result: PredictionAnalysisResult,
    *,
    backend_a_id: str,
    backend_b_id: str,
    backend_a_signals: tuple[PredictionSignal, ...] | None = None,
    backend_b_signals: tuple[PredictionSignal, ...] | None = None,
) -> PredictionBackendComparison:
    """Compare prediction dates, directions, and requested summary metrics."""
    if not backend_a_id or not backend_b_id or backend_a_id == backend_b_id:
        raise InvalidPredictionOutputError(
            "prediction comparison requires two distinct non-empty backend ids"
        )
    if (
        backend_a_result.market_data.dataset_id
        != backend_b_result.market_data.dataset_id
        or backend_a_result.market_data.bars_fingerprint
        != backend_b_result.market_data.bars_fingerprint
        or backend_a_result.strategy_id != backend_b_result.strategy_id
        or backend_a_result.strategy_implementation_version
        != backend_b_result.strategy_implementation_version
        or backend_a_result.strategy_configuration.get("parameters")
        != backend_b_result.strategy_configuration.get("parameters")
    ):
        raise InvalidPredictionOutputError(
            "prediction backend comparison requires the same dataset and logical rule"
        )
    rows_a = _signals_by_session(
        backend_a_result.rows if backend_a_signals is None else backend_a_signals,
        backend_a_id,
        backend_a_result,
    )
    rows_b = _signals_by_session(
        backend_b_result.rows if backend_b_signals is None else backend_b_signals,
        backend_b_id,
        backend_b_result,
    )
    signal_comparisons: list[PredictionSignalComparison] = []
    for signal_session in sorted(rows_a.keys() | rows_b.keys()):
        row_a = rows_a.get(signal_session)
        row_b = rows_b.get(signal_session)
        if row_a is None:
            status = PredictionSignalComparisonStatus.BACKEND_B_ONLY
        elif row_b is None:
            status = PredictionSignalComparisonStatus.BACKEND_A_ONLY
        elif row_a.direction is row_b.direction:
            status = PredictionSignalComparisonStatus.MATCHED
        else:
            status = PredictionSignalComparisonStatus.DIRECTION_CHANGED
        signal_comparisons.append(
            PredictionSignalComparison(
                signal_session=signal_session,
                status=status,
                backend_a_direction=None if row_a is None else row_a.direction,
                backend_b_direction=None if row_b is None else row_b.direction,
                backend_a_reason=None if row_a is None else row_a.reason,
                backend_b_reason=None if row_b is None else row_b.reason,
            )
        )
    accuracy_a = backend_a_result.metrics.accuracy
    accuracy_b = backend_b_result.metrics.accuracy
    average_signed_a = _average_signed_return(backend_a_result.rows)
    average_signed_b = _average_signed_return(backend_b_result.rows)
    return PredictionBackendComparison(
        backend_a_id=backend_a_id,
        backend_b_id=backend_b_id,
        backend_a_prediction_count=len(rows_a),
        backend_b_prediction_count=len(rows_b),
        backend_a_metric_sample_count=len(backend_a_result.rows),
        backend_b_metric_sample_count=len(backend_b_result.rows),
        signal_comparisons=tuple(signal_comparisons),
        metrics=(
            PredictionMetricDifference(
                "accuracy",
                accuracy_a,
                accuracy_b,
                _difference(accuracy_a, accuracy_b),
            ),
            PredictionMetricDifference(
                "average_signed_return",
                average_signed_a,
                average_signed_b,
                _difference(average_signed_a, average_signed_b),
            ),
        ),
    )


def run_overnight_gap_backend_comparison(
    dataset: MarketDataset,
    parameters: OvernightGapPredictionParameters | None = None,
    *,
    timeframe: Timeframe = DEFAULT_US_EQUITY_TIMEFRAME,
    backend_a_id: str = NATIVE_INDICATOR_BACKEND,
    backend_b_id: str = TALIB_INDICATOR_BACKEND,
    tolerances: IndicatorComparisonTolerances | None = None,
    backend_registry: IndicatorBackendRegistry | None = None,
) -> OvernightGapBackendComparisonResult:
    """Run the QF-11 baseline under two explicit backends and compare impact."""
    selected_parameters = parameters or OvernightGapPredictionParameters()
    rsi = WilderRelativeStrengthIndex(
        WilderRelativeStrengthIndexParameters(selected_parameters.rsi_period),
        backend_id=backend_a_id,
        backend_registry=backend_registry,
    )
    directional = WilderDirectionalMovement(
        WilderDirectionalMovementParameters(selected_parameters.adx_period),
        backend_id=backend_a_id,
        backend_registry=backend_registry,
    )
    indicator_comparisons = (
        compare_indicator_backends(
            dataset,
            rsi,
            timeframe=timeframe,
            backend_a_id=backend_a_id,
            backend_b_id=backend_b_id,
            tolerances=tolerances,
            backend_registry=backend_registry,
        ),
        compare_indicator_backends(
            dataset,
            directional,
            timeframe=timeframe,
            backend_a_id=backend_a_id,
            backend_b_id=backend_b_id,
            tolerances=tolerances,
            backend_registry=backend_registry,
        ),
    )
    strategy_a = OvernightGapPredictionStrategy(
        selected_parameters,
        backend_id=backend_a_id,
        backend_registry=backend_registry,
    )
    strategy_b = OvernightGapPredictionStrategy(
        selected_parameters,
        backend_id=backend_b_id,
        backend_registry=backend_registry,
    )
    study_a = run_prediction_study(
        dataset, create_overnight_gap_prediction_study(strategy_a)
    )
    study_b = run_prediction_study(
        dataset, create_overnight_gap_prediction_study(strategy_b)
    )
    prediction_comparison = compare_prediction_backends(
        run_prediction_analysis(dataset, strategy_a),
        run_prediction_analysis(dataset, strategy_b),
        backend_a_id=backend_a_id,
        backend_b_id=backend_b_id,
        backend_a_signals=study_a.signals,
        backend_b_signals=study_b.signals,
    )
    source_snapshot = PrimitiveMappingSnapshot.capture(indicator_comparisons[0].source)
    parameters_snapshot = PrimitiveMappingSnapshot.capture(
        selected_parameters.to_primitive()
    )
    identity_values: PrimitiveMapping = {
        "component": "quantforge_overnight_gap_backend_comparison",
        "engine_version": PREDICTION_BACKEND_COMPARISON_ENGINE_VERSION,
        "schema_version": PREDICTION_BACKEND_COMPARISON_SCHEMA_VERSION,
        "source": source_snapshot.to_primitive(),
        "parameters": parameters_snapshot.to_primitive(),
        "indicator_comparisons": [
            item.to_primitive() for item in indicator_comparisons
        ],
        "prediction_comparison": prediction_comparison.to_primitive(),
    }
    return OvernightGapBackendComparisonResult(
        comparison_id=configuration_identity(identity_values),
        source_snapshot=source_snapshot,
        parameters_snapshot=parameters_snapshot,
        indicator_comparisons=indicator_comparisons,
        prediction_comparison=prediction_comparison,
    )


def export_overnight_gap_backend_comparison(
    result: OvernightGapBackendComparisonResult, output_root: Path
) -> Path:
    """Atomically export the complete end-to-end comparison."""
    destination = output_root / result.comparison_id
    if destination.exists():
        raise PredictionExportError(
            f"prediction backend comparison already exists: {destination}"
        )
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{result.comparison_id}.", dir=str(output_root))
    )
    try:
        _write_json(temporary / "comparison.json", result.to_primitive())
        _write_rows(
            temporary / "indicator_fields.csv",
            [
                {
                    "indicator_name": indicator.definition.name,
                    **field.summary_row(),
                }
                for indicator in result.indicator_comparisons
                for field in indicator.field_comparisons
            ],
            fieldnames=_INDICATOR_FIELDNAMES,
        )
        _write_rows(
            temporary / "indicator_divergences.csv",
            [
                {
                    "indicator_name": indicator.definition.name,
                    **divergence.to_primitive(),
                }
                for indicator in result.indicator_comparisons
                for field in indicator.field_comparisons
                for divergence in field.divergences
            ],
            fieldnames=_INDICATOR_DIVERGENCE_FIELDNAMES,
        )
        _write_rows(
            temporary / "prediction_changes.csv",
            [
                item.to_primitive()
                for item in result.prediction_comparison.signal_comparisons
            ],
            fieldnames=_PREDICTION_CHANGE_FIELDNAMES,
        )
        _write_rows(
            temporary / "prediction_metrics.csv",
            [item.to_primitive() for item in result.prediction_comparison.metrics],
            fieldnames=_PREDICTION_METRIC_FIELDNAMES,
        )
        _write_text(temporary / "summary.txt", result.human_summary())
        os.rename(temporary, destination)
    except PredictionExportError:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    except (OSError, TypeError, ValueError) as error:
        shutil.rmtree(temporary, ignore_errors=True)
        raise PredictionExportError(
            "failed to export immutable prediction backend comparison"
        ) from error
    return destination


def validate_overnight_gap_backend_comparison_export(
    result: OvernightGapBackendComparisonResult, path: Path
) -> Path:
    """Require an existing end-to-end export to match regenerated bytes."""
    try:
        if path.name != result.comparison_id or not path.is_dir():
            raise PredictionExportError(
                "prediction backend export does not match the expected result"
            )
        entries = {entry.name: entry for entry in path.iterdir()}
        if set(entries) != set(PREDICTION_BACKEND_COMPARISON_ARTIFACT_FILENAMES):
            raise PredictionExportError(
                "prediction backend export does not match the expected result"
            )
        with tempfile.TemporaryDirectory(
            prefix="quantforge-prediction-backend-validation-"
        ) as temporary_root:
            expected = export_overnight_gap_backend_comparison(
                result, Path(temporary_root)
            )
            if any(
                entries[name].read_bytes() != (expected / name).read_bytes()
                for name in PREDICTION_BACKEND_COMPARISON_ARTIFACT_FILENAMES
            ):
                raise PredictionExportError(
                    "prediction backend export does not match the expected result"
                )
    except PredictionExportError:
        raise
    except OSError as error:
        raise PredictionExportError(
            "failed to validate immutable prediction backend export"
        ) from error
    return path


def _signals_by_session(
    signals: tuple[PredictionRow, ...] | tuple[PredictionSignal, ...],
    backend_id: str,
    analysis: PredictionAnalysisResult,
) -> dict[date, PredictionRow | PredictionSignal]:
    if any(
        signal.strategy_id != analysis.strategy_id
        or signal.strategy_implementation_version
        != analysis.strategy_implementation_version
        or signal.strategy_configuration_id != analysis.strategy_configuration_id
        for signal in signals
    ):
        raise InvalidPredictionOutputError(
            f"prediction signals do not match the analysis identity: {backend_id}"
        )
    mapped = {signal.signal_session: signal for signal in signals}
    if len(mapped) != len(signals):
        raise InvalidPredictionOutputError(
            f"prediction backend emitted duplicate signal sessions: {backend_id}"
        )
    return mapped


def _average_signed_return(rows: tuple[PredictionRow, ...]) -> Decimal | None:
    if not rows:
        return None
    with localcontext(_ARITHMETIC_CONTEXT):
        return sum(
            (row.signed_prediction_return for row in rows), Decimal(0)
        ) / Decimal(len(rows))


def _difference(value_a: Decimal | None, value_b: Decimal | None) -> Decimal | None:
    if value_a is None or value_b is None:
        return None
    with localcontext(_ARITHMETIC_CONTEXT):
        return value_b - value_a


def _optional_decimal(value: Decimal | None) -> str | None:
    return None if value is None else decimal_to_primitive(value)


def _optional_direction(value: PredictionDirection | None) -> str | None:
    return None if value is None else value.value


_INDICATOR_SUMMARY_FIELDS = (
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
_INDICATOR_FIELDNAMES = ("indicator_name", *_INDICATOR_SUMMARY_FIELDS)
_INDICATOR_DIVERGENCE_FIELDNAMES = (
    "indicator_name",
    "output_name",
    "timestamp",
    "backend_a_value",
    "backend_b_value",
    "absolute_difference",
    "relative_difference",
)
_PREDICTION_CHANGE_FIELDNAMES = (
    "signal_session",
    "status",
    "backend_a_direction",
    "backend_b_direction",
    "backend_a_reason",
    "backend_b_reason",
)
_PREDICTION_METRIC_FIELDNAMES = (
    "metric_name",
    "backend_a_value",
    "backend_b_value",
    "difference_backend_b_minus_a",
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
        raise PredictionExportError(
            f"prediction backend rows have inconsistent schema: {path.name}"
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
    "PREDICTION_BACKEND_COMPARISON_ARTIFACT_FILENAMES",
    "PREDICTION_BACKEND_COMPARISON_ENGINE_VERSION",
    "PREDICTION_BACKEND_COMPARISON_SCHEMA_VERSION",
    "OvernightGapBackendComparisonResult",
    "PredictionBackendComparison",
    "PredictionMetricDifference",
    "PredictionSignalComparison",
    "PredictionSignalComparisonStatus",
    "compare_prediction_backends",
    "export_overnight_gap_backend_comparison",
    "run_overnight_gap_backend_comparison",
    "validate_overnight_gap_backend_comparison_export",
]
