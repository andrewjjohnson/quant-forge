"""Primitive QF-40 schema checks; never rebuild aggregate statistics."""

from decimal import Decimal, InvalidOperation
from typing import cast

from quantforge.configuration import Primitive, PrimitiveMapping, configuration_identity
from quantforge.experiments._json import ManifestError, mapping, text
from quantforge.oos.models import OOSSource


def record(value: object, keys: set[str], label: str) -> PrimitiveMapping:
    if not isinstance(value, dict) or set(cast(dict[object, object], value)) != keys:
        raise ManifestError(f"OOS aggregate {label} fields differ from producer schema")
    return cast(PrimitiveMapping, value)


def records(value: object) -> list[PrimitiveMapping]:
    if not isinstance(value, list):
        raise ManifestError("OOS aggregate records must be arrays")
    return [mapping(item) for item in cast(list[object], value)]


def counter(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ManifestError("OOS aggregate counts must be nonnegative integers")
    return value


def decimal_field(
    value: object,
    *,
    nullable: bool = False,
    minimum: int | None = None,
    maximum: int | None = None,
) -> None:
    if value is None and nullable:
        return
    if not isinstance(value, str):
        raise ManifestError("OOS aggregate decimals must be strings")
    try:
        number = Decimal(value)
    except InvalidOperation as error:
        raise ManifestError("OOS aggregate decimal is invalid") from error
    if (
        not number.is_finite()
        or (minimum is not None and number < minimum)
        or (maximum is not None and number > maximum)
    ):
        raise ManifestError("OOS aggregate decimal is outside its domain")


def strings(value: object) -> None:
    if not isinstance(value, list):
        raise ManifestError("OOS aggregate text records must be arrays")
    for item in cast(list[object], value):
        text(item)


def validate_equity_rows(value: object) -> None:
    for row in records(value):
        record(
            row,
            {
                "fold_id",
                "run_id",
                "session",
                "timestamp_semantics",
                "window_start",
                "window_start_index",
                "equity_index",
                "benchmark_index",
                "drawdown",
                "benchmark_drawdown",
            },
            "normalized-equity row",
        )
        for field in ("fold_id", "run_id", "session"):
            text(row[field])
        if (
            row["timestamp_semantics"] != "exchange_session_close"
            or type(row["window_start"]) is not bool
        ):
            raise ManifestError("OOS aggregate equity timing is invalid")
        for field in ("window_start_index", "equity_index", "benchmark_index"):
            decimal_field(row[field], minimum=0)
        for field in ("drawdown", "benchmark_drawdown"):
            decimal_field(row[field], minimum=-1, maximum=0)


def validate_configuration_stability(value: object, source: OOSSource) -> None:
    stability = record(
        value,
        {
            "windows",
            "comparable_transitions",
            "configuration_changes",
            "configuration_change_frequency",
            "repeat_selections",
            "repeat_selection_frequency",
            "selection_counts",
            "parameter_change_counts",
            "interpretation",
        },
        "configuration stability",
    )
    transitions = counter(stability["comparable_transitions"])
    changes = counter(stability["configuration_changes"])
    repeats = counter(stability["repeat_selections"])
    if changes + repeats != transitions:
        raise ManifestError(
            "OOS aggregate stability transition counts are inconsistent"
        )
    for field in ("configuration_change_frequency", "repeat_selection_frequency"):
        if transitions == 0 and stability[field] is not None:
            raise ManifestError("OOS aggregate stability frequency must be unavailable")
        decimal_field(stability[field], nullable=transitions == 0, minimum=0, maximum=1)
    for field in ("selection_counts", "parameter_change_counts"):
        for name, count in mapping(stability[field]).items():
            text(name)
            counter(count)
    if stability["interpretation"] != "descriptive; no automatic quality judgement":
        raise ManifestError("OOS aggregate stability interpretation is invalid")
    windows = records(stability["windows"])
    if len(windows) != len(source.folds):
        raise ManifestError(
            "OOS aggregate stability windows differ from captured folds"
        )
    for window, fold in zip(windows, source.folds, strict=True):
        # These are copied frozen records, not newly computed turnover statistics.
        selected = (
            None if fold.selection is None else fold.selection.snapshot.to_primitive()
        )
        candidate = None if selected is None else mapping(selected["candidate"])
        evidence = (
            []
            if selected is None
            else [
                item
                for item in records(
                    mapping(selected["selection_evidence"])["stability"]
                )
                if item["trial_id"] == selected["selected_trial_id"]
            ]
        )
        expected: PrimitiveMapping = {
            "fold_id": fold.fold_id,
            "status": fold.status.value,
            "selection_id": None
            if fold.selection is None
            else fold.selection.selection_id,
            "candidate": candidate,
            "neighborhood_evidence": cast(list[Primitive], evidence),
        }
        if configuration_identity(window) != configuration_identity(expected):
            raise ManifestError(
                "OOS aggregate stability window differs from captured selection"
            )
