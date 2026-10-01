"""Transparent training-only preprocessing for event models (QF-68).

All learned state is fitted on the model's fitting rows only (development rows
of one fold with an available label and usable features) and persisted as
plain numbers:

- **Imputation** (``impute_training_mean`` only): each feature's arithmetic
  mean over its non-null fitting values replaces a null in any partition. A
  feature with no non-null fitting value cannot be imputed and fails training.
- **Scaling**: ``(x - center) / scale`` with ``center`` the fitting mean and
  ``scale`` the population standard deviation (``ddof = 0``).
- **Constant features**: a feature whose fitting values are all identical is
  centered at that exact value with ``scale = 1``; its coefficient is fixed at
  exactly ``0.0`` and it is not passed to the estimator.

Arithmetic is IEEE-754 binary64 in pure Python with ``math.fsum`` sums and
``math.sqrt``, all correctly rounded, so the fitted state is identical on every
platform for the same inputs. Non-finite values are rejected, never clipped.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

from quantforge.configuration import (
    Primitive,
    PrimitiveMapping,
    configuration_identity,
)
from quantforge.ml.modeling.configuration import (
    MissingFeaturePolicy,
    PreprocessingConfiguration,
)
from quantforge.ml.modeling.errors import ModelArtifactIntegrityError, ModelInputError
from quantforge.ml.modeling.status import TrainingLimitationError, TrainingStatus

type NumericRow = Sequence[float | None]


def finite(value: object, label: str) -> float:
    """A persisted float that is finite binary64 (never an int or bool)."""
    if type(value) is not float or not math.isfinite(value):
        raise ModelArtifactIntegrityError(f"{label} must be a finite float")
    return value


@dataclass(frozen=True, slots=True)
class FeatureTransform:
    """Fitted state of one feature column, in model column order."""

    name: str
    center: float
    scale: float
    constant_in_training: bool
    imputation_value: float | None
    fitting_observations: int
    fitting_missing_values: int

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "name": self.name,
            "center": self.center,
            "scale": self.scale,
            "constant_in_training": self.constant_in_training,
            "imputation_value": self.imputation_value,
            "fitting_observations": self.fitting_observations,
            "fitting_missing_values": self.fitting_missing_values,
        }

    @classmethod
    def from_primitive(cls, value: Primitive) -> "FeatureTransform":
        if not isinstance(value, dict):
            raise ModelArtifactIntegrityError("feature transform must be a mapping")
        try:
            imputation = value["imputation_value"]
            transform = cls(
                name=cast(str, value["name"]),
                center=finite(value["center"], "feature center"),
                scale=finite(value["scale"], "feature scale"),
                constant_in_training=cast(bool, value["constant_in_training"]),
                imputation_value=None
                if imputation is None
                else finite(imputation, "imputation value"),
                fitting_observations=cast(int, value["fitting_observations"]),
                fitting_missing_values=cast(int, value["fitting_missing_values"]),
            )
        except KeyError as error:
            raise ModelArtifactIntegrityError("invalid feature transform") from error
        if (
            transform.to_primitive() != value
            or type(value["constant_in_training"]) is not bool
            or transform.scale <= 0
            or (transform.constant_in_training and transform.scale != 1.0)
        ):
            raise ModelArtifactIntegrityError("feature transform is inconsistent")
        return transform


@dataclass(frozen=True, slots=True)
class TransformedRow:
    """Standardized model inputs of one scorable observation."""

    values: tuple[float, ...]
    imputed_features: int


@dataclass(frozen=True, slots=True)
class PreprocessingState:
    """Fitted, inspectable preprocessing state bound to one feature order."""

    configuration: PreprocessingConfiguration
    features: tuple[FeatureTransform, ...]

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.features)

    @property
    def informative(self) -> tuple[int, ...]:
        """Column positions passed to the estimator (non-constant features)."""
        return tuple(
            index
            for index, item in enumerate(self.features)
            if not item.constant_in_training
        )

    def transform(self, values: NumericRow, label: str) -> TransformedRow | None:
        """Standardize one row; ``None`` when a null cannot be scored."""
        if len(values) != len(self.features):
            raise ModelInputError(f"{label} does not match the model feature order")
        imputed = 0
        output: list[float] = []
        for raw, item in zip(values, self.features, strict=True):
            if raw is None:
                if item.imputation_value is None:
                    return None
                raw = item.imputation_value
                imputed += 1
            elif not math.isfinite(raw):
                raise ModelInputError(f"{label} feature {item.name!r} is not finite")
            standardized = (raw - item.center) / item.scale
            if not math.isfinite(standardized):
                raise ModelInputError(
                    f"{label} feature {item.name!r} is not finite after scaling"
                )
            output.append(standardized)
        return TransformedRow(tuple(output), imputed)

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "configuration": self.configuration.to_primitive(),
            "features": [item.to_primitive() for item in self.features],
        }

    @property
    def state_id(self) -> str:
        return configuration_identity(self.to_primitive())

    @classmethod
    def from_primitive(cls, value: Primitive) -> "PreprocessingState":
        if not isinstance(value, dict):
            raise ModelArtifactIntegrityError("preprocessing state must be a mapping")
        try:
            features = value["features"]
            state = cls(
                PreprocessingConfiguration.from_primitive(
                    cast(PrimitiveMapping, value["configuration"])
                ),
                tuple(
                    FeatureTransform.from_primitive(item)
                    for item in cast(list[Primitive], features)
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            if isinstance(error, ModelArtifactIntegrityError):
                raise
            raise ModelArtifactIntegrityError("invalid preprocessing state") from error
        imputing = (
            state.configuration.missing_features
            is MissingFeaturePolicy.IMPUTE_TRAINING_MEAN
        )
        if (
            state.to_primitive() != value
            or not state.features
            or len(set(state.columns)) != len(state.columns)
            or any(
                (item.imputation_value is not None) is not imputing
                for item in state.features
            )
        ):
            raise ModelArtifactIntegrityError("preprocessing state is inconsistent")
        return state


def _mean(values: Sequence[float], label: str) -> float:
    try:
        mean = math.fsum(values) / len(values)
    except OverflowError as error:
        raise ModelInputError(f"{label} overflows binary64") from error
    if not math.isfinite(mean):
        raise ModelInputError(f"{label} is not finite")
    return mean


def fit_preprocessing(
    columns: tuple[str, ...],
    rows: Sequence[NumericRow],
    configuration: PreprocessingConfiguration,
) -> PreprocessingState:
    """Fit on the fitting rows only; limitations raise ``TrainingLimitationError``."""
    imputing = (
        configuration.missing_features is MissingFeaturePolicy.IMPUTE_TRAINING_MEAN
    )
    transforms: list[FeatureTransform] = []
    for index, name in enumerate(columns):
        column = [row[index] for row in rows]
        present = [value for value in column if value is not None]
        missing = len(column) - len(present)
        if missing and not imputing:
            raise ModelInputError("excluded observations must not reach fitting")
        if not present:
            raise TrainingLimitationError(
                TrainingStatus.FEATURE_MISSING_IN_TRAINING,
                f"feature {name!r} has no non-null value among fitting rows",
            )
        if any(not math.isfinite(value) for value in present):
            raise ModelInputError(f"training feature {name!r} is not finite")
        imputation = _mean(present, f"feature {name!r} mean") if imputing else None
        values = (
            present
            if imputation is None
            else [imputation if value is None else value for value in column]
        )
        if all(value == values[0] for value in values):
            transforms.append(
                FeatureTransform(
                    name, values[0], 1.0, True, imputation, len(column), missing
                )
            )
            continue
        center = _mean(values, f"feature {name!r} mean")
        try:
            variance = math.fsum((value - center) ** 2 for value in values) / len(
                values
            )
        except OverflowError as error:
            raise ModelInputError(f"feature {name!r} variance overflows") from error
        scale = math.sqrt(variance)
        if not math.isfinite(scale) or scale <= 0:
            raise ModelInputError(
                f"feature {name!r} scale is not a positive finite binary64 value"
            )
        transforms.append(
            FeatureTransform(
                name, center, scale, False, imputation, len(column), missing
            )
        )
    state = PreprocessingState(configuration, tuple(transforms))
    if not state.informative:
        raise TrainingLimitationError(
            TrainingStatus.ALL_FEATURES_CONSTANT,
            "every feature is constant over the fitting rows",
        )
    return state


__all__ = [
    "FeatureTransform",
    "NumericRow",
    "PreprocessingState",
    "TransformedRow",
    "finite",
    "fit_preprocessing",
]
