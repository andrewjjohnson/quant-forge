"""Fixed, explicit event-model configuration and runtime identity (QF-68).

A ``ModelConfiguration`` is chosen before any data is read and is never tuned
here: one L2-regularized logistic regression (scikit-learn ``lbfgs``), one
preprocessing policy, fixed training/evaluation minimums, an optional fixed
class threshold and a fixed calibration binning. There is no hyperparameter or
threshold search.

Runtime versions that can change fitted coefficients (Python major.minor,
NumPy, SciPy, scikit-learn) are part of the bound model configuration identity.
The operating system and CPU are provenance only; see
``docs/event-ml-models.md`` for the cross-platform numerical contract.
"""

import platform
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from importlib import metadata
from typing import cast

from quantforge.configuration import (
    PrimitiveMapping,
    configuration_identity,
    decimal_to_primitive,
)
from quantforge.ml.modeling.errors import ModelConfigurationError

MODEL_CONFIGURATION_CONTRACT_VERSION = "1"
# Log loss clips probabilities to [epsilon, 1 - epsilon] (binary64).
LOG_LOSS_EPSILON = 1e-15
_MAXIMUM_SEED = 2**32 - 1


class MissingFeaturePolicy(StrEnum):
    """How null feature values are handled; never zero-filled."""

    # Exclude the observation from fitting; an evaluation row stays unscored.
    EXCLUDE_OBSERVATION = "exclude_observation"
    # Replace a null by the arithmetic mean of that feature over fitting rows.
    IMPUTE_TRAINING_MEAN = "impute_training_mean"


def _decimal(value: object, label: str) -> Decimal:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise ModelConfigurationError(f"{label} must be a finite Decimal")
    return value


def _positive_integer(value: object, label: str, *, minimum: int = 1) -> int:
    if type(value) is not int or value < minimum:
        raise ModelConfigurationError(f"{label} must be an integer >= {minimum}")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelConfigurationError(f"{label} must be nonempty text")
    return value


@dataclass(frozen=True, slots=True)
class LogisticRegressionConfiguration:
    """Fixed L2 logistic regression; the intercept is fitted and unpenalized.

    ``inverse_regularization`` is scikit-learn's ``C`` (larger is weaker
    regularization). ``lbfgs`` is deterministic for identical inputs and
    library versions; ``random_seed`` is still passed and recorded explicitly.
    """

    inverse_regularization: Decimal = Decimal(1)
    tolerance: Decimal = Decimal("0.00000001")
    maximum_iterations: int = 1000
    random_seed: int = 0

    def __post_init__(self) -> None:
        for value, label in (
            (self.inverse_regularization, "inverse regularization (C)"),
            (self.tolerance, "estimator tolerance"),
        ):
            if _decimal(value, label) <= 0:
                raise ModelConfigurationError(f"{label} must be positive")
        _positive_integer(self.maximum_iterations, "maximum iterations")
        _positive_integer(self.random_seed, "random seed", minimum=0)
        if self.random_seed > _MAXIMUM_SEED:
            raise ModelConfigurationError("random seed must fit in 32 bits")

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "estimator": "logistic_regression",
            "library": "scikit-learn",
            "implementation": "sklearn.linear_model.LogisticRegression",
            "solver": "lbfgs",
            "penalty": "l2",
            "l1_ratio": "0",
            "inverse_regularization_c": decimal_to_primitive(
                self.inverse_regularization
            ),
            "tolerance": decimal_to_primitive(self.tolerance),
            "maximum_iterations": self.maximum_iterations,
            "fit_intercept": True,
            "intercept_penalized": False,
            "class_weight": None,
            "random_seed": self.random_seed,
            "blas_threads_during_fit": 1,
            "output": "positive_class_probability",
            "positive_class": "target_true",
        }

    @classmethod
    def from_primitive(
        cls, value: PrimitiveMapping
    ) -> "LogisticRegressionConfiguration":
        try:
            configuration = cls(
                Decimal(cast(str, value["inverse_regularization_c"])),
                Decimal(cast(str, value["tolerance"])),
                cast(int, value["maximum_iterations"]),
                cast(int, value["random_seed"]),
            )
        except (KeyError, TypeError, InvalidOperation) as error:
            raise ModelConfigurationError("invalid estimator configuration") from error
        if configuration.to_primitive() != value:
            raise ModelConfigurationError("unsupported estimator configuration")
        return configuration


@dataclass(frozen=True, slots=True)
class PreprocessingConfiguration:
    """Training-only learned preprocessing; every rule below is fixed.

    - Scaling: ``(x - mean) / population_std`` over fitting rows.
    - Constant features (identical over fitting rows): center at that value,
      scale 1, coefficient fixed at exactly ``0.0``.
    - Non-finite numeric values (for example a decimal beyond binary64 range)
      are rejected; nothing is clipped or winsorized.
    """

    missing_features: MissingFeaturePolicy = MissingFeaturePolicy.EXCLUDE_OBSERVATION

    def __post_init__(self) -> None:
        if type(cast(object, self.missing_features)) is not MissingFeaturePolicy:
            raise ModelConfigurationError("missing-feature policy is unsupported")

    def to_primitive(self) -> PrimitiveMapping:
        imputing = self.missing_features is MissingFeaturePolicy.IMPUTE_TRAINING_MEAN
        return {
            "contract_version": "1",
            "numeric_view": "qf67_numeric_feature_rows_binary64",
            "missing_features": self.missing_features.value,
            "imputation": "training_mean_of_fitting_rows" if imputing else "none",
            "scaling": "standardize_fitting_mean_population_std",
            "constant_features": "center_only_coefficient_fixed_zero",
            "non_finite_values": "reject",
            "clipping": "none",
            "feature_selection": "none",
            "fit_rows": "fitting_observations_only",
        }

    @classmethod
    def from_primitive(cls, value: PrimitiveMapping) -> "PreprocessingConfiguration":
        try:
            configuration = cls(
                MissingFeaturePolicy(cast(str, value["missing_features"]))
            )
        except (KeyError, ValueError) as error:
            raise ModelConfigurationError(
                "invalid preprocessing configuration"
            ) from error
        if configuration.to_primitive() != value:
            raise ModelConfigurationError("unsupported preprocessing configuration")
        return configuration


@dataclass(frozen=True, slots=True)
class ModelConfiguration:
    """The complete fixed scientific choice, before binding to any dataset.

    ``class_threshold``, when set, is frozen here before fitting: a scored
    observation's class is ``probability >= threshold``. It is never searched.
    ``minimum_training_observations`` counts fitting rows (training rows with
    an available label and usable features). Metrics need at least
    ``minimum_evaluation_observations`` labeled scored rows.
    """

    name: str
    version: str
    estimator: LogisticRegressionConfiguration = field(
        default_factory=LogisticRegressionConfiguration
    )
    preprocessing: PreprocessingConfiguration = field(
        default_factory=PreprocessingConfiguration
    )
    minimum_training_observations: int = 20
    minimum_evaluation_observations: int = 1
    class_threshold: Decimal | None = None
    calibration_bins: int = 10

    def __post_init__(self) -> None:
        _text(self.name, "model configuration name")
        _text(self.version, "model configuration version")
        if type(cast(object, self.estimator)) is not LogisticRegressionConfiguration:
            raise ModelConfigurationError("estimator configuration is unsupported")
        if type(cast(object, self.preprocessing)) is not PreprocessingConfiguration:
            raise ModelConfigurationError("preprocessing configuration is unsupported")
        _positive_integer(
            self.minimum_training_observations,
            "minimum training observations",
            minimum=2,
        )
        _positive_integer(
            self.minimum_evaluation_observations, "minimum evaluation observations"
        )
        _positive_integer(self.calibration_bins, "calibration bins")
        if self.class_threshold is not None and not (
            0 < _decimal(self.class_threshold, "class threshold") < 1
        ):
            raise ModelConfigurationError("class threshold must be inside (0, 1)")

    @property
    def threshold(self) -> float | None:
        """The frozen threshold as binary64 (correctly rounded), if any."""
        return None if self.class_threshold is None else float(self.class_threshold)

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "contract_version": MODEL_CONFIGURATION_CONTRACT_VERSION,
            "name": self.name,
            "version": self.version,
            "estimator": self.estimator.to_primitive(),
            "preprocessing": self.preprocessing.to_primitive(),
            "training": {
                "training_role": "development_training",
                "fold_binding": "one_qf8_fold_per_model",
                "minimum_training_observations": self.minimum_training_observations,
                "unavailable_labels": "excluded_from_fitting_never_negative",
                "selection": "none_fixed_hyperparameters",
            },
            "evaluation": {
                "probability": "positive_class_binary64",
                "class_threshold": None
                if self.class_threshold is None
                else decimal_to_primitive(self.class_threshold),
                "threshold_rule": "probability_greater_than_or_equal",
                "threshold_selection": "fixed_before_fitting",
                "minimum_evaluation_observations": (
                    self.minimum_evaluation_observations
                ),
                "calibration_bins": self.calibration_bins,
                "calibration_binning": "equal_width_on_unit_interval",
                "log_loss_probability_clip": repr(LOG_LOSS_EPSILON),
                "baseline": "constant_training_prevalence",
            },
        }

    @property
    def configuration_id(self) -> str:
        return configuration_identity(self.to_primitive())

    @classmethod
    def from_primitive(cls, value: PrimitiveMapping) -> "ModelConfiguration":
        try:
            training = cast(PrimitiveMapping, value["training"])
            evaluation = cast(PrimitiveMapping, value["evaluation"])
            threshold = evaluation["class_threshold"]
            configuration = cls(
                name=cast(str, value["name"]),
                version=cast(str, value["version"]),
                estimator=LogisticRegressionConfiguration.from_primitive(
                    cast(PrimitiveMapping, value["estimator"])
                ),
                preprocessing=PreprocessingConfiguration.from_primitive(
                    cast(PrimitiveMapping, value["preprocessing"])
                ),
                minimum_training_observations=cast(
                    int, training["minimum_training_observations"]
                ),
                minimum_evaluation_observations=cast(
                    int, evaluation["minimum_evaluation_observations"]
                ),
                class_threshold=None
                if threshold is None
                else Decimal(cast(str, threshold)),
                calibration_bins=cast(int, evaluation["calibration_bins"]),
            )
        except (KeyError, TypeError, InvalidOperation) as error:
            raise ModelConfigurationError("invalid model configuration") from error
        if configuration.to_primitive() != value:
            raise ModelConfigurationError("unsupported model configuration")
        return configuration


def runtime_versions() -> PrimitiveMapping:
    """Library/runtime versions that can change fitted coefficients."""
    major, minor, _ = platform.python_version_tuple()
    return {
        "python": f"{platform.python_implementation()} {major}.{minor}",
        "numpy": metadata.version("numpy"),
        "scipy": metadata.version("scipy"),
        "scikit_learn": metadata.version("scikit-learn"),
    }


def runtime_platform() -> PrimitiveMapping:
    """Operational platform details; provenance only, never identity."""
    return {
        "python_version": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
    }


__all__ = [
    "LOG_LOSS_EPSILON",
    "MODEL_CONFIGURATION_CONTRACT_VERSION",
    "LogisticRegressionConfiguration",
    "MissingFeaturePolicy",
    "ModelConfiguration",
    "PreprocessingConfiguration",
    "runtime_platform",
    "runtime_versions",
]
