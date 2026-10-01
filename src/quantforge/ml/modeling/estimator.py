"""One L2 logistic-regression fit and a transparent binary64 scorer (QF-68).

scikit-learn is used only to *fit* the coefficients (``lbfgs``, one BLAS
thread). The fitted intercept and coefficients are persisted as plain floats,
and every probability (selection, out-of-sample, holdout and offline
reproduction) is computed here in pure Python from that persisted state:

    linear = fsum([intercept, coefficient_1 * z_1, ..., coefficient_k * z_k])
    p = 1 / (1 + exp(-linear))            if linear >= 0
    p = exp(linear) / (1 + exp(linear))   otherwise (no overflow)

``fsum`` is correctly rounded and order independent; ``exp`` is the platform
libm, which may differ by one unit in the last place across platforms. Reading
a model never deserializes a library object.
"""

import math
import warnings
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from importlib import import_module
from typing import Protocol, cast

from quantforge.configuration import (
    Primitive,
    PrimitiveMapping,
    configuration_identity,
)
from quantforge.ml.modeling.configuration import LogisticRegressionConfiguration
from quantforge.ml.modeling.errors import ModelArtifactIntegrityError, ModelInputError
from quantforge.ml.modeling.preprocessing import finite
from quantforge.ml.modeling.status import TrainingLimitationError, TrainingStatus


class _Array(Protocol):
    def tolist(self) -> object: ...


class _Classifier(Protocol):
    coef_: _Array
    intercept_: _Array
    n_iter_: _Array
    classes_: _Array

    def fit(self, design: list[list[float]], labels: list[int]) -> object: ...


class _LinearModels(Protocol):
    LogisticRegression: Callable[..., _Classifier]


class _ThreadPools(Protocol):
    threadpool_limits: Callable[..., AbstractContextManager[object]]


@dataclass(frozen=True, slots=True)
class LinearModelState:
    """Fitted logistic-regression parameters in model column order.

    ``coefficients`` apply to standardized features; a feature constant in
    training has a coefficient of exactly ``0.0``.
    """

    features: tuple[str, ...]
    intercept: float
    coefficients: tuple[float, ...]
    iterations: int

    def probability(self, values: Sequence[float], label: str) -> float:
        """Positive-class probability of one standardized row (binary64)."""
        try:
            linear = math.fsum(
                [
                    self.intercept,
                    *(
                        coefficient * value
                        for coefficient, value in zip(
                            self.coefficients, values, strict=True
                        )
                    ),
                ]
            )
        except (OverflowError, ValueError) as error:
            raise ModelInputError(f"{label} linear score overflows") from error
        if not math.isfinite(linear):
            raise ModelInputError(f"{label} linear score is not finite")
        if linear >= 0:
            return 1.0 / (1.0 + math.exp(-linear))
        exponential = math.exp(linear)
        return exponential / (1.0 + exponential)

    def to_primitive(self) -> PrimitiveMapping:
        return {
            "kind": "logistic_regression",
            "link": "logit",
            "probability_formula": "1 / (1 + exp(-(intercept + sum(coef * z))))",
            "input": "standardized_features_in_model_column_order",
            "intercept": self.intercept,
            "coefficients": [
                {"feature": name, "coefficient": value}
                for name, value in zip(self.features, self.coefficients, strict=True)
            ],
            "iterations": self.iterations,
        }

    @property
    def state_id(self) -> str:
        return configuration_identity(self.to_primitive())

    @classmethod
    def from_primitive(cls, value: Primitive) -> "LinearModelState":
        if not isinstance(value, dict):
            raise ModelArtifactIntegrityError("estimator state must be a mapping")
        try:
            entries = cast(list[PrimitiveMapping], value["coefficients"])
            state = cls(
                features=tuple(cast(str, item["feature"]) for item in entries),
                intercept=finite(value["intercept"], "intercept"),
                coefficients=tuple(
                    finite(item["coefficient"], "coefficient") for item in entries
                ),
                iterations=cast(int, value["iterations"]),
            )
        except (KeyError, TypeError) as error:
            raise ModelArtifactIntegrityError("invalid estimator state") from error
        if (
            state.to_primitive() != value
            or not state.features
            or type(value["iterations"]) is not int
        ):
            raise ModelArtifactIntegrityError("estimator state is inconsistent")
        return state


def _floats(value: object, label: str) -> list[float]:
    if not isinstance(value, list) or not all(
        type(item) is float for item in cast(list[object], value)
    ):
        raise TrainingLimitationError(
            TrainingStatus.ESTIMATOR_FAILED, f"{label} is invalid"
        )
    return cast(list[float], value)


def fit_logistic_regression(
    features: tuple[str, ...],
    informative: tuple[int, ...],
    design: list[list[float]],
    labels: list[bool],
    configuration: LogisticRegressionConfiguration,
) -> LinearModelState:
    """Fit on standardized informative columns; never substitute an estimator."""
    linear_models = cast(_LinearModels, import_module("sklearn.linear_model"))
    convergence = cast(
        type[Warning],
        getattr(import_module("sklearn.exceptions"), "ConvergenceWarning"),
    )
    thread_pools = cast(_ThreadPools, import_module("threadpoolctl"))
    estimator = linear_models.LogisticRegression(
        C=float(configuration.inverse_regularization),
        l1_ratio=0.0,
        tol=float(configuration.tolerance),
        max_iter=configuration.maximum_iterations,
        solver="lbfgs",
        fit_intercept=True,
        class_weight=None,
        random_state=configuration.random_seed,
    )
    matrix = [[row[index] for index in informative] for row in design]
    with (
        warnings.catch_warnings(record=True) as caught,
        thread_pools.threadpool_limits(limits=1),
    ):
        warnings.simplefilter("always")
        try:
            estimator.fit(matrix, [int(label) for label in labels])
        except (ValueError, ArithmeticError) as error:
            raise TrainingLimitationError(
                TrainingStatus.ESTIMATOR_FAILED, f"logistic regression failed: {error}"
            ) from error
    not_converged = [item for item in caught if issubclass(item.category, convergence)]
    for item in caught:
        if item not in not_converged:
            warnings.warn_explicit(
                item.message, item.category, item.filename, item.lineno
            )
    iterations = estimator.n_iter_.tolist()
    if not isinstance(iterations, list) or len(cast(list[object], iterations)) != 1:
        raise TrainingLimitationError(
            TrainingStatus.ESTIMATOR_FAILED, "iterations are invalid"
        )
    count = cast(list[object], iterations)[0]
    if type(count) is not int:
        raise TrainingLimitationError(
            TrainingStatus.ESTIMATOR_FAILED, "iterations are invalid"
        )
    if not_converged or count >= configuration.maximum_iterations:
        raise TrainingLimitationError(
            TrainingStatus.ESTIMATOR_DID_NOT_CONVERGE,
            f"lbfgs did not converge within {configuration.maximum_iterations} "
            "iterations",
        )
    if estimator.classes_.tolist() != [0, 1]:
        raise TrainingLimitationError(
            TrainingStatus.ESTIMATOR_FAILED, "classes are invalid"
        )
    rows = estimator.coef_.tolist()
    if not isinstance(rows, list) or len(cast(list[object], rows)) != 1:
        raise TrainingLimitationError(
            TrainingStatus.ESTIMATOR_FAILED, "coefficients invalid"
        )
    fitted = _floats(cast(list[object], rows)[0], "coefficients")
    intercepts = _floats(estimator.intercept_.tolist(), "intercept")
    if len(fitted) != len(informative) or len(intercepts) != 1:
        raise TrainingLimitationError(
            TrainingStatus.ESTIMATOR_FAILED, "coefficients invalid"
        )
    if not all(math.isfinite(value) for value in (*fitted, intercepts[0])):
        raise TrainingLimitationError(
            TrainingStatus.ESTIMATOR_FAILED, "fitted parameters are not finite"
        )
    coefficients = [0.0] * len(features)
    for index, value in zip(informative, fitted, strict=True):
        coefficients[index] = value
    return LinearModelState(features, intercepts[0], tuple(coefficients), count)


__all__ = ["LinearModelState", "fit_logistic_regression"]
