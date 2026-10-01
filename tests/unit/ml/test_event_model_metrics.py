"""QF-68 descriptive metrics, numerical conventions and configuration identity."""

import math
from decimal import Decimal
from importlib import import_module
from itertools import product
from typing import Any, cast

import pytest

from quantforge.configuration import PrimitiveMapping
from quantforge.ml.modeling import (
    LOG_LOSS_EPSILON,
    LogisticRegressionConfiguration,
    MissingFeaturePolicy,
    ModelConfiguration,
    ModelConfigurationError,
    PreprocessingConfiguration,
)
from quantforge.ml.modeling.configuration import runtime_versions
from quantforge.ml.modeling.metrics import (
    brier_score,
    describe_probabilities,
    log_loss,
    roc_auc,
)

metrics: Any = import_module("sklearn.metrics")
PROBABILITIES = [0.1, 0.4, 0.35, 0.8, 0.4, 0.65, 0.4, 0.95, 0.05]
LABELS = [False, False, True, True, True, False, False, True, False]


def brute_force_auc(probabilities: list[float], labels: list[bool]) -> float:
    positives = [p for p, y in zip(probabilities, labels, strict=True) if y]
    negatives = [p for p, y in zip(probabilities, labels, strict=True) if not y]
    wins = sum(
        1.0 if p > n else 0.5 if p == n else 0.0
        for p, n in product(positives, negatives)
    )
    return wins / (len(positives) * len(negatives))


def test_metrics_match_independent_references_including_ties() -> None:
    assert roc_auc(PROBABILITIES, LABELS) == pytest.approx(
        brute_force_auc(PROBABILITIES, LABELS), abs=1e-15
    )
    assert roc_auc(PROBABILITIES, LABELS) == pytest.approx(
        metrics.roc_auc_score(LABELS, PROBABILITIES), abs=1e-15
    )
    assert log_loss(PROBABILITIES, LABELS) == pytest.approx(
        metrics.log_loss(LABELS, PROBABILITIES), abs=1e-12
    )
    assert brier_score(PROBABILITIES, LABELS) == pytest.approx(
        metrics.brier_score_loss(LABELS, PROBABILITIES), abs=1e-15
    )
    # Perfectly wrong certain predictions are clipped, never infinite.
    clipped = log_loss([0.0, 1.0], [True, False])
    upper = 1.0 - LOG_LOSS_EPSILON  # binary64 clip, so 1 - upper != epsilon exactly
    assert clipped == (-math.log(LOG_LOSS_EPSILON) - math.log(1.0 - upper)) / 2
    assert math.isfinite(clipped)


def test_degenerate_partitions_stay_explicitly_undefined() -> None:
    empty = describe_probabilities([], [], threshold=None, bins=10, minimum=1)
    assert empty["labeled_observations"] == 0
    for name in ("log_loss", "brier_score", "roc_auc", "calibration"):
        assert empty[name] == {
            "status": "undefined",
            "reason": "no_labeled_observations",
        }
    small = describe_probabilities(
        [0.2, 0.7], [False, True], threshold=0.5, bins=10, minimum=3
    )
    assert small["log_loss"] == {
        "status": "undefined",
        "reason": "insufficient_labeled_observations",
    }
    assert small["classification"] == {
        "status": "undefined",
        "reason": "insufficient_labeled_observations",
    }
    single = describe_probabilities(
        [0.2, 0.7, 0.9], [True, True, True], threshold=0.5, bins=4, minimum=1
    )
    assert single["roc_auc"] == {"status": "undefined", "reason": "single_class_labels"}
    assert single["log_loss"]["status"] == "defined"  # pyright: ignore[reportIndexIssue,reportOptionalSubscript,reportArgumentType,reportCallIssue]
    classification = cast(PrimitiveMapping, single["classification"])
    assert classification["recall"] == {"status": "defined", "value": 2 / 3}
    no_positive_calls = describe_probabilities(
        [0.2, 0.3], [True, False], threshold=0.5, bins=4, minimum=1
    )
    calls = cast(PrimitiveMapping, no_positive_calls["classification"])
    assert calls["precision"] == {
        "status": "undefined",
        "reason": "no_predicted_positives",
    }
    no_actual = describe_probabilities(
        [0.2, 0.6], [False, False], threshold=0.5, bins=4, minimum=1
    )
    recall = cast(PrimitiveMapping, no_actual["classification"])["recall"]
    assert recall == {"status": "undefined", "reason": "no_actual_positives"}
    probability_only = describe_probabilities(
        [0.2, 0.6], [False, True], threshold=None, bins=4, minimum=1
    )
    assert probability_only["classification"] == {
        "status": "undefined",
        "reason": "no_frozen_threshold",
    }


def test_calibration_uses_fixed_equal_width_bins() -> None:
    summary = describe_probabilities(
        [0.0, 0.24, 0.25, 0.99, 1.0],
        [False, False, True, True, False],
        threshold=None,
        bins=4,
        minimum=1,
    )
    calibration = cast(PrimitiveMapping, summary["calibration"])
    bins = cast(list[PrimitiveMapping], calibration["bins"])
    assert [item["count"] for item in bins] == [2, 1, 0, 2]
    assert bins[2]["mean_probability"] is None
    assert bins[2]["observed_positive_rate"] is None
    assert bins[3]["observed_positive_rate"] == 0.5
    assert calibration["mean_probability_minus_observed_prevalence"] == pytest.approx(
        (0.0 + 0.24 + 0.25 + 0.99 + 1.0) / 5 - 2 / 5
    )


def test_configuration_identity_and_validation() -> None:
    base = ModelConfiguration("baseline", "1")
    assert base.configuration_id == ModelConfiguration("baseline", "1").configuration_id
    assert ModelConfiguration.from_primitive(base.to_primitive()) == base
    variants = (
        ModelConfiguration(
            "baseline",
            "1",
            LogisticRegressionConfiguration(inverse_regularization=Decimal("0.5")),
        ),
        ModelConfiguration(
            "baseline", "1", LogisticRegressionConfiguration(random_seed=7)
        ),
        ModelConfiguration(
            "baseline",
            "1",
            preprocessing=PreprocessingConfiguration(
                MissingFeaturePolicy.IMPUTE_TRAINING_MEAN
            ),
        ),
        ModelConfiguration("baseline", "1", class_threshold=Decimal("0.5")),
        ModelConfiguration("baseline", "1", minimum_training_observations=30),
        ModelConfiguration("baseline", "2"),
    )
    identities = {item.configuration_id for item in variants}
    assert len(identities) == len(variants)
    assert base.configuration_id not in identities
    for variant in variants:
        assert ModelConfiguration.from_primitive(variant.to_primitive()) == variant
    for bad in (
        lambda: LogisticRegressionConfiguration(inverse_regularization=Decimal(0)),
        lambda: LogisticRegressionConfiguration(tolerance=Decimal("NaN")),
        lambda: LogisticRegressionConfiguration(maximum_iterations=0),
        lambda: LogisticRegressionConfiguration(random_seed=-1),
        lambda: LogisticRegressionConfiguration(random_seed=cast(int, True)),
        lambda: ModelConfiguration("m", "1", class_threshold=Decimal(1)),
        lambda: ModelConfiguration("m", "1", class_threshold=cast(Decimal, 0.5)),
        lambda: ModelConfiguration("m", "1", minimum_training_observations=1),
        lambda: ModelConfiguration("", "1"),
        lambda: PreprocessingConfiguration(cast(MissingFeaturePolicy, "mean")),
    ):
        with pytest.raises(ModelConfigurationError):
            bad()
    tampered = base.to_primitive()
    tampered["name"] = "other"
    tampered["evaluation"] = {
        **cast(PrimitiveMapping, tampered["evaluation"]),
        "log_loss_probability_clip": "1e-7",
    }
    with pytest.raises(ModelConfigurationError):
        ModelConfiguration.from_primitive(tampered)


def test_material_runtime_versions_are_recorded() -> None:
    versions = runtime_versions()
    assert set(versions) == {"python", "numpy", "scipy", "scikit_learn"}
    assert str(versions["python"]).count(".") == 1  # major.minor only
