"""QF-68 training contracts: permitted rows only, chronology, bindings, limits.

Synthetic QF-67 datasets bound to the real QF-8 timestamp plan fixture (see
``model_fixtures``). Every dataset is exported and re-read by QF-67.
"""

import math
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from importlib import import_module
from pathlib import Path
from typing import Any

import pytest

from quantforge.ml import (
    EventDatasetIntegrityError,
    ForwardReturnBinaryTarget,
    NonAuthoritativeSourceError,
    read_event_dataset,
)
from quantforge.ml.modeling import (
    EventModelState,
    LogisticRegressionConfiguration,
    MissingFeaturePolicy,
    ModelBindingError,
    ModelConfigurationError,
    ModelInputError,
    ModelLeakageError,
    PreprocessingConfiguration,
    TrainingResult,
    TrainingStatus,
    fit_event_model,
)
from quantforge.validation import PartitionRole, TimestampBoundary, ValidationPlan
from tests.unit.ml.model_fixtures import (
    TRAINING_CLOCKS,
    Event,
    change,
    configuration,
    events,
    plan_fixture,
    standard_events,
    write_dataset,
)

linear_model: Any = import_module("sklearn.linear_model")


@pytest.fixture(scope="module")
def plan(tmp_path_factory: pytest.TempPathFactory) -> ValidationPlan:
    return plan_fixture(tmp_path_factory.mktemp("qf68-plan"))


def fit(
    plan: ValidationPlan, dataset: Path, *, fold: int = 0, **options: object
) -> TrainingResult:
    return fit_event_model(
        dataset,
        plan=plan,
        fold_id=plan.folds[fold].fold_id,
        configuration=configuration(**options),
    )


def fitted(plan: ValidationPlan, dataset: Path, **options: object) -> TrainingResult:
    result = fit_event_model(
        dataset,
        plan=plan,
        fold_id=plan.folds[0].fold_id,
        configuration=configuration(**options),
    )
    assert result.status is TrainingStatus.FITTED, result.detail
    assert result.model is not None
    return result


def state_of(result: TrainingResult) -> EventModelState:
    assert result.model is not None, result.detail
    return result.model.state


def test_only_labeled_development_rows_of_the_fold_fit(
    plan: ValidationPlan, tmp_path: Path
) -> None:
    items = [*standard_events(0, holdout=True), *standard_events(1)]
    dataset = write_dataset(plan, items, tmp_path)
    result = fitted(plan, dataset)
    assert result.model is not None
    state = result.model.state
    loaded = read_event_dataset(dataset)
    development = loaded.rows_for(
        PartitionRole.DEVELOPMENT, fold_id=plan.folds[0].fold_id
    )
    assert state.membership.fitting_observation_ids == tuple(
        loaded.rows[index].source_observation_id for index in development
    )
    assert state.membership.training_rows == len(development) == 34
    assert state.membership.exclusions == ()
    # Preprocessing statistics equal the training rows' own statistics.
    numeric = loaded.numeric_feature_rows()
    momentum = [numeric[index][0] for index in development]
    assert all(value is not None for value in momentum)
    values = [float(value) for value in momentum if value is not None]
    center = math.fsum(values) / len(values)
    scale = math.sqrt(math.fsum((v - center) ** 2 for v in values) / len(values))
    transform = state.preprocessing.features[0]
    assert (transform.center, transform.scale) == (center, scale)
    labels = [loaded.rows[index].label.value for index in development]
    assert state.baseline_probability == labels.count(True) / len(labels)
    chronology = state.chronology
    assert chronology.fold_id == plan.folds[0].fold_id
    protected = plan.folds[0].next_protected_window.interval.start
    assert isinstance(protected, TimestampBoundary)
    assert chronology.protected_start == protected.timestamp


def test_values_outside_training_never_change_fitted_state(
    plan: ValidationPlan, tmp_path: Path
) -> None:
    base = fitted(plan, write_dataset(plan, standard_events(0), tmp_path / "a"))
    altered = standard_events(0)
    for role in (PartitionRole.SELECTION, PartitionRole.WALK_FORWARD_TEST):
        altered = change(altered, role, features=("9", "40", 99, True), label=False)
    altered += events(PartitionRole.FINAL_HOLDOUT, None, ("14:00", "15:00"))
    altered += standard_events(1)
    other = fitted(plan, write_dataset(plan, altered, tmp_path / "b"))
    left, right = state_of(base), state_of(other)
    assert left.bound.dataset.dataset_id != right.bound.dataset.dataset_id
    assert left.fitted_state_id == right.fitted_state_id
    assert left.preprocessing.state_id == right.preprocessing.state_id
    assert left.estimator.state_id == right.estimator.state_id
    assert left.model_id != right.model_id  # bound to another dataset
    # A single development value does change it.
    first = standard_events(0)
    first[0] = replace(first[0], features=("0.7", *first[0].features[1:]))
    changed = state_of(fitted(plan, write_dataset(plan, first, tmp_path / "c")))
    assert changed.preprocessing.state_id != left.preprocessing.state_id


def test_outcome_reach_and_embargo_fail_closed(
    plan: ValidationPlan, tmp_path: Path
) -> None:
    # 11:25 + 35m reaches the 12:00 selection start: equality is leakage.
    late = [
        *standard_events(0),
        Event(PartitionRole.DEVELOPMENT, 0, "11:25", ("0.1", "1", 1, False), True),
    ]
    with pytest.raises(ModelLeakageError, match="purge rule"):
        fit(plan, write_dataset(plan, late, tmp_path / "late"))
    # With a 10 minute embargo the 11:20 rows are no longer permitted either.
    embargoed = plan_fixture(tmp_path / "embargo-plan", embargo=timedelta(minutes=10))
    with pytest.raises(ModelLeakageError, match="embargo"):
        fit(embargoed, write_dataset(embargoed, standard_events(0), tmp_path / "e"))
    early = [
        item
        for item in standard_events(0)
        if item.role is not PartitionRole.DEVELOPMENT or item.clock <= "11:10"
    ]
    assert fit(embargoed, write_dataset(embargoed, early, tmp_path / "ok")).fitted
    # A target reaching beyond the plan's purged label horizon is refused.
    with pytest.raises(ModelLeakageError, match="label horizon"):
        fit(
            plan,
            write_dataset(
                plan,
                standard_events(0),
                tmp_path / "reach",
                outcome_reach=timedelta(minutes=40),
            ),
        )


@pytest.mark.parametrize(
    ("role", "clock"),
    [
        (PartitionRole.DEVELOPMENT, "12:05"),  # development role in selection time
        (PartitionRole.SELECTION, "13:05"),  # selection role in test time
        (PartitionRole.WALK_FORWARD_TEST, "14:00"),  # test role after its window
        (PartitionRole.DEVELOPMENT, "09:55"),  # before the development window
    ],
)
def test_partition_boundary_violations_fail_closed(
    plan: ValidationPlan, tmp_path: Path, role: PartitionRole, clock: str
) -> None:
    items = [
        *standard_events(0),
        Event(role, 0, clock, ("0.1", "1", 1, False), True, 5),
    ]
    with pytest.raises(ModelLeakageError):
        fit(plan, write_dataset(plan, items, tmp_path))


def test_plan_fold_and_window_bindings_are_enforced(
    plan: ValidationPlan, tmp_path: Path
) -> None:
    dataset = write_dataset(plan, standard_events(0), tmp_path / "a")
    other_plan = plan_fixture(tmp_path / "other", embargo=timedelta(minutes=1))
    with pytest.raises(ModelBindingError, match="another QF-8 plan"):
        fit(other_plan, dataset)
    with pytest.raises(ModelBindingError, match="not a fold"):
        fit_event_model(
            dataset, plan=plan, fold_id="unknown", configuration=configuration()
        )
    with pytest.raises(ModelConfigurationError):
        fit_event_model(
            dataset,
            plan=plan,
            fold_id=plan.folds[0].fold_id,
            configuration=object(),  # pyright: ignore[reportArgumentType]
        )
    swapped = write_dataset(
        plan,
        standard_events(0),
        tmp_path / "b",
        window_overrides={(0, PartitionRole.SELECTION): plan.folds[0].test.window_id},
    )
    with pytest.raises(ModelBindingError, match="selection"):
        fit(plan, swapped)
    # A dataset directory that fails QF-67 validation never reaches training.
    corrupt = tmp_path / "corrupt"
    corrupt.mkdir()
    (corrupt / "manifest.json").write_text("{}\n")
    with pytest.raises(EventDatasetIntegrityError):
        fit(plan, corrupt)
    with pytest.raises(NonAuthoritativeSourceError):
        fit(plan, tmp_path / "scan.rapid.json")


def test_unavailable_labels_are_excluded_never_negative(
    plan: ValidationPlan, tmp_path: Path
) -> None:
    items = standard_events(0)
    unlabeled = {0, 5, 9}
    with_missing = [
        replace(item, label=None, status="dataset_end") if index in unlabeled else item
        for index, item in enumerate(items)
    ]
    result = fitted(plan, write_dataset(plan, with_missing, tmp_path / "a"))
    membership = result.membership
    assert membership.training_rows == 34
    assert len(membership.fitting) == 31
    assert [(e.reason, e.detail) for e in membership.exclusions] == [
        ("target_unavailable", "dataset_end")
    ] * 3
    labels = [
        item.label
        for index, item in enumerate(items)
        if index not in unlabeled and item.role is PartitionRole.DEVELOPMENT
    ]
    assert membership.positives == labels.count(True)
    assert membership.negatives == labels.count(False)
    # Identical fitted parameters to a dataset where those rows never existed.
    absent = [item for index, item in enumerate(items) if index not in unlabeled]
    reference = fitted(plan, write_dataset(plan, absent, tmp_path / "b"))
    assert state_of(result).estimator == state_of(reference).estimator
    assert state_of(result).preprocessing == state_of(reference).preprocessing


def test_null_features_follow_the_configured_policy(
    plan: ValidationPlan, tmp_path: Path
) -> None:
    items = standard_events(0)
    nulled = [
        replace(item, features=(None, *item.features[1:])) if index % 6 == 0 else item
        for index, item in enumerate(items)
    ]
    dataset = write_dataset(plan, nulled, tmp_path / "a")
    excluded = fitted(plan, dataset)
    reasons = {(e.reason, e.detail) for e in excluded.membership.exclusions}
    assert reasons == {("null_feature_value", "momentum")}
    assert all(
        t.imputation_value is None for t in state_of(excluded).preprocessing.features
    )
    imputing = PreprocessingConfiguration(MissingFeaturePolicy.IMPUTE_TRAINING_MEAN)
    imputed = fitted(plan, dataset, preprocessing=imputing)
    assert imputed.membership.exclusions == ()
    assert imputed.model is not None
    momentum = imputed.model.state.preprocessing.features[0]
    loaded = read_event_dataset(dataset)
    training = loaded.rows_for(PartitionRole.DEVELOPMENT)
    present = [
        value
        for value in (loaded.numeric_feature_rows()[index][0] for index in training)
        if value is not None
    ]
    assert momentum.imputation_value == math.fsum(present) / len(present)
    assert momentum.fitting_missing_values == 34 - len(present)
    # Nulls outside training cannot move the imputation value.
    later = change(
        nulled, PartitionRole.WALK_FORWARD_TEST, features=(None, None, None, None)
    )
    other = fitted(
        plan, write_dataset(plan, later, tmp_path / "b"), preprocessing=imputing
    )
    assert other.model is not None
    assert other.model.state.fitted_state_id == imputed.model.state.fitted_state_id
    # A feature with no training value at all.
    missing = change(items, PartitionRole.DEVELOPMENT, features=(None, "1", 1, False))
    empty = write_dataset(plan, missing, tmp_path / "c")
    assert fit(plan, empty, preprocessing=imputing).status is (
        TrainingStatus.FEATURE_MISSING_IN_TRAINING
    )
    assert fit(plan, empty).status is TrainingStatus.INSUFFICIENT_TRAINING_OBSERVATIONS


def test_constant_features_get_an_exact_zero_coefficient(
    plan: ValidationPlan, tmp_path: Path
) -> None:
    items = [
        replace(item, features=(*item.features[:3], False))
        if item.role is PartitionRole.DEVELOPMENT
        else item
        for item in standard_events(0)
    ]
    result = fitted(plan, write_dataset(plan, items, tmp_path / "a"))
    assert result.model is not None
    state = result.model.state
    gap = state.preprocessing.features[3]
    assert (gap.constant_in_training, gap.center, gap.scale) == (True, 0.0, 1.0)
    assert state.estimator.coefficients[3] == 0.0
    assert state.preprocessing.informative == (0, 1, 2)
    flat = change(items, PartitionRole.DEVELOPMENT, features=("0.1", "1", 2, True))
    assert fit(plan, write_dataset(plan, flat, tmp_path / "b")).status is (
        TrainingStatus.ALL_FEATURES_CONSTANT
    )


def test_degenerate_training_partitions_return_explicit_statuses(
    plan: ValidationPlan, tmp_path: Path
) -> None:
    single = change(standard_events(0), PartitionRole.DEVELOPMENT, label=True)
    result = fit(plan, write_dataset(plan, single, tmp_path / "a"))
    assert (result.status, result.model) == (
        TrainingStatus.SINGLE_CLASS_TRAINING_LABELS,
        None,
    )
    assert (
        fit(
            plan,
            write_dataset(plan, standard_events(0), tmp_path / "b"),
            minimum_training_observations=35,
        ).status
        is TrainingStatus.INSUFFICIENT_TRAINING_OBSERVATIONS
    )
    # QF-39 ran trials on selection, so development was never executed.
    no_training = [
        item
        for item in standard_events(0)
        if item.role is not PartitionRole.DEVELOPMENT
    ]
    exclusion = {
        "fold_id": plan.folds[0].fold_id,
        "role": PartitionRole.DEVELOPMENT.value,
        "reason": "role_not_executed_by_qf39_selection",
    }
    empty = fit(
        plan,
        write_dataset(plan, no_training, tmp_path / "c", excluded_sources=[exclusion]),
    )
    assert empty.status is TrainingStatus.NO_TRAINING_OBSERVATIONS
    assert "role_not_executed_by_qf39_selection" in empty.detail
    assert empty.to_primitive()["model_id"] is None
    # No silent fallback when the configured estimator cannot converge.
    stalled = fit(
        plan,
        write_dataset(plan, standard_events(0), tmp_path / "d"),
        estimator=LogisticRegressionConfiguration(
            tolerance=Decimal("0.000000000001"), maximum_iterations=1
        ),
    )
    assert (stalled.status, stalled.model) == (
        TrainingStatus.ESTIMATOR_DID_NOT_CONVERGE,
        None,
    )


def test_non_finite_numeric_inputs_fail_closed(
    plan: ValidationPlan, tmp_path: Path
) -> None:
    huge = "1" + "0" * 400  # a canonical finite decimal beyond binary64 range
    items = standard_events(0)
    items[0] = replace(items[0], features=(huge, *items[0].features[1:]))
    with pytest.raises(ModelInputError, match="finite"):
        fit(plan, write_dataset(plan, items, tmp_path))


def test_repeated_training_is_deterministic_and_matches_the_estimator(
    plan: ValidationPlan, tmp_path: Path
) -> None:
    dataset = write_dataset(plan, standard_events(0), tmp_path)
    first, second = fitted(plan, dataset), fitted(plan, dataset)
    assert state_of(first) == state_of(second)
    assert state_of(first).model_id == state_of(second).model_id
    assert first.model_configuration_id == second.model_configuration_id
    state = state_of(first)
    # The transparent scorer reproduces scikit-learn's own probabilities.
    loaded = read_event_dataset(dataset)
    rows = loaded.numeric_feature_rows()
    training = loaded.rows_for(PartitionRole.DEVELOPMENT)
    design = [state.preprocessing.transform(rows[index], "row") for index in training]
    matrix = [list(item.values) for item in design if item is not None]
    labels = [int(loaded.rows[index].label.value is True) for index in training]
    reference = linear_model.LogisticRegression(
        C=1.0, l1_ratio=0.0, tol=1e-8, max_iter=1000, solver="lbfgs", random_state=0
    ).fit(matrix, labels)
    expected = reference.predict_proba(matrix)[:, 1].tolist()
    actual = [state.estimator.probability(values, "row") for values in matrix]
    assert max(abs(a - b) for a, b in zip(actual, expected, strict=True)) < 1e-12


def test_each_fold_trains_only_on_its_own_development_rows(
    plan: ValidationPlan, tmp_path: Path
) -> None:
    both = write_dataset(
        plan, [*standard_events(0), *standard_events(1)], tmp_path / "both"
    )
    first = state_of(fitted(plan, both))
    later = state_of(fit(plan, both, fold=1))
    assert set(later.membership.fitting_observation_ids).isdisjoint(
        first.membership.fitting_observation_ids
    )
    assert later.chronology.fold_index == 1
    only_second = write_dataset(plan, standard_events(1), tmp_path / "second")
    alone = state_of(fit(plan, only_second, fold=1))
    assert alone.fitted_state_id == later.fitted_state_id
    only_first = state_of(
        fitted(plan, write_dataset(plan, standard_events(0), tmp_path / "first"))
    )
    assert only_first.fitted_state_id == first.fitted_state_id


def test_target_and_schema_are_bound_into_the_model_configuration(
    plan: ValidationPlan, tmp_path: Path
) -> None:
    base = fitted(plan, write_dataset(plan, standard_events(0), tmp_path / "a"))
    shifted = fitted(
        plan,
        write_dataset(
            plan,
            standard_events(0),
            tmp_path / "b",
            target=ForwardReturnBinaryTarget(threshold=Decimal("0.001")),
        ),
    )
    assert base.bound.dataset.target_configuration_id != (
        shifted.bound.dataset.target_configuration_id
    )
    assert base.model_configuration_id != shifted.model_configuration_id
    assert base.bound.dataset.feature_columns == (
        "momentum",
        "range_width",
        "bar_count",
        "gap_up",
    )
    primitive = base.bound.to_primitive()
    assert set(primitive) >= {
        "dataset",
        "feature_schema",
        "target",
        "chronology",
        "runtime",
    }
    assert len(TRAINING_CLOCKS) == 17
