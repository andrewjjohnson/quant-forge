"""QF-67 conditional event ML datasets from real QF-39 schema-4 studies, offline.

Synthetic QF-45-composition fixture (see ``event_dataset_fixtures``): one fold,
three EMA pairs on the selection window and the frozen pair on the test
window. Every pair triggers once per session; no-trigger decisions persist only
QF-64 coverage receipts. Nothing here is QF-45 research evidence.
"""

from collections.abc import Iterator, Sequence
from datetime import timedelta
from decimal import Decimal
from typing import cast

import pytest

from quantforge.configuration import PrimitiveMapping
from quantforge.data.prepared_canonical import canonical_preparation
from quantforge.examples.spy_ema_ml_dataset import (
    EMA_EVENT_FEATURES,
    ema_event_feature_schema,
)
from quantforge.ml import (
    FORBIDDEN_FEATURE_NAMES,
    DispositionPolicy,
    EventDataset,
    EventFeatureDefinition,
    EventFeatureSchema,
    EventFeatureSchemaError,
    EventPartitionError,
    EventPopulation,
    EventPopulationError,
    EventTargetError,
    FeatureValueType,
    ForwardReturnBinaryTarget,
    MissingValuePolicy,
    assemble_event_dataset,
    build_event_dataset,
)
from quantforge.ml.sources import (
    EventSourceWindow,
    HoldoutIsolation,
    load_study_event_sources,
    universe_candidate,
)
from quantforge.prediction.window_compact import CompactPredictionWindowDecision
from quantforge.prediction.window_coverage import PredictionWindowObservation
from quantforge.prediction.window_encoding import mapping
from quantforge.timeframes import IntradayInterval, Timeframe
from quantforge.validation import PartitionRole
from tests.integration.event_dataset_fixtures import (
    EventStudy,
    event_inputs,
    run_event_study,
)


@pytest.fixture(scope="module", autouse=True)
def prepared() -> Iterator[None]:
    """One QF-65 load session for direct source loads (operational only)."""
    with canonical_preparation():
        yield


@pytest.fixture(scope="module")
def study(tmp_path_factory: pytest.TempPathFactory, prepared: None) -> EventStudy:
    del prepared
    root = tmp_path_factory.mktemp("qf67-event-study")
    return run_event_study(root, event_inputs(root), schema="4")


@pytest.fixture(scope="module")
def dataset(study: EventStudy) -> EventDataset:
    return study.build()


def persisted_observations(
    study: EventStudy, pair: str
) -> Iterator[tuple[PartitionRole, PredictionWindowObservation]]:
    sources = load_study_event_sources(
        study.config.plan,
        study.study_path,
        combination_id=study.combination(pair),
        roles=frozenset({PartitionRole.SELECTION, PartitionRole.WALK_FORWARD_TEST}),
    )
    for source in sources.windows:
        for observation in source.reader.iterate_observations():
            yield source.role, observation


def assemble(
    study: EventStudy,
    pair: str,
    *,
    population_pair: str | None = None,
    policy: DispositionPolicy = DispositionPolicy.ACCEPTED_ONLY,
    windows: Sequence[EventSourceWindow] | None = None,
) -> EventDataset:
    """Assemble already-verified sources directly (bypassing the ledger)."""
    plan = study.config.plan
    sources = load_study_event_sources(
        plan,
        study.study_path,
        combination_id=study.combination(pair),
        roles=frozenset({PartitionRole.SELECTION, PartitionRole.WALK_FORWARD_TEST}),
    )
    population = EventPopulation.capture(
        plan,
        universe_candidate(sources.source, study.combination(population_pair or pair)),
        policy,
    )
    return assemble_event_dataset(
        sources.windows if windows is None else windows,
        plan=plan,
        population=population,
        feature_schema=ema_event_feature_schema(),
        target=ForwardReturnBinaryTarget(),
        isolation=HoldoutIsolation.capture(plan, (), population.symbol),
        exclusions=sources.exclusions,
        study_id=sources.source.study_id,
    )


def test_schema4_triggers_become_rows_and_no_trigger_receipts_stay_coverage(
    study: EventStudy, dataset: EventDataset, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert study.selected_pair == "8/48"
    plan = cast(PrimitiveMapping, dataset.scientific.to_primitive()["partition_plan"])
    sources = cast(list[PrimitiveMapping], plan["sources"])
    assert [(s["role"], s["decision_statuses"], s["rows"]) for s in sources] == [
        ("validation_selection", {"evaluated": 3, "no_prediction": 372}, 3),
        ("walk_forward_test", {"evaluated": 1, "no_prediction": 194}, 1),
    ]
    assert dataset.row_count == 4
    excluded = cast(list[PrimitiveMapping], plan["excluded_sources"])
    assert [item["reason"] for item in excluded] == [
        "role_not_executed_by_qf39_selection"
    ]
    # Count rich objects during assembly of already-verified sources: only the
    # four evaluated decisions are expanded (checked, then given their
    # catalogue view); the 566 no-trigger receipts never become rich objects.
    constructed: list[int] = []
    observed: list[int] = []
    validate = CompactPredictionWindowDecision.__post_init__
    original = PredictionWindowObservation.__init__

    def counted(self: CompactPredictionWindowDecision) -> None:
        constructed.append(1)
        validate(self)

    def counted_observation(
        self: PredictionWindowObservation, *args: object, **kwargs: object
    ) -> None:
        observed.append(1)
        original(self, *args, **kwargs)  # pyright: ignore[reportArgumentType]

    sources_loaded = load_study_event_sources(
        study.config.plan,
        study.study_path,
        combination_id=study.combination("8/48"),
        roles=frozenset({PartitionRole.SELECTION, PartitionRole.WALK_FORWARD_TEST}),
    )
    monkeypatch.setattr(CompactPredictionWindowDecision, "__post_init__", counted)
    monkeypatch.setattr(PredictionWindowObservation, "__init__", counted_observation)
    rebuilt = assemble_event_dataset(
        sources_loaded.windows,
        plan=study.config.plan,
        population=EventPopulation.capture(
            study.config.plan,
            universe_candidate(sources_loaded.source, study.combination("8/48")),
            DispositionPolicy.ACCEPTED_ONLY,
        ),
        feature_schema=ema_event_feature_schema(),
        target=ForwardReturnBinaryTarget(),
        isolation=HoldoutIsolation.capture(study.config.plan, (), "SPY"),
        exclusions=sources_loaded.exclusions,
        study_id=sources_loaded.source.study_id,
    )
    monkeypatch.undo()
    assert len(constructed) == 2 * 4
    assert len(observed) == 4
    # Same rows; identity differs only because fewer roles were requested
    # (the unexecuted development role is then not a recorded exclusion).
    assert rebuilt.rows == dataset.rows
    assert rebuilt.dataset_id != dataset.dataset_id


def test_features_are_exact_persisted_causal_values_in_schema_order(
    study: EventStudy, dataset: EventDataset
) -> None:
    assert dataset.feature_columns == tuple(name for name, _, _ in EMA_EVENT_FEATURES)
    persisted = [
        mapping(observation.signal()["features"])
        for _, observation in persisted_observations(study, "8/48")
    ]
    assert len(persisted) == dataset.row_count
    for name, source, _ in EMA_EVENT_FEATURES:
        assert dataset.feature_values(name) == tuple(item[source] for item in persisted)
    numeric = dataset.numeric_feature_rows()
    assert numeric == tuple(
        tuple(float(Decimal(cast(str, value))) for value in row.features)
        for row in dataset.rows
    )
    assert all(len(row) == 6 for row in numeric)


def test_target_semantics_keep_unavailable_labels_separate_from_negatives(
    dataset: EventDataset,
) -> None:
    target = dataset.target
    assert target.name == "forward_return_30m_positive"
    # 12-23 drifts up; the 12-24 13:00 early close overflows the 30m target;
    # 12-26 is exactly flat (zero is a negative); the 12-27 test drifts down.
    assert target.values == (True, None, False, False)
    assert target.statuses == (
        "available",
        "session_overflow",
        "available",
        "available",
    )
    assert target.source_values[1:3] == (None, "0")
    assert Decimal(cast(str, target.source_values[0])) > 0
    assert Decimal(cast(str, target.source_values[3])) < 0
    summary = cast(PrimitiveMapping, dataset.summaries.to_primitive()["overall"])
    assert summary == {
        "rows": 4,
        "positive": 1,
        "negative": 2,
        "unavailable": 1,
        "unavailable_by_status": {"session_overflow": 1},
    }


def test_partition_membership_is_verified_and_excluded_from_features(
    dataset: EventDataset,
) -> None:
    membership = dataset.partition_membership
    assert [(m.row_index, m.role) for m in membership] == [
        (0, PartitionRole.SELECTION),
        (1, PartitionRole.SELECTION),
        (2, PartitionRole.SELECTION),
        (3, PartitionRole.WALK_FORWARD_TEST),
    ]
    assert {m.fold_index for m in membership} == {0}
    assert dataset.rows_for(PartitionRole.SELECTION) == (0, 1, 2)
    assert dataset.rows_for(PartitionRole.FINAL_HOLDOUT) == ()
    layout = cast(PrimitiveMapping, dataset.scientific.to_primitive()["columns"])
    groups = {
        cast(str, item["name"]): item["group"]
        for item in cast(list[PrimitiveMapping], layout["columns"])
    }
    assert [name for name, group in groups.items() if group == "feature"] == list(
        dataset.feature_columns
    )
    assert not set(dataset.feature_columns) & FORBIDDEN_FEATURE_NAMES
    for name in ("partition_role", "fold_id", "target", "decision_timestamp"):
        assert groups[name] != "feature"


def test_every_persisted_outcome_provenance_and_partition_field_is_refused(
    study: EventStudy,
) -> None:
    _, observation = next(persisted_observations(study, "8/48"))
    row = observation.row()
    assert row is not None
    outcome, evaluation = mapping(row["outcome"]), mapping(row["evaluation"])
    future_fields = {
        *row,
        *outcome,
        *evaluation,
        *mapping(outcome["values"]),
        *mapping(evaluation["values"]),
        *mapping(outcome["temporal_resolution"]),
        "fold_id",
        "partition_role",
        "selection_id",
        "holdout_id",
        "experiment_id",
        "result_id",
        "lineage_id",
        "window_result_id",
    }
    for name in sorted(future_fields):
        for field in ("name", "source_field"):
            values = {"name": "safe_name", "source_field": "current_fast", field: name}
            with pytest.raises(EventFeatureSchemaError):
                EventFeatureDefinition(
                    name=values["name"],
                    source_field=values["source_field"],
                    value_type=FeatureValueType.DECIMAL,
                    missing_values=MissingValuePolicy.REJECT,
                    unit="price",
                    description="attempted future or provenance field",
                )


def test_population_is_bound_and_out_of_population_sources_fail(
    study: EventStudy, dataset: EventDataset
) -> None:
    # A non-selected pair has selection rows only; its OOS absence is recorded.
    other = study.build(pair="8/40")
    assert {row.partition_role for row in other.rows} == {PartitionRole.SELECTION}
    excluded = cast(
        list[PrimitiveMapping],
        cast(PrimitiveMapping, other.scientific.to_primitive()["partition_plan"])[
            "excluded_sources"
        ],
    )
    assert [(item["role"], item["reason"]) for item in excluded] == [
        ("development_training", "role_not_executed_by_qf39_selection"),
        ("walk_forward_test", "frozen_selection_is_another_candidate"),
    ]
    assert other.dataset_id != dataset.dataset_id
    population = cast(PrimitiveMapping, dataset.scientific.to_primitive()["population"])
    assert cast(PrimitiveMapping, population["strategy"])["parameters"] == {
        "daily": 50,
        "fast": 8,
        "slow": 48,
    }
    # 8/40 windows assembled under the 8/48 population: refused.
    with pytest.raises(EventPopulationError):
        assemble(study, "8/40", population_pair="8/48")
    with pytest.raises(EventPopulationError, match="universe"):
        build_event_dataset(
            plan=study.config.plan,
            study_path=study.study_path,
            workspace=study.workspace(),
            combination_id="not-a-candidate",
            feature_schema=ema_event_feature_schema(),
            target=ForwardReturnBinaryTarget(),
        )


def test_duplicate_sources_and_observations_fail_closed(study: EventStudy) -> None:
    sources = load_study_event_sources(
        study.config.plan,
        study.study_path,
        combination_id=study.combination("8/48"),
        roles=frozenset({PartitionRole.SELECTION}),
    )
    with pytest.raises(EventPartitionError, match="duplicate"):
        assemble(study, "8/48", windows=(*sources.windows, *sources.windows))


def test_feature_schema_and_target_mismatches_fail_closed(study: EventStudy) -> None:
    base = ema_event_feature_schema()
    missing = EventFeatureSchema(
        "mismatch",
        "1",
        (
            *base.features,
            EventFeatureDefinition(
                name="previous_close",
                source_field="previous_close",
                value_type=FeatureValueType.DECIMAL,
                missing_values=MissingValuePolicy.PRESERVE_NULL,
                unit="price",
                description="not persisted by this rule",
            ),
        ),
    )
    with pytest.raises(EventFeatureSchemaError, match="does not match"):
        study.build(feature_schema=missing)
    wrong_type = EventFeatureSchema(
        "wrong-type",
        "1",
        tuple(
            EventFeatureDefinition(
                name=item.name,
                source_field=item.source_field,
                value_type=FeatureValueType.INTEGER,
                missing_values=item.missing_values,
                unit=item.unit,
                description=item.description,
            )
            for item in base.features
        ),
    )
    with pytest.raises(EventFeatureSchemaError, match="integer"):
        study.build(feature_schema=wrong_type)
    five_minutes = Timeframe.us_equity(IntradayInterval(timedelta(minutes=5)))
    undeclared = EventFeatureSchema(
        "undeclared-timeframe",
        "1",
        (
            EventFeatureDefinition(
                name="ema_fast",
                source_field="current_fast",
                value_type=FeatureValueType.DECIMAL,
                missing_values=MissingValuePolicy.REJECT,
                unit="price",
                description="wrong source timeframe",
                timeframe=five_minutes,
            ),
        ),
    )
    with pytest.raises(EventFeatureSchemaError, match="timeframe"):
        study.build(feature_schema=undeclared)
    with pytest.raises(EventTargetError, match="horizon"):
        study.build(target=ForwardReturnBinaryTarget(horizon=timedelta(minutes=10)))


def test_rebuilds_and_source_order_produce_identical_identity(
    study: EventStudy, dataset: EventDataset
) -> None:
    again = study.build()
    assert again == dataset
    assert again.dataset_id == dataset.dataset_id
    sources = load_study_event_sources(
        study.config.plan,
        study.study_path,
        combination_id=study.combination("8/48"),
        roles=frozenset({PartitionRole.SELECTION, PartitionRole.WALK_FORWARD_TEST}),
    )
    reordered = assemble(study, "8/48", windows=tuple(reversed(sources.windows)))
    forward = assemble(study, "8/48")
    assert reordered.dataset_id == forward.dataset_id
    assert reordered.rows == forward.rows
    keys = [row.sort_key() for row in dataset.rows]
    assert keys == sorted(keys)
    stamps = [row.decision_timestamp for row in dataset.rows]
    assert stamps == sorted(stamps)
    # The disposition policy is part of population identity.
    everything = assemble(study, "8/48", policy=DispositionPolicy.ALL_GENERATED_SIGNALS)
    assert everything.rows == forward.rows
    assert everything.dataset_id != forward.dataset_id


def test_physical_window_schema_does_not_change_scientific_identity(
    study: EventStudy, dataset: EventDataset
) -> None:
    compact = run_event_study(study.root / "schema-2", study.inputs, schema="2")
    assert compact.study_path.name != study.study_path.name
    legacy = compact.build(workspace=study.workspace())
    assert legacy.rows == dataset.rows
    assert legacy.scientific == dataset.scientific
    assert legacy.dataset_id == dataset.dataset_id
    assert legacy.provenance != dataset.provenance
