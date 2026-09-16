"""Stability descriptions must agree with the frozen selection sequence."""

from copy import deepcopy
from dataclasses import replace
from decimal import localcontext
from itertools import pairwise

import pytest

from quantforge.configuration import PrimitiveMapping, PrimitiveMappingSnapshot
from quantforge.experiments import ManifestError
from quantforge.experiments._aggregate_schema import validate_configuration_stability
from quantforge.experiments._json import mapping
from quantforge.oos.common import configuration_stability
from quantforge.walk_forward.models import FoldResult, FrozenSelection
from tests.unit.experiments.test_adapters import block_research
from tests.unit.experiments.test_aggregate_schemas import Captured, inspect_record
from tests.unit.experiments.test_aggregate_schemas import exports as exports


@pytest.mark.parametrize("prediction", [False, True])
@pytest.mark.parametrize(
    "field",
    [
        "swap_transitions",
        "selection_counts",
        "parameter_change_counts",
        "configuration_change_frequency",
        "repeat_selection_frequency",
    ],
)
def test_rehashed_stability_totals_must_match_captured_folds(
    exports: dict[bool, Captured],
    monkeypatch: pytest.MonkeyPatch,
    prediction: bool,
    field: str,
) -> None:
    captured = exports[prediction]
    document = deepcopy(captured[1])
    stability = mapping(document["stability"])
    if field == "swap_transitions":
        stability["configuration_changes"], stability["repeat_selections"] = (
            stability["repeat_selections"],
            stability["configuration_changes"],
        )
        (
            stability["configuration_change_frequency"],
            stability["repeat_selection_frequency"],
        ) = (
            stability["repeat_selection_frequency"],
            stability["configuration_change_frequency"],
        )
    elif field.endswith("counts"):
        stability[field] = {"unrelated": 1}
    else:
        stability[field] = "0.5"
    block_research(monkeypatch)
    with pytest.raises(ManifestError, match="stability"):
        inspect_record(captured, document)


@pytest.mark.parametrize(
    "sequence", [(0, 0, 1, 1), (0, None, 1, None), (0, 1, 2, 3), (None, None), (0,)]
)
def test_stability_count_binding_preserves_adjacency_and_producer_ratios(
    exports: dict[bool, Captured],
    monkeypatch: pytest.MonkeyPatch,
    sequence: tuple[int | None, ...],
) -> None:
    source = exports[True][0].source
    base = source.folds[0]
    assert base.selection is not None
    folds: list[FoldResult] = []
    for index, choice in enumerate(sequence):
        selected = base.selection.snapshot.to_primitive()
        candidate = mapping(selected["candidate"])
        candidate["combination_id"] = str(choice)
        candidate["parameters"] = {
            "period": choice,
            **({"extra": None} if choice == 2 else {}),
        }
        selection = (
            None
            if choice is None
            else FrozenSelection(PrimitiveMappingSnapshot.capture(selected))
        )
        folds.append(replace(base, fold_id=str(index), selection=selection))
    source = replace(source, folds=tuple(folds))
    recorded: PrimitiveMapping = configuration_stability(source).to_primitive()
    block_research(monkeypatch)
    with localcontext() as context:
        context.prec = 3
        validate_configuration_stability(recorded, source)
    assert recorded["comparable_transitions"] == sum(
        a is not None and b is not None for a, b in pairwise(sequence)
    )
