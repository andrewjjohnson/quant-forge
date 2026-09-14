"""Configuration turnover is descriptive and uses only already-frozen records."""

from dataclasses import replace

from quantforge.configuration import PrimitiveMappingSnapshot
from quantforge.oos import configuration_stability
from quantforge.oos._records import mapping, records
from quantforge.walk_forward.models import FrozenSelection

from .conftest import CompletedStudy


def test_repeated_selection_and_preserved_parameters(
    completed_study: CompletedStudy,
) -> None:
    source = completed_study.source
    # Use a repeated logical candidate, while preserving different fold freezes.
    first, second = source.folds
    assert first.selection is not None
    assert second.selection is not None
    snapshot = second.selection.snapshot.to_primitive()
    snapshot["candidate"] = first.selection.snapshot.to_primitive()["candidate"]
    second = replace(
        second, selection=FrozenSelection(PrimitiveMappingSnapshot.capture(snapshot))
    )
    summary = configuration_stability(replace(source, folds=(first, second)))
    assert summary.comparable_transitions == 1
    assert summary.repeat_selections == 1
    assert summary.configuration_changes == 0
    assert summary.to_primitive()["repeat_selection_frequency"] == "1"
    assert all(
        value == 0 for value in summary.parameter_changes.to_primitive().values()
    )
    assert summary == configuration_stability(replace(source, folds=(first, second)))


def test_changed_configuration_parameter_counts_and_missing_gap(
    completed_study: CompletedStudy,
) -> None:
    source = completed_study.source
    first, second = source.folds
    assert first.selection is not None
    assert second.selection is not None
    definition = source.definition.to_primitive()
    universe = mapping(mapping(definition["adapter"])["universe"])
    first_candidate = first.selection.snapshot.to_primitive()["candidate"]
    alternate = next(c for c in records(universe["candidates"]) if c != first_candidate)
    snapshot = second.selection.snapshot.to_primitive()
    snapshot["candidate"] = alternate
    second = replace(
        second, selection=FrozenSelection(PrimitiveMappingSnapshot.capture(snapshot))
    )
    summary = configuration_stability(replace(source, folds=(first, second)))
    assert summary.configuration_changes == 1
    assert summary.repeat_selections == 0
    assert summary.to_primitive()["configuration_change_frequency"] == "1"
    assert any(
        value == 1 for value in summary.parameter_changes.to_primitive().values()
    )
    assert summary.windows[1].to_primitive()["candidate"] == alternate
    gap = configuration_stability(
        replace(source, folds=(replace(first, selection=None), second))
    )
    assert gap.comparable_transitions == 0
    assert gap.to_primitive()["configuration_change_frequency"] is None
