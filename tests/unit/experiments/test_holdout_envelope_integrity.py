"""A refreshed ledger reference cannot bless an incompatible result envelope."""

from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest

from quantforge.configuration import Primitive, PrimitiveMapping
from quantforge.experiments import (
    ArtifactType,
    ManifestError,
    inspect_validation,
    verify_artifacts,
)
from quantforge.oos import HoldoutEvaluation, HoldoutLedger
from quantforge.walk_forward.persistence import read_record, write_record
from tests.unit.experiments.test_adapters import block_research
from tests.unit.oos.conftest import CompletedStudy, complete_study

type ConsumedFixture = tuple[CompletedStudy, HoldoutLedger, Path, PrimitiveMapping]


@pytest.fixture(scope="module", params=[False, True], ids=["backtest", "prediction"])
def consumed_result(
    tmp_path_factory: pytest.TempPathFactory, request: pytest.FixtureRequest
) -> ConsumedFixture:
    root = tmp_path_factory.mktemp("holdout-envelope")
    completed = complete_study(root, prediction=bool(request.param))
    source = completed.source
    ledger = HoldoutLedger.create(root / "ledger")
    ledger.reserve(source)
    consumed = ledger.consume(
        HoldoutEvaluation.prepare(
            source, completed.evaluator, selection_fold_id=source.folds[-1].fold_id
        ),
        run_id="envelope-holdout",
    )
    assert consumed.result_reference is not None
    path = ledger.root / cast(str, consumed.result_reference.to_primitive()["path"])
    return completed, ledger, path, read_record(path)


@pytest.mark.parametrize("field", ["schema_version", "kind", "state"])
@pytest.mark.parametrize(
    "invalid",
    ["missing", None, "", " ", "future", "reserved_unconsumed", 1, True, [], {}],
)
def test_consumed_result_requires_supported_envelope_constants(
    consumed_result: ConsumedFixture,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    invalid: Primitive,
) -> None:
    completed, ledger, path, native = consumed_result
    original_bytes = path.read_bytes()
    original_state = ledger.state(completed.source)
    result = deepcopy(native)
    if invalid == "missing":
        result.pop(field)
    else:
        result[field] = invalid
    write_record(path, result)
    edited_bytes = path.read_bytes()
    try:
        current = ledger.state(completed.source)
        assert current.state.value == "consumed"
        assert current.consumption == original_state.consumption
        assert current.result_reference != original_state.result_reference
        block_research(monkeypatch)
        with pytest.raises(ManifestError, match="holdout result envelope"):
            inspect_validation(
                completed.source,
                completed.study.study_path,
                artifact_root=ledger.root.parent,
                ledger=ledger,
            )
        assert path.read_bytes() == edited_bytes
        assert ledger.state(completed.source) == current
    finally:
        path.write_bytes(original_bytes)
    assert ledger.state(completed.source) == original_state


def test_native_consumed_result_envelope_remains_indexable(
    consumed_result: ConsumedFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    completed, ledger, path, _ = consumed_result
    before = path.read_bytes()
    current = ledger.state(completed.source)
    block_research(monkeypatch)
    bundle = inspect_validation(
        completed.source,
        completed.study.study_path,
        artifact_root=ledger.root.parent,
        ledger=ledger,
    )
    entries = [
        entry
        for entry in bundle.index.entries
        if entry.artifact_type is ArtifactType.HOLDOUT_RESULT
    ]
    assert len(entries) == 1
    assert entries[0].schema_version == "1"
    assert verify_artifacts(bundle.index, ledger.root.parent).valid
    assert path.read_bytes() == before
    assert ledger.state(completed.source) == current
