import subprocess
import sys
from pathlib import Path

import pytest

from tests.unit.experiments.test_environment import (
    repository,  # noqa: F401  # pyright: ignore[reportUnusedImport]
)


@pytest.mark.parametrize(
    "module_name",
    [
        "quantforge.experiments",
        "quantforge.experiments.environment",
        "quantforge.experiments.persistence",
        "quantforge.experiments.adapters",
        "quantforge.experiments.validation",
    ],
)
def test_metadata_operations_do_not_import_research_or_numerical_backends(
    repository: Path,  # noqa: F811
    tmp_path: Path,
    module_name: str,
) -> None:
    # A fresh interpreter is essential: the main pytest process has already
    # imported producer fixtures and would hide eager import side effects.
    script = """
import importlib
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

forbidden = {
    'numpy', 'talib', 'pyarrow', 'quantforge.prediction',
    'quantforge.indicators', 'quantforge.backtesting', 'quantforge.optimization',
    'quantforge.oos', 'quantforge.walk_forward',
}
class RejectResearchImports:
    def find_spec(self, fullname, path=None, target=None):
        if any(
            fullname == name or fullname.startswith(name + '.') for name in forbidden
        ):
            raise AssertionError('metadata operation imported ' + fullname)

sys.meta_path.insert(0, RejectResearchImports())
importlib.import_module(sys.argv[1])
import quantforge.experiments as api
for name in api.__all__:
    getattr(api, name)
from quantforge.configuration import PrimitiveMappingSnapshot
from quantforge.experiments.adapters import (
    StudyArtifacts, create_manifest, inspect_study,
)
from quantforge.experiments.validation import inspect_validation
assert api.StudyArtifacts is StudyArtifacts
assert api.create_manifest is create_manifest
assert api.inspect_study is inspect_study
assert api.inspect_validation is inspect_validation

repository, root = Path(sys.argv[2]), Path(sys.argv[3])
code = api.capture_code_provenance(repository)
assert code.git_dirty is False and code.git_commit
dependencies = code.dependencies.to_primitive()
assert dependencies['numpy'] == version('numpy')
assert dependencies['TA-Lib'] == version('TA-Lib')
(root / 'result.json').write_text('{"study_id":"source","schema_version":"1"}\\n')
entry = api.index_artifact(
    root, path='result.json', artifact_type=api.ArtifactType.PREDICTION_RESULT,
    schema_version='1', producer_study_id='source', producer_artifact_id='result',
    bindings={'/study_id': 'source', '/schema_version': '1'},
)
study = api.StudyArtifacts(
    api.StudyProvenance(
        api.StudyType.PREDICTION, 'source',
        PrimitiveMappingSnapshot.capture({'rule': 'fixed'}),
    ),
    api.ArtifactIndex((entry,)),
)
execution = api.ExecutionProvenance(
    'execution', datetime(2026, 9, 15, tzinfo=UTC), code,
)
manifest = api.create_manifest(study, execution)
path = api.write_manifest(manifest, root / 'manifests', artifact_root=root)
loaded = api.read_manifest(path, artifact_root=root)
assert loaded.serialize() == manifest.serialize()
assert api.verify_artifacts(loaded.artifacts, root).valid
assert not any(name in sys.modules for name in forbidden)
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            script,
            module_name,
            str(repository),
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
