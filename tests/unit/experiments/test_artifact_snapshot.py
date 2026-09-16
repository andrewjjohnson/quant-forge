"""Artifact hashes and JSON contracts must describe the same captured bytes."""

import hashlib
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import IO, Any, cast

import pytest

from quantforge.configuration import PrimitiveMappingSnapshot
from quantforge.experiments import (
    ArtifactIndex,
    ManifestError,
    verify_artifacts,
    write_manifest,
)
from tests.unit.experiments.test_contracts import manifest


@pytest.mark.parametrize("publish", [False, True], ids=["verify", "publish"])
@pytest.mark.parametrize(
    ("original_bytes", "json_pointer", "expected_issue"),
    [
        (b'{"schema_version":"1"}', "", "incompatible_metadata"),
        (b'{"schema_version":"2"}', "/result", "invalid_artifact"),
        (b"{", "", "invalid_artifact"),
        (b'{"schema_version":"2","result":[]}', "/result", None),
    ],
    ids=["binding", "pointer", "invalid-json", "identical-replacement"],
)
def test_json_checks_use_the_bytes_that_were_hashed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    publish: bool,
    original_bytes: bytes,
    json_pointer: str,
    expected_issue: str | None,
) -> None:
    original = manifest(tmp_path)
    path = tmp_path / "result.json"
    path.write_bytes(original_bytes)
    entry = replace(
        original.artifacts.entries[0],
        sha256=hashlib.sha256(original_bytes).hexdigest(),
        json_pointer=json_pointer,
        bindings=PrimitiveMappingSnapshot.capture({"/schema_version": "2"}),
    )
    original = replace(original, artifacts=ArtifactIndex((entry,)))
    replacement_bytes = b'{"schema_version":"2","result":[]}'
    replacement_path = tmp_path / "replacement.json"
    replacement_path.write_bytes(replacement_bytes)
    open_file = Path.open
    swapped = False

    @contextmanager
    def replace_after_read(
        source: Path, *args: Any, **kwargs: Any
    ) -> Generator[IO[Any]]:
        nonlocal swapped
        with cast(IO[Any], open_file(source, *args, **kwargs)) as stream:
            yield stream
        if source == path and not swapped:
            # Replacement happens after the captured file is closed, before a
            # second open could parse a different version of the same path.
            replacement_path.replace(path)
            swapped = True

    monkeypatch.setattr(Path, "open", replace_after_read)
    output_root = tmp_path / "manifests"
    if publish:
        if expected_issue is None:
            assert write_manifest(
                original, output_root, artifact_root=tmp_path
            ).is_file()
        else:
            with pytest.raises(ManifestError, match=expected_issue):
                write_manifest(original, output_root, artifact_root=tmp_path)
            assert not output_root.exists()
    else:
        report = verify_artifacts(original.artifacts, tmp_path)
        assert [issue.code for issue in report.issues] == (
            [] if expected_issue is None else [expected_issue]
        )
    assert swapped
    assert path.read_bytes() == replacement_bytes
    assert entry.sha256 == hashlib.sha256(original_bytes).hexdigest()


def test_hash_mismatch_takes_precedence_over_invalid_json(tmp_path: Path) -> None:
    original = manifest(tmp_path)
    (tmp_path / "result.json").write_bytes(b"{")
    report = verify_artifacts(original.artifacts, tmp_path)
    assert [issue.code for issue in report.issues] == ["content_hash_mismatch"]
