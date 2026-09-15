"""Bind consumed producer metadata to the exact bytes represented by an index."""

from hashlib import sha256
from pathlib import Path

from quantforge.configuration import PrimitiveMapping
from quantforge.experiments._json import ManifestError
from quantforge.experiments.artifacts import ArtifactIndex, file_sha256, local_path
from quantforge.experiments.persistence import decode_producer_record


class ProducerReadSet:
    """Track one inspection's reads without caching mutable producer records."""

    def __init__(self) -> None:
        self._hashes: dict[Path, str] = {}

    def read(self, path: Path) -> tuple[PrimitiveMapping, str]:
        path = path.resolve()
        try:
            content = path.read_bytes()
        except OSError as error:
            raise ManifestError("cannot read producer metadata") from error
        record = decode_producer_record(content)
        self.expect(path, content)
        return record

    def expect(self, path: Path, content: bytes) -> None:
        """Bind an index entry to consumed or expected native export bytes."""
        self.expect_sha256(path, sha256(content).hexdigest())

    def expect_sha256(self, path: Path, fingerprint: str) -> None:
        """Bind a producer sidecar's file digest to indexed and final bytes."""
        path = path.resolve()
        if self._hashes.setdefault(path, fingerprint) != fingerprint:
            raise ManifestError("producer metadata changed during indexing; retry")

    def verify(self, index: ArtifactIndex, root: Path) -> None:
        """Require every consumed record to match its indexed and current bytes."""
        indexed: set[Path] = set()
        for entry in index.entries:
            path = local_path(root, entry.path).resolve()
            if path in self._hashes:
                if entry.sha256 != self._hashes[path]:
                    raise ManifestError(
                        "producer metadata changed during indexing; retry"
                    )
                indexed.add(path)
        try:
            if indexed != set(self._hashes) or any(
                file_sha256(path) != fingerprint
                for path, fingerprint in self._hashes.items()
            ):
                raise ManifestError("producer metadata changed during indexing; retry")
        except OSError as error:
            raise ManifestError(
                "producer metadata changed during indexing; retry"
            ) from error
