"""Optional explicit environment capture at execution time, without imports."""

import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from quantforge.configuration import PrimitiveMapping
from quantforge.experiments._json import snapshot
from quantforge.experiments.artifacts import file_sha256
from quantforge.experiments.models import CodeProvenance


def capture_code_provenance(
    repository: Path,
    *,
    dependency_names: tuple[str, ...] = (
        "TA-Lib",
        "numpy",
        "exchange-calendars",
        "pyarrow",
    ),
) -> CodeProvenance:
    """Capture at execution time and retain for later manifest generation.

    Reads package metadata, uv.lock, and git status; no environment variable
    values, provider configuration, diff content, or backend imports are read.
    """

    def installed(name: str) -> str | None:
        try:
            return version(name)
        except PackageNotFoundError:
            return None

    def git(*arguments: str) -> str | None:
        try:
            return subprocess.run(
                ["git", "-C", str(repository), *arguments],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    status = git("status", "--porcelain")
    dependencies: PrimitiveMapping = {
        name: installed(name) for name in dependency_names
    }
    lock = repository / "uv.lock"
    return CodeProvenance(
        installed("quantforge"),
        git("rev-parse", "HEAD"),
        None if status is None else bool(status),
        file_sha256(lock) if lock.is_file() else None,
        platform.python_version(),
        snapshot(dependencies),
    )
