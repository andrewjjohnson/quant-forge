"""Repository-wide test tiers, heavy-first dispatch and file sharding (QF-66).

Every collected test belongs to exactly one tier:

* ``heavy_acceptance``: explicitly marked long end-to-end scientific acceptance
  tests, currently the QF-45 pre-holdout runner;
* ``regression``: every other test under ``tests/integration`` or
  ``tests/performance``;
* ``fast``: every remaining test (``tests/unit`` and any new directory).

Tiers and shards only select and order tests; they never skip or weaken one.
``regression`` and ``fast`` are complementary by construction, so the three tier
expressions always cover the complete suite. CI runs every tier as a required
job; see ``docs/development.md``.
"""

import hashlib
from typing import cast

import pytest

HEAVY_ACCEPTANCE = "heavy_acceptance"
REGRESSION = "regression"
FAST = "fast"
_REGRESSION_ROOTS = (("tests", "integration"), ("tests", "performance"))


def tier_for(parts: tuple[str, ...]) -> str:
    """Automatic tier of an unmarked test from its repository-relative path."""
    return REGRESSION if parts[:2] in _REGRESSION_ROOTS else FAST


def shard_for(relative_path: str, count: int) -> int:
    """1-based shard of a repository-relative POSIX test-file path.

    SHA-256 rather than ``hash()`` keeps assignment identical across processes,
    xdist workers and CI jobs, so ``count`` shards always partition all files.
    """
    return int(hashlib.sha256(relative_path.encode()).hexdigest(), 16) % count + 1


def parse_file_shard(value: str) -> tuple[int, int]:
    index, separator, count = value.partition("/")
    if not (separator and index.isdigit() and count.isdigit()):
        raise pytest.UsageError("--file-shard must be INDEX/COUNT, e.g. 1/2")
    if not 1 <= int(index) <= int(count):
        raise pytest.UsageError("--file-shard INDEX must be between 1 and COUNT")
    return int(index), int(count)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--file-shard",
        default=None,
        metavar="INDEX/COUNT",
        help=(
            "run only test files whose stable SHA-256 path hash selects shard "
            "INDEX (1-based) of COUNT; the COUNT shards partition every file"
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    shard = cast(str | None, config.getoption("--file-shard"))
    if shard is not None:
        parse_file_shard(shard)  # Reject a malformed shard before collecting.


def _relative_parts(item: pytest.Item, config: pytest.Config) -> tuple[str, ...]:
    return item.path.relative_to(config.rootpath).parts


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    # Runs before ``-m`` deselection so the automatic tiers are selectable.
    for item in items:
        if item.get_closest_marker(HEAVY_ACCEPTANCE) is None:
            item.add_marker(tier_for(_relative_parts(item, config)))
    shard = cast(str | None, config.getoption("--file-shard"))
    if shard is not None:
        index, count = parse_file_shard(shard)
        selected: list[pytest.Item] = []
        deselected: list[pytest.Item] = []
        for item in items:
            # Whole files stay together, so module fixtures are never split.
            path = "/".join(_relative_parts(item, config))
            (selected if shard_for(path, count) == index else deselected).append(item)
        if deselected:
            config.hook.pytest_deselected(items=deselected)
            items[:] = selected
    # xdist ``--dist=loadfile --no-loadscope-reorder`` dispatches files in this
    # order. Starting the longest single-test files first keeps them off the end
    # of the critical path; the sort is stable, so all other order is unchanged.
    items.sort(key=lambda item: item.get_closest_marker(HEAVY_ACCEPTANCE) is None)
