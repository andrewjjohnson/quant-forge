"""QF-66 tiers and CI shards may select tests but never drop one."""

from pathlib import Path

import pytest

from tests.conftest import FAST, REGRESSION, parse_file_shard, shard_for, tier_for

ROOT = Path(__file__).resolve().parents[2]
TEST_FILES = tuple(
    sorted(
        path.relative_to(ROOT).as_posix() for path in ROOT.glob("tests/**/test_*.py")
    )
)


@pytest.mark.parametrize(
    ("path", "tier"),
    [
        ("tests/integration/test_spy_ema_compact.py", REGRESSION),
        ("tests/performance/test_prepared_features.py", REGRESSION),
        ("tests/unit/test_spy_ema_smoke.py", FAST),
        ("tests/unit/prediction/test_prepared_features.py", FAST),
        # A new directory is still selected by a required tier.
        ("tests/property/test_invariants.py", FAST),
    ],
)
def test_every_unmarked_path_has_one_tier(path: str, tier: str) -> None:
    assert tier_for(tuple(path.split("/"))) == tier


@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_shards_partition_every_test_file(count: int) -> None:
    assert len(TEST_FILES) > 200
    shards = [shard_for(path, count) for path in TEST_FILES]
    assert set(shards) == set(range(1, count + 1))
    selected = [
        path
        for index in range(1, count + 1)
        for path, shard in zip(TEST_FILES, shards, strict=True)
        if shard == index
    ]
    assert sorted(selected) == list(TEST_FILES)


def test_shard_assignment_is_stable_across_processes() -> None:
    # SHA-256 path buckets, never randomized ``hash()``: CI jobs must agree.
    assert [
        shard_for("tests/integration/test_spy_ema_runner.py", count)
        for count in (1, 2, 3, 4)
    ] == [1, 2, 2, 4]
    assert [
        shard_for("tests/unit/test_package.py", count) for count in (1, 2, 3, 4)
    ] == [1, 1, 3, 1]


@pytest.mark.parametrize("value", ["2", "1/", "/2", "0/2", "3/2", "a/b", "1/0"])
def test_malformed_file_shard_is_rejected(value: str) -> None:
    with pytest.raises(pytest.UsageError):
        parse_file_shard(value)


def test_file_shard_is_parsed() -> None:
    assert parse_file_shard("2/2") == (2, 2)
