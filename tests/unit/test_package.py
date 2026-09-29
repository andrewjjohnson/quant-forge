"""Tests for the top-level QuantForge package."""

import importlib
import logging
import sys

import pytest


def test_package_exposes_installed_version_without_side_effects(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Importing the package is quiet and does not configure global logging."""
    # Restore the original package at teardown. A leaked fresh package lacks its
    # already-imported submodule attributes, which broke later dotted-path
    # monkeypatches on the same worker (QF-66: order-dependent failure).
    monkeypatch.delitem(sys.modules, "quantforge", raising=False)
    root_handlers = tuple(logging.getLogger().handlers)

    quantforge = importlib.import_module("quantforge")

    captured = capsys.readouterr()
    assert quantforge.__version__
    assert captured.out == ""
    assert captured.err == ""
    assert tuple(logging.getLogger().handlers) == root_handlers
