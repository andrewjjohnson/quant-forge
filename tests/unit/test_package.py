"""Tests for the top-level QuantForge package."""

import importlib
import logging
import sys

import pytest


def test_package_exposes_installed_version_without_side_effects(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Importing the package is quiet and does not configure global logging."""
    sys.modules.pop("quantforge", None)
    root_handlers = tuple(logging.getLogger().handlers)

    quantforge = importlib.import_module("quantforge")

    captured = capsys.readouterr()
    assert quantforge.__version__
    assert captured.out == ""
    assert captured.err == ""
    assert tuple(logging.getLogger().handlers) == root_handlers
