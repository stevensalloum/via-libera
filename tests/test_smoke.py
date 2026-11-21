"""Smoke test: the package and its subpackages import."""

import importlib

import pytest

import tabpfn_spatial


def test_version_is_set() -> None:
    assert tabpfn_spatial.__version__ == "0.1.0"


@pytest.mark.parametrize("name", ["data", "models", "demo"])
def test_subpackages_import(name: str) -> None:
    module = importlib.import_module(f"tabpfn_spatial.{name}")
    assert module.__doc__
