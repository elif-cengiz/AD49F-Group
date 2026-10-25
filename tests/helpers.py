"""Shared data for the tests. Set SYNTHETIC = False to run the tests on the cached real data."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_data, split  # noqa: E402

SYNTHETIC = True
_CACHE = {}


def data():
    if "d" not in _CACHE:
        _CACHE["d"] = load_data(synthetic=SYNTHETIC)
    return _CACHE["d"]


def train_test():
    return split(data()[0])
