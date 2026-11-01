"""verify repeatable Python/NumPy/torch draws and optional torch import."""

import builtins
import random

import numpy as np
import pytest
import torch

from cgcr.utils.seeding import set_seed


@pytest.mark.parametrize("seed", [0, 42, 2**32 - 1])
def test_same_seed_repeats_python_numpy_and_torch(seed):
    set_seed(seed)
    python_draw = [random.random() for _ in range(5)]
    numpy_draw = np.random.random(5)
    torch_draw = torch.rand(5)
    set_seed(seed)
    assert [random.random() for _ in range(5)] == python_draw
    np.testing.assert_array_equal(np.random.random(5), numpy_draw)
    assert torch.equal(torch.rand(5), torch_draw)


def test_seeding_works_without_torch(monkeypatch):
    original_import = builtins.__import__

    def without_torch(name, *args, **kwargs):
        if name == "torch":
            raise ModuleNotFoundError("No module named torch", name="torch")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_torch)
    set_seed(42)
    expected = np.random.random(3)
    set_seed(42)
    np.testing.assert_array_equal(np.random.random(3), expected)


def test_broken_torch_dependency_is_not_silently_ignored(monkeypatch):
    original_import = builtins.__import__

    def broken_torch(name, *args, **kwargs):
        if name == "torch":
            raise ModuleNotFoundError("Missing torch dependency", name="torch_dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_torch)
    with pytest.raises(ModuleNotFoundError, match="Missing torch dependency"):
        set_seed(42)
