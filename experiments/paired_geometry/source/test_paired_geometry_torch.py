"""Paired-geometry tensor-loading contracts using synthetic PyTorch tensors."""

from __future__ import annotations

import pickle

import pytest
import torch
from paired_geometry import load_tensor_tree

EXECUTED: list[str] = []


def _record_execution() -> None:
    EXECUTED.append("unsafe")


class UnsafePayload:
    def __reduce__(self):
        return (_record_execution, ())


def test_tensor_loader_refuses_arbitrary_pickled_objects(tmp_path) -> None:
    path = tmp_path / "synthetic_unsafe.pt"
    torch.save({"persona": UnsafePayload()}, path)
    with pytest.raises(pickle.UnpicklingError):
        load_tensor_tree(path.read_bytes())
    assert EXECUTED == []
    safe = tmp_path / "synthetic_safe.pt"
    torch.save({"persona": {"x": {"pos": {26: torch.ones(2, 3)}, "neg": {26: torch.zeros(2, 3)}}}}, safe)
    assert load_tensor_tree(safe.read_bytes())["persona"]["x"]["neg"][26].shape == (2, 3)
