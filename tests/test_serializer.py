"""make_json_serializable handles the types our pipelines emit."""

import json

import numpy as np
import pytest
import torch

# Import after conftest sets ENVIRONMENT=development.
from src.api.app import make_json_serializable


def test_numpy_arrays_become_lists():
    out = make_json_serializable({"arr": np.array([1, 2, 3], dtype=np.int64)})
    assert out == {"arr": [1, 2, 3]}
    json.dumps(out)  # must be serializable end-to-end


def test_numpy_scalars_become_python_primitives():
    out = make_json_serializable({"i": np.int64(7), "f": np.float32(0.5)})
    assert isinstance(out["i"], int) and out["i"] == 7
    assert isinstance(out["f"], float) and out["f"] == pytest.approx(0.5)


def test_torch_tensors_are_detached_and_listed():
    # Tensor on the autograd graph must come out as a plain list of floats
    # so that downstream JSON serialization doesn't see a tensor.
    t = torch.tensor([0.1, 0.9], dtype=torch.float64, requires_grad=True)
    out = make_json_serializable({"probs": t})
    assert isinstance(out["probs"], list)
    assert out["probs"][0] == pytest.approx(0.1)
    assert out["probs"][1] == pytest.approx(0.9)
    json.dumps(out)


def test_nested_structures_recurse():
    out = make_json_serializable(
        {
            "outer": [np.array([1.0, 2.0]), {"inner": np.float64(3.5)}],
            "tuple": (np.int32(4), 5),
        }
    )
    assert out == {"outer": [[1.0, 2.0], {"inner": 3.5}], "tuple": [4, 5]}
