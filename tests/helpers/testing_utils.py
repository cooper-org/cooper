"""Utilities for writing tests."""

import functools
from collections.abc import Collection
from typing import Any, Sequence

import torch


def frozen_rand_generator(seed=2147483647):
    """Creates a pseudo random number generator object with a fixed seed for
    reproducible tests.
    """

    # TODO(juan43ramirez): Random number generator producing error when using cuda.
    device = "cpu"
    generator = torch.Generator(device)
    generator.manual_seed(seed)
    return generator


def get_device_or_skip(aim_device: torch.device, cuda_available: bool) -> tuple[torch.device, bool, str]:
    """Verifies availability of a GPU and sets a flag to skip a test if GPU execution
    was requested, but no GPU was available.
    """

    device = torch.device("cuda") if (aim_device == "cuda" and cuda_available) else torch.device("cpu")

    skip, skip_reason = False, None

    if aim_device == "cuda" and not cuda_available:
        skip = True
        skip_reason = "Aim device 'cuda' is not available."
    elif aim_device not in ["cuda", "cpu"]:
        raise ValueError(f"aim_device = {aim_device} not understood.")

    return device, skip, skip_reason


def build_params_from_init(init: Sequence[float], device: torch.device) -> list[torch.nn.Parameter]:
    """Builds a list of `torch.nn.Parameter`\\s from a list of initial values."""
    return [torch.nn.Parameter(torch.tensor([elem], device=device, requires_grad=True)) for elem in init]


def mktensor(device=None, dtype=None, requires_grad=False):
    return functools.partial(torch.tensor, device=device, dtype=dtype, requires_grad=requires_grad)


def compare_values(val1: Any, val2: Any) -> bool:
    """
    Compares whether two objects match. Can be applied to tensors, iterables, or dicts.
    """

    if isinstance(val1, torch.Tensor) and isinstance(val2, torch.Tensor):
        # convert both to the same CUDA device
        if str(val1.device) != "cuda:0":
            val1 = val1.to("cuda:0" if torch.cuda.is_available() else "cpu")
        if str(val2.device) != "cuda:0":
            val2 = val2.to("cuda:0" if torch.cuda.is_available() else "cpu")

        return torch.allclose(val1, val2)

    if isinstance(val1, dict) and isinstance(val2, dict):
        if val1.keys() != val2.keys():
            return False
        else:
            return all([compare_values(val1[k], val2[k]) for k in val1.keys()])

    if isinstance(val1, Collection) and isinstance(val2, Collection):
        if len(val1) != len(val2):
            return False
        return all([compare_values(ii, jj) for ii, jj in zip(val1, val2)])

    else:
        return val1 == val2


def validate_state_dicts(model_state_dict_1: dict, model_state_dict_2: dict) -> bool:
    """
    Verifies whether two state_dicts match.
    """
    # Edited from: https://gist.github.com/rohan-varma/a0a75e9a0fbe9ccc7420b04bff4a7212

    if model_state_dict_1 is None and model_state_dict_2 is None:
        return True

    if model_state_dict_1 == {} or model_state_dict_2 == {}:
        return (model_state_dict_1 == {}) and (model_state_dict_2 == {})

    if len(model_state_dict_1) != len(model_state_dict_2):
        print(f"Length mismatch: {len(model_state_dict_1)}, {len(model_state_dict_2)}")
        return False

    if isinstance(model_state_dict_1, list) and isinstance(model_state_dict_2, list):
        zipped_dicts = zip(model_state_dict_1, model_state_dict_2)
        is_each_valid = [validate_state_dicts(ii, jj) for ii, jj in zipped_dicts]
        return all(is_each_valid)

    # Replicate modules have "module" attached to their keys, so strip these off when comparing to local model.
    if next(iter(model_state_dict_1.keys())).startswith("module"):
        model_state_dict_1 = {k[len("module") + 1 :]: v for k, v in model_state_dict_1.items()}

    if next(iter(model_state_dict_2.keys())).startswith("module"):
        model_state_dict_2 = {k[len("module") + 1 :]: v for k, v in model_state_dict_2.items()}

    for ((k_1, val1), (k_2, val2)) in zip(model_state_dict_1.items(), model_state_dict_2.items()):
        if k_1 != k_2:
            print(f"Key mismatch: {k_1} vs {k_2}")
            return False

        if not compare_values(val1, val2):
            print(f"Attribute mismatch: {val1} vs {val2}")
            return False

    return True
