"""Utilities for writing tests."""

import functools
from typing import List, Sequence, Tuple

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


def get_device_or_skip(aim_device: torch.device, cuda_available: bool) -> Tuple[torch.device, bool, str]:
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


def build_params_from_init(init: Sequence[float], device: torch.device) -> List[torch.nn.Parameter]:
    """Builds a list of `torch.nn.Parameter`\\s from a list of initial values."""
    return [torch.nn.Parameter(torch.tensor([elem], device=device, requires_grad=True)) for elem in init]


def mktensor(device=None, dtype=None, requires_grad=False):
    return functools.partial(torch.tensor, device=device, dtype=dtype, requires_grad=requires_grad)
