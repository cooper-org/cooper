"""Utilities for writing tests."""

from dataclasses import dataclass

import torch


@dataclass
class SkipTest:
    do_skip: bool
    skip_reason: str = None


def get_device_skip(aim_device, cuda_available):

    device = "cuda" if (aim_device and torch.cuda.is_available()) else "cpu"

    if aim_device == "cuda":
        if cuda_available:
            device = "cuda"
            skip = SkipTest(do_skip=False)
        else:
            # Intended to test GPU execution, but GPU not available. Skipping
            # test.
            device = None
            skip = SkipTest(do_skip=True, skip_reason="CUDA is not available")
    else:
        device = "cpu"
        skip = SkipTest(do_skip=False)

    return device, skip
