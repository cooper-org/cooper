#!/usr/bin/env python

"""Tests for Multiplier class."""

import torch
from torch_coop import torch_coop


def test_multipliers_init():

    init_tensor = torch.randn(100, 1)
    multiplier = torch_coop.DenseMultiplier(init_tensor)
    assert torch.allclose(multiplier(), init_tensor)

    init_tensor = torch.tensor([0.0, -1.0, 2.5])
    multiplier = torch_coop.DenseMultiplier(init_tensor)
    assert torch.allclose(multiplier(), init_tensor)

    # For inequality constraints, the multipliers should be non-negative
    init_tensor = torch.tensor([0.1, -1.0, 2.5])
    multiplier = torch_coop.DenseMultiplier(init_tensor, positive=True)
    assert torch.allclose(multiplier(), torch.tensor([0.1, 0.0, 2.5]))
