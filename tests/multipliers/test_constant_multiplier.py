import torch

from cooper import multipliers


def test_constant_multiplier_init_and_forward(init_tensor):
    multiplier = multipliers.ConstantMultiplier(init_tensor)
    assert torch.allclose(multiplier(), init_tensor)
