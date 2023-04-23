import torch

from cooper import multipliers


def test_constant_multiplier_init_and_forward(_init_tensor):
    multiplier = multipliers.ConstantMultiplier(_init_tensor)
    assert torch.allclose(multiplier(), _init_tensor)
