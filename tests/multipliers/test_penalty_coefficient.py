import torch

from cooper import multipliers


def test_penalty_coefficient_init_and_forward(init_tensor):
    penalty_coefficient = multipliers.DensePenaltyCoefficient(init_tensor)
    assert torch.allclose(penalty_coefficient(), init_tensor)
