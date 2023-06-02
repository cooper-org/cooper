import multiplier_test_utils
import pytest
import torch

from cooper import multipliers


def test_penalty_coefficient_init_and_forward(_init_tensor):
    penalty_coefficient = multipliers.DensePenaltyCoefficient(_init_tensor)
    assert torch.allclose(penalty_coefficient(), _init_tensor)
