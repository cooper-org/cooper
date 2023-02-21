import pytest
import torch

import cooper
from tests.helpers.cooper_test_utils import Toy2dCMP


def test_cmp():

    params = torch.nn.Parameter(torch.tensor([0.1, 0.5]))
    cmp = Toy2dCMP(use_ineq_constraints=True)

    cmp_state = cmp.closure(params)

    lagrangian, multipliers = cmp_state.populate_lagrangian(return_multipliers=True)
    cmp_state.backward()

    breakpoint()
